import json
import os
from functools import partial
from typing import List, Tuple

import numpy as np
import safetensors.numpy
import safetensors.torch
import torch
from einops import rearrange
from jaxtyping import Float
from manim import (
    BLUE,
    DEGREES,
    GRAY,
    GREEN,
    ORIGIN,
    RED,
    TAU,
    WHITE,
    Arrow,
    Arrow3D,
    Axes,
    Circle,
    Create,
    DashedLine,
    DiGraph,
    Dot,
    ManimColor,
    Mobject,
    PMobject,
    PointCloudDot,
    Polygon,
    Rotating,
    Scene,
    ThreeDAxes,
    ThreeDScene,
    TipableVMobject,
    Triangle,
    VGroup,
    VMobject,
    smooth,
)

from markov import HiddenMarkovModel, circle_matrices, messn_matrices, sample_hmm
from markov.predict.numpy import get_optimal_beliefs
from markov.sequence_model import SequenceModel, SingleHeadFixedAttention

# manim -qh --renderer=opengl scripts/animations.py DemoScene
# manim -qh --renderer=opengl scripts/animations.py BeliefSpaceScene
# manim -pql --renderer=opengl --write_to_movie scripts/manim_animations.py BeliefUpdatingScene


def load_model(model_dir: str):
    # Load config
    with open(os.path.join(model_dir, "config.json"), mode="r") as f:
        config_dict = json.load(f)

    # Initialize model
    model = SequenceModel(
        **config_dict["model"],
        attn_layer=SingleHeadFixedAttention(**config_dict["attn_layer"]),
    )

    # Load parameters into model
    missing, unexpected = safetensors.torch.load_model(
        model, os.path.join(model_dir, "model.safetensors")
    )
    print(missing, unexpected)

    # Load HMM
    hmm_np = safetensors.numpy.load_file(os.path.join(model_dir, "hmm.safetensors"))
    hmm = {
        "transition_matrix": torch.from_numpy(hmm_np["transition_matrix"]).float(),
        "emission_matrix": torch.from_numpy(hmm_np["emission_matrix"]).float(),
    }

    return model, config_dict, hmm


class DemoScene(Scene):
    def construct(self):
        circle = Circle()

        self.play(Create(circle))


class BeliefSpaceScene(ThreeDScene):
    @torch.inference_mode()
    def construct(self):
        # Parameters
        model_dir = "data/mess3/custom"
        colors = [RED, GREEN, BLUE]
        scale = 2.5

        # Load model
        model, config, hmm = load_model(model_dir)

        # Prepare data
        token_seq = torch.arange(config["model"]["num_tokens"] - 1)

        e = model.to_embeddings(token_seq)
        o = model.to_logits.weight

        # Get orthogonal basis for X
        e_svd = torch.linalg.svd(e, full_matrices=False)
        e_basis = e_svd.Vh
        e_projected = e_svd.U

        # Get reading directions for the output logit head
        o_projected = o[:-1] @ e_basis.T

        # Get simplex center
        e_simplex_center = torch.mean(e_projected, dim=0, keepdim=True)

        # Get vectors projected onto simplex
        e_simplex_projected = e_projected - e_simplex_center

        # Add objects
        axes = ThreeDAxes()
        self.add(axes)

        for i, vec in enumerate(e_projected[:3].numpy()):
            arrow = Arrow(
                start=np.asarray([0, 0, 0]), end=vec * scale, buff=0, color=colors[i]
            )
            # arrow = Arrow3D(start=np.asarray([0, 0, 0]), end=vec, resolution=4)
            self.add(arrow)

        for i, vec in enumerate(e_projected[:3].numpy()):
            arrow = Arrow(
                start=e_simplex_center[0] * scale,
                end=vec * scale,
                buff=0,
                color=colors[i],
            )
            self.add(arrow)

        for i, vec in enumerate(o_projected[:3].numpy()):
            line = DashedLine(
                start=np.asarray([0, 0, 0]), end=vec * scale, color=colors[i]
            )
            self.add(line)

        triangle = Polygon(*e_projected[:3].numpy() * scale)
        triangle.set_stroke(WHITE)
        triangle.set_fill(WHITE, opacity=0.25)
        self.add(triangle)

        self.set_camera_orientation(phi=70 * DEGREES, theta=-135 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.2, about="theta")
        self.wait(duration=5)


class Fractal(PMobject):
    def __init__(
        self,
        vectors: np.ndarray,
        depth: int,
        scale: float | Tuple[float, ...],
        decay: float = 1.0,
        show_all: bool = False,
        **kwargs
    ):
        self._vectors = vectors
        self._depth = depth
        self._scale = scale
        self._decay = decay
        self._show_all = show_all

        super().__init__(color=WHITE, **kwargs)

    def set_vectors(self, vectors: np.ndarray):
        self._vectors = vectors
        self._update()

        return self

    def set_scale(self, scale: float | Tuple[float, ...]):
        self._scale = scale
        self._update()

        return self

    def set_decay(self, decay: float):
        self._decay = decay
        self._update()

        return self

    def _update(self):
        if isinstance(self._scale, float):
            scale_list = (self._scale,) * self._depth
        elif isinstance(self._scale, tuple):
            scale_list = self._scale
        else:
            raise TypeError(self._scale)

        if self._show_all:
            dot_index = 0
            for i in range(1, self._depth + 1):
                for j in range(3**i):
                    pos = np.zeros(3)
                    seq_index = j

                    for k in range(i):
                        vec_index = seq_index % 3
                        seq_index //= 3

                        scale = scale_list[k] * self._decay**k
                        pos += self._vectors[vec_index] * scale

                    self.points[dot_index] = pos
                    dot_index += 1
        else:
            for i in range(3**self._depth):
                pos = np.zeros(3)
                seq_index = i

                for j in range(self._depth):
                    vec_index = seq_index % 3
                    seq_index //= 3

                    scale = scale_list[j] * self._decay**j
                    pos += self._vectors[vec_index] * scale

                self.points[i] = pos

    def init_points(self):
        self.reset_points()
        self.generate_points()

    def generate_points(self):
        if self._show_all:
            num_points = 0
            for i in range(self._depth):
                num_points += 3 ** (i + 1)
        else:
            num_points = 3**self._depth

        points = np.zeros((num_points, 3))
        self.add_points(points)
        self._update()


def ternary_to_cartesian(t: Float[np.ndarray, "... 3"]) -> Float[np.ndarray, "... 2"]:
    x = 1 / 2 * (2 * t[..., 1] + t[..., 2]) / (t[..., 0] + t[..., 1] + t[..., 2])
    y = np.sqrt(3) / 2 * t[..., 2] / (t[..., 0] + t[..., 1] + t[..., 2])

    return np.stack((x, y), axis=-1)


class OptimalBeliefSimplex(PMobject):
    def __init__(
        self,
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
        num_samples: int,
        seq_len: int,
        **kwargs
    ):
        self._transition_matrix = transition_matrix
        self._emission_matrix = emission_matrix
        self._num_samples = num_samples
        self._seq_len = seq_len

        super().__init__(color=WHITE, **kwargs)

    def set_transition_matrix(self, transition_matrix: np.ndarray):
        self._transition_matrix = transition_matrix
        self._update()

        return self

    def set_emission_matrix(self, emission_matrix: np.ndarray):
        self._emission_matrix = emission_matrix
        self._update()

        return self

    def set_transition_matrix(self, transition_matrix: np.ndarray):
        self._transition_matrix = transition_matrix
        self._update()

        return self

    def set_transmission_and_emission_matrix(
        self, transition_matrix: np.ndarray, emission_matrix: np.ndarray
    ):
        self._transition_matrix = transition_matrix
        self._emission_matrix = emission_matrix
        self._update()

        return self

    def _update(self):
        hmm = HiddenMarkovModel(self._transition_matrix, self._emission_matrix)
        rng = np.random.default_rng(1234)

        states, outputs = sample_hmm(hmm, self._seq_len, self._num_samples, rng)
        beliefs, probs, loss = get_optimal_beliefs(
            outputs, self._transition_matrix, self._emission_matrix
        )

        """
        angles = np.arange(3) * 2 * np.pi / 3
        vectors = np.stack(
            [np.sin(angles), np.cos(angles), np.zeros_like(angles)], axis=0
        )

        self.points = (
            vectors
            @ rearrange(beliefs, "batch sequence belief -> belief (batch sequence)")
        ).T
        """

        # To simplex
        self.points[:, :2] = ternary_to_cartesian(
            rearrange(beliefs, "batch sequence belief -> (batch sequence) belief")
        )

        # Center
        center = ternary_to_cartesian(np.ones(3) / np.linalg.norm(np.ones(3)))
        self.points[:, :2] -= center[None]

        # Normal Vector: 1, 1, 1
        # Tangent Vector 1: -1, 1, 1
        # Tangent Vector 2: 0, -2, 2
        """
        proj_matrix = np.asarray([[0, -2, 2], [-1, 1, 1]], dtype=np.float64)
        proj_matrix /= np.linalg.norm(proj_matrix, ord=2, axis=1, keepdims=True)

        self.points[:, :2] = (
            proj_matrix
            @ rearrange(beliefs, "batch sequence belief -> belief (batch sequence)")
        ).T
        self.points[:, 1] *= -1
        """

        # Scale
        self.points *= 2.5

    def init_points(self):
        self.reset_points()
        self.generate_points()

    def generate_points(self):
        points = np.zeros((self._num_samples * self._seq_len, 3))
        self.add_points(points)
        self._update()


class BeliefUpdatingScene(Scene):
    def construct(self):
        # axes = Axes()
        # self.add(axes)

        depth = 7
        colors = [RED, GREEN, BLUE]

        simplex = Triangle(color=WHITE).scale(2.5)
        vectors: List[Arrow] = []
        for i in range(3):
            vector = Arrow(
                start=np.asarray([0, 0, 0]),
                end=simplex.get_vertices()[i],
                buff=0.0,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.1,
                color=colors[i],
            )
            vectors.append(vector)

        scale = np.asarray([0.5**i for i in range(depth)])
        scale /= scale.sum()

        fractal = Fractal(
            vectors=np.asarray(simplex.get_vertices()),
            depth=depth,
            scale=tuple(scale.tolist()),
            decay=1.0,
            show_all=True,
        )

        def fractal_updater(fractal: Fractal):
            fractal.set_vectors(np.asarray([vec.get_end() for vec in vectors]))

        fractal.add_updater(fractal_updater, call_updater=True)

        self.add(*vectors, fractal, simplex)

        # Animate single vector scaling and rotation
        self.play(vectors[0].animate.scale(0.25, about_point=ORIGIN))
        self.pause(0.25)
        self.play(Rotating(vectors[0], about=ORIGIN, run_time=2.0, rate_func=smooth))
        self.pause(0.25)
        self.play(vectors[0].animate.scale(1 / 0.25, about_point=ORIGIN))
        self.pause(0.25)

        # Animate movement of all three vectors
        self.play(
            vectors[0]
            .animate.scale(0.25, about_point=ORIGIN)
            .rotate_about_origin(TAU * 0.1),
            vectors[1]
            .animate.scale(0.4, about_point=ORIGIN)
            .rotate_about_origin(TAU * 0.25),
            vectors[2]
            .animate.scale(0.55, about_point=ORIGIN)
            .rotate_about_origin(TAU * -0.2),
        )
        self.pause(0.25)

        self.play(
            *[
                vectors[i].animate.put_start_and_end_on(
                    ORIGIN, simplex.get_vertices()[i]
                )
                for i in range(3)
            ]
        )
        self.pause(0.25)

        # Animate scale
        # self.play(fractal.animate.set_decay(0.1))

        """
        scale = np.ones(depth)
        scale /= scale.sum()
        self.play(fractal.animate.set_scale(tuple(scale.tolist())))

        scale = np.asarray([0.9**i if i % 2 == 0 else 0.6**i for i in range(depth)])
        scale /= scale.sum()
        self.play(fractal.animate.set_scale(tuple(scale.tolist())))
        """


class OptimalBeliefScene(Scene):
    def construct(self):
        transition_matrix, emission_matrix = messn_matrices()
        beliefs = OptimalBeliefSimplex(
            transition_matrix,
            emission_matrix,
            num_samples=1024 * 16,
            seq_len=64,
            stroke_width=0.1,
        )

        """
        angles = np.arange(3) * 2 * np.pi / 3
        vectors = np.stack(
            [np.sin(angles), np.cos(angles), np.zeros_like(angles)], axis=1
        )
        """

        edges = ternary_to_cartesian(np.eye(3))
        center = ternary_to_cartesian(np.ones(3) / np.linalg.norm(np.ones(3)))

        edges = np.concatenate((edges, np.zeros_like(edges[:, :1])), axis=-1)
        center = np.concatenate((center, np.zeros_like(center[:1])), axis=-1)
        vertices = (edges - center[None]) * 2.5

        simplex_boundary = Polygon(*vertices, color=WHITE)

        self.add(beliefs, simplex_boundary)
        self.play(
            beliefs.animate.set_emission_matrix(messn_matrices(alpha=0.5)[1]),
            run_time=2.5,
        )
        # """
        self.pause(0.5)
        self.play(
            beliefs.animate.set_transition_matrix(messn_matrices(x=0.001)[0]),
            run_time=2.5,
        )
        self.pause(0.5)
        self.play(
            beliefs.animate.set_transmission_and_emission_matrix(*circle_matrices()),
            run_time=2.5,
        )
        # """
