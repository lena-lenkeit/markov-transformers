import json
import os
from functools import partial
from typing import List

import numpy as np
import safetensors.numpy
import safetensors.torch
import torch
from manim import (
    BLUE,
    DEGREES,
    GRAY,
    GREEN,
    ORIGIN,
    RED,
    WHITE,
    Arrow,
    Arrow3D,
    Axes,
    Circle,
    Create,
    DashedLine,
    Dot,
    ManimColor,
    Mobject,
    Polygon,
    Rotating,
    Scene,
    ThreeDAxes,
    ThreeDScene,
    Triangle,
    VGroup,
    VMobject,
    smooth,
)

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


class Fractal(VGroup):
    def __init__(
        self, vectors: np.ndarray, init_scale: float, scale: float, depth: int, **kwargs
    ):
        super().__init__(**kwargs)

        self._vectors = vectors
        self._init_scale = init_scale
        self._scale = scale
        self._depth = depth

        dots: List[Dot] = []
        for i in range(3**depth):
            dot = Dot(radius=0.01)
            dots.append(dot)

        self._dots = dots
        self.add(*dots)

        self._update()

    def set_vectors(self, vectors: np.ndarray):
        self._vectors = vectors
        self._update()

        return self

    def _update(self):
        for i in range(3**self._depth):
            pos = np.zeros(3)
            dot_index = i
            scale = self._init_scale

            for j in range(self._depth):
                vec_index = dot_index % 3
                dot_index //= 3

                pos += self._vectors[vec_index] * scale
                scale *= self._scale
                self._dots[i].move_to(pos)


class BeliefUpdatingScene(Scene):
    def construct(self):
        # axes = Axes()
        # self.add(axes)

        depth = 6
        colors = [RED, GREEN, BLUE]

        simplex = Triangle(color=GRAY).scale(2.5)
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

        """
        def dot_updater(dot: Dot, i: int):
            scale = 0.5
            scale_mult = 0.5
            pos = np.zeros(3)
            for _ in range(depth):
                vec_id = i % 3
                i //= 3

                pos += vectors[vec_id].get_end() * scale
                scale *= scale_mult

            dot.move_to(pos)

        dots = []
        for i in range(3**depth):
            dot = Dot(radius=0.01)
            dot.add_updater(partial(dot_updater, i=i), call_updater=True)
            dots.append(dot)

        self.add(simplex, *vectors, *dots)
        """

        fractal = Fractal(np.asarray(simplex.get_vertices()), 0.5, 0.5, 6)

        def fractal_updater(fractal: Fractal):
            fractal.set_vectors(np.asarray([vec.get_end() for vec in vectors]))

        fractal.add_updater(fractal_updater, call_updater=True)

        self.add(simplex, *vectors, fractal)

        self.play(vectors[0].animate.scale(0.25, about_point=ORIGIN))
        self.play(Rotating(vectors[0], about=ORIGIN, run_time=2.0, rate_func=smooth))
        self.play(vectors[0].animate.scale(1 / 0.25, about_point=ORIGIN))
