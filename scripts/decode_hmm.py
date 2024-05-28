import json
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int
from sklearn.decomposition import PCA
from tqdm.auto import trange
from x_transformers import Decoder, TransformerWrapper

from markov import mess3_matrices
from markov.predict.torch import (
    get_stationary_distribution,
    propagate_values,
    update_posterior,
)
from markov.train import (
    hmm_from_belief_states,
    hmm_from_observations,
    hmm_from_residuals,
)


def l2_normalize(x: torch.Tensor, dim: int, keepdim: bool):
    return x / torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=keepdim)


@torch.inference_mode()
def plot_pca():
    with open("config.json", mode="r") as f:
        config_dict = json.load(f)

    model = TransformerWrapper(
        **config_dict["transformer"],
        attn_layers=Decoder(**config_dict["decoder"]),
    )

    missing, unexpected = safetensors.torch.load_model(model, "model.safetensors")
    print(missing, unexpected)

    eval_data = safetensors.torch.load_file("eval.safetensors")

    embeddings = eval_data["embeddings"]
    embeddings = rearrange(
        embeddings, "batch sequence features -> (batch sequence) features"
    )
    embeddings = embeddings - torch.mean(embeddings, dim=0, keepdim=True)

    to_logits = model.to_logits.weight
    to_logits = to_logits - torch.mean(to_logits, dim=0, keepdim=True)
    u, s, v = torch.svd(to_logits)
    # print(u, s, v)

    logit_components = v[:, :3]

    # PCA dimensionality reduction
    u, s, v = torch.svd(embeddings)
    # print(s)
    residual_components = v[:, :2]
    reduced = embeddings @ v[:, :2]

    print(
        l2_normalize(logit_components, dim=0, keepdim=True).T
        @ l2_normalize(residual_components, dim=0, keepdim=True)
    )

    # print(logit_components)
    # print(residual_components)

    plt.figure()
    plt.scatter(*reduced.T, s=1.0)
    plt.show()


@torch.no_grad()
def inverse_linear(
    y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    x = y - bias
    x = np.linalg.lstsq(weight.numpy(), x.numpy().T)[0].T

    return torch.from_numpy(x)


def main():
    # observations = torch.randint(3, size=(1024 * 16, 16))

    # eval_data = safetensors.torch.load_file("data/eval.safetensors")
    # observations = eval_data["outputs"]

    # transition_matrix, emission_matrix, prior = hmm_from_observations(
    #    observations, dim_state=3, dim_observations=3
    # )
    # print(transition_matrix, emission_matrix, prior)

    """
    hmm = HiddenMarkovModel(*mess3_matrices())

    rng = np.random.default_rng(1234)
    num_samples = 1024 * 1
    seq_len = 256

    init_state = rng.integers(hmm.num_states, size=num_samples)
    init_observation = hmm.sample_output(init_state, rng)

    prior = get_prior(torch.from_numpy(hmm.transition_matrix))
    prior = repeat(prior, "states -> batch states", batch=num_samples)

    states = [init_state]
    outputs = [init_observation]
    beliefs = [prior]

    for i in range(seq_len):
        prior = beliefs[-1]
        posterior = get_posterior(
            prior, torch.from_numpy(outputs[-1]), torch.from_numpy(hmm.output_matrix)
        )
        posterior = propagate_posterior(
            posterior, torch.from_numpy(hmm.transition_matrix)
        )
        beliefs.append(posterior)

        states.append(hmm.next_state(states[-1], rng))
        outputs.append(hmm.sample_output(states[-1], rng))

    states = np.stack(states, axis=1)  # (batch, sequence)
    outputs = np.stack(outputs, axis=1)  # (batch, sequence)
    beliefs = torch.stack(beliefs, dim=1)  # (batch, sequence, states)

    transition_matrix, emission_matrix, prior = hmm_from_belief_states(
        beliefs.float(), torch.from_numpy(outputs), dim_observations=3
    )
    print(transition_matrix, emission_matrix, prior)
    """

    # """
    eval_data = safetensors.torch.load_file("data/eval.safetensors")
    transition_matrix, emission_matrix, prior, residual_belief_mapping = (
        hmm_from_residuals(
            eval_data["embeddings"],
            eval_data["outputs"],
            dim_state=3,
            dim_observations=3,
            num_train_steps=1024,
        )
    )
    print(transition_matrix, emission_matrix, prior)

    with torch.no_grad():
        belief_states = residual_belief_mapping(eval_data["embeddings"])
        min_belief = torch.min(belief_states, dim=-1, keepdim=True).values
        belief_states = belief_states - min_belief.clamp(max=0.0)
        belief_states = belief_states / torch.sum(belief_states, dim=-1, keepdim=True)

        with open("data/config.json", mode="r") as f:
            config_dict = json.load(f)

        model = TransformerWrapper(
            **config_dict["transformer"],
            attn_layers=Decoder(**config_dict["decoder"]),
        )

        missing, unexpected = safetensors.torch.load_model(
            model, "data/model.safetensors"
        )
        print(missing, unexpected)

        residuals = inverse_linear(
            torch.eye(3),
            residual_belief_mapping.weight,
            residual_belief_mapping.bias,
        )
        em = model.to_logits(residuals)

        min_em = torch.min(em, dim=-1, keepdim=True).values
        em = em - min_em.clamp(max=0.0)
        em = em / torch.sum(em, dim=-1, keepdim=True)
        print("EM: ", em)
        print("EM->P", residual_belief_mapping(residuals))

        p = belief_states.numpy()

        plt.figure()
        plt.scatter(p[..., 0].flatten(), p[..., 1].flatten(), s=1.0)
        plt.show()

        plt.figure()
        plt.scatter(p[..., 0].flatten(), p[..., 2].flatten(), s=1.0)
        plt.show()

        plt.figure()
        plt.scatter(p[..., 1].flatten(), p[..., 2].flatten(), s=1.0)
        plt.show()

        embeddings = eval_data["embeddings"]
        embeddings = rearrange(
            embeddings, "batch sequence features -> (batch sequence) features"
        )
        embeddings = embeddings - torch.mean(embeddings, dim=0, keepdim=True)

        u, s, v = torch.svd(embeddings)
        residual_components = v[:, :2]
        reduced = embeddings @ residual_components

        plt.figure()
        plt.scatter(*reduced.T, s=1.0)
        plt.show()

    # """


if __name__ == "__main__":
    main()
