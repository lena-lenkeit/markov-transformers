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
from markov import HiddenMarkovModel, sample_matrix
from sklearn.decomposition import PCA
from tqdm.auto import trange
from x_transformers import Decoder, TransformerWrapper


def l2_normalize(x: torch.Tensor, dim: int, keepdim: bool):
    return x / torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=keepdim)


def get_posterior(
    prior: Float[torch.Tensor, "... state"],
    observation: Int[torch.Tensor, "..."],
    emission_matrix: Float[torch.Tensor, "state emission"],
) -> Float[torch.Tensor, "... state"]:
    # Bayes' Theorem: p(Hi|Oj) = p(Oj|Hi)p(Hi)/p(Oj)
    # p(Oj|Hi) is the emission matrix

    likelihood = emission_matrix[:, observation]
    posterior = einsum(likelihood, prior, "i ..., ... i -> ... i")
    posterior = posterior / torch.sum(posterior, dim=-1, keepdim=True)
    return posterior


def propagate_posterior(
    p: Float[torch.Tensor, "..."],
    transition_matrix: Float[torch.Tensor, "state next_state"],
) -> Float[torch.Tensor, "..."]:
    p = einsum(p, transition_matrix, "... i, i j -> ... j")
    # Not necessary, but might avoid accumulation errors
    p = p / torch.sum(p, dim=-1, keepdim=True)

    return p


def get_prior(transition_matrix: Float[torch.Tensor, "state next_state"]):
    values, vectors = torch.linalg.eig(transition_matrix)

    values = torch.real(values)
    vectors = torch.real(vectors)

    one_distance = torch.abs(1 - values)
    one_index = torch.argmin(one_distance)
    one_vector = vectors[:, one_index]

    prior = one_vector / torch.sum(one_vector)

    return prior


def hmm_from_belief_states(
    belief_states: Float[torch.Tensor, "batch sequence state"],
    observations: Int[torch.Tensor, "batch sequence"],
    dim_observations: int,
):
    """Attempts to decode the full HMM (transition and emission matrix) given sequences
    of belief states and observations by assuming the belief states
    were generated by an optimal predictor."""

    dim_batch, dim_sequence, dim_state = belief_states.shape

    # TODO: Like hmm_from_observations, but with the belief state supplied externally


def hmm_from_observations(
    observations: Int[torch.Tensor, "batch sequence"],
    dim_state: int,
    dim_observations: int,
):
    dim_batch, dim_sequence = observations.shape

    transition_logit_matrix = torch.empty((dim_state, dim_state), requires_grad=True)
    emission_logit_matrix = torch.empty(
        (dim_state, dim_observations), requires_grad=True
    )

    with torch.no_grad():
        transition_logit_matrix.data.normal_(std=1e-1)
        emission_logit_matrix.data.normal_(std=1e-1)

    optimizer = optim.AdamW(
        [transition_logit_matrix, emission_logit_matrix], lr=1e-1, weight_decay=0.0
    )

    pbar = trange(256)
    for i in pbar:
        transition_matrix = F.softmax(transition_logit_matrix, dim=1)
        emission_matrix = F.softmax(emission_logit_matrix, dim=1)
        prior = get_prior(transition_matrix)

        # Get initial state
        belief_state = repeat(prior, "states -> batch states", batch=dim_batch)
        probs = einsum(belief_state, emission_matrix, "... i, i j -> ... j")

        seq_probs = [probs]
        for j in range(dim_sequence - 1):
            tokens = observations[:, j]

            belief_state = get_posterior(belief_state, tokens, emission_matrix)
            belief_state = propagate_posterior(belief_state, transition_matrix)

            probs = einsum(belief_state, emission_matrix, "... i, i j -> ... j")
            seq_probs.append(probs)

        seq_probs = torch.stack(seq_probs, dim=1)
        seq_log_probs = torch.log(seq_probs)
        loss = F.nll_loss(
            rearrange(
                seq_log_probs, "batch sequence logits -> (batch sequence) logits"
            ),
            rearrange(observations, "batch sequence -> (batch sequence)"),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss:.2e}")

    with torch.no_grad():
        transition_matrix = F.softmax(transition_logit_matrix, dim=1)
        emission_matrix = F.softmax(emission_logit_matrix, dim=1)
        prior = get_prior(transition_matrix)

    return transition_matrix, emission_matrix, prior


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


def main():
    # observations = torch.randint(3, size=(128, 256))
    eval_data = safetensors.torch.load_file("data/eval.safetensors")
    observations = eval_data["outputs"]

    transition_matrix, emission_matrix, prior = hmm_from_observations(
        observations, dim_state=3, dim_observations=3
    )
    print(transition_matrix, emission_matrix, prior)


if __name__ == "__main__":
    main()
