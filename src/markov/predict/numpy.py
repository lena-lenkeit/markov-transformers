from typing import Tuple

import numpy as np
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int


def update_posterior(
    prior: Float[np.ndarray, "... state"],
    observation: Int[np.ndarray, "..."],
    emission_matrix: Float[np.ndarray, "state emission"],
) -> Float[np.ndarray, "... state"]:
    # Bayes' Theorem: p(Hi|Oj) = p(Oj|Hi)p(Hi)/p(Oj)
    # p(Oj|Hi) is the emission matrix

    likelihood = emission_matrix[:, observation]
    posterior = einsum(likelihood, prior, "i ..., ... i -> ... i")
    posterior = posterior / np.sum(posterior, axis=-1, keepdims=True)
    return posterior


def propagate_values(
    x: Float[np.ndarray, "..."],
    transition_matrix: Float[np.ndarray, "state next_state"],
    normalize: bool = False,
) -> Float[np.ndarray, "..."]:
    x = einsum(x, transition_matrix, "... i, i j -> ... j")
    if normalize:
        x = x / np.sum(x, axis=-1, keepdims=True)

    return x


def get_stationary_distribution(
    transition_matrix: Float[np.ndarray, "state next_state"],
):
    values, vectors = np.linalg.eig(transition_matrix)

    values = np.real(values)
    vectors = np.real(vectors)

    one_distance = np.abs(1 - values)
    one_index = np.argmin(one_distance)
    one_vector = vectors[:, one_index]

    prior = one_vector / np.sum(one_vector)

    return prior


def nll_loss(input, target):
    # TODO: Fix this
    return -np.mean(np.log(input[np.arange(len(target)), target]))


def get_optimal_beliefs(
    observations: Int[np.ndarray, "batch sequence"],
    transition_matrix: Float[np.ndarray, "state next_state"],
    emission_matrix: Float[np.ndarray, "state emission"],
):
    dim_batch, dim_sequence = observations.shape

    # Get initial state
    prior = get_stationary_distribution(transition_matrix)
    belief_state = repeat(prior, "states -> batch states", batch=dim_batch)
    probs = einsum(belief_state, emission_matrix, "... i, i j -> ... j")

    seq_beliefs = [belief_state]
    seq_probs = [probs]
    for j in range(dim_sequence - 1):
        tokens = observations[:, j]

        belief_state = update_posterior(belief_state, tokens, emission_matrix)
        belief_state = propagate_values(belief_state, transition_matrix, normalize=True)

        probs = einsum(belief_state, emission_matrix, "... i, i j -> ... j")

        seq_beliefs.append(belief_state)
        seq_probs.append(probs)

    seq_beliefs = np.stack(seq_beliefs, axis=1)
    seq_probs = np.stack(seq_probs, axis=1)

    seq_log_probs = np.log(seq_probs)
    loss = nll_loss(
        rearrange(seq_log_probs, "batch sequence logits -> (batch sequence) logits"),
        rearrange(observations, "batch sequence -> (batch sequence)"),
    )

    return seq_beliefs, seq_probs, loss
