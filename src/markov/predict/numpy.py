import numpy as np
from einops import einsum
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
