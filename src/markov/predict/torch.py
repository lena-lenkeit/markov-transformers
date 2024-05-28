import torch
from einops import einsum
from jaxtyping import Float, Int


def update_posterior(
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


def propagate_values(
    x: Float[torch.Tensor, "..."],
    transition_matrix: Float[torch.Tensor, "state next_state"],
    normalize: bool = False,
) -> Float[torch.Tensor, "..."]:
    x = einsum(x, transition_matrix, "... i, i j -> ... j")
    if normalize:
        x = x / torch.sum(x, dim=-1, keepdim=True)

    return x


def get_stationary_distribution(
    transition_matrix: Float[torch.Tensor, "state next_state"],
):
    values, vectors = torch.linalg.eig(transition_matrix)

    values = torch.real(values)
    vectors = torch.real(vectors)

    one_distance = torch.abs(1 - values)
    one_index = torch.argmin(one_distance)
    one_vector = vectors[:, one_index]

    prior = one_vector / torch.sum(one_vector)

    return prior
