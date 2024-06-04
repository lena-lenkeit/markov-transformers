import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
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


def get_optimal_beliefs(
    observations: Int[torch.Tensor, "batch sequence"],
    transition_matrix: Float[torch.Tensor, "state next_state"],
    emission_matrix: Float[torch.Tensor, "state emission"],
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

    seq_beliefs = torch.stack(seq_beliefs, dim=1)
    seq_probs = torch.stack(seq_probs, dim=1)

    seq_log_probs = torch.log(seq_probs)
    loss = F.nll_loss(
        rearrange(seq_log_probs, "batch sequence logits -> (batch sequence) logits"),
        rearrange(observations, "batch sequence -> (batch sequence)"),
    )

    return seq_beliefs, seq_probs, loss
