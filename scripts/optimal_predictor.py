import matplotlib.pyplot as plt
import numpy as np
from einops import einsum
from markov import HiddenMarkovModel


def mess3_matrices(x: float = 0.05, alpha: float = 0.85):
    transition_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                transition_matrix[i, j] = 1 - 2 * x
            else:
                transition_matrix[i, j] = x

    emission_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                emission_matrix[i, j] = alpha
            else:
                emission_matrix[i, j] = (1 - alpha) / 2

    return transition_matrix, emission_matrix


def get_posterior(
    prior: np.ndarray,
    observation: np.ndarray,
    emission_matrix: np.ndarray,
) -> np.ndarray:
    # Bayes' Theorem: p(Hi|Oj) = p(Oj|Hi)p(Hi)/p(Oj)
    # p(Oj|Hi) is the emission matrix

    likelihood = emission_matrix[:, observation]
    posterior = einsum(likelihood, prior, "i ..., ... i -> ... i")
    posterior = posterior / np.sum(posterior, axis=-1, keepdims=True)
    return posterior


def propagate_posterior(p: np.ndarray, transition_matrix: np.ndarray):
    p = einsum(p, transition_matrix, "... i, i j -> ... j")
    # Not necessary, but might avoid accumulation errors
    p = p / np.sum(p, axis=-1, keepdims=True)

    return p


hmm = HiddenMarkovModel(*mess3_matrices())

rng = np.random.default_rng(1234)
num_samples = 1024 * 1
seq_len = 256

init_state = rng.integers(hmm.num_states, size=num_samples)
init_observation = hmm.sample_output(init_state, rng)

# TODO: Prior from stationary distribution of HMM (Eigenvector of the HMM transition
# matrix with eigenvalue 1)
# print(np.linalg.eig(hmm.transition_matrix))
prior = np.full((num_samples, hmm.num_states), 1 / hmm.num_states)

states = [init_state]
outputs = [init_observation]
beliefs = [prior]

for i in range(seq_len):
    prior = beliefs[-1]
    posterior = get_posterior(prior, outputs[-1], hmm.output_matrix)
    posterior = propagate_posterior(posterior, hmm.transition_matrix)
    beliefs.append(posterior)

    states.append(hmm.next_state(states[-1], rng))
    outputs.append(hmm.sample_output(states[-1], rng))

states = np.stack(states, axis=1)  # (batch, sequence)
beliefs = np.stack(beliefs, axis=1)  # (batch, sequence, states)

plt.figure(figsize=(8, 8))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.scatter(
    beliefs[..., 0].flatten() + beliefs[..., 1].flatten() * 0.5,
    beliefs[..., 1].flatten(),
    s=1.0,
    c=states.flatten(),
)
plt.show()
