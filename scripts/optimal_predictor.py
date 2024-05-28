import matplotlib.pyplot as plt
import numpy as np
from einops import repeat

from markov import HiddenMarkovModel, mess3_matrices
from markov.predict.numpy import (
    get_stationary_distribution,
    propagate_values,
    update_posterior,
)

hmm = HiddenMarkovModel(*mess3_matrices())

rng = np.random.default_rng(1234)
num_samples = 1024 * 1
seq_len = 256

init_state = rng.integers(hmm.num_states, size=num_samples)
init_observation = hmm.sample_output(init_state, rng)

prior = get_stationary_distribution(hmm.transition_matrix)
prior = repeat(prior, "state -> batch state", batch=num_samples)

states = [init_state]
outputs = [init_observation]
beliefs = [prior]

for i in range(seq_len):
    prior = beliefs[-1]
    posterior = update_posterior(prior, outputs[-1], hmm.output_matrix)
    posterior = propagate_values(posterior, hmm.transition_matrix, normalize=True)
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
