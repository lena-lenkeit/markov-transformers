"""Utilities for generating and sampling from random (hidden) Markov models."""

from dataclasses import dataclass

import numpy as np


def vectorized_categorical(pdf: np.ndarray, rng: np.random.Generator):
    """Samples from a vector of categorical distributions, specified by a matrix with
    probabilities summing to one in the rows. Based on inverse transform sampling."""

    uniform = rng.uniform(size=(pdf.shape[0], 1))
    cdf = np.cumsum(pdf, axis=1)
    idx = np.sum(cdf <= uniform, axis=1)

    return idx


@dataclass
class MarkovModel:
    transition_matrix: np.ndarray

    @property
    def num_states(self):
        return self.transition_matrix.shape[0]

    def next_state(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        next_state_probs = self.transition_matrix[state]
        return vectorized_categorical(next_state_probs, rng)


@dataclass
class HiddenMarkovModel(MarkovModel):
    output_matrix: np.ndarray

    @property
    def num_outputs(self):
        return self.output_matrix.shape[0]

    def sample_output(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        output_probs = self.output_matrix[state]
        return vectorized_categorical(output_probs, rng)


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    """Softmax along one axis."""

    return np.exp(x) / (np.sum(np.exp(x), axis=axis, keepdims=True))


def sample_matrix(
    num_states: int,
    *,
    num_outputs: int | None = None,
    temperature: float = 1.0,
    rng: np.random.Generator
) -> np.ndarray:
    """Samples a random transition or output matrix with column-normalized
    probabilities."""

    if num_outputs is None:
        num_outputs = num_states

    logits = rng.standard_normal(size=(num_states, num_outputs)) * temperature
    probs = softmax(logits, axis=1)

    return probs
