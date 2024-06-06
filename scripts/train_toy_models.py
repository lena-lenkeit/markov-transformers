import json
import os
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import safetensors.numpy
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import einsum, rearrange
from opt_einsum import contract
from sklearn.decomposition import PCA
from tqdm.auto import trange
from x_transformers import Decoder, TransformerWrapper
from x_transformers.x_transformers import SimpleRMSNorm, TokenEmbedding

from markov import (
    HiddenMarkovModel,
    circle_matrices,
    messn_matrices,
    sample_hmm,
    sample_matrix,
)
from markov.predict.torch import get_optimal_beliefs


class SequenceModel(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim_model: int,
        attn_layer: nn.Module,
        attn_norm: bool,
        final_norm: bool,
    ):
        super().__init__()

        self.to_embeddings = TokenEmbedding(dim_model, num_tokens, l2norm_embed=True)
        self.to_logits = nn.Linear(dim_model, num_tokens)
        self.norm = SimpleRMSNorm(dim_model)
        self.attn_layer = attn_layer
        self.attn_norm = attn_norm
        self.final_norm = final_norm

        with torch.no_grad():
            self.to_embeddings.emb.weight.normal_(std=1e-4)

    def forward(self, token_ids: torch.Tensor):
        token_embeddings = self.to_embeddings(token_ids)
        attn_outs = self.attn_layer(token_embeddings)

        if self.attn_norm:
            attn_outs = self.norm(attn_outs)

        final_outs = token_embeddings + attn_outs
        if self.final_norm:
            final_outs = self.norm(final_outs)

        logits = self.to_logits(final_outs)
        return logits


# TBD
class FWA(nn.Module):
    """Single-headed fixed sliding attention, with a data-independent mixing vector"""

    def __init__(
        self,
        dim_input: int,
        dim_v: int,
        seq_len: int,
        bias: bool,
        mult: float = 1e2,
    ):
        super().__init__()

        self.mult = mult

        self.v_project = nn.Linear(dim_input, dim_v, bias=bias)
        self.m = nn.Parameter(torch.zeros((seq_len, seq_len)))

        with torch.no_grad():
            self.m.data.normal_(std=1e-4)

    def get_mixing_matrix(self):
        m = self.m

        if self.causal:
            m = torch.where(torch.tril(torch.ones_like(m)) == 0, -torch.inf, m)

        m = F.softmax(m * self.mult, dim=1)

        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.get_mixing_matrix()
        v = self.v_project(x)
        o = einsum(m, v, "t s, ... s p -> ... t p")

        return o


class FA(nn.Module):
    """Single-headed fixed attention, with a data-independent mixing matrix"""

    def __init__(
        self,
        dim_input: int,
        dim_v: int,
        seq_len: int,
        bias: bool,
        causal: bool,
        mult: float = 1e2,
    ):
        super().__init__()

        self.causal = causal
        self.mult = mult

        self.v_project = nn.Linear(dim_input, dim_v, bias=bias)
        self.m = nn.Parameter(torch.zeros((seq_len, seq_len)))

        with torch.no_grad():
            self.m.data.normal_(std=1e-4)

    def get_mixing_matrix(self):
        m = self.m

        if self.causal:
            m = torch.where(torch.tril(torch.ones_like(m)) == 0, -torch.inf, m)

        m = F.softmax(m * self.mult, dim=1)

        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.get_mixing_matrix()
        v = self.v_project(x)
        o = einsum(m, v, "t s, ... s p -> ... t p")

        return o


# TBD
class MA(nn.Module):
    """Single-headed masked attention"""

    def __init__(
        self,
        dim_input: int,
        dim_qk: int,
        dim_v: int,
        bias: bool,
        seq_len: int,
        kernel_fn: Callable,
    ):
        super().__init__()

        self.q_project = nn.Linear(dim_input, dim_qk, bias=bias)
        self.k_project = nn.Linear(dim_input, dim_qk, bias=bias)
        self.v_project = nn.Linear(dim_input, dim_v, bias=bias)
        self.l = nn.Parameter(torch.zeros((seq_len, seq_len)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_project(x)
        k = self.k_project(x)
        v = self.v_project(x)

        o = contract("tn, sn, sp, ts -> tp", q, k, v, self.l.tril(), backend="torch")
        return o


# TBD
class SMA(nn.Module):
    """Single-headed structured masked attention"""


# TBD
class SSM(nn.Module):
    "Discrete time-invariant state space model"

    # def __init__(self,)


def main():
    # TODO: Resolve the non-linear output issue somehow (linear interpolation over
    # logits != linear interpolation over probabilites)

    # Paths
    save_dir = "data/mess3/2layer_halfalpha"
    os.makedirs(save_dir, exist_ok=True)

    # Parameters
    device = "cuda"
    batch_size = 128
    eval_batch_size = 1024

    num_states = 3
    num_outputs = num_states
    hmm_temperature = 2.0
    seq_len = 256

    rng = np.random.default_rng(14)

    # Derived parameters
    num_tokens = num_outputs + 1  # HMM output tokens + BOS token
    bos_token_id = num_tokens - 1

    # Initialize HMM
    # hmm = HiddenMarkovModel(*messn_matrices(n=num_states, alpha=0.5))

    hmm = HiddenMarkovModel(
        sample_matrix(num_states, temperature=hmm_temperature, rng=rng),
        sample_matrix(
            num_states, num_outputs=num_outputs, temperature=hmm_temperature, rng=rng
        ),
    )

    print(hmm.transition_matrix)
    print(hmm.output_matrix)

    # Initialize model and optimizer
    model = SequenceModel(
        num_tokens=num_tokens,
        dim_model=256,
        attn_layer=FA(
            dim_input=256, dim_v=256, seq_len=seq_len, bias=True, causal=True, mult=1e2
        ),
        attn_norm=True,
        final_norm=True,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)

    # Get optimal belief states and loss
    with torch.no_grad():
        optimal_states, optimal_outputs = sample_hmm(hmm, seq_len, eval_batch_size, rng)
        optimal_states = torch.from_numpy(optimal_states).to(device)
        optimal_outputs = torch.from_numpy(optimal_outputs).to(device)

        optimal_beliefs, optimal_probs, optimal_loss = get_optimal_beliefs(
            optimal_outputs.cpu(),
            torch.from_numpy(hmm.transition_matrix),
            torch.from_numpy(hmm.output_matrix),
        )

        random_loss = F.cross_entropy(
            rearrange(
                torch.zeros_like(optimal_probs),
                "batch sequence logits -> (batch sequence) logits",
            ),
            rearrange(optimal_outputs.cpu(), "batch sequence -> (batch sequence)"),
        )

    print(f"Optimal Eval Loss: {optimal_loss:.4e}, Random Eval Loss: {random_loss:.4e}")

    # Training
    pbar = trange(1024)
    for i in pbar:
        states, outputs = sample_hmm(hmm, seq_len, batch_size, rng)
        states = torch.from_numpy(states).to(device)
        outputs = torch.from_numpy(outputs).to(device)

        tokens = outputs.clone()
        targets = outputs.clone()

        tokens = torch.roll(tokens, shifts=1, dims=1)
        tokens[:, 0] = bos_token_id

        logits = model.forward(tokens)
        loss = F.cross_entropy(
            rearrange(logits, "batch sequence logits -> (batch sequence) logits"),
            rearrange(targets, "batch sequence -> (batch sequence)"),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        pbar.set_postfix_str(
            f"Loss: {loss:.4e}, Regret: {loss - optimal_loss:.4e}, LR: {lr:.2e}"
        )

    # Evaluation
    with torch.no_grad():
        tokens = optimal_outputs.clone()
        targets = optimal_outputs.clone()

        tokens = torch.roll(tokens, shifts=1, dims=1)
        tokens[:, 0] = bos_token_id

        logits = model.forward(tokens)
        eval_loss = F.cross_entropy(
            rearrange(logits, "batch sequence logits -> (batch sequence) logits"),
            rearrange(targets, "batch sequence -> (batch sequence)"),
        )

        print(f"Eval Loss: {eval_loss:.4e}, Regret: {eval_loss - optimal_loss:.4e}")

    with torch.no_grad():
        plt.figure()
        plt.imshow(model.attn_layer.get_mixing_matrix().cpu().numpy())
        plt.show()


if __name__ == "__main__":
    main()
