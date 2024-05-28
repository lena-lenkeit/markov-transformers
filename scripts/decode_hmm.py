import json
import os
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
from x_transformers import AutoregressiveWrapper, Decoder, TransformerWrapper

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


@torch.no_grad()
def sample_dataset(
    model: TransformerWrapper,
    num_batches: int,
    batch_size: int,
    seq_len: int,
    device: str = "cuda",
):
    def filter_bos(logits: torch.Tensor, **kwargs):
        logits[:, -1] = -torch.inf
        return logits

    autoregressive = AutoregressiveWrapper(model)
    autoregressive.to(device)

    outputs = []
    prompts = torch.full((batch_size, 1), model.num_tokens - 1, device=device)

    for i in trange(num_batches):
        generation = autoregressive.generate(
            prompts, seq_len, filter_logits_fn=filter_bos
        )
        outputs.append(generation)

    outputs = torch.cat(outputs, dim=0)

    inputs = torch.cat(
        (
            repeat(prompts, "batch token -> (repeat batch) token", repeat=num_batches),
            outputs[:, :-1],
        ),
        dim=1,
    )
    logits, embeddings = model(inputs, return_logits_and_embeddings=True)

    return outputs, embeddings, logits


def main():
    # Paths
    model_dir = "data/1layer_attn-only"

    # Load model
    with open(os.path.join(model_dir, "config.json"), mode="r") as f:
        config_dict = json.load(f)

    model = TransformerWrapper(
        **config_dict["transformer"],
        attn_layers=Decoder(**config_dict["decoder"]),
    )

    missing, unexpected = safetensors.torch.load_model(
        model, os.path.join(model_dir, "model.safetensors")
    )
    print(missing, unexpected)

    outputs, embeddings, logits = sample_dataset(model, 4, 128, 256)
    outputs = outputs.cpu()
    embeddings = embeddings.cpu()
    logits = logits.cpu()

    print(outputs)
    print(logits)

    transition_matrix, emission_matrix, prior, residual_belief_mapping = (
        hmm_from_residuals(
            embeddings,
            outputs,
            dim_state=3,
            dim_observations=3,
            num_train_steps=1024,
        )
    )

    print(transition_matrix, emission_matrix, prior)


if __name__ == "__main__":
    main()
