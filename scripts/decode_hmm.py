import json
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from markov import HiddenMarkovModel, sample_matrix
from sklearn.decomposition import PCA
from tqdm.auto import trange
from x_transformers import Decoder, TransformerWrapper


def l2_normalize(x: torch.Tensor, dim: int, keepdim: bool):
    return x / torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=keepdim)


@torch.inference_mode()
def main():
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


main()
