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
from einops import rearrange
from sklearn.decomposition import PCA
from tqdm.auto import trange
from x_transformers import Decoder, TransformerWrapper

from markov import HiddenMarkovModel, messn_matrices, sample_hmm


def main():
    # TODO: Resolve the non-linear output issue somehow (linear interpolation over
    # logits != linear interpolation over probabilites)

    # Paths
    os.makedirs(save_dir, exist_ok=True)

    # Parameters
    device = "cuda"
    batch_size = 128
    eval_batch_size = 1024

    num_states = 3
    num_outputs = num_states
    hmm_temperature = 1.0
    seq_len = 256

    transformer_kwargs: Dict[str, Any] = dict(
        l2norm_embed=True,
    )
    decoder_kwargs: Dict[str, Any] = dict(
        dim=128,
        depth=1,
        heads=2,
        use_simple_rmsnorm=True,
        ff_glu=True,
        rotary_pos_emb=True,
    )

    # Derived parameters
    num_tokens = num_outputs + 1  # HMM output tokens + BOS token
    bos_token_id = num_tokens - 1

    # Update kwargs
    transformer_kwargs.update(
        dict(
            num_tokens=num_tokens,
            max_seq_len=seq_len,
        )
    )

    # Initialize HMM
    rng = np.random.default_rng(1)
    # hmm = HiddenMarkovModel(
    #    sample_matrix(num_states, temperature=hmm_temperature, rng=rng),
    #    sample_matrix(
    #        num_states, num_outputs=num_outputs, temperature=hmm_temperature, rng=rng
    #    ),
    # )

    hmm = HiddenMarkovModel(*messn_matrices(n=num_states))

    # Initialize model and optimizer
    model = TransformerWrapper(
        **transformer_kwargs,
        attn_layers=Decoder(**decoder_kwargs),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    # Training
    pbar = trange(512)
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

        """
        probs = logits
        min_probs = torch.min(probs, dim=-1, keepdim=True).values
        probs = 1e-2 + probs - min_probs.clamp(max=0.0)
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)
        print(probs)
        log_probs = torch.log(probs)

        loss = F.nll_loss(
            rearrange(log_probs, "batch sequence logits -> (batch sequence) logits"),
            rearrange(targets, "batch sequence -> (batch sequence)"),
        )
        """

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss:.2e}")

    # Save model
    config_dict = {"transformer": transformer_kwargs, "decoder": decoder_kwargs}
    safetensors.torch.save_model(
        model,
        os.path.join(save_dir, "model.safetensors"),
        metadata={"config": json.dumps(config_dict)},
    )
    with open(os.path.join(save_dir, "config.json"), mode="w") as f:
        json.dump(config_dict, f, indent=4)

    # Sample hidden states
    with torch.no_grad():
        states, outputs = sample_hmm(hmm, seq_len, eval_batch_size, rng)
        states = torch.from_numpy(states).to(device)
        outputs = torch.from_numpy(outputs).to(device)

        tokens = outputs.clone()
        targets = outputs.clone()

        tokens = torch.roll(tokens, shifts=1, dims=1)
        tokens[:, 0] = bos_token_id

        logits, embeddings = model.forward(tokens, return_logits_and_embeddings=True)
        loss = F.cross_entropy(
            rearrange(logits, "batch sequence logits -> (batch sequence) logits"),
            rearrange(targets, "batch sequence -> (batch sequence)"),
        )

        print(f"Eval Loss: {loss:.2e}")

    # Save eval data
    safetensors.torch.save_file(
        {
            "states": states,
            "outputs": outputs,
            "logits": logits,
            "embeddings": embeddings,
        },
        os.path.join(save_dir, "eval.safetensors"),
    )

    # Perform PCA
    embeddings = embeddings.cpu().numpy()
    states = states.cpu().numpy()

    pca = PCA(n_components=16, whiten=False)
    embeddings_2d = pca.fit_transform(
        rearrange(embeddings, "batch sequence features -> (batch sequence) features")
    )

    # Plot PCA
    plt.figure()
    plt.plot(
        np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_
    )
    plt.show()

    plt.figure()
    plt.scatter(
        *embeddings_2d[:, :2].T,
        s=1.0,
        c=rearrange(states, "batch sequence -> (batch sequence)"),
    )
    plt.show()


if __name__ == "__main__":
    main()
