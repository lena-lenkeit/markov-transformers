import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from markov import HiddenMarkovModel, sample_matrix
from sklearn.decomposition import PCA
from tqdm.auto import trange
from x_transformers import Decoder, TransformerWrapper


def sample_hmm(
    hmm: HiddenMarkovModel, seq_len: int, batch_size: int, rng: np.random.Generator
):
    state = rng.integers(hmm.num_states, size=batch_size)
    output = hmm.sample_output(state, rng)

    states = [state]
    outputs = [output]
    for i in range(seq_len - 1):
        state = hmm.next_state(state, rng)
        output = hmm.sample_output(state, rng)

        states.append(state)
        outputs.append(output)

    states = np.stack(states, axis=1)
    outputs = np.stack(outputs, axis=1)

    return states, outputs


# https://arxiv.org/abs/1702.08565
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


def messn_matrices(x: float = 0.05, alpha: float = 0.85, n: int = 3):
    transition_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                transition_matrix[i, j] = 1 - (n - 1) * x
            else:
                transition_matrix[i, j] = x

    emission_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                emission_matrix[i, j] = alpha
            else:
                emission_matrix[i, j] = (1 - alpha) / (n - 1)

    return transition_matrix, emission_matrix


def main():
    # Parameters
    device = "cuda"
    batch_size = 128
    eval_batch_size = 1024

    num_states = 3
    num_outputs = num_states
    hmm_temperature = 1.0
    seq_len = 64

    # Derived parameters
    num_tokens = num_outputs + 1  # HMM output tokens + BOS token
    bos_token_id = num_tokens - 1

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
        num_tokens=num_tokens,
        max_seq_len=seq_len,
        l2norm_embed=True,
        attn_layers=Decoder(
            dim=128,
            depth=1,
            heads=2,
            use_simple_rmsnorm=True,
            ff_glu=True,
            rotary_pos_emb=True,
        ),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss:.2e}")

    # Sample hidden states
    with torch.no_grad():
        states, outputs = sample_hmm(hmm, seq_len, eval_batch_size, rng)
        # states = torch.from_numpy(states).to(device)
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

        embeddings = embeddings.cpu().numpy()

        print(f"Eval Loss: {loss:.2e}")

    # Perform PCA
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
