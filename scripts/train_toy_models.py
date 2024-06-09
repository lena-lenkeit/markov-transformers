import json
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import safetensors.numpy
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from tqdm.auto import trange

from markov import (
    HiddenMarkovModel,
    circle_matrices,
    messn_matrices,
    sample_hmm,
    sample_matrix,
    trapn_matrices,
)
from markov.predict.torch import get_optimal_beliefs
from markov.sequence_model import SequenceModel, SingleHeadFixedAttention


def main():
    # TODO: Resolve the non-linear output issue somehow (linear interpolation over
    # logits != linear interpolation over probabilites)

    # Paths
    save_dir = "data/trap2/custom/"
    os.makedirs(save_dir, exist_ok=True)

    # Parameters
    device = "cuda"
    batch_size = 128
    eval_batch_size = 1024
    num_train_steps = 1024 * 8
    lr_start = 3e-4
    lr_end = 3e-5
    weight_decay = 0.0

    num_states = 2
    num_outputs = num_states + 1
    hmm_temperature = 2.0
    seq_len = 256

    dim_model = 128

    rng = np.random.default_rng(14)

    # Derived parameters
    num_tokens = num_outputs + 1  # HMM output tokens + BOS token
    bos_token_id = num_tokens - 1

    # Model parameter dicts
    model_kwargs: Dict[str, Any] = dict(
        num_tokens=num_tokens,
        dim_model=dim_model,
        l2norm_embed=False,  # For tiny models (w.o. a proj.), false is more expressive
        # At least one norm true necessary
        attn_norm=True,
        final_norm=False,
        logit_bias=False,  # Doesn't matter
        has_residual=False,  # Works even with no residual
    )
    attn_layer_kwargs: Dict[str, Any] = dict(
        dim_input=dim_model,
        dim_v=dim_model,
        seq_len=seq_len,
        bias=False,  # Doesn't matter
        causal=True,
        mult=1e2,  # 1e2 trains well (this is just a gradient boosting hack)
        # At least one projection true necessary (if there is a residual, otherwise not)
        has_v=False,
        has_o=False,
    )

    # Initialize HMM
    # hmm = HiddenMarkovModel(*messn_matrices(n=num_states, alpha=0.85)) # MessN
    hmm = HiddenMarkovModel(*trapn_matrices(n=num_states, x=0.1, alpha=0.25))  # TrapN
    # TODO: Also try the one-sided trap2 variant, easy to visualize in 2d

    if False:
        hmm = HiddenMarkovModel(
            sample_matrix(num_states, temperature=hmm_temperature, rng=rng),
            sample_matrix(
                num_states,
                num_outputs=num_outputs,
                temperature=hmm_temperature,
                rng=rng,
            ),
        )

    print(hmm.transition_matrix)
    print(hmm.output_matrix)

    # Save HMM
    safetensors.numpy.save_file(
        {
            "transition_matrix": hmm.transition_matrix,
            "emission_matrix": hmm.output_matrix,
        },
        os.path.join(save_dir, "hmm.safetensors"),
    )

    # Initialize model and optimizer
    model = SequenceModel(
        **model_kwargs, attn_layer=SingleHeadFixedAttention(**attn_layer_kwargs)
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr_start, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_train_steps, eta_min=lr_end
    )

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

    # Save optimal prediction data
    safetensors.torch.save_file(
        {
            "states": optimal_states,
            "outputs": optimal_outputs,
            "beliefs": optimal_beliefs,
            "probs": optimal_probs,
            "loss": optimal_loss,
            "random": random_loss,
        },
        os.path.join(save_dir, "optimal.safetensors"),
    )

    # Training
    pbar = trange(num_train_steps)
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

    # Save model
    config_dict = {"model": model_kwargs, "attn_layer": attn_layer_kwargs}
    safetensors.torch.save_model(
        model,
        os.path.join(save_dir, "model.safetensors"),
        metadata={"config": json.dumps(config_dict)},
    )
    with open(os.path.join(save_dir, "config.json"), mode="w") as f:
        json.dump(config_dict, f, indent=4)

    # Evaluation
    with torch.no_grad():
        tokens = optimal_outputs.clone()
        targets = optimal_outputs.clone()

        tokens = torch.roll(tokens, shifts=1, dims=1)
        tokens[:, 0] = bos_token_id

        logits, embeddings = model.forward(tokens, return_final=True)
        eval_loss = F.cross_entropy(
            rearrange(logits, "batch sequence logits -> (batch sequence) logits"),
            rearrange(targets, "batch sequence -> (batch sequence)"),
        )

        print(f"Eval Loss: {eval_loss:.4e}, Regret: {eval_loss - optimal_loss:.4e}")

    # Save eval data
    safetensors.torch.save_file(
        {
            "states": optimal_states,
            "outputs": optimal_outputs,
            "logits": logits,
            "embeddings": embeddings.contiguous(),
            "loss": eval_loss,
            "regret": eval_loss - optimal_loss,
        },
        os.path.join(save_dir, "eval.safetensors"),
    )


if __name__ == "__main__":
    main()
