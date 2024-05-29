"""This is an attempt at fully reverse engineering HMM-trained transformers. I'm noting
my ideas on how I think the belief-updating procedure is implemented internally, which
I'll also use as a basis to develop my first attempt:

* Evidence of transformer inner workings:
* - Belief states are represented linearly in the residual stream
* - Even 1-layer 1-head attn-only transformers work (on mess3, training a full 2-layer
    2-head transformer doesn't improve the loss; 1-layer 2-head attn+ff possibly
    improves the loss slightly, but hard to tell)
* - Turning off the positional embedding makes the loss higher, and no fractal structure
    is present in the belief states

* Knowledge from the optimal prediction mechanism:
* - Operations
*   - 1x Indexing/Selection based on current input observation (equivalent to /
      realizable by token embedding matrix?)
*   - 1x vector-dot product to update the posterior from the observations (natural to
      implement in FF layer)
*   - 1x vector-matrix multiply (which are multiple vector-dot products and sums) to
      propagate the posterior according to the transition matrix (natural to implement
      in FF layer)
*   - 1x normalization (via the final layer norm?); the norm is an observation-dependent
      multiplier though, so could also be encoded in the embedding layer?
* - Depends only on the previous belief state (assuming order-1 HMM)
* - Including all of the above, if we have the belief state and observation emission
    probabilities, we can calculate the next belief state via one matmul + normalization
*   - It would therefore make sense to encode the emission probabilities in the token
      embedding matrix along the belief state vectors
*   - However, how would you get the previous complete belief state? The attention layer
      can grab the previous state, but you'd need to simultaneously perform the update
      so the new state is in the cache...
*     - Maybe the attention layer computes the belief state update for all possible
        observations, encoded in different subspaces, and the following FF layer just
        extracts the correct subspace based on the actual observation?

* Transformer mechanism ideas:
* - The embeddings of the input tokens probably at least already contain the conditional
    updating probabilites of the belief states, along a set of "basis" vectors
*   - But how is the updating done? It's only linear in log space, and only the part of
      updating the posterior, not the propagation along the transition matrix (which
      contains a sum and thus can't be performed in log space)
* - Maybe a representation of complex numbers or transfer to another domain is involved
    in the attention layer, in which a sum over the sequence (which the attn-layer can
    implement) is equivalent to applying the linear update + propagation operator
    multiple times? This would also naturally allow the rotary embedding to work...
*   - But how is the belief state linearly represented then? And how are there only 2
      principle components?
* - Also, isn't the full updating procedure a dicrete time-invariant state space model?
    Can attention layers implement state space models?

Testing is necessary...

* Observations:
* - Attention scores for the 1-layer 1-head attn-only HMM-transformer attend weakly to
    the current token, most strongly to the previous token, and decay quickly after a
    few tokens
*   - This is only the case for mess3 (due to symmetry, or strongly bimodal probs?)! On
      a random 3-state, 3-output (also 16/16) HMM, attn-scores were uniform. Didn't test
      yet if recovering the system was possible...
*   - Similar short attention structure for circle4
* - QKV layers have no strong singular values
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch
from einops import repeat
from tqdm.auto import trange
from x_transformers import AutoregressiveWrapper, Decoder, TransformerWrapper


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
    (logits, embeddings), attn_maps = model(
        inputs, return_logits_and_embeddings=True, return_attn=True
    )

    return inputs, outputs, logits, embeddings, attn_maps


@torch.no_grad()
def main():
    # Paths
    model_dir = "data/random/circle3/1layer_attn-only"

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

    # Look at some singular values
    if False:
        u, s, v = torch.linalg.svd(model.token_emb.emb.weight, full_matrices=False)
        print(u, s, v)

        u, s, v = torch.linalg.svd(
            model.attn_layers.layers[0][1].to_k.weight, full_matrices=False
        )
        print(u, s, v)

        u, s, v = torch.linalg.svd(
            model.attn_layers.layers[0][1].to_out.weight, full_matrices=False
        )
        print(u, s, v)

    # Look at token encodings at different layers in terms of logits
    to_embeddings = model.token_emb
    to_logits = model.to_logits
    attn_layer = model.attn_layers.layers[0][1]
    norm_layer = model.attn_layers.layers[0][0][0]
    final_norm = model.attn_layers.final_norm

    token_embeddings = to_embeddings(torch.arange(4))
    token_v = attn_layer.to_v(norm_layer(token_embeddings))
    token_v_out = attn_layer.to_out(token_v)

    print(to_logits(final_norm(token_embeddings)))
    print(to_logits(final_norm(token_v)))
    print(to_logits(final_norm(token_embeddings + token_v_out)))

    # This reproduces the correct output
    # print(to_logits(final_norm(token_embeddings + token_v_out)))

    # Sample data
    if True:
        inputs, outputs, logits, embeddings, attn_maps = sample_dataset(
            model, 1, 128, 256
        )

        print(outputs)
        print(logits)
        print(len(attn_maps))
        print(attn_maps[0].shape)

        # Plot attention scores
        plt.figure()
        plt.imshow(attn_maps[0][0, 0].cpu().numpy())
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.imshow(torch.mean(attn_maps[0], dim=0)[0].cpu().numpy())
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()
