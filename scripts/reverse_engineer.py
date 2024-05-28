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
"""
