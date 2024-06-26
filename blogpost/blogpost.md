# Reverse-engineering optimal belief updating in HMM-Transformers

## Summary

I reverse engineer how models trained to predict HMM outputs internally implement the optimal belief state updating mechanism. I find even linear single-head attention-only transformers to be capable of approximating the HMM and the fractal belief state geometry, via a simple scaled-vector addition fractal formation mechanism. In addition, these findings naturally map onto State Space Models (as such, a large variety of models can approximate optimal belief updating over HMMs, i.e. a single attention head, a group of filters in a convolutional layer, or linear recurrent networks).

### Key Animated Eyecatch Figure

HMM + Optimal Belief State + Fractal Approx., side-by-side, when interpolating over different HMMs

## Introduction

Even small Transformer models are capable of approximating HMMs. Recently, in "Transformers Represent...", this has been linked to Computation Mechanics. CMech requires every optimal predictor of an HMM to internally represent the probability of being in state X for every state of the HMM (the belief state). For most HMMs, the optimal belief state has a non-trivial fractal structure, and this structure is indeed linearly encoded inside the residual stream activations of transformers trained to be predictors of HMMs!

However, the optimal belief state is represented by a non-linear recurrent function of the HMM outputs, while attention layers within Transformers can only perform linear mixing across the sequence (In fact, transformers rose to the top in the first place, precisely because they can be evaluated in parallel). So how can even simple transformers approximate HMMs that well and also have a fractal belief state geometry?

### (Hidden) Markov Models
Let's first review HMMs and the optimal belief state updating algorithm.

A Markov model is a graph or network, where the vertices / nodes are the states of the model, and the edges represent transition probabilities between the states. We have a probabilistic model of the form $p(X_{i}|X_{j})=T_{ij}$, i.e. the probability of moving into state $Xi$, given that we are currently in the state $Xj$, is a fixed number $Tij$, where $T$ is the matrix of all possible current- and next-state transition probabilities. The output of an HMM is then simply formed by choosing a starting state, continuously sampling new states, and noting down each state the model was in as the output.

A hidden Markov model adds another probabilistic output step on top of the state of the model. For a given state $X_j$, we similarly define the probability of emitting the output $O_i$ as $p(O_i|X_j) = E_{ij}$, where E now refers to the matrix containing the emission probabilities of possible outputs over all states. HMMs generalize over standard Markov models: For a Markov model with $N$ states, we can construct an equivalent hidden Markov model by adding a diagonal emission matrix of shape $\mathbb{R}^{N \times N}$. Intuitively, this assigns a unique output to each state, such that we have perfect certainty of which state the HMM is in given an observation. However, we can also construct emission matrices which allow multiple outputs to be emitted even when the model is in the same (hidden) state. We then only obtain a limited amount of information about the current state of the model from observing a single model output.

TODO Include examples and images of actual HMMs here.

### Optimal Belief State Updating

TODO Check order-of-operations here for the matrix-vector products, I just wrote them down quickly without keeping track of ordering.

Inferring the state in a Markov model is simple: Since the model outputs are directly given by the sequence of sampled states, we by definition have perfect certainty of which state the model was in during the entire sequence. For a hidden Markov model, the amount of information gained from observing an emission can vary. If we wish to model the internal state of the hidden Markov model given a sequence of probabilistic emissions, we can naturally represent our uncertainty of the internal state by assigning and keeping track of the probability $p(X_i)$ of each state. For a hidden Markov model with $N$ states, we end up with a belief state vector $p(X) = [p(X_1), p(X_2), ..., p(X_n)]$, with the additional constraint that all probabilities sum to one [^1].

If we now observe an output $O_j$ from our hidden Markov model, how does our belief of the hidden state of the model $p(X)$ change? We are looking for $p(X_i|O_j)$. Expanding via Bayes' Theorem, we find

$$p(X_i|O_j)=\frac{p(O_j|X_i)p(X_i)}{p(O_j)},$$

where $p(X_i)$ is our prior belief of the model being in state $X_i$ before receiving the observation $O_j$ and crucially, $p(O_j|X_i)$ an entry $E_{ji}$ of the emission matrix of the HMM [^2]! Rewriting the Bayesian updating step for our entire belief state vector in matrix notation, we then have 

$$p(X|O_j)=\frac{p(X) \odot E_{:,j}}{p(O_j)}.$$

After emitting an output, the HMM then also samples the next hidden state according to the transition matrix. In turn, our belief states also "diffuse" along all possible transitions of the HMM. Our belief state after a transition is then given by a linear mixture of all next-state transition probabilities (the columns of the transition matrix) weighted by the probability of each state, as

$$p(X_{t+1}|X_{t})=p(X_t)T.$$

Our combined update-and-transition operator, omitting normalization, then becomes

$$p(X_{t+1}|O_j,X_{t})=(p(X) \odot E_{:,j}) T.$$

This operator gives us the optimal forward belief states, assuming knowledge of the HMM matrices, for any sequence of HMM outputs. An optimal predictor of HMM outputs then returns

$$p(O) = p(X)E,$$

where $p(X)$ is given by the optimal updating procedure. For any model trained purely as a predictor of HMM outputs, two components must necessarily be implemented to obtain optimal predictive perfomance:

TODO Check if true

1. The predictive model must infer and implicitly represent the HMM transition and emission matrices (Learn the correct HMM).
2. The model must also represent and update an internal belief state of the HMM hidden state, to predict optimal next-emission probabilities (Learn to synchronize to the HMM).

TODO Optimal Belief Simplex Figure here

## Finding the Minimal Model

WIP

Small transformer (4 layers) -> 2 layers -> 1 layer, 1 layer attn-only: Show attention matrix is fixed -> simple model with fixed learnable attention matrix

attn-only + no-skip, attn-only + no-skip + no-norm

## Explaining Fractal Formation

### Intuitions for the Fixed Attention Matrix

TODO "Forgetting" past information / past belief states

## Constructing a Model from Scratch

## Extra

WIP

Is MEO^T the best linear operator? What about the case where we allow position-specific embeddings of E, i.e. we process the one-hot Ehat as MhatEhat, where MHat maps from one-hot vectors directly to logits over the entire sequence (and then we stack multiple of these for every position). Also, going back to the real world case this is what multi-layer transformers could do, by first adding position embeddings early on, then constructing HMMs in an attention layer, and performing extra state updating with pattern-matching heads.

TODO Explain the above at the end more, especially the part with having additional pattern-matching heads to complement the HMM. Maybe look for an HMM where this is super obvious, like the trap-HMM with rare state switching (here, once observing the state-specific token, you'd immediately assign near-full certainty to that state, but the linear fractal mechanism can't really erase past tokens conditionally, only in expectation via exponential decay of attention scores. A larger circuit might be able to search for only the most-recent token though, which would perform better. Maybe there are other HMMs where this is clearer).

Also what about hunting for fixed-attention heads in actual LLMs? They should exist if LLMs model HMM-like constructs internally.

## Final Thoughts

## Footnotes

[^1]: As we'll later see, this is also the reason why belief states of a model with $N$ states lie in a simplex with $N-1$ dimensions. The sum-to-one constraint removes a degree of freedom (one of the state probabilities can be inferred by knowing all other probabilities), such that all possible belief state vectors lie on a $N-1$ dimensional surface. The geometrical interpretation is simplest here: Imagine a model with $3$ states, where each belief state probability represents one of the axes in a standard Cartesian coordinate system. The sum-to-one constraint $p(X_1) + p(X_2) + p(X_3) = 1$ forms the equation for a 2-dimensional plane with normal vector $(1, 1, 1)$ and containing the tips of all three basis vectors. Furthermore, probabilities are always $\geq 0$, such that every belief vector will always lie in the $(+, +, +)$-region of the belief space. The final space in which belief state vectors can lie is then constrained to an equilateral triangle, the corners of which are spanned by the belief state basis vectors. This reasoning generalizes to any number of dimensions.
[^2]: $p(0_j)$ can be omitted as usual.