import json
import os

import safetensors.numpy
import safetensors.torch
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm.auto import trange

from markov.predict.torch import (
    get_optimal_beliefs,
    get_stationary_distribution,
    propagate_values,
    update_posterior,
)
from markov.sequence_model import SequenceModel, SingleHeadFixedAttention


def load_model(model_dir: str):
    # Load config
    with open(os.path.join(model_dir, "config.json"), mode="r") as f:
        config_dict = json.load(f)

    # Initialize model
    model = SequenceModel(
        **config_dict["model"],
        attn_layer=SingleHeadFixedAttention(**config_dict["attn_layer"]),
    )

    # Load parameters into model
    missing, unexpected = safetensors.torch.load_model(
        model, os.path.join(model_dir, "model.safetensors")
    )
    print(missing, unexpected)

    # Load HMM
    hmm_np = safetensors.numpy.load_file(os.path.join(model_dir, "hmm.safetensors"))
    hmm = {
        "transition_matrix": torch.from_numpy(hmm_np["transition_matrix"]).float(),
        "emission_matrix": torch.from_numpy(hmm_np["emission_matrix"]).float(),
    }

    return model, config_dict, hmm


def remove_component(x: torch.Tensor, component: torch.Tensor) -> torch.Tensor:
    # Normalize the component
    component = component / torch.linalg.vector_norm(component, dim=-1)

    # Compute the projection of x onto the component
    factor = torch.linalg.vecdot(x, component, dim=-1)[..., None]
    projection = factor * component

    # Subtract the projection from x to remove the component
    x_removed = x - projection

    return x_removed


def basic_sim_analysis(model: SequenceModel, config: dict):
    o = model.to_logits.weight
    v = model.attn_layer.v_project.weight

    token_seq = torch.arange(config["model"]["num_tokens"])
    e = model.to_embeddings(token_seq)
    eo = model.to_logits(e)

    x = model.attn_layer.v_project(e)
    xo = model.to_logits(x)

    ee_sim = F.cosine_similarity(e[None], e[:, None], dim=-1)
    xx_sim = F.cosine_similarity(x[None], x[:, None], dim=-1)
    oo_sim = F.cosine_similarity(o[None], o[:, None], dim=-1)

    ex_sim = F.cosine_similarity(e[None], x[:, None], dim=-1)
    eo_sim = F.cosine_similarity(e[None], o[:, None], dim=-1)
    xo_sim = F.cosine_similarity(x[None], o[:, None], dim=-1)

    ewo = e.clone()
    for _o in model.to_logits.weight:
        ewo = remove_component(ewo, _o[None])

    xwo = x.clone()
    for _o in model.to_logits.weight:
        xwo = remove_component(xwo, _o[None])

    ewo_sim = F.cosine_similarity(ewo[None], ewo[:, None], dim=-1)
    xwo_sim = F.cosine_similarity(xwo[None], xwo[:, None], dim=-1)

    print(torch.linalg.svd(torch.cat((e, x, o), dim=0)))
    print(torch.linalg.svd(v))

    print(torch.linalg.matrix_rank(e))
    print(torch.linalg.matrix_rank(x))
    print(torch.linalg.matrix_rank(torch.cat((e, x), dim=0)))

    print(torch.linalg.matrix_rank(ewo))
    print(torch.linalg.matrix_rank(xwo))
    print(torch.linalg.matrix_rank(torch.cat((ewo, xwo), dim=0)))

    print(eo)
    print(xo)

    print("EE", ee_sim)  # Not orthogonal
    print("XX", xx_sim)  # Weak-ish orthogonal
    print("OO", oo_sim)  # Weak-ish orthogonal

    print("EX", ex_sim)  # Almost orthogonal
    print("EO", eo_sim)  # Not orthogonal (weak diagonal)
    print("XO", xo_sim)  # Not orthogonal (diagonal)

    print(ewo_sim)
    print(xwo_sim)


def plot_attention_matrix(model: SequenceModel, config: dict):
    plt.figure()
    plt.imshow(model.attn_layer.get_mixing_matrix(), cmap="magma")
    plt.colorbar()
    plt.show()


def plot_arrow_sequence(model: SequenceModel, config: dict):
    o = model.to_logits.weight
    v = model.attn_layer.v_project.weight
    m = model.attn_layer.get_mixing_matrix()

    token_seq = torch.arange(config["model"]["num_tokens"] - 1)
    e = model.to_embeddings(token_seq)
    x = model.attn_layer.v_project(e)

    # Get orthogonal basis for X
    x_svd = torch.linalg.svd(x, full_matrices=False)
    x_basis = x_svd.Vh
    x_projected = x_svd.U

    # Get reading directions for the output logit head
    o_projected = o[:2] @ x_basis.T

    # Map a sequence to the basis space
    token_seq = torch.randint(config["model"]["num_tokens"] - 1, size=(256,))
    x_sequence = x_projected[:, token_seq].T * m[-1][:, None]
    x_sequence = torch.cumsum(x_sequence, dim=0)
    x_sequence = torch.cat((torch.zeros_like(x_sequence[:1]), x_sequence), dim=0)

    plt.figure()
    plt.plot(*x_sequence.T)
    plt.plot(*x_sequence[[0, -1]].T, color="tab:red")
    plt.plot([0, o_projected[0, 0]], [0, o_projected[0, 1]], color="tab:green")
    plt.plot([0, o_projected[1, 0]], [0, o_projected[1, 1]], color="tab:green")
    plt.show()


def plot_arrow_sequence_minimal_model(model: SequenceModel, config: dict):
    o = model.to_logits.weight
    m = model.attn_layer.get_mixing_matrix()

    token_seq = torch.arange(config["model"]["num_tokens"] - 1)
    e = model.to_embeddings(token_seq)

    # Get orthogonal basis for X
    e_svd = torch.linalg.svd(e, full_matrices=False)
    e_basis = e_svd.Vh
    e_projected = e_svd.U

    # Get reading directions for the output logit head
    o_projected = o[:2] @ e_basis.T

    # Map a sequence to the basis space
    token_seq = torch.randint(config["model"]["num_tokens"] - 1, size=(256,))
    e_sequence = e_projected[:, token_seq].T * m[-1][:, None]
    e_sequence = torch.cumsum(e_sequence, dim=0)
    e_sequence = torch.cat((torch.zeros_like(e_sequence[:1]), e_sequence), dim=0)

    # Plot basis space
    plt.figure()
    plt.title("Basis Space Vectors")
    plt.xlabel("Basis Vector 1")
    plt.ylabel("Basis Vector 2")
    plt.plot([0, e_projected[0, 0]], [0, e_projected[0, 1]], label="Token 1 Write")
    plt.plot([0, e_projected[1, 0]], [0, e_projected[1, 1]], label="Token 2 Write")
    plt.plot([0, o_projected[0, 0]], [0, o_projected[0, 1]], label="Token 1 Read")
    plt.plot([0, o_projected[1, 0]], [0, o_projected[1, 1]], label="Token 2 Read")
    plt.legend()
    plt.show()

    # Plot sequence
    plt.figure()
    plt.title("Token Sequence in Basis Space")
    plt.plot(*e_sequence.T, label="Sequence")
    plt.plot(*e_sequence[[0, -1]].T, label="Result")
    plt.plot(
        [0, e_projected[0, 0]],
        [0, e_projected[0, 1]],
        linestyle="--",
        label="Token 1 Write",
    )
    plt.plot(
        [0, e_projected[1, 0]],
        [0, e_projected[1, 1]],
        linestyle="--",
        label="Token 2 Write",
    )
    plt.plot(
        [0, o_projected[0, 0]],
        [0, o_projected[0, 1]],
        linestyle="--",
        label="Token 1 Read",
    )
    plt.plot(
        [0, o_projected[1, 0]],
        [0, o_projected[1, 1]],
        linestyle="--",
        label="Token 2 Read",
    )
    plt.legend()
    plt.show()


def plot_expected_information_gain(model: SequenceModel, config: dict, hmm: dict):
    num_samples = 1024 * 128
    num_tokens = config["model"]["num_tokens"] - 1
    seq_len = config["attn_layer"]["seq_len"]

    num_states = hmm["transition_matrix"].shape[1]
    num_emissions = hmm["emission_matrix"].shape[1]

    # * Via a closed form method, deriving the EIG of the T and E matrices
    # * We are looking for the expected information gain. Let's first derive the
    #   information gain for a single transition or emission event.
    # * Assume we have a prior p and observe some evidence o, such that we now have the
    #   posterior q. The amount of information we gained is the weighted surprise / the
    #   KL-divergence KL(p||q) = \sum q_i \log_2 q_i/p_i.
    # * The expected information gain is the expectation of KL(p||q) when sampling p and
    #   o over an unbiased prior. Sampling over o is easy: Since it's discrete, we just
    #   average. However, p is continuous, so we need to find the corresponding
    #   integral.
    # * Sampling is pssible here, but we don't have the true prior of p...

    priors = torch.rand((num_samples, num_states))
    priors = priors / torch.sum(priors, dim=1, keepdim=True)
    emissions = torch.randint(num_tokens, size=(num_samples,))

    posteriors = update_posterior(priors, emissions, hmm["emission_matrix"])
    posteriors_propagated = propagate_values(
        posteriors, hmm["transition_matrix"], normalize=True
    )
    priors_propagated = propagate_values(
        priors, hmm["transition_matrix"], normalize=True
    )

    kl = F.kl_div(
        torch.log(priors),
        torch.log(posteriors_propagated),
        log_target=True,
        reduction="batchmean",
    )
    kl_no_update = F.kl_div(
        torch.log(priors),
        torch.log(priors_propagated),
        log_target=True,
        reduction="batchmean",
    )
    kl_no_propagate = F.kl_div(
        torch.log(priors),
        torch.log(posteriors),
        log_target=True,
        reduction="batchmean",
    )
    print(kl, kl_no_update, kl_no_propagate)

    # Like below, but batched for efficiency
    token_seq = torch.randint(num_tokens, size=(num_samples, seq_len))
    optimal_belief_seq, optimal_prob_seq, optimal_loss = get_optimal_beliefs(
        token_seq, hmm["transition_matrix"], hmm["emission_matrix"]
    )

    eig = []
    for i in trange(seq_len):
        optimal_prob = optimal_prob_seq[:, -1]
        optimal_belief = optimal_belief_seq[:, -1]
        start_belief = optimal_belief_seq[:, i]

        chained_matrix = torch.linalg.matrix_power(
            hmm["transition_matrix"], seq_len - i
        )
        belief = propagate_values(start_belief, chained_matrix, normalize=True)
        prob = propagate_values(belief, hmm["emission_matrix"], normalize=True)

        """
        optimal_loss_at_i = F.nll_loss(
            torch.log(optimal_prob_seq[:, i]), token_seq[:, i]
        )
        base_loss_at_i = F.nll_loss(torch.log(prob), token_seq[:, i])

        eig.append(optimal_loss_at_i - base_loss_at_i)
        """

        # eig.append(torch.mean(torch.sum(optimal_prob * torch.log(prob), dim=-1)))

        # This (the cross-entropy between the probabilites derived with and without
        # including model emissions) seems to match the M matrix!
        eig.append(torch.mean(torch.sum(prob * torch.log(optimal_prob), dim=-1)))

        """
        eig.append(
            -F.kl_div(
                torch.log(prob),
                torch.log(optimal_prob),
                log_target=True,
                reduction="batchmean",
            )
        )
        """

        """
        eig.append(
            -F.kl_div(
                torch.log(belief),
                torch.log(optimal_belief),
                log_target=True,
                reduction="batchmean",
            )
        )
        """

    eig = torch.FloatTensor(eig)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(eig)
    ax[1].plot(model.attn_layer.get_mixing_matrix()[-1])
    plt.show()

    # Via a sampling based method over entire sequences
    eig = torch.zeros(seq_len)
    for sample_id in trange(num_samples):
        # * Sample a random sequence of tokens and get the optimal belief states for
        # that sequence. Then, look at the predictive power of each optimal belief state
        # at time step t for the final time step when propagated forward without
        # applying observations (i.e. only applying the transition matrix). This
        # quantity should be related to the expected information gain from also
        # including the observations vs. simply using prior knowledge of the transition
        # and emission matrix.
        # * My idea is that this metric is related to M, the sequence kernel.

        token_seq = torch.randint(num_tokens, size=(seq_len,))
        optimal_belief_seq, optimal_prob_seq, optimal_loss = get_optimal_beliefs(
            token_seq[None], hmm["transition_matrix"], hmm["emission_matrix"]
        )

        ig = []
        for i in range(seq_len):
            belief = optimal_belief_seq[0, i]
            chained_matrix = torch.linalg.matrix_power(
                hmm["transition_matrix"], seq_len - i
            )
            belief = propagate_values(belief, chained_matrix, normalize=True)
            prob = propagate_values(belief, hmm["emission_matrix"], normalize=True)

            optimal_loss_at_i = F.nll_loss(
                torch.log(optimal_prob_seq[0, i]), token_seq[i]
            )
            base_loss_at_i = F.nll_loss(torch.log(prob), token_seq[i])

            ig.append(optimal_loss_at_i - base_loss_at_i)

        eig = eig + torch.FloatTensor(ig)

    eig = eig / num_samples

    plt.figure()
    plt.plot(eig)
    plt.show()


@torch.no_grad()
def main():
    model_dir = "data/mess2/custom"
    model, config, hmm = load_model(model_dir)

    plot_expected_information_gain(model, config, hmm)
    # plot_arrow_sequence(model, config)
    plot_arrow_sequence_minimal_model(model, config)
    # plot_attention_matrix(model, config)
    # basic_sim_analysis(model, config)


if __name__ == "__main__":
    main()
