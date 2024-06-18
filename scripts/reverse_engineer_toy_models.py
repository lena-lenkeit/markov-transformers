import json
import os

import mpl_toolkits.mplot3d as plt3d
import numpy as np
import safetensors.numpy
import safetensors.torch
import torch
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import trange

from markov import sample_hmm
from markov.markov import HiddenMarkovModel
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


def plot_subspace(model: SequenceModel, config: dict, hmm: dict, plot_3d: bool = False):
    token_seq = torch.arange(config["model"]["num_tokens"] - 1)

    e = model.to_embeddings(token_seq)
    o = model.to_logits.weight

    # Get orthogonal basis for X
    e_svd = torch.linalg.svd(e, full_matrices=False)
    e_basis = e_svd.Vh
    e_projected = e_svd.U

    # Get reading directions for the output logit head
    o_projected = o[:-1] @ e_basis.T

    # Plot basis space
    if plot_3d:
        # Get simplex center
        e_simplex_center = torch.mean(e_projected, dim=0, keepdim=True)

        # Get vectors projected onto simplex
        e_simplex_projected = e_projected - e_simplex_center

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.set_title("Basis Space Vectors")
        ax.set_xlabel("Basis Vector 1")
        ax.set_ylabel("Basis Vector 2")
        ax.set_zlabel("Basis Vector 3")

        ax.set_aspect("equal", adjustable="box")

        # Token Write
        for i, vec in enumerate(e_projected):
            ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], label=f"Token {i} Write")

        # Simplex Center
        ax.plot(
            [0, e_simplex_center[0, 0]],
            [0, e_simplex_center[0, 1]],
            [0, e_simplex_center[0, 2]],
            c="black",
            label="_",
        )

        # Simplex Write
        ax.set_prop_cycle(None)
        for i, vec in enumerate(e_projected):
            ax.plot(
                [e_simplex_center[0, 0], vec[0]],
                [e_simplex_center[0, 1], vec[1]],
                [e_simplex_center[0, 2], vec[2]],
                label="_",
            )

        # Token Read
        ax.set_prop_cycle(None)
        for i, vec in enumerate(o_projected):
            plt.plot(
                [0, vec[0]],
                [0, vec[1]],
                [0, vec[2]],
                linestyle="--",
                label=f"Token {i} Read",
            )

        triangle = plt3d.art3d.Poly3DCollection([e_projected])
        triangle.set_alpha(0.25)
        triangle.set_color("tab:grey")
        triangle.set_edgecolor("black")
        ax.add_collection3d(triangle)

        ax.legend()
        plt.show()
    else:
        plt.figure()
        plt.title("Basis Space Vectors")
        plt.xlabel("Basis Vector 1")
        plt.ylabel("Basis Vector 2")
        for i, vec in enumerate(e_projected):
            plt.plot([0, vec[0]], [0, vec[1]], label=f"Token {i} Write")
        plt.gca().set_prop_cycle(None)
        for i, vec in enumerate(o_projected):
            plt.plot([0, vec[0]], [0, vec[1]], linestyle="--", label=f"Token {i} Read")
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


def emission_eig_at(
    model: SequenceModel, config: dict, hmm: dict, at: int, num_samples: int
):
    num_tokens = config["model"]["num_tokens"] - 1

    # Like below, but batched for efficiency
    # TODO: This is the wrong expectation, these should be HMM sequences
    # token_seq = torch.randint(num_tokens, size=(num_samples, at))
    _, token_seq = sample_hmm(
        HiddenMarkovModel(
            hmm["transition_matrix"].numpy(), hmm["emission_matrix"].numpy()
        ),
        at,
        num_samples,
        np.random.default_rng(1234),
    )
    token_seq = torch.from_numpy(token_seq)

    optimal_belief_seq, optimal_prob_seq, optimal_loss = get_optimal_beliefs(
        token_seq, hmm["transition_matrix"], hmm["emission_matrix"]
    )

    eig = []
    for i in trange(at):
        optimal_prob = optimal_prob_seq[:, -1]
        optimal_belief = optimal_belief_seq[:, -1]
        start_belief = optimal_belief_seq[:, i]

        chained_matrix = torch.linalg.matrix_power(hmm["transition_matrix"], at - i - 1)
        belief = propagate_values(start_belief, chained_matrix, normalize=True)
        prob = propagate_values(belief, hmm["emission_matrix"], normalize=True)

        # This (the cross-entropy between the probabilites derived with and without
        # including model emissions) seems to match the M matrix!
        eig.append(torch.mean(torch.sum(prob * torch.log(optimal_prob), dim=-1)))

    return torch.stack(eig)


def construct_from_hmm(
    model: SequenceModel, config: dict, hmm: dict, verbose: bool = True
):
    seq_len = config["attn_layer"]["seq_len"]

    transition_matrix = hmm["transition_matrix"]
    emission_matrix = hmm["emission_matrix"]
    prob_matrix = (transition_matrix @ emission_matrix).T
    # prob_matrix = -torch.log(prob_matrix)

    # emission_eig = -torch.log(emission_matrix).T
    emission_eig = torch.eye(emission_matrix.shape[0]).float()
    emission_eig += 5.0
    # emission_eig = emission_matrix.T

    if verbose:
        plt.figure(figsize=(6, 6))
        for eig in emission_eig:
            plt.plot([0, eig[0]], [0, eig[1]])
        for prob_vec in prob_matrix:
            plt.plot([0, prob_vec[0]], [0, prob_vec[1]])
        plt.show()

    def to_m(x):
        x = x - torch.min(x)
        x = x / torch.sum(x)
        x = torch.cat((x, torch.zeros(seq_len - x.shape[0])), dim=0)
        return x

    m_vec = to_m(emission_eig_at(model, config, hmm, seq_len, 1024 * 128))
    m_mat = [m_vec]
    for i in range(seq_len - 1):
        m_vec = torch.roll(m_vec, -1)
        m_vec[0] += m_vec[-1]
        m_vec[-1] = 0

        m_mat.append(m_vec)

    m_mat = torch.stack(m_mat, dim=0)
    m_mat = torch.flip(m_mat, dims=(0,))

    # m_vec = to_m(emission_eig_at(model, config, hmm, seq_len, 1024 * 128))
    # plt.figure()
    # plt.plot(m_vec)
    # plt.show()

    # mixing_eig = torch.stack(
    #    [
    #        to_m(emission_eig_at(model, config, hmm, i + 1, 1024 * 16))
    #        for i in range(seq_len)
    #    ]
    # )

    if verbose:
        plt.figure()
        plt.imshow(m_mat, cmap="magma")
        plt.show()

    # Test synthetic model
    num_tokens = config["model"]["num_tokens"] - 1

    if verbose:
        num_samples = 1024 * 128

        # Like below, but batched for efficiency
        # token_seq = torch.randint(num_tokens, size=(num_samples, seq_len))
        _, token_seq = sample_hmm(
            HiddenMarkovModel(transition_matrix.numpy(), emission_matrix.numpy()),
            seq_len,
            num_samples,
            np.random.default_rng(1234),
        )
        token_seq = torch.from_numpy(token_seq)

        optimal_belief_seq, optimal_prob_seq, optimal_loss = get_optimal_beliefs(
            token_seq, hmm["transition_matrix"], hmm["emission_matrix"]
        )

    synth_model = SequenceModel(
        num_tokens + 1,
        dim_model=emission_eig.shape[0],
        attn_layer=SingleHeadFixedAttention(
            dim_input=emission_eig.shape[0],
            dim_v=emission_eig.shape[0],
            seq_len=seq_len,
            bias=False,
            causal=True,
            has_v=False,
            has_o=False,
            # m=m_mat,
            m=model.attn_layer.get_mixing_matrix(),
        ),
        attn_norm=True,
        final_norm=False,
        logit_bias=False,
        has_residual=False,
        l2norm_embed=False,
        norm_p=1.0,
    )

    if verbose:
        print(emission_eig, prob_matrix)

    synth_model.to_embeddings.emb.weight.data.copy_(
        F.normalize(
            torch.cat((emission_eig, torch.ones_like(emission_eig[:1])), dim=0), dim=-1
        )
    )
    synth_model.to_logits.weight.data.copy_(
        F.normalize(
            torch.cat((prob_matrix, torch.ones_like(prob_matrix[:1])), dim=0), dim=-1
        )
        * emission_eig.shape[0] ** -0.5
    )

    if verbose:
        print(synth_model.to_embeddings.emb.weight.data)
        print(synth_model.to_logits.weight.data)

        bos_token_id = num_tokens

        tokens = token_seq.clone()
        tokens = torch.roll(tokens, shifts=1, dims=1)
        tokens[:, 0] = bos_token_id

        synth_probs, synth_post_norm = synth_model(token_seq, return_final=True)
        synth_probs = F.softmax(synth_probs, dim=-1)

        print(synth_probs.min(), synth_probs.max())
        print(synth_post_norm.min(), synth_post_norm.max())
        synth_loss = F.nll_loss(
            rearrange(
                torch.log(synth_probs),
                "batch sequence logits -> (batch sequence) logits",
            ),
            rearrange(token_seq, "batch sequence -> (batch sequence)"),
            reduction="none",
        )

        synth_loss = rearrange(
            synth_loss,
            "(batch sequence) -> batch sequence",
            batch=num_samples,
            sequence=seq_len,
        )

        plt.figure()
        plt.plot(torch.mean(synth_loss, dim=0))
        plt.show()

        print(optimal_loss, torch.mean(synth_loss))

    return synth_model


def plot_belief_states(model: SequenceModel, config: dict, hmm: dict):
    # HPs
    num_samples = 1024 * 16
    seq_start = 64
    rng = np.random.default_rng(1234)

    # Fetch data
    seq_len = config["attn_layer"]["seq_len"]

    transition_matrix = hmm["transition_matrix"]
    emission_matrix = hmm["emission_matrix"]

    hmm_markov = HiddenMarkovModel(transition_matrix.numpy(), emission_matrix.numpy())

    # Sample ground truth HMM outputs
    _, token_seq = sample_hmm(hmm_markov, seq_len, num_samples, rng)
    token_seq = torch.from_numpy(token_seq)

    # Sample optimal belief states
    optimal_beliefs, optimal_probs, optimal_loss = get_optimal_beliefs(
        token_seq, transition_matrix, emission_matrix
    )

    # Sample model beliefs
    model_outputs, model_features = model(token_seq, return_final=True)

    # Train linear probe
    optimal_beliefs = optimal_beliefs[:, seq_start:]
    model_features = model_features[:, seq_start:]

    probe = make_pipeline(StandardScaler(), LinearRegression())
    probe.fit(
        rearrange(
            model_features.numpy(),
            "batch sequence features -> (batch sequence) features",
        ),
        rearrange(
            optimal_beliefs.numpy(),
            "batch sequence beliefs -> (batch sequence) beliefs",
        ),
    )
    model_beliefs = probe.predict(
        rearrange(
            model_features.numpy(),
            "batch sequence features -> (batch sequence) features",
        )
    )

    # Plot
    plt.figure(figsize=(8, 8))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.hist2d(
        optimal_beliefs[..., 0].flatten() + optimal_beliefs[..., 1].flatten() * 0.5,
        optimal_beliefs[..., 1].flatten(),
        bins=(256, 256),
        norm="log",
        cmap="magma",
    )
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.hist2d(
        model_beliefs[..., 0].flatten() + model_beliefs[..., 1].flatten() * 0.5,
        model_beliefs[..., 1].flatten(),
        bins=(256, 256),
        norm="log",
        cmap="magma",
    )
    plt.colorbar()
    plt.show()


@torch.no_grad()
def main():
    model_dir = "data/mess3/custom"
    model, config, hmm = load_model(model_dir)

    # plot_attention_matrix(model, config)
    # plot_arrow_sequence_minimal_model(model, config)
    synth_model = construct_from_hmm(model, config, hmm, verbose=False)
    plot_subspace(model, config, hmm, plot_3d=True)
    plot_subspace(synth_model, config, hmm, plot_3d=True)
    plot_belief_states(synth_model, config, hmm)
    # plot_expected_information_gain(model, config, hmm)
    # plot_arrow_sequence(model, config)
    # basic_sim_analysis(model, config)


if __name__ == "__main__":
    main()
