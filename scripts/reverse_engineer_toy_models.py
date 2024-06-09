import json
import os

import safetensors.torch
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

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

    return model, config_dict


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


@torch.no_grad()
def main():
    model_dir = "data/mess2/custom"
    model, config = load_model(model_dir)

    # plot_arrow_sequence(model, config)
    plot_arrow_sequence_minimal_model(model, config)
    # plot_attention_matrix(model, config)
    # basic_sim_analysis(model, config)


if __name__ == "__main__":
    main()
