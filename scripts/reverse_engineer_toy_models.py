import json
import os

import safetensors.torch
import torch
import torch.nn.functional as F

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


@torch.no_grad()
def main():
    model_dir = "data/mess3/custom"
    model, config = load_model(model_dir)
    basic_sim_analysis(model, config)


if __name__ == "__main__":
    main()
