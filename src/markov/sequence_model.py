import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from x_transformers.x_transformers import SimpleRMSNorm, TokenEmbedding


class SequenceModel(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim_model: int,
        attn_layer: nn.Module,
        attn_norm: bool,
        final_norm: bool,
        logit_bias: bool = True,
    ):
        super().__init__()

        self.to_embeddings = TokenEmbedding(dim_model, num_tokens, l2norm_embed=True)
        self.to_logits = nn.Linear(dim_model, num_tokens, bias=logit_bias)
        self.norm = SimpleRMSNorm(dim_model)
        self.attn_layer = attn_layer
        self.attn_norm = attn_norm
        self.final_norm = final_norm

        with torch.no_grad():
            self.to_embeddings.emb.weight.normal_(std=1e-4)

    def forward(self, token_ids: torch.Tensor, return_final: bool = False):
        token_embeddings = self.to_embeddings(token_ids)
        attn_outs = self.attn_layer(token_embeddings)

        if self.attn_norm:
            attn_outs = self.norm(attn_outs)

        final_outs = token_embeddings + attn_outs
        if self.final_norm:
            final_outs = self.norm(final_outs)

        logits = self.to_logits(final_outs)

        if return_final:
            return logits, final_outs
        else:
            return logits


class SingleHeadFixedAttention(nn.Module):
    """Single-headed fixed attention, with a data-independent mixing matrix"""

    def __init__(
        self,
        dim_input: int,
        dim_v: int,
        seq_len: int,
        bias: bool,
        causal: bool,
        mult: float = 1e2,
        has_v: bool = True,
        has_o: bool = True,
    ):
        super().__init__()

        self.causal = causal
        self.mult = mult
        self.has_v = has_v
        self.has_o = has_o

        self.v_project = nn.Linear(dim_input, dim_v, bias=bias)
        self.o_project = nn.Linear(dim_v, dim_input, bias=bias)
        self.m = nn.Parameter(torch.zeros((seq_len, seq_len)))

        with torch.no_grad():
            self.m.data.normal_(std=1e-4)

    def get_mixing_matrix(self):
        m = self.m

        if self.causal:
            m = torch.where(torch.tril(torch.ones_like(m)) == 0, -torch.inf, m)

        m = F.softmax(m * self.mult, dim=1)

        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_v:
            v = self.v_project(x)
        else:
            v = x

        m = self.get_mixing_matrix()
        o = einsum(m, v, "t s, ... s p -> ... t p")

        if self.has_o:
            o = self.o_project(o)

        return o
