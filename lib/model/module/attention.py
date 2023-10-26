from collections import namedtuple
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .rotary import RotaryEmbedding


class KVCacheStableAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int, context_size: int = 512
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.context_size = context_size

        self.q_proj = nn.Linear(hidden_size, head_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(hidden_size, head_dim * num_heads, bias=False)
        self.o_proj = nn.Linear(head_dim * num_heads, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(head_dim)

    def split_heads(self, x: Tensor) -> Tensor:
        return (
            x.view(x.size(0), x.size(1), self.num_heads, self.head_dim)
            .transpose(-2, -3)
            .contiguous()
        )

    def merge_heads(self, x: Tensor) -> Tensor:
        return (
            x.transpose(-2, -3)
            .contiguous()
            .view(x.size(0), -1, self.head_dim * self.num_heads)
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        past_key_value: Tuple[Tensor, Tensor] | None = None,
    ):
        q = self.q_proj(x)
        aq = self.q_proj(self.anchor_tokens)
        xk = self.k_proj(x)
        ak = self.k_proj(self.anchor_tokens)
        xv = self.v_proj(x)
        av = self.v_proj(self.anchor_tokens)
        q = self.split_heads(q)
        aq = self.split_heads(aq)
        xk = self.split_heads(xk)
        ak = self.split_heads(ak)
        xv = self.split_heads(xv)
        av = self.split_heads(av)
        k, v = None, None
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, xk], dim=-2)
            v = torch.cat([pv, xv], dim=-2)
            if k.shape[-2] > self.context_size:
                k = k[:, :, -self.context_size :, :]
                v = v[:, :, -self.context_size :, :]
                attn_mask = attn_mask[:, :, -self.context_size :]
        else:
            k, v = xk, xv
        past_key_value = (k, v)
        q = torch.cat([aq.repeat(q.shape[0], 1, 1, 1), q], dim=-2)
        k = torch.cat([ak.repeat(k.shape[0], 1, 1, 1), k], dim=-2)
        v = torch.cat([av.repeat(v.shape[0], 1, 1, 1), v], dim=-2)
        attn_mask = F.pad(
            attn_mask, (self.anchor_size, 0, 0, self.anchor_size), value=True
        )
        q, k = self.rotary(q, k)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask.unsqueeze(-3).type(torch.bool)
        )
        out = out[:, :, self.anchor_size :, :]
        out = self.merge_heads(out)
        out = self.o_proj(out)
        return out, past_key_value
