from collections import namedtuple
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn

from .module import KVCacheStableAttention, RMSNorm, SwiGLU


@tensorclass
class TransformerOutput:
    logits: Optional[Tensor] = None
    last_hidden_states: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    past_key_values: Optional[List[List[Tensor]]] = None


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_head_size: int,
    ):
        super().__init__()
        self.pre_attn_norm = RMSNorm(hidden_size)
        self.attention = KVCacheStableAttention(
            hidden_size, num_attention_heads, attention_head_size
        )
        self.pre_mlp_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(
        self, x: Tensor, attn_mask: Tensor, past_key_value: Tensor | None = None
    ) -> Tensor:
        x0 = x
        x = self.pre_attn_norm(x0)
        x, past_key_value = self.attention(
            x, past_key_value=past_key_value, attn_mask=attn_mask
        )
        x0 = x0 + x
        x = self.pre_mlp_norm(x0)
        x = self.mlp(x)
        x = x0 + x
        return x, past_key_value


class ModernTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        output_size: int,
        anchor_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.output_size = output_size
        self.anchor_size = anchor_size
        self.anchor_tokens = nn.Parameter(torch.randn(1, anchor_size, hidden_size))
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(
                DecoderLayer(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    hidden_size // num_attention_heads,
                )
            )
        self.head = nn.Linear(hidden_size, output_size, bias=False)

    def build_attn_mask(
        self,
        inputs_embeds: Tensor,
        attn_mask: Tensor | None = None,
        past_key_values: List[List[Tensor]] | None = None,
        is_causal: bool = True,
    ) -> Tensor:
        if attn_mask is None:
            attn_mask = torch.ones(
                inputs_embeds.size(0),
                inputs_embeds.size(1),
                inputs_embeds.size(1),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        if is_causal:
            causal_mask = torch.tril(
                torch.ones(
                    (inputs_embeds.size(1), inputs_embeds.size(1)),
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                ),
                diagonal=0,
            )
            attn_mask = attn_mask & causal_mask[None, ...]
            if past_key_values is not None and past_key_values[0] is not None:
                pkv_length = past_key_values[0][0].size(-2)
                pkv_mask = torch.ones(
                    inputs_embeds.shape[1],
                    pkv_length,
                    dtype=torch.bool,
                    device=attn_mask.device,
                )
                pkv_mask = pkv_mask.unsqueeze(0).repeat(attn_mask.shape[0], 1, 1)
                attn_mask = torch.cat(
                    [pkv_mask, attn_mask],
                    dim=-1,
                )
        return attn_mask

    def forward(
        self,
        inputs_embeds: int,
        attn_mask: Tensor | None = None,
        past_key_values: List[List[Tensor]] | None = None,
        is_causal: bool = True,
    ) -> TransformerOutput:
        inputs_embeds = torch.cat([self.anchor_tokens, inputs_embeds], dim=1)
        attn_mask = self.build_attn_mask(
            inputs_embeds, attn_mask, past_key_values, is_causal
        )
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        for i in range(len(self.layers)):
            inputs_embeds, (curr_key, curr_value) = self.layers[i](
                inputs_embeds, attn_mask, past_key_values[i]
            )
            curr_key = curr_key[..., self.anchor_size :, :]
            curr_value = curr_value[..., self.anchor_size :, :]
            past_key_values[i] = (curr_key, curr_value)

        inputs_embeds = inputs_embeds[:, self.anchor_size]

        logits = self.head(inputs_embeds)
        return TransformerOutput(logits=logits, past_key_values=past_key_values)
