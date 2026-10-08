"""Original Sidecar injection implementation from generate_sidecar_compare_standalone.py.

Keep these operations in sync with the validated standalone generator.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class IdentityProjector(nn.Module):
    def __init__(self, identity_dim: int = 512, rank: int = 512, num_tokens: int = 8):
        super().__init__()
        self.rank = rank
        self.num_tokens = num_tokens
        self.proj = nn.Linear(identity_dim, num_tokens * rank, bias=True)
        self.norm = nn.LayerNorm(rank)

    def forward(self, identity: torch.Tensor) -> torch.Tensor:
        x = self.proj(identity.float()).view(identity.shape[0], self.num_tokens, self.rank)
        return self.norm(x)


class IdentityAttentionResidual(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 512):
        super().__init__()
        self.rank = rank
        self.hidden_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_q = nn.Linear(hidden_dim, rank, bias=False)
        self.to_k = nn.Linear(rank, rank, bias=False)
        self.to_v = nn.Linear(rank, rank, bias=False)
        self.out_proj = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, hidden: torch.Tensor, identity_tokens: torch.Tensor) -> torch.Tensor:
        q = self.to_q(self.hidden_norm(hidden.float()))
        k = self.to_k(identity_tokens)
        v = self.to_v(identity_tokens)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rank), dim=-1)
        return self.out_proj(torch.matmul(attn, v)).to(hidden.dtype)


class IdentitySidecar(nn.Module):
    def __init__(self, hidden_dim: int, rank: int, num_tokens: int, layers: Iterable[int], scales: Iterable[float]):
        super().__init__()
        layers = tuple(int(x) for x in layers)
        scales = tuple(float(x) for x in scales)
        if len(layers) != len(scales) or len(set(layers)) != len(layers):
            raise ValueError('layers/scales must pair, with unique layers')
        self.layers = layers
        self.scales = dict(zip(layers, scales))
        self.projector = IdentityProjector(512, rank, num_tokens)
        self.slots = nn.ModuleDict({str(i): IdentityAttentionResidual(hidden_dim, rank) for i in layers})
        self.identity: torch.Tensor | None = None
        self.image_tokens: int | None = None
        self.global_scale = 1.0
        self._handles = []

    def set_context(self, identity: torch.Tensor, image_tokens: int, global_scale: float) -> None:
        self.identity = identity
        self.image_tokens = int(image_tokens)
        self.global_scale = float(global_scale)

    def _hook(self, layer_idx: int):
        def hook(_module, _args, output):
            if self.identity is None or self.image_tokens is None or self.global_scale == 0.0:
                return output
            if self.image_tokens > output.shape[1]:
                raise ValueError('real image token count exceeds transformer block sequence length')
            n = self.image_tokens
            image_hidden = output[:, :n]
            identity_tokens = self.projector(self.identity)
            delta = self.slots[str(layer_idx)](image_hidden, identity_tokens)
            delta = delta * (self.global_scale * self.scales[layer_idx])
            return torch.cat([image_hidden + delta, output[:, n:]], dim=1)
        return hook

    def attach(self, transformer) -> None:
        if self._handles:
            return
        for layer in self.layers:
            if not 0 <= layer < len(transformer.layers):
                raise ValueError(f'invalid injection layer {layer}; model has {len(transformer.layers)} layers')
        self._handles = [transformer.layers[i].register_forward_hook(self._hook(i)) for i in self.layers]

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
