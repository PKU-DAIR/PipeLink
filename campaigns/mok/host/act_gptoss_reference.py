"""Activation-owner reference; clamp boundary gradients follow the pinned torch."""
from __future__ import annotations

import torch


def gptoss_swiglu(gate: torch.Tensor, up: torch.Tensor,
                   swiglu_limit: float | None = None) -> torch.Tensor:
    """Alpha=1.702 SwiGLU with optional scalar-10 clamps.

    Low-precision inputs compute in FP32; FP64 inputs stay FP64 for the
    independent numerical diagnosis. The output retains the input dtype.
    Use torch.autograd.grad(output, (gate, up), d_hidden) for backward.
    None corresponds to the unchanged ABI's clamp=0 variant.
    """
    dtype = torch.float64 if gate.dtype == torch.float64 else torch.float32
    g, u = gate.to(dtype), up.to(dtype)
    if swiglu_limit is not None:
        g = g.clamp(max=swiglu_limit)
        u = u.clamp(min=-swiglu_limit, max=swiglu_limit)
    return ((u + 1.0) * (g * torch.sigmoid(1.702 * g))).to(gate.dtype)
