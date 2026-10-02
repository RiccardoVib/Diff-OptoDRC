# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato

import math
import torch
import torch.nn as nn


def inverse_softplus(y: float) -> float:
    """theta such that softplus(theta) == y, computed stably for y > 0."""
    if y <= 0:
        raise ValueError("softplus is strictly positive")
    # y + log(1 - exp(-y))  == y + log(-expm1(-y))
    return y + math.log(-math.expm1(-y))


def logit(p: float) -> float:
    """theta such that sigmoid(theta) == p."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    return math.log(p / (1.0 - p))


def soft_lower_bound(x: torch.Tensor, lo: float, beta: float = 4.0) -> torch.Tensor:
    """Smooth, everywhere-differentiable version of ``x.clamp(min=lo)``.
    """
    return lo + torch.nn.functional.softplus(beta * (x - lo)) / beta


def zero_init_head(linear: nn.Linear, bias_value) -> None:
    """Zero the weights of an output head and set its bias.
    """
    nn.init.zeros_(linear.weight)
    with torch.no_grad():
        if isinstance(bias_value, torch.Tensor):
            linear.bias.copy_(bias_value)
        else:
            linear.bias.fill_(float(bias_value))


def amp_to_db(x: torch.Tensor, floor_db: float = -120.0) -> torch.Tensor:
    """|x| -> dB, with a hard floor.
    """
    floor_amp = 10.0 ** (floor_db / 20.0)
    return 20.0 * torch.log10(x.abs().clamp_min(floor_amp))


def db_to_gain(gain_db: torch.Tensor) -> torch.Tensor:
    return torch.pow(10.0, gain_db / 20.0)


def match_cond_shape(c, seq_len: int):
    """Normalise conditioning to (B, Lc, cond_dim) with Lc in {1, seq_len}.
    """
    if c is None:
        return None
    if c.dim() == 2:                      # (B, cond_dim)
        return c.unsqueeze(1)
    if c.dim() != 3:
        raise ValueError(f"conditioning must be 2D or 3D, got shape {tuple(c.shape)}")
    if c.shape[1] > 1:
        first = c[:, :1]
        if torch.equal(c, first.expand_as(c)):
            return first
    return c
