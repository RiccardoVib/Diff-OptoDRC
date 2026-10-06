"""
grey-box: code adapted from https://github.com/Alec-Wright/GreyBoxDRC
grey-box DRC (Wright & Valimaki, DAFx 2022)
"""

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# dB helpers
# --------------------------------------------------------------------------

def lin_to_db(x, min_db=-80.0):
    """|x| in dBFS, floored at ``min_db``.
    """
    eps = 10.0 ** (min_db / 20.0)
    return 20.0 * torch.log10(torch.clamp(x.abs(), min=eps))


def db_to_lin(g_db):
    return torch.pow(10.0, g_db / 20.0)


def tau_to_alpha(tau_s, sample_rate):
    """One-pole coefficient for a time constant in seconds."""
    return torch.exp(-1.0 / (tau_s * sample_rate))


def alpha_to_tau(alpha, sample_rate):
    return -1.0 / (torch.log(alpha) * sample_rate)


# --------------------------------------------------------------------------
# conditioning
# --------------------------------------------------------------------------

def as_cond(cond, batch, device, dtype):
    """Normalise a conditioning tensor to ``[B, C]``.
    """
    if cond is None:
        return torch.zeros(batch, 0, device=device, dtype=dtype)
    if not torch.is_tensor(cond):
        cond = torch.as_tensor(cond, device=device, dtype=dtype)
    cond = cond.to(device=device, dtype=dtype)
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    if cond.dim() == 3:
        cond = cond[:, 0, :]
    if cond.dim() != 2:
        raise ValueError(f"conditioning must be 1D/2D/3D, got shape {tuple(cond.shape)}")
    if cond.shape[0] == 1 and batch > 1:
        cond = cond.expand(batch, -1)
    if cond.shape[0] != batch:
        raise ValueError(f"conditioning batch {cond.shape[0]} != signal batch {batch}")
    return cond


class ParamHead(nn.Module):
    """Maps a conditioning vector to ``out_dim`` raw parameters.
    """

    def __init__(self, cond_dim, out_dim, hidden=20, n_hidden=2, activation=nn.Tanh,
                 init_bias=None):
        """``init_bias``: raw parameter values the head should emit at init.
        """
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.out_dim = int(out_dim)
        if init_bias is not None:
            init_bias = torch.as_tensor(init_bias, dtype=torch.float32).reshape(-1)
            if init_bias.numel() != self.out_dim:
                raise ValueError(f"init_bias has {init_bias.numel()} entries, "
                                 f"expected out_dim={self.out_dim}")
        if self.cond_dim == 0:
            self.bias = nn.Parameter(torch.zeros(out_dim) if init_bias is None
                                     else init_bias.clone())
            self.net = None
        else:
            layers, d = [], self.cond_dim
            for _ in range(n_hidden):
                layers += [nn.Linear(d, hidden), activation()]
                d = hidden
            last = nn.Linear(d, out_dim)
            if init_bias is not None:
                with torch.no_grad():
                    last.weight.zero_()
                    last.bias.copy_(init_bias)
            layers += [last]
            self.net = nn.Sequential(*layers)
            self.bias = None

    def forward(self, cond, batch=None, device=None, dtype=None):
        if self.net is None:
            b = batch if batch is not None else (1 if cond is None else cond.shape[0])
            p = self.bias
            if device is not None:
                p = p.to(device=device, dtype=dtype)
            return p.unsqueeze(0).expand(b, -1)
        if cond is None or cond.shape[-1] != self.cond_dim:
            got = "None" if cond is None else cond.shape[-1]
            raise ValueError(
                f"ParamHead expects cond_dim={self.cond_dim}, got {got}. "
                "Check cond_routing / cond_names."
            )
        return self.net(cond)


# --------------------------------------------------------------------------
# conditioning routing
# --------------------------------------------------------------------------

def resolve_routing(cond_names, routing):
    """Turn ``{'curve': ['peak_reduction'], ...}`` into index tensors.
    """
    blocks = ("curve", "ballistics", "makeup")
    if routing is None:
        n = len(cond_names) if cond_names else 0
        return {b: list(range(n)) for b in blocks}
    name_to_idx = {n: i for i, n in enumerate(cond_names or [])}
    out = {}
    for b in blocks:
        names = routing.get(b, [])
        idx = []
        for nm in names:
            if nm not in name_to_idx:
                raise KeyError(
                    f"cond_routing['{b}'] wants '{nm}' but cond_names={list(cond_names)}"
                )
            idx.append(name_to_idx[nm])
        out[b] = idx
    return out


def slice_cond(cond, idx):
    """Select the control columns a block is allowed to see."""
    if cond is None:
        return None
    if len(idx) == 0:
        return cond[:, :0]
    return cond[:, idx]


def next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


__all__ = [
    "lin_to_db", "db_to_lin", "tau_to_alpha", "alpha_to_tau",
    "as_cond", "ParamHead", "resolve_routing", "slice_cond", "next_pow2",
]