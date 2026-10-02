# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato

import torch
import torch.nn as nn
import torch.nn.functional as F
from CompressorUtils import inverse_softplus, logit, zero_init_head, match_cond_shape


class SoftKneeGainCurve(nn.Module):
    def __init__(
        self,
        cond_dim=None,
        hidden: int = 16,
        threshold_range=(-60.0, 0.0),
        threshold_init_db: float = -20.0,
        ratio_init: float = 4.0,
        knee_init_db: float = 6.0,
        knee_type: str = "softplus",
        n_knots: int = 8,
        residual_db: float = 0.0,
        residual_hidden: int = 16,
    ):
        super().__init__()
        if knee_type not in ("softplus", "quadratic", "free_monotone"):
            raise ValueError("knee_type must be 'softplus', 'quadratic' or 'free_monotone'")
        self.knee_type = knee_type
        self.cond_dim = cond_dim if cond_dim else 0
        self.t_min, self.t_max = float(threshold_range[0]), float(threshold_range[1])
        self.residual_db = float(residual_db)

        slope_init = 1.0 - 1.0 / float(ratio_init)

        if knee_type == "free_monotone":
            self.n_knots = int(n_knots)
            knots = torch.linspace(self.t_min, self.t_max, self.n_knots)
            self.register_buffer("knots", knots)
            spacing = float(knots[1] - knots[0]) if self.n_knots > 1 else knee_init_db
            theta0 = -((knots - threshold_init_db) / max(knee_init_db, 1e-3)) ** 2
            shape_bias = torch.cat([
                torch.tensor([logit(slope_init)]),                      # nu -> max ratio
                theta0,                                                 # where compression happens
                torch.tensor([inverse_softplus(max(spacing, 1e-3))]),   # knot width
            ])
        else:
            shape_bias = torch.tensor([
                logit((threshold_init_db - self.t_min) / (self.t_max - self.t_min)),
                logit(slope_init),
                inverse_softplus(knee_init_db),
            ])

        self.n_shape = shape_bias.numel()
        bias = shape_bias

        if self.cond_dim > 0:
            self.net = nn.Sequential(nn.Linear(self.cond_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, bias.numel()))
            zero_init_head(self.net[-1], bias)          # start exactly at the prior
        else:
            self.net = None
            self.raw = nn.Parameter(bias.view(1, 1, -1))

        if self.residual_db > 0.0:
            self.res_net = nn.Sequential(nn.Linear(1 + self.cond_dim, residual_hidden),
                                         nn.SiLU(), nn.Linear(residual_hidden, 1))
            zero_init_head(self.res_net[-1], 0.0)
        else:
            self.res_net = None

    # ------------------------------------------------------------------ #

    def compute_params(self, c=None, seq_len: int = 1, c_makeup=None):
        """-> dict of (B, Lc, .) tensors.  Call once per chunk."""
        if self.net is not None:
            c = match_cond_shape(c, seq_len)
            if c is None:
                raise ValueError("this curve was built with cond_dim > 0 but c is None")
            raw = self.net(c)
        else:
            raw = self.raw

        shape, tail = raw[..., :self.n_shape], raw[..., self.n_shape:]

        p = {}

        if self.knee_type == "free_monotone":
            nu = torch.sigmoid(shape[..., :1])                   # total slope <= nu
            p["weights"] = nu * torch.softmax(shape[..., 1:1 + self.n_knots], dim=-1)
            p["nu"] = nu
            p["knot_width_db"] = F.softplus(shape[..., 1 + self.n_knots:2 + self.n_knots]) + 1e-2
        else:
            t_raw, s_raw, k_raw = shape.split(1, dim=-1)
            p["threshold_db"] = self.t_min + (self.t_max - self.t_min) * torch.sigmoid(t_raw)
            p["slope"] = torch.sigmoid(s_raw)                    # 1 - 1/ratio
            p["knee_db"] = F.softplus(k_raw) + 1e-3              # strictly > 0
        return p

    def apply_curve(self, level_db, p):
        """Pure arithmetic.  ``level_db`` broadcastable against the params.
        """

        if self.knee_type == "free_monotone":
            # level_db (..., C) -> (..., C, 1);  params (..., 1|K) -> (..., 1, 1|K)
            s = p["knot_width_db"].unsqueeze(-2)
            w = p["weights"].unsqueeze(-2)
            hinge = s * F.softplus((level_db.unsqueeze(-1) - self.knots) / s)
            return -(w * hinge).sum(-1)

        T, W = p["threshold_db"], p["knee_db"]
        over = level_db - T

        if self.knee_type == "softplus":
            reduction = W * F.softplus(over / W)
        else:
            quad = (over + 0.5 * W) ** 2 / (2.0 * W)             # finite: W >= 1e-3
            reduction = torch.where(
                2.0 * over < -W, torch.zeros_like(over),
                torch.where(2.0 * over > W, over, quad))
        return -p["slope"] * reduction

    def forward(self, level_db, c=None, params=None, c_makeup=None, with_makeup: bool = True):
        """level_db: (B, L, C) -> gain_db: (B, L, C)."""
        if params is None:
            params = self.compute_params(c, seq_len=level_db.shape[1], c_makeup=c_makeup)
        gain_db = self.apply_curve(level_db, params)

        if self.res_net is not None:
            feats = [level_db.unsqueeze(-1) / 20.0]
            if self.cond_dim > 0:
                cc = match_cond_shape(c, level_db.shape[1])
                cc = cc.expand(-1, level_db.shape[1], -1) if cc.shape[1] == 1 else cc
                feats.append(cc.unsqueeze(2).expand(-1, -1, level_db.shape[2], -1))
            gain_db = gain_db + self.residual_db * torch.tanh(
                self.res_net(torch.cat(feats, dim=-1)).squeeze(-1))
        return gain_db, params

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def measure(self, c=None, c_makeup=None, lo_db: float = None, hi_db: float = None,
                step: float = 0.25):
        """Read threshold / max ratio / knee width back *off* the fitted curve.
        """
        lo = self.t_min - 20.0 if lo_db is None else lo_db
        hi = self.t_max + 20.0 if hi_db is None else hi_db
        p = self.compute_params(c, seq_len=1, c_makeup=c_makeup)
        p0 = {k: v[0:1, 0:1] for k, v in p.items()}

        lvl = torch.arange(lo, hi, step, device=next(iter(p0.values())).device).view(1, -1, 1)
        out = lvl + self.apply_curve(lvl, p0)
        comp = 1.0 - torch.diff(out, dim=1) / step          # compression slope in [0, 1]
        centres = lvl[0, :-1, 0] + step / 2
        comp = comp[0, :, 0]

        nu = float(comp.max())
        res = {"ratio_max": 1.0 / max(1.0 - nu, 1e-6), "slope_max": nu}
        if nu > 1e-6:
            def cross(frac):
                idx = torch.nonzero(comp >= frac * nu)
                return float(centres[idx[0]]) if len(idx) else float("nan")
            res["threshold_db"] = cross(0.5)
            res["knee_db"] = cross(0.9) - cross(0.1)
        return res

    def extra_repr(self):
        return (f"cond_dim={self.cond_dim}, "
                f"knee_type={self.knee_type}, "
                f"threshold_range=({self.t_min:g}, {self.t_max:g}), "
                f"residual_db={self.residual_db:g}")
