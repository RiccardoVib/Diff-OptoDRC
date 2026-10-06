"""
grey-box: code adapted from https://github.com/Alec-Wright/GreyBoxDRC
grey-box DRC (Wright & Valimaki, DAFx 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from GreyCommon import ParamHead, lin_to_db


class StaticCurve(nn.Module):
    """Instantaneous level (dB) -> gain (dB).
    """

    def __init__(self, curve_type="sk", cond_size=1, hidden=20, min_db=-80.0,
                 max_ratio=30.0, max_knee_db=30.0, knee_type="quadratic",
                 min_knee_db=1e-3, HC_hidden=None,
                 threshold_init_db=None, ratio_init=None, knee_init_db=None):
        super().__init__()

        if HC_hidden is not None:
            hidden = HC_hidden
        if curve_type not in ("hk", "sk"):
            raise ValueError(f"curve_type must be 'hk' or 'sk', got {curve_type!r}")
        if knee_type not in ("quadratic", "softplus"):
            raise ValueError(f"knee_type must be 'quadratic' or 'softplus', got {knee_type!r}")
        self.curve_type = curve_type
        self.knee_type = knee_type
        self.min_db = float(min_db)
        self.max_ratio = float(max_ratio)
        self.max_knee_db = float(max_knee_db)
        self.min_knee_db = float(min_knee_db)
        self.cond_size = int(cond_size)

        n_out = 2 if curve_type == "hk" else 3

        init_bias = None
        if any(v is not None for v in (threshold_init_db, ratio_init, knee_init_db)):
            import math

            def logit(u):
                u = min(max(float(u), 1e-6), 1 - 1e-6)
                return math.log(u / (1 - u))

            t = -20.0 if threshold_init_db is None else float(threshold_init_db)
            r = 4.0 if ratio_init is None else float(ratio_init)
            k = 6.0 if knee_init_db is None else float(knee_init_db)
            init_bias = [logit(t / self.min_db), logit((r - 1.0) / self.max_ratio)]
            if n_out == 3:
                init_bias.append(logit((k - self.min_knee_db) / self.max_knee_db))

        self.head = ParamHead(cond_size, n_out, hidden=hidden, init_bias=init_bias)

    # ------------------------------------------------------------------
    def get_pars(self, cond, batch=None, device=None, dtype=None):
        """Return (threshold_db, ratio, knee_db), each ``[B, 1, 1]``."""
        raw = self.head(cond, batch=batch, device=device, dtype=dtype)
        threshold = self.min_db * torch.sigmoid(raw[:, 0:1]).unsqueeze(1)
        ratio = self.max_ratio * torch.sigmoid(raw[:, 1:2]).unsqueeze(1) + 1.0
        if self.curve_type == "hk":
            knee = torch.zeros_like(ratio)
        else:
            knee = self.max_knee_db * torch.sigmoid(raw[:, 2:3]).unsqueeze(1) + self.min_knee_db
        return threshold, ratio, knee

    # ------------------------------------------------------------------
    def gain_from_db(self, x_db, cond=None):
        """Gain in dB for an input level in dB. ``x_db`` is ``[B, L, 1]``."""
        b = x_db.shape[0]
        threshold, ratio, knee = self.get_pars(
            cond, batch=b, device=x_db.device, dtype=x_db.dtype)
        over = x_db - threshold
        slope = 1.0 - 1.0 / ratio                      # dB of reduction per dB over

        if self.curve_type == "hk":
            return -slope * torch.clamp(over, min=0.0)

        if self.knee_type == "softplus":
            # W acts as the softness scale; W -> 0 recovers the hard knee.
            w = knee
            return -slope * w * F.softplus(over / w)

        # Quadratic soft knee, exactly as in the paper: knee region is
        # |2*(x - T)| <= W, i.e. |x - T| <= W/2.
        w = knee
        in_knee = (2.0 * over).abs() <= w
        above = (2.0 * over) > w
        # Guard the 1/(2W) so the unselected branch cannot be inf (NaN grads).
        knee_red = slope * (over + w / 2.0) ** 2 / (2.0 * w)
        red = torch.where(in_knee, knee_red, torch.zeros_like(knee_red))
        red = red + torch.where(above, slope * over, torch.zeros_like(over))
        return -red

    def forward(self, x, cond=None):
        """Linear audio ``[B, L, 1]`` -> static gain in dB ``[B, L, 1]``."""
        return self.gain_from_db(lin_to_db(x, self.min_db), cond)

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------
    def curve(self, cond=None, n=1000, device=None, dtype=torch.float32):
        """Return (input_db, output_db) for the characteristic, ``[n]`` each."""
        device = device or next(self.parameters()).device
        x_db = torch.linspace(self.min_db, 0.0, n, device=device,
                              dtype=dtype).view(1, n, 1)
        g_db = self.gain_from_db(x_db, cond)
        return x_db.squeeze(), (x_db + g_db).squeeze()

    @torch.no_grad()
    def measure(self, cond=None, n=4000, device=None):
        """Read threshold / ratio / knee back off the rendered curve.
        """
        x_db, y_db = self.curve(cond=cond, n=n, device=device)
        dx = x_db[1] - x_db[0]
        slope = torch.gradient(y_db, spacing=(x_db,))[0]
        top = slope[int(0.99 * n):].mean()
        ratio = float(1.0 / torch.clamp(top, min=1e-6))
        # asymptote y = top * x + c fitted at the top of the range
        c = float(y_db[-1] - top * x_db[-1])
        threshold = float(c / (1.0 - top)) if abs(1.0 - float(top)) > 1e-6 else float("nan")
        lo, hi = 1.0 - 0.05 * (1.0 - float(top)), 1.0 - 0.95 * (1.0 - float(top))
        in_knee = (slope <= lo) & (slope >= hi)
        knee = float(in_knee.sum() * dx) if bool(in_knee.any()) else 0.0
        return {"threshold_db": threshold, "ratio": ratio, "knee_db": knee}

    def describe(self, cond=None, device=None):
        t, r, k = self.get_pars(cond, batch=1, device=device or torch.device("cpu"),
                                dtype=torch.float32)
        return {
            "threshold_db": float(t.detach().reshape(-1)[0]),
            "ratio": float(r.detach().reshape(-1)[0]),
            "knee_db": float(k.detach().reshape(-1)[0]),
            "curve_type": self.curve_type,
            "knee_type": self.knee_type if self.curve_type == "sk" else "hard",
        }

    def get_leg(self, cond=None):
        d = self.describe(cond)
        s = f" T={-d['threshold_db']:.0f}, R={min(d['ratio'], 30.0):.0f},"
        if self.curve_type == "sk":
            s += f" W={d['knee_db']:.0f},"
        return s