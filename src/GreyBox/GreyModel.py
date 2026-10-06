"""
grey-box: code adapted from https://github.com/Alec-Wright/GreyBoxDRC
grey-box DRC (Wright & Valimaki, DAFx 2022)
"""

import torch
import torch.nn as nn

from GreyBallistics import GainSmooth
from GreyCommon import as_cond, db_to_lin, resolve_routing, slice_cond
from GreyMakeup import MakeUp
from GreyStaticCurve import StaticCurve



class GreyBoxComp(nn.Module):
    def __init__(self, cond_dim=None, static_comp=None, gain_smooth=None,
                 make_up=None, sample_rate=44100, min_db=-80.0, cond_names=None,
                 cond_routing=None, seq_len=None, trunc_steps=None):
        super().__init__()


        if cond_names is None and cond_dim is not None:
            cond_names = [f"c{i}" for i in range(int(cond_dim))]
        if cond_dim is not None and cond_names is not None \
                and len(cond_names) != int(cond_dim):
            raise ValueError(
                f"cond_dim={cond_dim} but cond_names={list(cond_names)}")
        trunc_steps = trunc_steps if trunc_steps is not None else seq_len
        static_comp = dict(static_comp or {"type": "sk"})
        gain_smooth = dict(gain_smooth or {"type": "OnePoleAttRel", "cond": False})
        make_up = dict(make_up or {"type": "Static"})

        self.sample_rate = float(sample_rate)
        self.min_db = float(min_db)
        self.cond_names = list(cond_names) if cond_names else []
        self.cond_dim = len(self.cond_names)
        self.routing = resolve_routing(self.cond_names, cond_routing)

        curve_type = static_comp.pop("type")
        self.static_comp = StaticCurve(
            curve_type=curve_type, cond_size=len(self.routing["curve"]),
            min_db=min_db, **static_comp)

        if "cond_size" not in gain_smooth and gain_smooth.get("cond", False):
            gain_smooth["cond_size"] = len(self.routing["ballistics"])
        self.gain_smooth = GainSmooth(gain_smooth, trunc_steps=trunc_steps,
                                      min_db=min_db, sample_rate=sample_rate)

        self.make_up = MakeUp(make_up)


    # ------------------------------------------------------------------
    def _split_cond(self, cond, batch, device, dtype):
        c = as_cond(cond, batch, device, dtype)
        if self.cond_dim and c.shape[-1] != self.cond_dim:
            raise ValueError(
                f"expected {self.cond_dim} controls {self.cond_names}, "
                f"got {c.shape[-1]}")
        return (slice_cond(c, self.routing["curve"]),
                slice_cond(c, self.routing["ballistics"]),
                slice_cond(c, self.routing["makeup"]))

    def forward(self, x, cond=None):
        """``x`` is ``[B, L, 1]`` linear audio.  Returns ``[B, L, 1]``."""
        return self.forward_verbose(x, cond)["y"]

    def forward_verbose(self, x, cond=None):
        if x.dim() != 3 or x.shape[-1] != 1:
            raise ValueError(f"expected [B, L, 1] audio, got {tuple(x.shape)}")
        c_curve, c_ball, _ = self._split_cond(cond, x.shape[0], x.device, x.dtype)

        gc = self.static_comp(x, c_curve)                 # static gain, dB
        if self.gain_smooth.tvop:
            gs, taus = self.gain_smooth.verbose_forward(gc, c_ball)
        else:
            gs, taus = self.gain_smooth(gc, c_ball), None
        x_comp = x * db_to_lin(gs)
        y = self.make_up(x_comp)
        return {"x": x, "gain_static_db": gc, "gain_db": gs,
                "x_comp": x_comp, "y": y, "taus": taus}

    def gain_db(self, x, cond=None):

        return self.forward_verbose(x, cond)["gain_db"]

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def reset_hidden_states(self, batch_size=1):
        self.gain_smooth.reset_state(batch_size)
        self.make_up.reset_state(batch_size)

    def detach_hidden_states(self):
        self.gain_smooth.detach_state()
        self.make_up.detach_state()

    @torch.no_grad()
    def prime_from(self, x, cond=None):
        """Seed the smoother from the static gain at the first sample of ``x``.

        """
        c_curve, _, _ = self._split_cond(cond, x.shape[0], x.device, x.dtype)
        gc0 = self.static_comp(x[:, :1], c_curve)
        self.gain_smooth.prime(gc0.reshape(x.shape[0], 1, 1))

    reset_states = reset_hidden_states
    detach_states = detach_hidden_states

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------
    def _cond_for_inspection(self, cond, device):
        c = as_cond(cond, 1, device, torch.float32)
        return (slice_cond(c, self.routing["curve"]),
                slice_cond(c, self.routing["ballistics"]))

    @torch.no_grad()
    def describe(self, cond=None, device=None):
        device = device or next(self.parameters()).device
        c_curve, c_ball = self._cond_for_inspection(cond, device)
        d = {"n_params": sum(p.numel() for p in self.parameters()),
             "sample_rate": self.sample_rate,
             "cond_names": self.cond_names,
             "routing": {k: [self.cond_names[i] for i in v]
                         for k, v in self.routing.items()}}
        d["curve"] = self.static_comp.describe(c_curve, device=device)
        d["curve_measured"] = self.static_comp.measure(c_curve, device=device)
        d["ballistics"] = self.gain_smooth.describe(c_ball, device=device)
        d["make_up"] = self.make_up.describe()
        return d

    def static_curve(self, cond=None, n=1000, device=None):
        device = device or next(self.parameters()).device
        c_curve, _ = self._cond_for_inspection(cond, device)
        return self.static_comp.curve(c_curve, n=n, device=device)

    def step_response(self, cond=None, **kw):
        device = kw.pop("device", None) or next(self.parameters()).device
        _, c_ball = self._cond_for_inspection(cond, device)
        return self.gain_smooth.step_response(c_ball, device=device, **kw)