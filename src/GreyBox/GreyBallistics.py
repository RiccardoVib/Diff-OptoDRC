"""
grey-box: code adapted from https://github.com/Alec-Wright/GreyBoxDRC
grey-box DRC (Wright & Valimaki, DAFx 2022)
"""

import math
import torch
import torch.nn as nn
from GreyCommon import ParamHead, next_pow2


# --------------------------------------------------------------------------

def one_minus_alpha(log_alpha):
    return -torch.expm1(log_alpha)


def onepole_parallel(x, log_alpha, h0):
    L = x.shape[1]
    n = torch.arange(L, device=x.device, dtype=x.dtype).view(1, L, 1)
    ir = one_minus_alpha(log_alpha) * torch.exp(n * log_alpha)  # [B|1, L, F]
    nfft = next_pow2(2 * L)
    y = torch.fft.irfft(
        torch.fft.rfft(x, n=nfft, dim=1) * torch.fft.rfft(ir, n=nfft, dim=1),
        n=nfft, dim=1)[:, :L, :]
    return y + h0 * torch.exp((n + 1) * log_alpha)


def onepole_recursive(x, log_alpha, h0):
    """Sample-by-sample reference implementation."""
    oma = one_minus_alpha(log_alpha)
    h = h0
    out = []
    for n in range(x.shape[1]):
        h = h + oma * (x[:, n:n + 1, :] - h)
        out.append(h)
    return torch.cat(out, dim=1)


# --------------------------------------------------------------------------

def geometric_map(raw, lo, hi):
    return lo * (hi / lo) ** torch.sigmoid(raw)


def geometric_map_inverse(tau, lo, hi):
    u = math.log(tau / lo) / math.log(hi / lo)
    u = min(max(u, 1e-6), 1 - 1e-6)
    return math.log(u / (1 - u))


class _Smoother(nn.Module):
    """Common state handling."""

    n_filters = 1
    cond = False

    def _init_state(self):
        self.register_buffer("iir_state", torch.zeros(1, 1, self.n_filters),
                             persistent=False)

    def reset_state(self, batch_size=1):
        dev = self.iir_state.device
        dtype = self.iir_state.dtype
        self.iir_state = torch.zeros(batch_size, 1, self.n_filters,
                                     device=dev, dtype=dtype)

    def detach_state(self):
        self.iir_state = self.iir_state.detach()

    def prime(self, value):

        v = value if torch.is_tensor(value) else torch.tensor(float(value))
        v = v.reshape(v.shape[0] if v.dim() else 1, 1, 1)
        self.iir_state = v.expand(-1, 1, self.n_filters).contiguous()

    def _state_for(self, x):
        s = self.iir_state
        if s.shape[0] != x.shape[0] or s.device != x.device or s.dtype != x.dtype:
            s = torch.zeros(x.shape[0], 1, self.n_filters,
                            device=x.device, dtype=x.dtype)
            self.iir_state = s
        return s

    def verbose_forward(self, x, cond=None):
        """Return (output, per-sample tau in seconds)."""
        y = self(x, cond)
        taus = self.taus(cond, batch=x.shape[0], device=x.device, dtype=x.dtype)
        return y, taus.mean(dim=-1, keepdim=True).expand_as(y)


# --------------------------------------------------------------------------

class OnePoleAttOnly(_Smoother):
    """Single time constant, attack == release.  LTI -> parallelisable."""

    def __init__(self, cond=False, cond_size=1, n_filters=1, hidden=20,
                 sample_rate=44100, tau_init=0.05, tau_min=1e-4, tau_max=1.0,
                 parallel_max_len=40000, trunc_steps=None, tau_param="sigmoid"):
        super().__init__()
        if tau_param not in ("sigmoid", "log"):
            raise ValueError("tau_param must be 'sigmoid' or 'log'")
        self.tau_param = tau_param
        self.cond = bool(cond)
        self.n_filters = int(n_filters)
        self.sample_rate = float(sample_rate)
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)
        self.parallel_max_len = int(parallel_max_len)
        if self.cond:
            ib = None
            if tau_param == "log":
                ib = [geometric_map_inverse(tau_init, tau_min, tau_max)] * int(n_filters)
            self.head = ParamHead(cond_size, self.n_filters, hidden=hidden,
                                  init_bias=ib)
            self.raw_taus = None
        else:
            init = torch.full((1, 1, self.n_filters), float(tau_init))
            self.raw_taus = nn.Parameter(init)
            self.head = None
        self._init_state()

    def taus(self, cond=None, batch=1, device=None, dtype=None):
        if self.cond:
            raw = self.head(cond, batch=batch, device=device, dtype=dtype).unsqueeze(1)
            if self.tau_param == "log":
                return geometric_map(raw, self.tau_min, self.tau_max)
            return 0.5 * torch.sigmoid(raw)               # [B,1,F], paper form
        t = self.raw_taus.clamp(min=self.tau_min, max=self.tau_max)
        if device is not None:
            t = t.to(device=device, dtype=dtype)
        return t

    def forward(self, x, cond=None):
        h0 = self._state_for(x)
        taus = self.taus(cond, batch=x.shape[0], device=x.device, dtype=x.dtype)
        log_alpha = -1.0 / (taus * self.sample_rate)
        if self.n_filters > 1:
            x = x.expand(-1, -1, self.n_filters)
        y = (onepole_parallel(x, log_alpha, h0)
             if x.shape[1] <= self.parallel_max_len
             else onepole_recursive(x, log_alpha, h0))
        self.iir_state = y[:, -1:, :]
        return y.mean(dim=2, keepdim=True)

    def describe(self, cond=None, device=None):
        t = self.taus(cond, batch=1, device=device, dtype=torch.float32).detach()
        return {"tau_ms": [float(v) * 1e3 for v in t.reshape(-1)],
                "attack_ms": float(t.reshape(-1).mean()) * 1e3,
                "release_ms": float(t.reshape(-1).mean()) * 1e3}


class OnePoleAttRel(_Smoother):
    """Independent attack and release time constants.
    """

    def __init__(self, cond=False, cond_size=1, hidden=20, sample_rate=44100,
                 tau_attack_init=0.005, tau_release_init=0.150,
                 order_taus=False, trunc_steps=None,
                 tau_param="sigmoid",
                 tau_attack_range=(5e-4, 0.3),
                 tau_release_range=(0.05, 10.0)):
        super().__init__()
        if tau_param not in ("sigmoid", "log"):
            raise ValueError("tau_param must be 'sigmoid' or 'log'")
        self.cond = bool(cond)
        self.n_filters = 1
        self.sample_rate = float(sample_rate)
        self.order_taus = bool(order_taus)
        self.tau_param = tau_param
        self.tau_attack_range = tuple(float(v) for v in tau_attack_range)
        self.tau_release_range = tuple(float(v) for v in tau_release_range)
        if tau_param == "log":
            lo_a, hi_a = self.tau_attack_range
            lo_r, hi_r = self.tau_release_range
            if not (lo_a <= tau_attack_init <= hi_a):
                raise ValueError(f"tau_attack_init={tau_attack_init} outside "
                                 f"tau_attack_range={self.tau_attack_range}")
            if not (lo_r <= tau_release_init <= hi_r):
                raise ValueError(f"tau_release_init={tau_release_init} outside "
                                 f"tau_release_range={self.tau_release_range}")

        def _raw_init():
            def inv_sig(v):
                v = min(max(v, 1e-6), 1 - 1e-6)
                return math.log(v / (1 - v))
            if tau_param == "log":
                a0 = geometric_map_inverse(tau_attack_init, *self.tau_attack_range)
                if self.order_taus:
                    gap = math.log(tau_release_init / tau_attack_init)
                    r0 = math.log(math.expm1(max(gap, 1e-6)))
                else:
                    r0 = geometric_map_inverse(tau_release_init, *self.tau_release_range)
            else:
                a0, r0 = inv_sig(tau_attack_init), inv_sig(tau_release_init)
            return [a0, r0]

        if self.cond:
            self.head = ParamHead(cond_size, 2, hidden=hidden,
                                  init_bias=_raw_init())
            self.raw_taus = None
        else:
            # sigmoid^-1 of the target time constants (taus = sigmoid(raw))
            def inv_sig(v):
                v = min(max(v, 1e-6), 1 - 1e-6)
                return math.log(v / (1 - v))

            self.raw_taus = nn.Parameter(torch.tensor([_raw_init()]).unsqueeze(0))
            self.head = None
        self._init_state()

    def taus(self, cond=None, batch=1, device=None, dtype=None):
        if self.cond:
            raw = self.head(cond, batch=batch, device=device, dtype=dtype).unsqueeze(1)
        else:
            raw = self.raw_taus
            if device is not None:
                raw = raw.to(device=device, dtype=dtype)
        if self.tau_param == "log":
            att = geometric_map(raw[..., 0:1], *self.tau_attack_range)
            if self.order_taus:
                # release >= attack by construction, and bounded by the range
                rel = att * torch.exp(torch.nn.functional.softplus(raw[..., 1:2]))
                rel = rel.clamp(max=self.tau_release_range[1])
            else:
                rel = geometric_map(raw[..., 1:2], *self.tau_release_range)
            return torch.cat([att, rel], dim=-1)

        t = torch.sigmoid(raw)  # [B,1,2], seconds
        if self.order_taus:
            att = t[..., 0:1]
            rel = att + torch.nn.functional.softplus(raw[..., 1:2])
            t = torch.cat([att, rel], dim=-1)
        return t

    def forward(self, x, cond=None):
        h = self._state_for(x)
        taus = self.taus(cond, batch=x.shape[0], device=x.device, dtype=x.dtype)
        log_alpha = -1.0 / (taus * self.sample_rate)
        oma = one_minus_alpha(log_alpha)          # stable at long tau
        att, rel = oma[..., 0:1], oma[..., 1:2]
        out = []
        for n in range(x.shape[1]):
            gc = x[:, n:n + 1, :]
            k = torch.where(h > gc, att, rel)
            h = h + k * (gc - h)
            out.append(h)
        y = torch.cat(out, dim=1)
        self.iir_state = y[:, -1:, :]
        return y

    def describe(self, cond=None, device=None):
        t = self.taus(cond, batch=1, device=device, dtype=torch.float32).detach().reshape(-1)
        return {"attack_ms": float(t[0]) * 1e3, "release_ms": float(t[1]) * 1e3}

    def verbose_forward(self, x, cond=None):
        y = self(x, cond)
        t = self.taus(cond, batch=x.shape[0], device=x.device, dtype=x.dtype)
        return y, t[..., 0:1].expand_as(y)


class TimeVaryOP(_Smoother):
    """A recurrent unit predicts the time constant at every sample.
    """

    def __init__(self, hidden_size=8, rec="gru", cond=False, cond_size=1,
                 sample_rate=44100, tau_max=1.0, tau_min=1e-4,
                 tau_param="sigmoid", trunc_steps=None):
        super().__init__()
        rec = rec.lower()
        if rec == "gru":
            self.rec = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        elif rec == "rnn":
            self.rec = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        else:
            raise ValueError(f"rec must be 'gru' or 'rnn', got {rec!r}")
        self.lin = nn.Linear(hidden_size, 1)
        self.n_filters = 1
        self.cond = False  # tau comes from the signal, not c
        self.sample_rate = float(sample_rate)
        self.tau_max = float(tau_max)
        self.tau_min = float(tau_min)
        self.tau_param = tau_param
        self.state = None
        self._init_state()

    def reset_state(self, batch_size=1):
        super().reset_state(batch_size)
        self.state = None

    def detach_state(self):
        super().detach_state()
        if self.state is not None:
            self.state = self.state.detach()

    def _run(self, g):
        h0 = self._state_for(g)
        taus, self.state = self.rec(g, self.state)
        raw = self.lin(taus)
        taus = (geometric_map(raw, self.tau_min, self.tau_max)
                if self.tau_param == "log"
                else self.tau_max * torch.sigmoid(raw))
        oma = one_minus_alpha(-1.0 / (taus * self.sample_rate))
        h = h0
        out = []
        for n in range(g.shape[1]):
            k = oma[:, n:n + 1, :]
            h = h + k * (g[:, n:n + 1, :] - h)
            out.append(h)
        y = torch.cat(out, dim=1)
        self.iir_state = y[:, -1:, :]
        return y, taus

    def forward(self, g, cond=None):
        return self._run(g)[0]

    def verbose_forward(self, g, cond=None):
        return self._run(g)

    def describe(self, cond=None, device=None):
        return {"attack_ms": float("nan"), "release_ms": float("nan"),
                "note": "time-varying; tau is signal dependent"}


SMOOTHERS = {
    "OnePoleAttOnly": OnePoleAttOnly,
    "OnePoleAttRel": OnePoleAttRel,
    "TimeVaryOP": TimeVaryOP,
}


class GainSmooth(nn.Module):
    """Wrapper matching the original ``GainSmooth`` API.
    """

    def __init__(self, params, trunc_steps=None, min_db=-80.0, sample_rate=44100):
        super().__init__()
        params = dict(params)
        self.smooth_type = params.pop("type")
        if self.smooth_type not in SMOOTHERS:
            raise ValueError(f"unknown gain_smooth type {self.smooth_type!r}; "
                             f"expected one of {sorted(SMOOTHERS)}")
        params.setdefault("sample_rate", sample_rate)
        if self.smooth_type == "TimeVaryOP":
            params.setdefault("hidden_size", 8)
            params.setdefault("rec", "gru")
        self.model = SMOOTHERS[self.smooth_type](trunc_steps=trunc_steps, **params)
        self.tvop = self.smooth_type == "TimeVaryOP"
        self.min_db = float(min_db)
        self.data_range = abs(float(min_db))
        self.sample_rate = float(sample_rate)

    @property
    def cond(self):
        return self.model.cond

    def forward(self, gains, cond=None):
        g = gains / self.data_range
        g = self.model(g, cond) if self.model.cond else self.model(g)
        return g * self.data_range

    def verbose_forward(self, gains, cond=None):
        g = gains / self.data_range
        g, taus = (self.model.verbose_forward(g, cond) if self.model.cond
                   else self.model.verbose_forward(g))
        return g * self.data_range, taus

    def reset_state(self, batch_size=1):
        self.model.reset_state(batch_size)

    def detach_state(self):
        self.model.detach_state()

    def prime(self, gain_db):
        """Seed the smoother state from a gain in dB (pre-scaling applied)."""
        self.model.prime(gain_db / self.data_range)

    def describe(self, cond=None, device=None):
        return self.model.describe(cond, device=device)

    # ------------------------------------------------------------------
    def make_step_in(self, step_samps=4410, level_db=-40.0, device=None):
        z = torch.zeros(step_samps, 1, device=device)
        return torch.cat((z, level_db * torch.ones(step_samps, 1, device=device), z))

    @torch.no_grad()
    def step_response(self, cond=None, step_samps=8820, level_db=-40.0, device=None):
        """Gain-reduction step response, for measuring attack/release."""
        device = device or next(self.parameters()).device
        saved = self.model.iir_state
        self.reset_state(1)
        x = self.make_step_in(step_samps, level_db, device).unsqueeze(0)
        y = self(x, None if not self.model.cond else cond)
        self.model.iir_state = saved
        return y[0, :, 0], x[0, :, 0]

    get_step_resp = step_response