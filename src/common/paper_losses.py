# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato

"""
Loss functions as used in the original papers of each baseline

Which paper uses what:

  micro-tcn (Steinmetz & Reiss, AES 2022)
      L1 + STFT                       -> preset "l1+stft"
      base.py default train_loss; STFT is auraloss.freq.STFTLoss() with
      its defaults (fft 1024 / hop 256 / win 1024), i.e. spectral
      convergence + log-magnitude, both weighted 1.0.

  gcn-tfilm (Comunita et al., ICASSP 2023)
      0.5*L1 + 0.5*MRSTFT             -> preset "gcntf"
      train.py: loss_fcns={"L1":0.5,"MSTFT":0.5}, prefilt=None.

  sptmod (Bourdin et al., DAFx 2025)
      100*MAE + ESR + MRSTFT + MREESR -> preset "sptmod"
      MR windows 512/1024/2048.

  Mamba/S6 (Simionato & Fasciani, JAES 2025)
      -> preset "mse"

  grey-box DRC (Wright & Valimaki, DAFx 2022)
      ESR (configs 4-6) computed on a DC-blocking pre-emphasised
      signal (one-pole, R=0.995)

Shape convention: pred/target arrive as [B, C, L] (or [B, L]); all losses
flatten to [N, L] internally, so they work with whatever the wrappers
emit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import CustomLoss

EPS = 1e-8


def _flatten(x):
    """[B, C, L] -> [B*C, L]; [B, L] passes through."""
    if x.dim() == 3:
        return x.reshape(-1, x.shape[-1])
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


class PreEmphasis(nn.Module):
    """First-order filter applied to pred and target before the loss.
    """

    def __init__(self, coefs=(-0.95, 1.0)):
        super().__init__()
        kernel = torch.tensor(coefs, dtype=torch.float32).flip(0).view(1, 1, -1)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        x_ = _flatten(x).unsqueeze(1)
        y = F.conv1d(x_, self.kernel, padding=self.kernel.shape[-1] - 1)
        return y[..., :x_.shape[-1]].squeeze(1)


class L1Loss(nn.Module):
    """Mean absolute error"""

    def forward(self, pred, target):
        return F.l1_loss(_flatten(pred), _flatten(target))


class MSELoss(nn.Module):
    def forward(self, pred, target):
        return F.mse_loss(_flatten(pred), _flatten(target))


class ESRLoss(nn.Module):
    """Error-to-signal ratio: sum (y - y_hat)^2 / sum y^2."""

    def forward(self, pred, target):
        p, t = _flatten(pred), _flatten(target)
        num = torch.sum((t - p) ** 2, dim=-1)
        den = torch.sum(t ** 2, dim=-1) + EPS
        return torch.mean(num / den)


class DCLoss(nn.Module):
    """DC offset error, as in Wright et al. / gcn-tfilm's "DC" term."""

    def forward(self, pred, target):
        p, t = _flatten(pred), _flatten(target)
        num = (torch.mean(t - p, dim=-1)) ** 2
        den = torch.mean(t ** 2, dim=-1) + EPS
        return torch.mean(num / den)


class STFTLoss(nn.Module):
    """Single-resolution STFT loss: spectral convergence + log magnitude.
    """

    def __init__(self, fft_size=1024, hop_size=256, win_length=1024,
                 w_sc=1.0, w_log_mag=1.0):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.register_buffer("window", torch.hann_window(win_length))

    def _mag(self, x):
        X = torch.stft(x, n_fft=self.fft_size, hop_length=self.hop_size,
                       win_length=self.win_length, window=self.window,
                       center=True, return_complex=True)
        return X.abs()

    def forward(self, pred, target):
        p, t = _flatten(pred), _flatten(target)
        if p.shape[-1] < self.win_length:      # window longer than signal
            return p.new_zeros(())
        P, T = self._mag(p), self._mag(t)

        # spectral convergence (Frobenius norm ratio)
        sc = torch.norm(T - P, p="fro") / (torch.norm(T, p="fro") + EPS)
        # log-magnitude L1
        lm = F.l1_loss(torch.log(P + EPS), torch.log(T + EPS))
        return self.w_sc * sc + self.w_log_mag * lm


class MultiResolutionSTFTLoss(nn.Module):
    """Average of STFT losses over several resolutions.
    """

    def __init__(self,
                 fft_sizes=(1024, 2048, 512),
                 hop_sizes=(120, 240, 50),
                 win_lengths=(600, 1200, 240)):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(fft_size=fs, hop_size=hs, win_length=wl)
            for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, pred, target):
        return sum(l(pred, target) for l in self.losses) / len(self.losses)


class EESRLoss(nn.Module):
    """Energy error-to-signal ratio at one window size (SPTMod, eq. 4).

        E_k = (1/W) * sum_{tau in window k} y[tau]^2
        L   = (1/K) * sum_k |E_hat_k - E_k| / E_k
        K   = floor(L / (W/4))    i.e. 75% overlap
    """

    def __init__(self, win_length=1024):
        super().__init__()
        self.win_length = win_length
        self.hop = win_length // 4

    def _energy(self, x):
        # frames: [N, K, W] -> mean square per frame
        frames = x.unfold(-1, self.win_length, self.hop)
        return frames.pow(2).mean(dim=-1)

    def forward(self, pred, target):
        p, t = _flatten(pred), _flatten(target)
        if p.shape[-1] < self.win_length:
            return p.new_zeros(())
        Ep, Et = self._energy(p), self._energy(t)
        return torch.mean((Ep - Et).abs() / (Et + EPS))


class MultiResolutionEESRLoss(nn.Module):
    """MR-EESR: EESR averaged over the same window sizes as MR-STFT."""

    def __init__(self, win_lengths=(512, 1024, 2048)):
        super().__init__()
        self.losses = nn.ModuleList([EESRLoss(w) for w in win_lengths])

    def forward(self, pred, target):
        return sum(l(pred, target) for l in self.losses) / len(self.losses)


class DCPreEmphasis(nn.Module):
    """One-pole DC-blocking pre-emphasis, (1 - z^-1)/(1 - R z^-1).
    """

    def __init__(self, R=0.995, n_taps=2000):
        super().__init__()
        import scipy.signal as signal
        _, ir = signal.dimpulse(signal.dlti([1, -1], [1, -R]), n=n_taps)
        ir = torch.tensor(ir[0][:, 0], dtype=torch.float32).flip(0)
        self.pad = len(ir) - 1
        self.register_buffer("kernel", ir.view(1, 1, -1))

    def forward(self, x):
        x_ = _flatten(x).unsqueeze(1)
        x_ = F.pad(x_, (self.pad, 0))
        k = self.kernel.to(dtype=x_.dtype, device=x_.device)
        return F.conv1d(x_, k).squeeze(1)


class RMSLoss(nn.Module):
    """Windowed level error at one window size (Wright & Valimaki).
    """

    def __init__(self, window=50, hop_div=4, domain="power", eps=1e-10):
        super().__init__()
        self.window = int(window)
        self.hop = max(1, int(window) // int(hop_div))
        self.domain = domain
        self.eps = eps

    def _power(self, x):
        return x.unfold(-1, self.window, self.hop).pow(2).mean(dim=-1)

    def forward(self, pred, target):
        p, t = _flatten(pred), _flatten(target)
        if p.shape[-1] < self.window:
            return p.new_zeros(())
        Pp, Pt = self._power(p), self._power(t)
        if self.domain == "power":
            return torch.mean(torch.abs(Pt - Pp))
        if self.domain == "db":
            to_db = lambda z: 10.0 * torch.log10(z + self.eps)
            return torch.mean(torch.abs(to_db(Pt) - to_db(Pp)))
        if self.domain == "rms":
            return torch.mean(torch.abs((Pt + self.eps).sqrt() - (Pp + self.eps).sqrt()))
        raise ValueError(f"domain must be power/db/rms, got {self.domain!r}")


class RMSMLoss(nn.Module):
    """Multi-window RMS loss"""

    def __init__(self, window_sizes=(8, 16, 32, 64), domain="power"):
        super().__init__()
        self.losses = nn.ModuleList([RMSLoss(w, domain=domain)
                                     for w in window_sizes])

    def forward(self, pred, target):
        return sum(l(pred, target) for l in self.losses) / len(self.losses)


class LogEnvelopeLoss(nn.Module):
    """L1 on the smoothed log envelope.
    """

    def __init__(self, sample_rate=48000, tau=0.010, min_db=-80.0):
        super().__init__()
        import math
        self.alpha = math.exp(-1.0 / (tau * sample_rate))
        self.floor = (10.0 ** (min_db / 20.0)) ** 2

    def _env_db(self, x):
        p = _flatten(x).pow(2)
        L = p.shape[-1]
        n = torch.arange(L, device=x.device, dtype=p.dtype)
        ir = (1 - self.alpha) * self.alpha ** n
        nfft = 1 << (2 * L - 1).bit_length()
        env = torch.fft.irfft(torch.fft.rfft(p, n=nfft) * torch.fft.rfft(ir, n=nfft),
                              n=nfft)[..., :L]
        return 10.0 * torch.log10(torch.clamp(env, min=self.floor))

    def forward(self, pred, target):
        return torch.mean(torch.abs(self._env_db(target) - self._env_db(pred)))


class WeightedLoss(nn.Module):
    """Weighted sum of named terms
    """

    def __init__(self, terms: dict, prefilt=None):
        super().__init__()
        self.names = list(terms.keys())
        self.fns = nn.ModuleList([terms[n][0] for n in self.names])
        self.weights = [terms[n][1] for n in self.names]
        self.prefilt = prefilt
        self.last_terms = {}

    def forward(self, pred, target):
        if pred.dim() == 3 and pred.shape[1] != 1 and pred.shape[-1] == 1:
            raise ValueError(f"expected [B, C, L], got {tuple(pred.shape)} — looks like [B, L, C]")

        if self.prefilt is not None:
            pred, target = self.prefilt(pred), self.prefilt(target)

        total = None
        self.last_terms = {}
        for name, fn, w in zip(self.names, self.fns, self.weights):
            val = fn(pred, target)
            self.last_terms[name] = float(val.detach())
            term = w * val
            total = term if total is None else total + term
        return total

    def __repr__(self):
        parts = [f"{w}*{n}" for n, w in zip(self.names, self.weights)]
        return f"WeightedLoss({' + '.join(parts)})"


# ------------------------------------------------------------------
# presets
# ------------------------------------------------------------------

def _preset(name: str, seq_len=None, prefilt=None) -> nn.Module:
    name = name.lower()

    # grey-box DRC
    if name in ("greybox", "esr"):
        return WeightedLoss({"ESR": (ESRLoss(), 1.0)},
                            prefilt if prefilt is not None else DCPreEmphasis())
    # diff-opto DRC
    if name in ("diff-opto", "mdl"):
        if seq_len is None:
            raise ValueError("preset 'mdl' needs seq_len (frame lengths are "
                             "derived from it); pass build_loss(..., seq_len=...)")
        return WeightedLoss({"MDL": (CustomLoss(frame_lengths=[seq_len//8, seq_len//4, seq_len//2], hop_length=seq_len//8), 1.0)}, prefilt)

    # mamba
    if name == ("l1+stft", "mamba"):
        return WeightedLoss({"MSE": (MSELoss(), 1.0)}, prefilt)

    # micro-tcn"
    if name in ("l1+stft", "tcn"):
        return WeightedLoss({
            "L1": (L1Loss(), 1.0),
            "STFT": (STFTLoss(), 1.0),
        }, prefilt)

    # gcn-tfilm: {"L1": 0.5, "MSTFT": 0.5}
    if name in ("gcntf", "l1+mrstft"):
        return WeightedLoss({
            "L1": (L1Loss(), 0.5),
            "MRSTFT": (MultiResolutionSTFTLoss(), 0.5),
        }, prefilt)

    # sptmod: 100*MAE + ESR + MRSTFT + MREESR, MR windows 512/1024/2048
    if name in ("sptmod"):
        return WeightedLoss({
            "MAE": (L1Loss(), 100.0),
            "ESR": (ESRLoss(), 1.0),
            "MRSTFT": (MultiResolutionSTFTLoss(
                fft_sizes=(512, 1024, 2048),
                hop_sizes=(128, 256, 512),
                win_lengths=(512, 1024, 2048)), 1.0),
            "MREESR": (MultiResolutionEESRLoss((512, 1024, 2048)), 1.0),
        }, prefilt)

    raise ValueError(f"unknown loss preset: {name!r}")


def build_loss(model_type: str = None, preset: str = None, seq_len=None, prefilt=None):
    """Build the loss for a model.
    """
    if preset is not None:

        return _preset(preset, seq_len, prefilt)

    paper_loss = {
        "tcn": "l1+stft",
        "lstm": "l1+stft",
        "gcntf": "l1+mrstft",
        "sptmod": "sptmod",
        "mamba": "mse",
        "greybox": "greybox",
        "diff-opto": "mdl",

    }
    if model_type is None:
        raise ValueError("pass model_type or preset")
    key = model_type.lower()
    if key not in paper_loss:
        raise ValueError(f"no paper loss registered for {model_type!r}")
    return _preset(paper_loss[key], seq_len, prefilt)