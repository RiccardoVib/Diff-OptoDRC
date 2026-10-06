"""Loss functions.
grey-box: code adapted from https://github.com/Alec-Wright/GreyBoxDRC
grey-box DRC (Wright & Valimaki, DAFx 2022)
"""


import math

import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F


class DC_PreEmph(nn.Module):
    """First-order DC-blocking pre-emphasis, applied to output and target."""

    def __init__(self, R=0.995, n_taps=2000):
        super().__init__()
        _, ir = signal.dimpulse(signal.dlti([1, -1], [1, -R]), n=n_taps)
        ir = ir[0][:, 0]
        self.zPad = len(ir) - 1
        kernel = torch.flipud(torch.tensor(ir, dtype=torch.float32))
        self.register_buffer("pars", kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, output, target):
        output = output.permute(0, 2, 1)
        target = target.permute(0, 2, 1)
        pad_o = output.new_zeros(output.shape[0], 1, self.zPad)
        pad_t = target.new_zeros(target.shape[0], 1, self.zPad)
        output = torch.cat((pad_o, output), dim=2)
        target = torch.cat((pad_t, target), dim=2)
        k = self.pars.to(dtype=output.dtype, device=output.device)
        output = F.conv1d(output, k, bias=None)
        target = F.conv1d(target, k, bias=None)
        return output.permute(0, 2, 1), target.permute(0, 2, 1)


class ESRLoss(nn.Module):
    """Error-to-signal ratio."""

    def __init__(self, dc_pre=True, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.dc_pre = DC_PreEmph() if dc_pre else None

    def forward(self, output, target):
        if self.dc_pre is not None:
            output, target = self.dc_pre(output, target)
        err = torch.mean((target - output) ** 2)
        energy = torch.mean(target ** 2) + self.epsilon
        return err / energy


class RMSLoss(nn.Module):
    """Windowed level error at a single window size."""

    def __init__(self, window=50, hop_div=4, domain="power", epsilon=1e-10):
        super().__init__()
        self.window = int(window)
        self.hop = max(1, int(window) // int(hop_div))
        self.domain = domain
        self.epsilon = epsilon

    def _frames(self, x):
        # x: [B, T, C] -> [B, n_frames, window] mean power
        f = x.unfold(1, self.window, self.hop)          # [B, n, C, window]
        p = f.pow(2).mean(dim=-1)                       # [B, n, C]
        return p

    def forward(self, output, target):
        po, pt = self._frames(output), self._frames(target)
        if self.domain == "power":
            return torch.mean(torch.abs(pt - po))
        if self.domain == "rms":
            return torch.mean(torch.abs(torch.sqrt(pt + self.epsilon)
                                        - torch.sqrt(po + self.epsilon)))
        if self.domain == "db":
            to_db = lambda p: 10.0 * torch.log10(p + self.epsilon)
            return torch.mean(torch.abs(to_db(pt) - to_db(po)))
        raise ValueError(f"domain must be power/rms/db, got {self.domain!r}")


class RMSMLoss(nn.Module):
    """Multi-window version of :class:`RMSLoss`."""

    def __init__(self, window_sizes=(8, 16, 32, 64), dc_pre=True, domain="power"):
        super().__init__()
        self.rms_loss = nn.ModuleList([RMSLoss(s, domain=domain)
                                       for s in window_sizes])
        self.dc_pre = DC_PreEmph() if dc_pre else None

    def forward(self, output, target):
        if self.dc_pre is not None:
            output, target = self.dc_pre(output, target)
        total = sum(f(output, target) for f in self.rms_loss)
        return total / len(self.rms_loss)


class LogEnvelopeLoss(nn.Module):
    """L1 on the smoothed log envelope.

    """

    def __init__(self, sample_rate=44100, tau=0.010, min_db=-80.0):
        super().__init__()
        self.alpha = math.exp(-1.0 / (tau * sample_rate))
        self.eps = 10.0 ** (min_db / 20.0)

    def _env_db(self, x):
        p = x.pow(2).permute(0, 2, 1)                   # [B, C, T]
        # one-pole smoother as an FFT convolution (see ballistics.onepole_parallel)
        T = p.shape[-1]
        n = torch.arange(T, device=x.device, dtype=x.dtype)
        ir = (1 - self.alpha) * self.alpha ** n
        nfft = 1 << (2 * T - 1).bit_length()
        env = torch.fft.irfft(torch.fft.rfft(p, n=nfft) * torch.fft.rfft(ir, n=nfft),
                              n=nfft)[..., :T]
        return 10.0 * torch.log10(torch.clamp(env, min=self.eps ** 2))

    def forward(self, output, target):
        return torch.mean(torch.abs(self._env_db(target) - self._env_db(output)))


class MultiResSTFTLoss(nn.Module):
    """Spectral convergence + log-magnitude at several resolutions."""

    def __init__(self, fft_sizes=(512, 1024, 2048), hop_ratio=0.25, eps=1e-7):
        super().__init__()
        self.fft_sizes = tuple(int(n) for n in fft_sizes)
        self.hop_ratio = hop_ratio
        self.eps = eps
        for n in self.fft_sizes:
            self.register_buffer(f"win_{n}", torch.hann_window(n), persistent=False)

    def _stft(self, x, n):
        w = getattr(self, f"win_{n}").to(x.device, x.dtype)
        s = torch.stft(x.squeeze(-1), n_fft=n, hop_length=max(1, int(n * self.hop_ratio)),
                       win_length=n, window=w, return_complex=True, center=True)
        return s.abs()

    def forward(self, output, target):
        total = 0.0
        for n in self.fft_sizes:
            if output.shape[1] < n:
                continue
            so, st = self._stft(output, n), self._stft(target, n)
            sc = torch.linalg.norm(st - so) / (torch.linalg.norm(st) + self.eps)
            mag = F.l1_loss(torch.log(so + self.eps), torch.log(st + self.eps))
            total = total + sc + mag
        return total / max(1, len(self.fft_sizes))


class CombinedLoss(nn.Module):
    """Weighted sum of named losses, e.g. ``{'ESR': 1.0, 'env': 0.5}``."""

    def __init__(self, weights, **kwargs):
        super().__init__()
        self.weights = dict(weights)
        self.terms = nn.ModuleDict({k: build_loss(k, **kwargs.get(k, {}))
                                    for k in self.weights})

    def forward(self, output, target):
        return sum(w * self.terms[k](output, target)
                   for k, w in self.weights.items())


LOSSES = {
    "ESR": ESRLoss,
    "MAE": nn.L1Loss,
    "MSE": nn.MSELoss,
    "rms": RMSLoss,
    "rmsm": RMSMLoss,
    "env": LogEnvelopeLoss,
    "stft": MultiResSTFTLoss,
}


def build_loss(name, **kwargs):
    if isinstance(name, dict):                 # {'ESR': 1.0, 'env': 0.5}
        return CombinedLoss(name, **kwargs)
    if name not in LOSSES:
        raise KeyError(f"unknown loss {name!r}; have {sorted(LOSSES)}")
    return LOSSES[name](**kwargs)