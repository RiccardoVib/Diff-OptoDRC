# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from CompressorUtils import inverse_softplus, logit


def _fft_causal_conv(u, k):
    """u: (B, C, L) real, k: (L,) real -> causal convolution, (B, C, L)."""
    L = u.shape[-1]
    n = 1
    while n < 2 * L:
        n *= 2
    U = torch.fft.rfft(u, n=n)
    K = torch.fft.rfft(k, n=n)
    return torch.fft.irfft(U * K, n=n)[..., :L]


class ComplexDiagonalSSM(nn.Module):
    """Diagonal SSM with ``n_pairs`` conjugate pole pairs, real input/output.

    y[t] = 2*Re( sum_n C_n h_n[t] ) + D u[t],   h_n[t] = A_n h_n[t-1] + B_n u[t]

    """

    def __init__(self, n_pairs: int = 4, sr: int = 48000,
                 f_range=(30.0, 8000.0), zeta_init: float = 0.7,
                 c_init_std: float = 0.01, d_init: float = 1.0):
        super().__init__()
        self.n_pairs = int(n_pairs)
        self.sr = float(sr)
        self.dt = 1.0 / float(sr)

        if self.n_pairs == 1:
            f0 = torch.tensor([math.sqrt(f_range[0] * f_range[1])])
        else:
            f0 = torch.logspace(math.log10(f_range[0]), math.log10(f_range[1]), self.n_pairs)
        # omega = (pi/dt) * sigmoid(raw)  ->  always strictly below Nyquist
        self.raw_omega = nn.Parameter(torch.tensor([logit(min(f / (0.5 * sr), 0.999)) for f in f0]))
        self.raw_zeta = nn.Parameter(torch.full((self.n_pairs,), inverse_softplus(zeta_init)))

        self.b = nn.Parameter(torch.ones(self.n_pairs))
        self.c_re = nn.Parameter(torch.randn(self.n_pairs) * c_init_std)
        self.c_im = nn.Parameter(torch.randn(self.n_pairs) * c_init_std)
        self.D = nn.Parameter(torch.tensor(float(d_init)))

        self.register_buffer("state", torch.zeros(1, 1, self.n_pairs, dtype=torch.cfloat),
                             persistent=False)

    # ------------------------------------------------------------------ #

    def poles(self):
        """Continuous-time poles ``s = -zeta*omega + i*omega`` (rad/s)."""
        omega = (math.pi / self.dt) * torch.sigmoid(self.raw_omega)
        zeta = F.softplus(self.raw_zeta) + 1e-4
        return -zeta * omega, omega

    def resonances_hz(self):
        """Interpretability hook: (frequency in Hz, Q) of each pole pair."""
        sigma, omega = self.poles()
        return omega / (2 * math.pi), omega / (2 * (-sigma) + 1e-12)

    def _discrete(self):
        sigma, omega = self.poles()
        s = torch.complex(sigma, omega).to(torch.cdouble)
        A = torch.exp(s * self.dt)                                  # |A| < 1 always
        B = (1.0 - A) * self.b.to(torch.cdouble)                    # DC gain of pole n = C_n b_n
        C = torch.complex(self.c_re, self.c_im).to(torch.cdouble)
        return A, B, C, s

    def reset_hidden_states(self, batch_size=None, n_channels=None, device=None):
        b = batch_size if batch_size is not None else self.state.shape[0]
        c = n_channels if n_channels is not None else self.state.shape[1]
        dev = device if device is not None else self.state.device
        self.state = torch.zeros(b, c, self.n_pairs, dtype=torch.cfloat, device=dev)

    def forward(self, u, truncate_state: bool = True):
        """u: (B, L, C) -> y: (B, L, C).  State carries across chunks exactly."""
        B, L, C = u.shape
        if self.state.shape[0] != B or self.state.shape[1] != C or self.state.device != u.device:
            self.reset_hidden_states(B, C, u.device)

        A, Bc, Cc, _ = self._discrete()
        l = torch.arange(L, device=u.device, dtype=torch.double)
        # A^l computed as exp(l * s * dt)
        A_pow = torch.exp(l.unsqueeze(-1) * torch.log(A).unsqueeze(0))   # (L, N)

        # impulse response of the SSM part
        k = 2.0 * (Cc * Bc).unsqueeze(0).mul(A_pow).sum(-1).real        # (L,)

        u_t = u.transpose(1, 2).contiguous()                            # (B, C, L)
        y = _fft_causal_conv(u_t.double(), k) + self.D.double() * u_t.double()

        # zero-input response of the carried state: 2*Re(sum_n C_n A_n^{t+1} h_n[-1])
        h0 = self.state.to(torch.cdouble)
        if h0.abs().max() > 0:
            zir = 2.0 * torch.einsum("bcn,ln->bcl", Cc * h0, A_pow * A.unsqueeze(0)).real
            y = y + zir

        # new state: h[L-1] = A^L h[-1] + B * sum_k A^(L-1-k) u[k]
        acc = torch.einsum("bcl,ln->bcn", u_t.to(torch.cdouble), A_pow.flip(0))
        h_last = (A ** L).unsqueeze(0).unsqueeze(0) * h0 + Bc.unsqueeze(0).unsqueeze(0) * acc
        h_last = h_last.to(torch.cfloat)
        self.state = h_last.detach() if truncate_state else h_last

        return y.to(u.dtype).transpose(1, 2)


class OnePoleTilt(nn.Module):
    """First-order sidechain tilt: ``y = u - g * lowpass(u)``, ``g in [0, 1]``.
    """

    def __init__(self, sr: int = 48000, f_init: float = 80.0, tilt_init: float = 0.5,
                 learnable: bool = True):
        super().__init__()
        self.sr = float(sr)
        self.dt = 1.0 / float(sr)
        raw = torch.tensor(math.log(2.0 * math.pi * f_init))
        raw_g = torch.tensor(logit(min(max(tilt_init, 1e-9), 1 - 1e-9)))
        if learnable:
            self.log_omega = nn.Parameter(raw)
            self.raw_tilt = nn.Parameter(raw_g)
        else:
            self.register_buffer("log_omega", raw)
            self.register_buffer("raw_tilt", raw_g)
        self.register_buffer("state", torch.zeros(1, 1), persistent=False)

    def cutoff_hz(self):
        return torch.exp(self.log_omega) / (2 * math.pi)

    def tilt(self):
        """Amount of low-frequency removal in the sidechain, 0 = flat."""
        return torch.sigmoid(self.raw_tilt)

    def reset_hidden_states(self, batch_size=None, n_channels=None, device=None):
        b = batch_size if batch_size is not None else self.state.shape[0]
        c = n_channels if n_channels is not None else self.state.shape[1]
        dev = device if device is not None else self.state.device
        self.state = torch.zeros(b, c, device=dev)

    def forward(self, u, truncate_state: bool = True):
        B, L, C = u.shape
        if self.state.shape != (B, C) or self.state.device != u.device:
            self.reset_hidden_states(B, C, u.device)

        omega = torch.exp(self.log_omega).clamp(max=0.45 * math.pi / self.dt)
        a = torch.exp(-omega * self.dt).double()                     # in (0, 1)

        l = torch.arange(L, device=u.device, dtype=torch.double)
        a_pow = torch.exp(l * torch.log(a))
        k = (1.0 - a) * a_pow                                        # unity DC gain

        u_t = u.transpose(1, 2).contiguous().double()
        lp = _fft_causal_conv(u_t, k)
        lp = lp + self.state.double().unsqueeze(-1) * (a_pow * a).unsqueeze(0)

        acc = torch.einsum("bcl,l->bc", u_t, a_pow.flip(0))
        new_state = (a ** L) * self.state.double() + (1.0 - a) * acc
        self.state = (new_state.detach() if truncate_state else new_state).to(u.dtype)

        return (u_t - self.tilt().double() * lp).to(u.dtype).transpose(1, 2)