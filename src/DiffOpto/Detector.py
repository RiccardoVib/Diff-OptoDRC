# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato


import torch
import torch.nn as nn
import torch.nn.functional as F

from CompressorUtils import inverse_softplus


class LogDomainDetector(nn.Module):
    """Parameters: only the attack/release switch sharpness.
    """

    def __init__(self, switch_sharpness: float = 4.0, hard: bool = False,
                 learn_sharpness: bool = True):
        super().__init__()
        self.hard = bool(hard)
        raw = torch.tensor(inverse_softplus(switch_sharpness))
        if learn_sharpness and not hard:
            self.raw_k = nn.Parameter(raw)
        else:
            self.register_buffer("raw_k", raw)

    @property
    def sharpness(self):
        return F.softplus(self.raw_k)

    def forward(self, x_db, alpha_att, alpha_rel, h0,
                program_step=None, mix=None, mem0=None):
        """
        Parameters
        ----------
        x_db      : (B, L, C)   input level in dB
        alpha_att : (B, Lc, N)  attack coefficients, Lc in {1, L}
        alpha_rel : (B, Lc, N)  release coefficients
        h0        : (B, C, N)   carried state (dB)
        program_step : callable or None
            ``(level_db, memory, t) -> (alpha_att, alpha_release, memory)`` from
            ``PhysicsInformedDiagonalA.program_stepper``.  When given, the
            coefficients are recomputed each sample from the *envelope* (opto /
            T4 behaviour) and the ``alpha_*`` arguments are ignored.
        mix       : (N,) convex weights used to form the scalar level that
            drives the program dependence.
        mem0      : (B, C) carried light-history state.

        Returns
        -------
        h_seq : (B, L, C, N),  h_last : (B, C, N),  mem_last : (B, C) or None
        """
        B, L, C = x_db.shape
        static = alpha_att.shape[1] == 1
        k = self.sharpness

        if static:
            a_att_s = alpha_att[:, 0].unsqueeze(1)          # (B, 1, N)
            a_rel_s = alpha_rel[:, 0].unsqueeze(1)

        h = h0
        mem = mem0
        out = []
        for t in range(L):
            xt = x_db[:, t].unsqueeze(-1)                   # (B, C, 1)

            if program_step is not None:
                # Modulate from the envelope as it stood at t-1: using h[t] would
                # make the recursion implicit.  h is smooth on the timescale of
                # the ballistics, so the one-sample lag is immaterial.
                level = (h * mix).sum(-1)                   # (B, C)
                a_att, a_rel, mem = program_step(level, mem, t)
            elif static:
                a_att, a_rel = a_att_s, a_rel_s
            else:
                a_att = alpha_att[:, t].unsqueeze(1)        # (B, 1, N)
                a_rel = alpha_rel[:, t].unsqueeze(1)

            diff = xt - h                                   # (B, C, N)
            if self.hard:
                s = (diff > 0).to(h.dtype)
            else:
                s = torch.sigmoid(k * diff)
            a = s * a_att + (1.0 - s) * a_rel
            h = a * h + (1.0 - a) * xt
            out.append(h)

        h_seq = torch.stack(out, dim=1)                     # (B, L, C, N)
        return h_seq, h, mem

    def step(self, x_db_t, a_att, a_rel, h):
        """Single sample -- used by the feedback topology.

        x_db_t : (B, C)     a_att/a_rel : (B, 1, N)     h : (B, C, N)
        """
        xt = x_db_t.unsqueeze(-1)
        diff = xt - h
        s = (diff > 0).to(h.dtype) if self.hard else torch.sigmoid(self.sharpness * diff)
        a = s * a_att + (1.0 - s) * a_rel
        return a * h + (1.0 - a) * xt

    def extra_repr(self):
        return f"hard={self.hard}"
