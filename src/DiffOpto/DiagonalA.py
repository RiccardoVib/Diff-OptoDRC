# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from CompressorUtils import inverse_softplus, soft_lower_bound, zero_init_head, match_cond_shape


class PhysicsInformedDiagonalA(nn.Module):
    """Per-state attack and release one-pole coefficients, conditioned on ``c``.

    Parameters
    ----------
    n_state : int
        Number of parallel detector branches, each with its own attack *and*
        release time constant.  ``n_state=1`` is a classic single-stage
        detector; 2-4 gives the multi-stage response an opto cell needs.
    cond_dim : int or None
        Width of the conditioning vector (knob settings).  0/None = static.
    program_dependent : bool
        Enable envelope-dependent time constants (see module docstring).
    """

    def __init__(
        self,
        n_state: int,
        cond_dim=None,
        sr: int = 48000,
        tau_attack_range=(1e-4, 5e-2),      # 0.1 ms - 50 ms
        tau_release_range=(1e-2, 1.0),      # 10 ms - 1 s
        max_log_shift: float = math.log(10.0),
        cond_hidden: int = 16,
        tau_floor_factor: float = 2.0,
        program_dependent: bool = False,
        program_max_log_shift: float = math.log(20.0),
        tau_memory_init: float = 1.0,       # seconds
        drive_ref_db_init: float = -20.0,
        drive_width_db_init: float = 10.0,
    ):
        super().__init__()
        self.n_state = int(n_state)
        self.sr = float(sr)
        self.dt = 1.0 / float(sr)
        self.max_log_shift = float(max_log_shift)
        self.cond_dim = cond_dim if cond_dim else 0
        self.program_dependent = bool(program_dependent)
        self.program_max_log_shift = float(program_max_log_shift)

        # --- physics prior: log-spaced attack taus, paired with release taus ---
        if self.n_state == 1:
            log_tau_att = torch.tensor([math.log(math.sqrt(tau_attack_range[0] * tau_attack_range[1]))])
            log_tau_rel = torch.tensor([math.log(math.sqrt(tau_release_range[0] * tau_release_range[1]))])
        else:
            log_tau_att = torch.linspace(math.log(tau_attack_range[0]),
                                         math.log(tau_attack_range[1]), self.n_state)
            log_tau_rel = torch.linspace(math.log(tau_release_range[0]),
                                         math.log(tau_release_range[1]), self.n_state)

        self.log_tau_attack = nn.Parameter(log_tau_att)

        rho_init = torch.tensor([inverse_softplus(max(v, 1e-3))
                                 for v in (log_tau_rel - log_tau_att).tolist()])
        self.rho = nn.Parameter(rho_init)

        self.log_tau_min = math.log(tau_floor_factor * self.dt)

        if self.cond_dim > 0:
            self.cond_net = nn.Sequential(
                nn.Linear(self.cond_dim, cond_hidden),
                nn.SiLU(),
                nn.Linear(cond_hidden, 2 * self.n_state),
            )
            zero_init_head(self.cond_net[-1], 0.0)   # start at the physics prior
        else:
            self.cond_net = None

        if self.program_dependent:
            self.ref_db = nn.Parameter(torch.tensor(float(drive_ref_db_init)))
            self.raw_width = nn.Parameter(torch.tensor(inverse_softplus(drive_width_db_init)))
            self.log_tau_memory = nn.Parameter(torch.tensor(math.log(tau_memory_init)))
            self.drive_att = nn.Parameter(torch.zeros(self.n_state))
            self.drive_rel = nn.Parameter(torch.zeros(self.n_state))
            self.mem_att = nn.Parameter(torch.zeros(self.n_state))
            self.mem_rel = nn.Parameter(torch.zeros(self.n_state))

    # ------------------------------------------------------------------ #
    # base (envelope-independent) ballistics
    # ------------------------------------------------------------------ #

    def _base_log(self, c=None, seq_len: int = 1):
        """-> (log_tau_attack, rho_effective), each (B, Lc, N)."""
        c = match_cond_shape(c, seq_len)

        log_att = self.log_tau_attack.view(1, 1, -1)
        rho = self.rho.view(1, 1, -1)

        if self.cond_net is not None:
            if c is None:
                raise ValueError("this module was built with cond_dim > 0 but c is None")
            d_att, d_rel = self.cond_net(c).split(self.n_state, dim=-1)

            log_att = log_att + self.max_log_shift * torch.tanh(d_att)
            rho = rho + self.max_log_shift * torch.tanh(d_rel)

        return log_att, rho

    @staticmethod
    def _alphas(log_att, rho, log_tau_min, dt):
        """Coefficients only, skipping tau.

        alpha_attack  = exp(-dt/tau_att)      = exp(-dt * inv_tau)
        alpha_release = exp(-dt/tau_rel)      = exp(-dt * inv_tau * exp(-gap))

        """
        log_att = soft_lower_bound(log_att, log_tau_min)
        inv_tau = torch.exp(-log_att)
        scaled = inv_tau * torch.exp(-F.softplus(rho))
        return torch.exp(-dt * inv_tau), torch.exp(-dt * scaled)

    @staticmethod
    def _finish(log_att, rho, log_tau_min, dt):
        log_att = soft_lower_bound(log_att, log_tau_min)
        log_rel = log_att + F.softplus(rho)                 # gap > 0 always
        tau_att, tau_rel = torch.exp(log_att), torch.exp(log_rel)
        return torch.exp(-dt / tau_att), torch.exp(-dt / tau_rel), tau_att, tau_rel

    def time_constants(self, c=None, seq_len: int = 1):
        """(tau_attack, tau_release) in seconds, (B, Lc, N), at zero drive.
        """
        log_att, rho = self._base_log(c, seq_len)
        _, _, tau_att, tau_rel = self._finish(log_att, rho, self.log_tau_min, self.dt)
        return tau_att, tau_rel

    def forward(self, c=None, seq_len: int = 1, dt: float = None):
        """One-pole coefficients ``alpha = exp(-dt / tau)`` in (0, 1).

        ``dt`` overrides the sample period: pass ``dt * R`` to run the sidechain
        decimated and the same taus give the correct coefficients at sr/R.
        """
        dt = self.dt if dt is None else float(dt)
        log_att, rho = self._base_log(c, seq_len)
        log_tau_min = math.log(math.exp(self.log_tau_min) * (dt / self.dt))
        return self._finish(log_att, rho, log_tau_min, dt)

    # ------------------------------------------------------------------ #
    # program dependence
    # ------------------------------------------------------------------ #

    def program_stepper(self, c=None, seq_len: int = 1, dt: float = None,
                        update_every: int = 1):
        """Build the per-sample ballistics closure, or None if disabled.

        Returns ``step(level_db, memory, t) -> (alpha_attack, alpha_release, memory)``
        with ``level_db`` and ``memory`` of shape (B, C) and the alphas (B, C, N).

        ``update_every=K`` re-evaluates the modulation only every K samples and
        holds the coefficients in between -- a control-rate approximation, exact
        at K=1.  For chunk-exactness the chunk length must be a multiple of K.
        """
        if not self.program_dependent:
            return None

        dt = self.dt if dt is None else float(dt)
        log_att0, rho0 = self._base_log(c, seq_len)                # (B, Lc, N)
        Lc = log_att0.shape[1]
        log_tau_min = math.log(math.exp(self.log_tau_min) * (dt / self.dt))

        S = self.program_max_log_shift
        s_att = S * torch.tanh(self.drive_att)
        s_rel = S * torch.tanh(self.drive_rel)
        m_att = S * torch.tanh(self.mem_att)
        m_rel = S * torch.tanh(self.mem_rel)

        ref = self.ref_db
        inv_width = 1.0 / (F.softplus(self.raw_width) + 1e-2)
        every = max(1, int(update_every))
        dt_mem = dt * every
        mem_lerp = 1.0 - torch.exp(-dt_mem / torch.exp(self.log_tau_memory).clamp_min(2 * dt_mem))
        static_c = Lc == 1
        if static_c:
            la_const = log_att0[:, 0].unsqueeze(1)                 # (B, 1, N)
            rh_const = rho0[:, 0].unsqueeze(1)

        cache = {}

        def step(level_db, memory, t):

            if every > 1 and t % every and cache:
                a_att, a_rel = cache["alphas"]
                return a_att, a_rel, cache["memory"]

            if static_c:
                la0, rh0 = la_const, rh_const
            else:
                idx = min(t, Lc - 1)
                la0 = log_att0[:, idx].unsqueeze(1)
                rh0 = rho0[:, idx].unsqueeze(1)

            drive = torch.sigmoid((level_db - ref) * inv_width)    # (B, C)
            memory = torch.lerp(memory, drive, mem_lerp)
            d = drive.unsqueeze(-1)                                # (B, C, 1)
            m = memory.unsqueeze(-1)

            la = torch.addcmul(torch.addcmul(la0, d, s_att), m, m_att)
            rh = torch.addcmul(torch.addcmul(rh0, d, s_rel), m, m_rel)
            a_att, a_rel = self._alphas(la, rh, log_tau_min, dt)
            if every > 1:
                cache["alphas"] = (a_att, a_rel)
                cache["memory"] = memory
            return a_att, a_rel, memory

        return step

    def settled_memory(self, level_db):
        """Light-history state a cell sitting at ``level_db`` would equilibrate to.

        """
        if not self.program_dependent:
            return None
        width = F.softplus(self.raw_width) + 1e-2
        return torch.sigmoid((level_db - self.ref_db) / width)

    @torch.no_grad()
    def release_vs_level(self, levels_db, c=None, settled: bool = True):
        """tau_release (seconds) as a function of detector level -- the T4 curve.

        ``settled=True`` assumes the memory state has equilibrated at the given
        level (memory == drive), which is the right comparison for a sustained
        tone; ``settled=False`` gives the instantaneous response of a cell that
        has been dark (memory == 0), i.e. the fast first-hit release.
        """
        levels = torch.as_tensor(levels_db, dtype=torch.float32).view(-1, 1)
        log_att0, rho0 = self._base_log(c, 1)
        log_att0, rho0 = log_att0[0, 0], rho0[0, 0]                # (N,)

        if self.program_dependent:
            S = self.program_max_log_shift
            drive = torch.sigmoid((levels - self.ref_db) / (F.softplus(self.raw_width) + 1e-2))
            mem = drive if settled else torch.zeros_like(drive)
            la = log_att0 + drive * (S * torch.tanh(self.drive_att)) + mem * (S * torch.tanh(self.mem_att))
            rh = rho0 + drive * (S * torch.tanh(self.drive_rel)) + mem * (S * torch.tanh(self.mem_rel))
        else:
            la = log_att0.expand(levels.shape[0], -1)
            rh = rho0.expand(levels.shape[0], -1)

        _, _, tau_att, tau_rel = self._finish(la, rh, self.log_tau_min, self.dt)
        return tau_att, tau_rel

    def extra_repr(self):
        return (f"n_state={self.n_state}, cond_dim={self.cond_dim}, sr={self.sr:g}, "
                f"program_dependent={self.program_dependent}")