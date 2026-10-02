# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato

"""Interpretable SSM dynamic range compressor (mono).

    u ─┬─► [sidechain tilt] ─► |·| → dB ─► detector A(c) ─► mix ─► curve(c) ─┐
       │                                                                     │
       └──────────────────► × ◄────────────────────── 10^(gain_dB/20) ───────┘
                            └─► [complex-pole signal path] ─► y

Conditioning is split into two independent vectors rather than routed by name:
``c_ballistics`` reaches the time constants, ``c_curve`` reaches the static
curve.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from CompressorUtils import amp_to_db, db_to_gain, match_cond_shape
from DiagonalA import PhysicsInformedDiagonalA
from Detector import LogDomainDetector
from GainCurve import SoftKneeGainCurve
from SignalPath import ComplexDiagonalSSM, OnePoleTilt


class DiffOpto(nn.Module):
    """u: (B, L, d_model) audio -> y: (B, L, d_model).

    Conditioning may be passed either way::

        model(u, {"ballistics": c_b, "curve": c_c})     # explicit
        model(u, c)                                     # (B, ballistics_dim + curve_dim)

    """

    def __init__(
        self,
        d_model: int = 1,
        n_state: int = 2,
        curve_dim: int = 1,
        ballistics_dim: int = 0,
        batch_size: int = 1,
        sr: int = 48000,
        device: str = "cpu",
        topology: str = "feedback",         # 'feedforward' | 'feedback'
        sidechain_decimation: int = 1,
        decimation_mode: str = "block",     # 'block' | 'causal'
        program_update_every: int = 1,      # control rate for the opto modulation
        use_sidechain_hpf: bool = True,
        hpf_init_hz: float = 80.0,
        sidechain_tilt_init: float = 0.5,
        signal_path_pairs: int = 0,         # 0 disables the complex SSM
        signal_path_position: str = "post", # 'post' | 'pre'
        db_floor: float = -96.0,
        state_init: str = "first_sample",   # 'first_sample' | 'floor'
        hard_switch: bool = False,
        ballistics_kwargs: dict = None,
        curve_kwargs: dict = None,
    ):
        super().__init__()
        if topology not in ("feedforward", "feedback"):
            raise ValueError("topology must be 'feedforward' or 'feedback'")
        if decimation_mode not in ("block", "causal"):
            raise ValueError("decimation_mode must be 'block' or 'causal'")
        if topology == "feedback" and sidechain_decimation != 1 and decimation_mode != "causal":
            raise ValueError("feedback topology needs decimation_mode='causal'")

        self.d_model = int(d_model)
        self.n_state = int(n_state)
        self.ballistics_dim = int(ballistics_dim) if ballistics_dim else 0
        self.curve_dim = int(curve_dim) if curve_dim else 0
        self.cond_dim = self.ballistics_dim + self.curve_dim
        self.sr = float(sr)
        self.dt = 1.0 / float(sr)
        self.topology = topology
        self.R = int(sidechain_decimation)
        self.decimation_mode = decimation_mode
        self.shift_blocks = topology == "feedback" or decimation_mode == "causal"
        self.program_update_every = max(1, int(program_update_every))
        self.db_floor = float(db_floor)
        self.state_init = state_init
        self.n_det_ch = self.d_model

        self.A_module = PhysicsInformedDiagonalA(
            n_state=self.n_state, cond_dim=self.ballistics_dim, sr=sr,
            **(ballistics_kwargs or {}))
        self.detector = LogDomainDetector(hard=hard_switch)
        self.curve = SoftKneeGainCurve(cond_dim=self.curve_dim, **(curve_kwargs or {}))
        self.program_dependent = self.A_module.program_dependent

        self.det_mix = nn.Parameter(torch.zeros(self.n_state))

        self.hpf = (OnePoleTilt(sr=sr, f_init=hpf_init_hz, tilt_init=sidechain_tilt_init)
                    if use_sidechain_hpf else None)
        self.signal_path_position = signal_path_position
        self.signal_path = (ComplexDiagonalSSM(n_pairs=signal_path_pairs, sr=sr)
                            if signal_path_pairs else None)

        self.register_buffer("detector_state",
                             torch.full((batch_size, self.n_det_ch, self.n_state),
                                        self.db_floor, device=device), persistent=False)
        self.register_buffer("light_memory",
                             torch.zeros(batch_size, self.n_det_ch, device=device),
                             persistent=False)
        self.register_buffer("last_output",
                             torch.zeros(batch_size, self.d_model, device=device),
                             persistent=False)
        self.register_buffer("gain_hold", torch.zeros(batch_size, 1, self.n_det_ch,
                                                      device=device), persistent=False)
        self.register_buffer("level_hold", torch.full((batch_size, 1, self.n_det_ch),
                                                      self.db_floor, device=device),
                             persistent=False)
        self.register_buffer("peak_hold", torch.full((batch_size, self.n_det_ch),
                                                     self.db_floor, device=device),
                             persistent=False)
        self.register_buffer("primed", torch.zeros((), dtype=torch.bool, device=device),
                             persistent=False)
        self.to(device)

    # ------------------------------------------------------------------ #
    # conditioning
    # ------------------------------------------------------------------ #

    def _split_cond(self, c, seq_len):
        """-> (c_ballistics, c_curve), each (B, Lc, dim) or None."""
        if c is None:
            return None, None
        if isinstance(c, dict):
            ball = match_cond_shape(c.get("ballistics"), seq_len)
            curve = match_cond_shape(c.get("curve"), seq_len)
            return ball, curve

        c = match_cond_shape(c, seq_len)
        if c.shape[-1] < self.cond_dim:
            raise ValueError(f"c has {c.shape[-1]} controls, expected at least "
                             f"{self.cond_dim} (ballistics {self.ballistics_dim} "
                             f"+ curve {self.curve_dim})")
        ball = c[..., :self.ballistics_dim] if self.ballistics_dim else None
        curve = (c[..., self.ballistics_dim:self.ballistics_dim + self.curve_dim]
                 if self.curve_dim else None)
        return ball, curve

    def _decimate_cond(self, c, length):
        if c is None or c.shape[1] == 1:
            return c
        return c[:, ::self.R][:, :length]

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #

    def reset_hidden_states(self, batch_size: int = None, device=None):
        b = batch_size if batch_size is not None else self.detector_state.shape[0]
        dev = device if device is not None else self.detector_state.device
        self.detector_state = torch.full((b, self.n_det_ch, self.n_state), self.db_floor,
                                         device=dev)
        self.light_memory = torch.zeros(b, self.n_det_ch, device=dev)
        self.last_output = torch.zeros(b, self.d_model, device=dev)
        self.gain_hold = torch.zeros(b, 1, self.n_det_ch, device=dev)
        self.level_hold = torch.full((b, 1, self.n_det_ch), self.db_floor, device=dev)
        self.peak_hold = torch.full((b, self.n_det_ch), self.db_floor, device=dev)
        self.primed = torch.zeros((), dtype=torch.bool, device=dev)
        if self.hpf is not None:
            self.hpf.reset_hidden_states(b, self.d_model, dev)
        if self.signal_path is not None:
            self.signal_path.reset_hidden_states(b, self.d_model, dev)

    def _ensure_state(self, batch, device):
        if self.detector_state.shape[0] != batch or self.detector_state.device != device:
            self.reset_hidden_states(batch, device)

    def _initial_state(self, first_level_db):
        """Detector state to start this chunk from.

       """
        if self.state_init != "first_sample" or bool(self.primed):
            return self.detector_state
        self.primed = torch.ones((), dtype=torch.bool, device=self.detector_state.device)
        level = first_level_db.detach()
        # The opto light history relaxes with a multi-second time constant, so
        # charging it by running audio would need ~10 s of warm-up.  Seed it with
        # the value it would equilibrate to at this level instead.
        settled = self.A_module.settled_memory(level)
        if settled is not None:
            self.light_memory = settled
        return level.unsqueeze(-1).expand(-1, -1, self.n_state)

    @staticmethod
    def _carry(value, truncate):
        return value.detach() if truncate else value

    def _sidechain_input(self, signal, truncate_state):
        filtered = self.hpf(signal, truncate_state) if self.hpf is not None else signal
        return amp_to_db(filtered, self.db_floor)

    # ------------------------------------------------------------------ #

    def forward(self, u, c=None, target=None, teacher_forcing: float = 1.0,
                return_aux: bool = False, truncate_state: bool = True):
        """``target`` enables teacher forcing in the feedback loop (training only).

       ``teacher_forcing`` in [0, 1] blends: 1.0 is pure teacher forcing, 0.0 is
        free-running, in between is scheduled sampling (anneal it to 0 so the
        model ends up trained on the signal it will actually see at inference).
        """
        if u.dim() == 2:
            u = u.unsqueeze(-1)
        batch, length, channels = u.shape
        if channels != self.d_model:
            raise ValueError(f"expected d_model={self.d_model} channels, got {channels}")

        period = self.R * self.program_update_every
        if period > 1 and length % period:
            raise ValueError(
                f"chunk length {length} must be a multiple of "
                f"sidechain_decimation * program_update_every = "
                f"{self.R} * {self.program_update_every} = {period}, "
                "or chunked output will not match whole-sequence output")

        self._ensure_state(batch, u.device)

        if target is not None:
            if not self.training:
               raise ValueError("target= is teacher forcing and is training-only; "
                                 "pass target=None in eval")
            if self.topology != "feedback":
                raise ValueError("teacher forcing only applies to the feedback "
                                 "topology; the feedforward sidechain never sees y")
            if target.dim() == 2:
                target = target.unsqueeze(-1)
            if target.shape != u.shape:
                raise ValueError(f"target {tuple(target.shape)} must match u {tuple(u.shape)}")

        c_ballistics, c_curve = self._split_cond(c, length)

        if self.signal_path is not None and self.signal_path_position == "pre":
            u = self.signal_path(u, truncate_state)

        if self.topology == "feedback" and target is not None and teacher_forcing == 1.0:
            y, aux = self._forward_feedforward(u, c_ballistics, c_curve, truncate_state,
                                               sidechain=target, shift_blocks=True)
            self.last_output = self._carry(target[:, -1], truncate_state)
        elif self.topology == "feedback":
            y, aux = self._forward_feedback(u, c_ballistics, c_curve, truncate_state,
                                            target, teacher_forcing)
        else:
            y, aux = self._forward_feedforward(u, c_ballistics, c_curve, truncate_state,
                                               shift_blocks=self.shift_blocks)

        if self.signal_path is not None and self.signal_path_position == "post":
            y = self.signal_path(y, truncate_state)
        return (y, aux) if return_aux else y

    # ------------------------------------------------------------------ #

    def _forward_feedforward(self, u, c_ballistics, c_curve, truncate_state,
                             sidechain=None, shift_blocks=False):
        """``sidechain`` is what the detector measures; ``u`` is what gets the gain.
        """
        length = u.shape[1]
        fresh = self.state_init == "first_sample" and not bool(self.primed)
        prime_level = amp_to_db(u[:, :1], self.db_floor)[:, 0]
        input_db = self._sidechain_input(u if sidechain is None else sidechain,
                                         truncate_state)

        if self.R > 1:
            padded = F.pad(input_db.transpose(1, 2), (0, (-length) % self.R),
                           value=self.db_floor)
            pooled = F.max_pool1d(padded, self.R, self.R).transpose(1, 2)
            dt = self.dt * self.R
        else:
            pooled, dt = input_db, self.dt

        if shift_blocks:
            first = prime_level if fresh else self.peak_hold
            level_in = torch.cat([first.unsqueeze(1), pooled[:, :-1]], dim=1)
            self.peak_hold = self._carry(pooled[:, -1], truncate_state)
        else:
            level_in = pooled

        n = level_in.shape[1]
        c_ballistics = self._decimate_cond(c_ballistics, n) if self.R > 1 else c_ballistics
        c_curve = self._decimate_cond(c_curve, n) if self.R > 1 else c_curve

        alpha_attack, alpha_release, tau_attack, tau_release = self.A_module(
            c_ballistics, seq_len=n, dt=dt)
        mix = torch.softmax(self.det_mix, dim=-1)
        program = self.A_module.program_stepper(c_ballistics, seq_len=n, dt=dt,
                                                update_every=self.program_update_every)

        states, state_last, memory = self.detector(
            level_in, alpha_attack, alpha_release, self._initial_state(prime_level),
            program_step=program, mix=mix, mem0=self.light_memory)
        self.detector_state = self._carry(state_last, truncate_state)
        if memory is not None:
            self.light_memory = self._carry(memory, truncate_state)

        level_db = (states * mix).sum(-1)
        gain_db, curve_params = self.curve(level_db, c_curve)

        if self.R > 1:
            gain_db, gain_hold = self._upsample(gain_db, length, self.gain_hold, fresh)
            level_db, level_hold = self._upsample(level_db, length, self.level_hold, fresh)
            self.gain_hold = self._carry(gain_hold, truncate_state)
            self.level_hold = self._carry(level_hold, truncate_state)

        y = u * db_to_gain(gain_db)
        aux = dict(gain_db=gain_db, level_db=level_db, input_db=input_db,
                   tau_attack=tau_attack, tau_release=tau_release,
                   detector_states_db=states, mix=mix, **curve_params)
        if memory is not None:
            aux["light_memory"] = memory
        return y, aux

    def _upsample(self, x, length, hold, fresh):
        """Control rate -> audio rate, causally.
        """
        batch, n, channels = x.shape
        previous = x[:, :1] if fresh else hold
        left = torch.cat([previous, x[:, :-1]], dim=1)          # value at block j-1
        ramp = torch.arange(1, self.R + 1, device=x.device, dtype=x.dtype) / self.R
        out = left.unsqueeze(2) + (x - left).unsqueeze(2) * ramp.view(1, 1, -1, 1)
        return out.reshape(batch, n * self.R, channels)[:, :length], x[:, -1:]

    def _forward_feedback(self, u, c_ballistics, c_curve, truncate_state,
                          target=None, teacher_forcing: float = 1.0):
        """Detector sees the output, delayed by one control block.
        """
        batch, length, channels = u.shape
        R = self.R
        n_blocks = length // R

        alpha_attack, alpha_release, tau_attack, tau_release = self.A_module(
            c_ballistics, seq_len=n_blocks, dt=self.dt * R)
        curve_params = self.curve.compute_params(c_curve, seq_len=n_blocks)
        mix = torch.softmax(self.det_mix, dim=-1)
        program = self.A_module.program_stepper(c_ballistics, seq_len=n_blocks,
                                                dt=self.dt * R,
                                                update_every=self.program_update_every)
        alpha_is_static = alpha_attack.shape[1] == 1

        # The sidechain filter has to run recursively here; the FFT path used in
        # the feedforward topology cannot be evaluated inside a sample loop.
        if self.hpf is not None:
            if self.hpf.state.shape != (batch, channels) or self.hpf.state.device != u.device:
                self.hpf.reset_hidden_states(batch, channels, u.device)
            omega = torch.exp(self.hpf.log_omega).clamp(max=0.45 * math.pi / self.dt)
            hpf_pole, hpf_tilt = torch.exp(-omega * self.dt), self.hpf.tilt()
            lowpass = self.hpf.state
        else:
            hpf_pole = hpf_tilt = lowpass = None

        fresh = self.state_init == "first_sample" and not bool(self.primed)
        prime_level = amp_to_db(u[:, :1], self.db_floor)[:, 0]
        state = self._initial_state(prime_level)
        memory = self.light_memory

        def control_step(peak_db, state, memory, j):
            """One detector step + one curve evaluation, at the control rate."""
            if program is not None:
                a_attack, a_release, memory = program((state * mix).sum(-1), memory, j)
            else:
                k = 0 if alpha_is_static else min(j, alpha_attack.shape[1] - 1)
                a_attack = alpha_attack[:, k].unsqueeze(1)
                a_release = alpha_release[:, k].unsqueeze(1)
            state = self.detector.step(peak_db, a_attack, a_release, state)
            level = (state * mix).sum(-1)
            # index per key: params can have different time extents
            p = {k: v[:, 0 if v.shape[1] == 1 else min(j, v.shape[1] - 1)]
                 for k, v in curve_params.items()}
            return state, memory, level, self.curve.apply_curve(level, p)

        peak = prime_level if fresh else self.peak_hold
        state, memory, level, gain_cur = control_step(peak, state, memory, 0)
        gain_prev = gain_cur if fresh else self.gain_hold[:, 0]
        block_peak = torch.full_like(peak, self.db_floor)
        previous = u[:, 0].detach() if fresh else self.last_output

        outputs, gains, levels = [], [], []
        for t in range(length):
            block, offset = divmod(t, R)
            gain_db = (gain_cur if R == 1 else
                       gain_prev + (gain_cur - gain_prev) * ((offset + 1) / R))
            y_t = u[:, t] * db_to_gain(gain_db)

            outputs.append(y_t)
            gains.append(gain_db)
            levels.append(level)

            previous = y_t if target is None else torch.lerp(y_t, target[:, t],
                                                             teacher_forcing)
            if lowpass is not None:
                lowpass = hpf_pole * lowpass + (1.0 - hpf_pole) * previous
                sidechain = previous - hpf_tilt * lowpass
            else:
                sidechain = previous
            measured = amp_to_db(sidechain, self.db_floor)
            block_peak = measured if offset == 0 else torch.maximum(block_peak, measured)

            if offset == R - 1:
                if block < n_blocks - 1:
                    gain_prev = gain_cur
                    state, memory, level, gain_cur = control_step(
                        block_peak, state, memory, block + 1)
                else:                      # carry it; the next chunk consumes it
                    self.peak_hold = self._carry(block_peak, truncate_state)

        self.detector_state = self._carry(state, truncate_state)
        self.light_memory = self._carry(memory, truncate_state)
        self.last_output = self._carry(previous, truncate_state)
        self.gain_hold = self._carry(gain_cur.unsqueeze(1), truncate_state)
        if lowpass is not None:
            self.hpf.state = self._carry(lowpass, truncate_state)

        aux = dict(gain_db=torch.stack(gains, 1), level_db=torch.stack(levels, 1),
                   tau_attack=tau_attack, tau_release=tau_release, mix=mix, **curve_params)
        if program is not None:
            aux["light_memory"] = memory
        return torch.stack(outputs, dim=1), aux

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def describe(self, c=None):
        """Everything the model claims about the device, in physical units."""
        c_ballistics, c_curve = self._split_cond(c, 1)

        tau_attack, tau_release = self.A_module.time_constants(c_ballistics, seq_len=1)
        info = {
            "tau_attack_ms": (tau_attack.flatten() * 1e3).tolist(),
            "tau_release_ms": (tau_release.flatten() * 1e3).tolist(),
            "detector_mix": torch.softmax(self.det_mix, -1).tolist(),
        }
        # For a feedback model these describe the *internal* curve, not the
        # closed-loop characteristic measured at the terminals.
        info.update(self.curve.measure(c_curve))

        if self.program_dependent:
            # The T4 curve: release time vs detector level, for a cell that has
            # been dark (first hit) and one that has equilibrated.
            levels = [-50.0, -40.0, -30.0, -20.0, -10.0, 0.0]
            _, cold = self.A_module.release_vs_level(levels, c_ballistics, settled=False)
            _, warm = self.A_module.release_vs_level(levels, c_ballistics, settled=True)
            mix = torch.softmax(self.det_mix, -1)
            info["release_vs_level_db"] = levels
            info["release_ms_first_hit"] = (cold * mix * 1e3).sum(-1).tolist()
            info["release_ms_settled"] = (warm * mix * 1e3).sum(-1).tolist()
            info["light_memory_tau_s"] = float(torch.exp(self.A_module.log_tau_memory))

        if self.hpf is not None:
            info["sidechain_hpf_hz"] = float(self.hpf.cutoff_hz())
            info["sidechain_tilt"] = float(self.hpf.tilt())      # 0 = flat sidechain
        if self.signal_path is not None:
            freq, q = self.signal_path.resonances_hz()
            info["signal_path_hz"] = freq.tolist()
            info["signal_path_Q"] = q.tolist()
        return info