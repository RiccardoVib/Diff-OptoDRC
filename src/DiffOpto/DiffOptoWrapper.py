# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato

import torch
from src.common.common import BaselineModel
from DiffOpto import DiffOpto


class DiffOptoWrapper(BaselineModel):
    stateful = False
    supports_conditioning = True
    cat_input = True

    def __init__(self, cond_dim: int, buffer: int = 0, seq_len: int = 2048,
                 ballistics_dim: int = None, curve_dim: int = None,
                 teacher_forcing: float = 1.0, **model_kwargs):
        super().__init__()

        ballistics_dim = 0 if ballistics_dim is None else int(ballistics_dim)
        if curve_dim is None:
            curve_dim = int(cond_dim) - ballistics_dim
        if ballistics_dim + curve_dim > int(cond_dim):
            raise ValueError(
                f"ballistics_dim + curve_dim = {ballistics_dim + curve_dim} > "
                f"cond_dim = {cond_dim}")

        self.model = DiffOpto(ballistics_dim=ballistics_dim,
                               curve_dim=curve_dim, **model_kwargs)
        self.buffer = int(buffer)
        self.target_length = int(seq_len)
        self.criterion = None

        # 1.0 = pure teacher forcing (parallel, fast), 0.0 = free-running.
        self.teacher_forcing = float(teacher_forcing)
        self.can_teacher_force = self.model.topology == "feedback"

        self.period = self.model.R * self.model.program_update_every
        if self.period > 1 and self.target_length % self.period:
            raise ValueError(
                f"seq_len={self.target_length} must be a multiple of "
                f"sidechain_decimation * program_update_every = {self.period}")

    # ------------------------------------------------------------------ #

    def set_teacher_forcing(self, p: float):
        """Scheduled sampling: anneal from 1.0 to 0.0 over training."""
        self.teacher_forcing = float(max(0.0, min(1.0, p)))

    def reset_hidden_states(self, *args, **kwargs):

        self.model.reset_hidden_states(*args, **kwargs)

    def _trim_history(self, history):
        """Drop the oldest samples so the warm-up covers whole control periods."""
        if self.period > 1:
            extra = history.shape[1] % self.period
            if extra:
                history = history[:, extra:]
        return history

    # ------------------------------------------------------------------ #

    def forward(self, x, c=None, target=None):
        """x: (B, buffer + seq_len, 1).  ``target`` is the same span of y, or None."""
        T = self.target_length
        history, current = x[:, :-T], x[:, -T:]
        history = self._trim_history(history)

        use_tf = (target is not None and self.training
                  and self.can_teacher_force and self.teacher_forcing > 0.0)
        if use_tf:
            tgt_hist, tgt_current = target[:, :-T], target[:, -T:]
            tgt_hist = self._trim_history(tgt_hist)
        else:
            tgt_hist = tgt_current = None

        self.model.reset_hidden_states(x.shape[0], x.device)

        if history.shape[1] > 0:
            with torch.no_grad():

                self.model(history, c, target=tgt_hist,
                           teacher_forcing=self.teacher_forcing if use_tf else 1.0)

        return self.model(current, c, target=tgt_current,
                          teacher_forcing=self.teacher_forcing if use_tf else 1.0)

    # ------------------------------------------------------------------ #

    def train_step(self, x, y, mem, c, optimizer, criterion=None, mem_y=None):
        criterion = criterion or self.criterion
        optimizer.zero_grad()

        x_full = torch.cat([mem, x], dim=1)
        if mem_y is not None and self.can_teacher_force:
            target_full = torch.cat([mem_y, y], dim=1)
        else:
            target_full = None

        pred = self.forward(x_full, c, target=target_full).permute(0, 2, 1)
        loss = criterion(pred, y.permute(0, 2, 1))              # [B, 1, L]
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def val_step(self, x, y, mem, c, criterion=None):
        criterion = criterion or self.criterion
        x_full = torch.cat([mem, x], dim=1)
        pred = self.forward(x_full, c).permute(0, 2, 1)         # free-running
        loss = criterion(pred, y.permute(0, 2, 1))
        return loss.item()