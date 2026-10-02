# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato

"""
Common interface for all baseline models, so that training.py never has to branch on
which architecture it's training.

Batch convention:
    x   : [B, C_in,  L_in]   input audio window (+ sidechain if present)
    y   : [B, C_out, L_out]  target audio window
    mem : [B, ...]           side/memory features -- only consumed by your
                              own model; baselines accept and ignore it,
                              purely to keep the call signature uniform.
    c   : [B, P]             conditioning parameters (device controls)

"""
import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    stateful: bool = False
    supports_conditioning: bool = True
    cat_input: bool = True

    def reset_hidden_states(self):
        pass

    def detach_states(self):
        pass

    def set_criterion(self, criterion):
        self.criterion = criterion

    @staticmethod
    def _match_length(y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Center-crop the target so it lines up with the model's output."""
        if y.shape[-1] == pred.shape[-1]:
            return y
        start = (y.shape[-1] - pred.shape[-1]) // 2
        return y[..., start:start + pred.shape[-1]]

    def forward(self, x, c=None):
        raise NotImplementedError

    def train_step(self, x, y, mem, c, optimizer, criterion=None):
        criterion = criterion or self.criterion
        optimizer.zero_grad()
        if self.cat_input:
            x = torch.cat([mem, x], dim=1).permute(0, 2, 1)
            pred = self(x, c[:, None, :])
            y = y.permute(0, 2, 1)
        else:
            pred = self(x, mem, c)
        target = self._match_length(y, pred)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def val_step(self, x, y, mem, c, criterion=None):
        criterion = criterion or self.criterion
        if self.cat_input:
            x = torch.cat([mem, x], dim=1).permute(0, 2, 1)
            pred = self(x, c[:, None, :])
            y = y.permute(0, 2, 1)
        else:
            pred = self(x, mem, c)
        target = self._match_length(y, pred)
        loss = criterion(pred, target)
        return loss.item()
