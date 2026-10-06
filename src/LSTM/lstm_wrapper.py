"""
Wrapper for lstm: https://github.com/csteinmetz1/micro-tcn
LSTM micro-tcn(Steinmetz & Reiss, AES 2022)
"""

import torch
from src.common import BaselineModel
from lstm import LSTMModel


class LSTMWrapper(BaselineModel):
    supports_conditioning = True
    cat_input = True

    def __init__(self, cond_dim: int, buffer: int = 0, seq_len=2048, **tcn_kwargs):
        super().__init__()
        self.model = LSTMModel(nparams=cond_dim, **tcn_kwargs)
        self.criterion = None
        self.buffer = buffer
        self.target_length = seq_len

    def reset_hidden_states(self):
        self.model.reset_states()

    def detach_states(self):
        self.model.detach_states()

    def forward(self, x, c=None):
        T = self.target_length

        warm, tgt = x[:, :-T], x[:, -T:]
        self.model.reset_states()

        with torch.no_grad():
            L_warm = warm.shape[1]
            warmup_chunk_size = 2400
            for start in range(0, L_warm, warmup_chunk_size):
                end = min(start + warmup_chunk_size, L_warm)
                _ = self.model(warm[:, start:end], c)
                self.model.detach_states()

        out = self.model(tgt, c)                 # [B, L, 1]
        out = out[:, :, -self.target_length:]
        return out

    def train_step(self, x, y, mem, c, optimizer, criterion=None):
        criterion = criterion or self.criterion
        optimizer.zero_grad()
        y = y.permute(0, 2, 1)
        x = torch.cat([mem, x], dim=1)
        pred = self.forward(x, c[:, None, :])         # [B, L, 1]
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        self.detach_states()                       # truncated BPTT across chunks
        return loss.item()

    @torch.no_grad()
    def val_step(self, x, y, mem, c, criterion=None):
        criterion = criterion or self.criterion
        y = y.permute(0, 2, 1)
        x = torch.cat([mem, x], dim=1)
        pred = self.forward(x, c[:, None, :])
        pred = self._match_length(pred, y)
        loss = criterion(pred, y)
        return loss.item()
