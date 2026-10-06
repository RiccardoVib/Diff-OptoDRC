"""
Wrapper for mamba/S6 -- https://github.com/RiccardoVib/Optical-DRC-with-Selective-SSMs
mamba -- Simionato & Fasciani, JAES 73(3), 2025
"""
import torch
from src.common.common import BaselineModel
from mamba_torch import MambaModel


class MambaWrapper(BaselineModel):
    supports_conditioning = True
    cat_input = True

    def __init__(self, cond_dim: int, buffer: int = 0, seq_len=2048, **mamba_kwargs):
        super().__init__()
        self.model = MambaModel(nparams=cond_dim, **mamba_kwargs)
        self.buffer = buffer
        self.target_length = seq_len
        self.criterion = None

    def reset_hidden_states(self):
        self.model.reset_states()

    def detach_states(self):
        self.model.detach_states()

    def _make_windows(self, x):
        """[B, L, 1] + [B, buffer, 1] -> [B, L, window] sliding windows."""
        # history FIRST, then the current chunk
        x = x.squeeze(-1)                    # [B, buffer + L]
        W = self.model.window
        if x.shape[1] < W:
            raise ValueError(
                f"need at least {W} samples of context (buffer + chunk), "
                f"got {x.shape[1]}; check get_input_length wiring"
            )
        # each row t is full[t : t+W], i.e. the W samples ending at t+W-1
        return x.unfold(dimension=1, size=W, step=1)  # [B, L, W]

    def forward(self, x, c=None):
        x_win = self._make_windows(x)
        T = self.target_length
        warm, tgt = x_win[:, :-T], x_win[:, -T:]
        self.model.reset_states()

        with torch.no_grad():
            L_warm = warm.shape[1]
            warmup_chunk_size = 2400
            for start in range(0, L_warm, warmup_chunk_size):
                end = min(start + warmup_chunk_size, L_warm)
                _ = self.model(warm[:, start:end], c)
                self.model.detach_states()

        out = self.model(tgt, c)                 # [B, L, 1]
        out = out[:, -self.target_length:, :]
        return out                # [B, L, 1] to match pipeline

    def train_step(self, x, y, mem, c, optimizer, criterion=None):
        criterion = criterion or self.criterion
        optimizer.zero_grad()
        y = y.permute(0, 2, 1)
        x = torch.cat([mem, x], dim=1)
        pred = self.forward(x, c).permute(0, 2, 1)        # [B, L, 1]
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
        pred = self.forward(x, c).permute(0, 2, 1)
        loss = criterion(pred, y)
        return loss.item()
