"""Wrapper for grey-box: https://github.com/Alec-Wright/GreyBoxDRC
grey-box DRC (Wright & Valimaki, DAFx 2022)
"""

import torch
import torch.nn as nn
from GreyModel import GreyBoxComp



# Keys consumed by this wrapper; everything else goes to GreyBoxComp.
_WRAPPER_KEYS = ("config", "warmup_max", "prime_state", "reset_per_batch",
                 "trim_output", "grad_clip", "criterion")

# Keys GreyBoxComp accepts from a config preset.
_MODEL_KEYS = ("static_comp", "gain_smooth", "make_up", "cond_names",
               "cond_routing", "sample_rate", "min_db")


def to_bcl(x):
    """[B, L, 1] -> [B, 1, L]. The losses in paper_losses want [B, C, L]."""
    return x.permute(0, 2, 1).contiguous()


def to_blc(x):
    """Accept either layout and return [B, L, 1].
    """
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.shape[-1] == 1:
        return x
    if x.shape[1] == 1:
        return x.permute(0, 2, 1)
    raise ValueError(f"expected a single-channel signal, got {tuple(x.shape)}")


class GreyCompWrapper(nn.Module):
    """Factory-compatible wrapper around :class:`GreyBoxComp`.
    """

    def __init__(self, cond_dim: int, buffer: int = 0, seq_len=2048,
                 **greybox_kwargs):
        super().__init__()
        opts = {k: greybox_kwargs.pop(k) for k in _WRAPPER_KEYS
                if k in greybox_kwargs}

        self.model = GreyBoxComp(cond_dim=cond_dim, seq_len=seq_len,
                                 **greybox_kwargs)

        self.cond_dim = int(cond_dim)
        self.buffer = int(buffer)
        self.seq_len = seq_len
        self.sample_rate = float(self.model.sample_rate)
        self.warmup_max = opts.get("warmup_max")
        self.prime_state = bool(opts.get("prime_state", True))
        self.reset_per_batch = bool(opts.get("reset_per_batch", True))
        self.trim_output = bool(opts.get("trim_output", True))
        self.grad_clip = opts.get("grad_clip")

        self._criterion = [opts.get("criterion")]

    # ------------------------------------------------------------------
    @property
    def criterion(self):
        return self._criterion[0]

    def set_criterion(self, criterion):
        self._criterion[0] = criterion

    def get_input_length(self):
        """Warmup the model actually needs: 5 tau of the slowest release."""
        rel = self.model.gain_smooth.describe(None).get("release_ms")
        if rel is None or rel != rel:
            rel = 1000.0
        return int(5.0 * rel * 1e-3 * self.sample_rate)

    def reset_hidden_states(self, batch_size=None):
        self.model.reset_hidden_states(batch_size or 1)

    def detach_hidden_states(self):
        self.model.detach_hidden_states()

    # ------------------------------------------------------------------
    def _cond(self, c):
        if c is None:
            return None
        if c.dim() == 3:
            c = c[:, 0, :]
        return c

    def forward(self, x, c=None):
        """``x`` [B, L, 1] (or [B, 1, L]), ``c`` [B, C] or [B, 1, C] -> [B, L, 1].
        """
        y = self.model(to_blc(x), self._cond(c))
        if self.trim_output and self.seq_len and y.shape[1] > self.seq_len:
            y = y[:, -int(self.seq_len):]
        return y

    def gain_db(self, x, c=None):
        """Smoothed gain envelope in dB, for plotting against measured GR."""
        return self.model.gain_db(to_blc(x), self._cond(c))

    # ------------------------------------------------------------------
    def _warmup(self, mem, c):
        """Prime the state on ``mem`` without building a graph over it."""
        if mem is None or mem.numel() == 0:
            return
        mem = to_blc(mem)
        if self.warmup_max is not None:
            n = int(self.warmup_max)
            if n <= 0:
                return
            mem = mem[:, -n:] if mem.shape[1] > n else mem
        with torch.no_grad():
            if self.prime_state:
                self.model.prime_from(mem, c)
            self.model(mem, c)

    def _run(self, x, mem, c):
        c = self._cond(c)
        if self.reset_per_batch:
            self.model.reset_hidden_states(x.shape[0])
        self._warmup(mem, c)
        self.model.detach_hidden_states()
        return self.model(to_blc(x), c)

    def train_step(self, x, y, mem=None, c=None, optimizer=None, criterion=None):
        criterion = criterion or self.criterion
        if criterion is None:
            raise ValueError("no criterion: pass one or call set_criterion()")
        self.train()
        pred = self._run(x, mem, c)
        loss = criterion(to_bcl(pred), to_bcl(to_blc(y)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        optimizer.step()
        self.model.detach_hidden_states()
        return float(loss.detach())

    @torch.no_grad()
    def val_step(self, x, y, mem=None, c=None, criterion=None):
        criterion = criterion or self.criterion
        self.eval()
        pred = self._run(x, mem, c)
        return float(criterion(to_bcl(pred), to_bcl(to_blc(y))).detach())

    # ------------------------------------------------------------------
    def describe(self, c=None):
        return self.model.describe(self._cond(c) if c is not None else None)