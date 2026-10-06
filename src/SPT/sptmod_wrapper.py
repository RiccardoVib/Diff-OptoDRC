"""
Wrapper for SPTmod net:
sptmod -- Bourdin, Legrand & Roche, DAFx 2025 ("SPTMod25")
"""
import torch
from torch import Tensor
from src.common.common import BaselineModel
from sptmod import Model as SPTModCore, CachedPadding1d


class SPTModWrapper(BaselineModel):
    stateful = False
    supports_conditioning = True
    cat_input = True

    def __init__(self, cond_dim: int, use_spn: bool = True,
                 eval_mode: str = "st", buffer: int = 0, seq_len=2048,
                 reset_states_each_chunk: bool = True, warmup_no_grad: bool = True,
                 warmup_chunk: int = 32768,
                 **sptmod_kwargs):
        super().__init__()
        self.model = SPTModCore(n_params=cond_dim, **sptmod_kwargs)

        self.buffer = int(buffer)
        self.reset_states_each_chunk = reset_states_each_chunk and self.buffer > 0
        self.use_spn = use_spn and self.model.spnet is not None

        self.eval_mode = eval_mode
        self.criterion = None
        self.target_length = seq_len
        self._spn_pending = True
        self.warmup_no_grad = warmup_no_grad and buffer > 0
        self.warmup_chunk = warmup_chunk
        if self.warmup_no_grad:
            block = self.model.tfilm_block_size
            if self.model.spnet is not None:
                raise ValueError(
                    "warmup_no_grad is the replacement for the state "
                    "predictor, not a companion to it. Build with "
                    "spnet=False (the SPN already avoids this cost by "
                    "downsampling the lookback), or set warmup_no_grad=False."
                )
            for name, v in (("seq_len", seq_len), ("buffer", buffer),
                            ("warmup_chunk", warmup_chunk)):
                if v and v % block:
                    raise ValueError(
                        f"{name} ({v}) must be a multiple of "
                        f"tfilm_block_size ({block}): every forward call "
                        f"is pooled by that factor "
                        f"(assert z.shape[2] % block_size == 0)."
                    )

        self.get_input_length()

    def get_input_length(self) -> int:
        """Number of input samples the DATASET must supply per chunk.
        """
        if self.warmup_no_grad:
            self.model.set_target_length(self.target_length)
            return self.buffer

        total_out = self.target_length + self.buffer

        # SPTMod's pooling size must divide the sequence length
        block = self.model.tfilm_block_size
        if total_out % block != 0:
            raise ValueError(
                f"seq_len + warmup = {total_out} must be a multiple of "
                f"tfilm_block_size ({block}). Nearest valid warmup: "
                f"{((total_out // block) * block) - self.target_length} or "
                f"{(((total_out // block) + 1) * block) - self.target_length}."
            )
        if self.buffer > 0 and self.model.spnet is not None:
            print("[SPTModWrapper] warning: warmup > 0 together with "
                  "spnet=True adds the SPN lookback ON TOP of the warmup. "
                  "Build with spnet=False when using warmup.")

        self.model.set_target_length(total_out)
        return self.model.input_length - self.target_length

    def reset_hidden_states(self):
        self.model.reset_states()
        self.model.reset_caches()
        self._spn_pending = True

    def detach_states(self):
        self.model.detach_states()

    @staticmethod
    def _as_2d(c):
        """SPTMod's condnet/film modules (PReLU-based) require plain
        [B, n_params]. Flatten any extra broadcast dims added upstream."""
        if c is not None and c.dim() > 2:
            c = c.reshape(c.shape[0], -1)
        return c

    def forward(self, x, c=None, y_true=None):
        if self.target_length is None:
            raise RuntimeError(
                "call get_input_length(target_length) once (the dataset "
                "does this when it sizes its windows) before forward()"
            )
        c = self._as_2d(c)

        if self.warmup_no_grad:
            return self._forward_warmup(x, c)


        use_spn = self.use_spn and (
            self.training or self.eval_mode == "wt" or self._spn_pending)

        if use_spn and y_true is None:
            raise ValueError(
                "SPTMod's state predictor needs y_true (ground-truth "
                "lookback) for this call; pass it, or set use_spn=False / "
                "eval_mode='st' after the first chunk."
            )

        if self.reset_states_each_chunk:

           self.model.reset_states()

        out = self.model(
            x, c,
            y_true=y_true if use_spn else None,
            use_spn=use_spn,
            paddingmode=CachedPadding1d.NoPadding,
        )
        self._spn_pending = False

        # drop the warm-up outputs; keep the target region
        if self.buffer > 0:
            out = out[..., -self.target_length:]
        return out

    def _forward_warmup(self, x, c):
        """Warm-up under no_grad, then the target region with grad.
        """
        T = self.target_length
        warm, tgt = x[..., :-T], x[..., -T:]

        self.model.reset_states()
        self.model.reset_caches()

        if warm.shape[-1] > 0:
            step = self.warmup_chunk or warm.shape[-1]
            with torch.no_grad():
                for i in range(0, warm.shape[-1], step):
                    self.model(warm[..., i:i + step], c, y_true=None,
                               use_spn=False,
                               paddingmode=CachedPadding1d.CachedPadding)
        self.model.detach_states()

        return self.model(tgt, c, y_true=None, use_spn=False,
                          paddingmode=CachedPadding1d.CachedPadding)

    def train_step(self, x, y, mem, c, optimizer, criterion=None):
        criterion = criterion or self.criterion
        optimizer.zero_grad()
        y = y.permute(0, 2, 1)
        x = torch.cat([mem, x], dim=1).permute(0, 2, 1)
        pred = self(x, c, y_true=y)
        target = self._match_length(y, pred)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def val_step(self, x, y, mem, c, criterion=None):
        criterion = criterion or self.criterion
        x = torch.cat([mem, x], dim=1).permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        pred = self(x, c, y_true=y)
        target = self._match_length(y, pred)
        loss = criterion(pred, target)
        return loss.item()