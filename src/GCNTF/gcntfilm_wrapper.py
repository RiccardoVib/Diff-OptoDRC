"""
Wrapper for GCNTF-TFiLM: https://github.com/mcomunita/gcn-tfilm/
gcn-tfilm -- Comunità et al., ICASSP 2023 (config "GCNTF3")
"""

from src.common.common import BaselineModel
from gcntfilm_cond import GCNTFCond


class GCNTFWrapper(BaselineModel):
    stateful = False
    supports_conditioning = True
    cat_input = True

    def __init__(self, cond_dim: int = 0, buffer: int = 0, seq_len=2048, **gcn_kwargs):
        super().__init__()

        self.model = GCNTFCond(**gcn_kwargs)
        self.criterion = None
        self.buffer = buffer
        self.target_length = seq_len

    def get_input_length(self) -> int:
        rf = self.model.compute_receptive_field()
        block = self.model.tfilm_block_size

        total = self.target_length + rf
        remainder = total % block
        if remainder != 0:
            total += block - remainder  # round up to a multiple of block

        self.buffer = total - self.target_length
        return self.buffer

    def reset_hidden_states(self):
        self.model.reset_states()

    def detach_states(self):
        self.model.detach_states()

    def forward(self, x, c=None):
        # GCNTF expects [length, batch, channels]
        x_ = x.permute(2, 0, 1)
        y = self.model(x_, c)
        y = y.permute(1, 2, 0)
        return y[..., self.buffer:]
