"""
Wrapper for TCN: https://github.com/csteinmetz1/micro-tcn
TCN micro-tcn(Steinmetz & Reiss, AES 2022)
"""
from src.common.common import BaselineModel
from tcn import TCNModel


class TCNWrapper(BaselineModel):
    stateful = False
    supports_conditioning = True
    cat_input = True

    def __init__(self, cond_dim: int, buffer: int = 0, seq_len=2048, **tcn_kwargs):
        super().__init__()
        self.model = TCNModel(nparams=cond_dim, **tcn_kwargs)
        self.criterion = None
        self.buffer = buffer
        self.target_length = seq_len


    def forward(self, x, c=None):
        return self.model(x, c)
