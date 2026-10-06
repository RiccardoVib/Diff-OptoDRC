"""
lstm: code adapted from https://github.com/csteinmetz1/micro-tcn
LSTM micro-tcn(Steinmetz & Reiss, AES 2022)
"""

import torch
from src.TCN.base import Base


class LSTMModel(Base):
    def __init__(self,
                 nparams,
                 ninputs=1,
                 noutputs=1,
                 hidden_size=32,
                 num_layers=1,
                 ):
        super(LSTMModel, self).__init__()
        input_size = ninputs + nparams
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.noutputs = noutputs
        self.lstm = torch.nn.LSTM(input_size,
                                  self.hidden_size,
                                  self.num_layers,
                                  batch_first=False,
                                  bidirectional=False)

        self.linear = torch.nn.Linear(self.hidden_size,
                                      self.noutputs)

        self.state = None

    def reset_states(self):
        self.state = None

    def detach_states(self):
        if self.state is not None:
            h, c = self.state
            self.state = (h.clone().detach(), c.clone().detach())

    def forward(self, x, p):
        # bs = x.size(0) # batch size
        s = x.size(1)  # samples
        x = x.permute(1, 0, 2)  # shape for LSTM (seq, batch, channel)

        if self.state is None:
            h = torch.zeros((1, x.shape[1], self.hidden_size), dtype=x.dtype, device=x.device)
            c = torch.zeros((1, x.shape[1], self.hidden_size), dtype=x.dtype, device=x.device)
            state = (h, c)
        else:
            state = self.state

        if p is not None:
            p = p.permute(1, 0, 2)  # change channel to seq dim
            p = p.repeat(s, 1, 1)  # expand to every time step
            x = torch.cat((x, p), dim=-1)  # append to input along feature dim

        out, state = self.lstm(x, state)
        self.state = state
        out = torch.tanh(self.linear(out))
        out = out.permute(1, 2, 0)  # put shape back (batch, channel, seq)

        return out