"""
grey-box: code adapted from https://github.com/Alec-Wright/GreyBoxDRC
grey-box DRC (Wright & Valimaki, DAFx 2022)
"""

import torch
import torch.nn as nn


class StaticAmp(nn.Module):
    def __init__(self, dc_bias=False):
        super().__init__()
        self.cond = False
        self.dc_bias = bool(dc_bias)
        self.gain = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(())) if dc_bias else None

    def forward(self, x, cond=None):
        if self.dc_bias:
            x = x + torch.tanh(self.bias)
        return x * (torch.tanh(self.gain) + 1.0)

    def reset_state(self, batch_size=1):
        pass

    def detach_state(self):
        pass

    def describe(self):
        g = float((torch.tanh(self.gain) + 1.0).detach())
        d = {"makeup_gain": g,
             "makeup_db": 20.0 * torch.log10(torch.tensor(max(g, 1e-9))).item()}
        if self.dc_bias:
            d["dc_bias"] = float(torch.tanh(self.bias).detach())
        return d


class GRUAmp(nn.Module):
    def __init__(self, hidden_size=8, init_scale=1e-3):
        super().__init__()
        self.cond = False
        self.rec = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, 1)

        with torch.no_grad():
            self.lin.weight.mul_(init_scale)
            self.lin.bias.zero_()
        self.state = None

    def forward(self, x, cond=None):
        res = x
        h, self.state = self.rec(x, self.state)
        return self.lin(h) + res

    def reset_state(self, batch_size=1):
        self.state = None

    def detach_state(self):
        if self.state is not None:
            self.state = self.state.detach()

    def describe(self):
        return {"makeup": "gru", "hidden_size": self.rec.hidden_size}


MAKEUPS = {"Static": StaticAmp, "GRU": GRUAmp}


class MakeUp(nn.Module):
    def __init__(self, params):
        super().__init__()
        params = dict(params)
        self.type = params.pop("type")
        if self.type not in MAKEUPS:
            raise ValueError(f"unknown make_up type {self.type!r}; "
                             f"expected one of {sorted(MAKEUPS)}")
        self.model = MAKEUPS[self.type](**params)

    def forward(self, x, cond=None):
        return self.model(x, cond)

    def reset_state(self, batch_size=1):
        self.model.reset_state(batch_size)

    def detach_state(self):
        self.model.detach_state()

    def describe(self):
        return self.model.describe()