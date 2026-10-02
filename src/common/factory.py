# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato

from src.LSTM.lstm_wrapper import LSTMWrapper
from src.TCN.tcn_wrapper import TCNWrapper
from src.GCN.gcntfilm_wrapper import GCNTFWrapper
from src.SPT.sptmod_wrapper import SPTModWrapper
from src.Mamba.mamba_wrapper import MambaWrapper
from src.GrayBox.GreyCompWrapper import GreyCompWrapper
from src.DiffOpto.DiffOptoWrapper import DiffOptoWrapper

def build_model(model_type: str, cond_dim: int, buffer: int = 0, seq_len: int = 2048, **kwargs):
    model_type = model_type.lower()

    if model_type == "greycomp":
        return GreyCompWrapper(cond_dim=cond_dim, buffer=buffer, seq_len=seq_len, **kwargs)

    if model_type == "tcn":
        return TCNWrapper(cond_dim=cond_dim, buffer=buffer, seq_len=seq_len, **kwargs)

    if model_type == "lstm":
        return LSTMWrapper(cond_dim=cond_dim, buffer=buffer, seq_len=seq_len, **kwargs)

    if model_type in ("gcntfilm"):
        return GCNTFWrapper(cond_dim=cond_dim, buffer=buffer, seq_len=seq_len, **kwargs)

    if model_type in ("sptmod"):
        return SPTModWrapper(cond_dim=cond_dim, buffer=buffer, seq_len=seq_len, **kwargs)

    if model_type in ("mamba"):
        return MambaWrapper(cond_dim=cond_dim, buffer=buffer, seq_len=seq_len, **kwargs)

    if model_type in ("diffopto"):
        return DiffOptoWrapper(cond_dim=cond_dim, buffer=buffer, seq_len=seq_len, **kwargs)

    raise ValueError(f"unknown model_type: {model_type!r}")
