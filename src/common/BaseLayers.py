# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato

import torch.nn as nn
from torch.nn import init

def init_linear_like_keras(linear_layer):
    """Initialize Linear layer like Keras Dense (glorot_uniform)"""
    init.xavier_uniform_(linear_layer.weight)
    if linear_layer.bias is not None:
        init.zeros_(linear_layer.bias)

def init_lstm_like_keras(lstm_layer):
    """
    Initialize LSTM weights to match Keras/TensorFlow defaults:
    - Input weights (weight_ih): glorot_uniform (Xavier uniform)
    - Recurrent weights (weight_hh): orthogonal
    - Biases: zeros
    """
    for name, param in lstm_layer.named_parameters():
        if 'weight_ih' in name:
            # Input-hidden weights: Xavier/Glorot uniform
            init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            # Hidden-hidden (recurrent) weights: Orthogonal
            init.orthogonal_(param.data)
        elif 'bias' in name:
            # Biases: zeros (optionally set forget gate bias to 1)
            param.data.fill_(0.0)
            # Optional: Initialize forget gate bias to 1 (Jozefowicz et al., 2015)
            # This improves training stability
            n = param.size(0)
            param.data[n//4: n//2].fill_(1.0)


def defineLSTMLayer(input_dim, output_dim, bias=True):

    layer = nn.LSTM(input_dim, output_dim, batch_first=True, bias=bias)
    init_lstm_like_keras(layer)
    return layer


def defineLinearLayer(input_dim, output_dim, bias=True):
    layer = nn.Linear(input_dim, output_dim, bias=bias)
    init_linear_like_keras(layer)
    return layer
