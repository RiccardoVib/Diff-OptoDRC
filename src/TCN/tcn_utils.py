"""
Code adapted from TCN: https://github.com/csteinmetz1/micro-tcn
TCN micro-tcn(Steinmetz & Reiss, AES 2022)
"""

def center_crop(x, length: int):
    start = (x.shape[-1]-length)//2
    stop  = start + length
    return x[...,start:stop]

def causal_crop(x, length: int):
    stop = x.shape[-1] - 1
    start = stop - length
    return x[...,start:stop]
