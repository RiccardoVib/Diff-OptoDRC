"""
Discretisation of a diagonal continuous-time system  x' = a x + b u,  a < 0.
Code adapted from Mamba/S6 -- https://github.com/RiccardoVib/Optical-DRC-with-Selective-SSMs

mamba -- Simionato & Fasciani, JAES 73(3), 2025
"""

import math
import torch


def phi1(z, eps=1e-6):
    """(exp(z) - 1) / z, accurate at z = 0 and differentiable there."""
    small = z.abs() < eps
    z_safe = torch.where(small, torch.ones_like(z), z)
    return torch.where(small, 1.0 + z_safe * 0.0 + z / 2.0, torch.expm1(z_safe) / z_safe)


def zoh_diag(a, T):
    """Exact ZOH for a diagonal continuous-time `a` (< 0).  Returns (a_d, b_factor)
    with `b_d = b_factor * b`.  b_factor = T * phi1(a T) -> T as a -> 0."""
    z = a * T
    return torch.exp(z), T * phi1(z)


def bilinear_diag(a, T):
    """Tustin.  a_d = (1 + aT/2)/(1 - aT/2); b_factor = T / (1 - aT/2)."""
    half = a * T / 2.0
    den = 1.0 - half
    return (1.0 + half) / den, T / den


def alpha_from_log_tau(log_tau, T):
    """log a_d = -T/tau, computed as -T*exp(-log_tau): no division, no overflow,
    monotone and well conditioned over the whole 0.1 ms .. 10 s range."""
    return -T * torch.exp(-log_tau)


def one_minus_exp(log_a):
    """1 - a  for a = exp(log_a), exact as log_a -> 0."""
    return -torch.expm1(log_a)


def db_from_amp(x, floor_db=-100.0):
    return 20.0 * torch.log10(x.abs().clamp_min(10.0 ** (floor_db / 20.0)))


def amp_from_db(x_db):
    return torch.pow(10.0, x_db / 20.0)


LOG10 = math.log(10.0)
