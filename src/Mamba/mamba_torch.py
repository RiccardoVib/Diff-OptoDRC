"""
Code adapted from Mamba/S6 -- https://github.com/RiccardoVib/Optical-DRC-with-Selective-SSMs
mamba -- Simionato & Fasciani, JAES 73(3), 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """Gated linear unit.
    """

    def __init__(self, in_size):
        super(GLU, self).__init__()
        self.in_size = in_size
        self.dense = nn.Linear(in_size, in_size * 2)
        self.act = torch.nn.Softsign()
    def forward(self, x):
        x = self.dense(x)
        a, b = torch.chunk(x, 2, dim=-1)
        return a * self.act(b)


class FiLM(nn.Module):
    """Static feature-wise modulation: x = a*x + b, then GLU."""

    def __init__(self, in_size, cond_dim, bias=True):
        super(FiLM, self).__init__()
        self.in_size = in_size
        self.dense = nn.Linear(cond_dim, in_size * 2, bias=bias)
        self.glu = GLU(in_size=in_size)

    def forward(self, x, c):
        c = self.dense(c)
        a, b = torch.chunk(c, 2, dim=-1)
        x = a * x
        x = x + b
        return self.glu(x)


class TemporalFiLM(nn.Module):
    """Time-varying modulation: a GRU turns the conditioning sequence into
    a per-timestep (a, b), then x = a*x + b, then GLU.
    """

    def __init__(self, in_size, cond_dim, bias=True):
        super(TemporalFiLM, self).__init__()
        self.in_size = in_size
        self.gru = nn.GRU(input_size=cond_dim,
                          hidden_size=in_size * 2,
                          num_layers=1,
                          batch_first=True,
                          bias=bias)
        self.glu = GLU(in_size=in_size)
        self.hidden_state = None

    def reset_state(self):
        self.hidden_state = None

    def detach_state(self):
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.clone().detach()

    def forward(self, x, c):
        h = self.hidden_state
        if (h is not None) and (h.shape[1] != c.shape[0]):
            h = None
        if h is None:
            h = torch.zeros(1, c.shape[0], self.in_size * 2,
                            device=c.device, dtype=c.dtype)
        c_out, h_new = self.gru(c, h)
        self.hidden_state = h_new

        a, b = torch.chunk(c_out, 2, dim=-1)
        x = a * x
        x = x + b
        return self.glu(x)


def selective_scan(u, delta, A, B, C, D, last_state=None,
                   chunk_size=32, dA_clamp=8.0):
    """Chunked selective scan -- numerically stable replacement for the
    single-shot cumsum form used in the TF original / mamba-tiny.

    The recurrence being solved is exactly the same one:

        x_t = exp(dA_t) * x_{t-1} + dB_u_t          dA_t <= 0
        y_t = <x_t, C_t>  (+ u_t * D)

    Returns (y, final_state) exactly like the original.
    """
    b, L, d = u.shape
    n = A.shape[-1]
    K = min(int(chunk_size), L)

    if K * float(dA_clamp) > 300.0:
        raise ValueError(
            f"chunk_size ({K}) * dA_clamp ({dA_clamp}) = {K * float(dA_clamp):.0f} "
            f"exceeds the safe range (300). Forward alone tolerates ~709 (the "
            f"float64 exp limit), but BACKWARD differentiates cumsum(dB_u / Q) "
            f"and therefore evaluates 1/Q**2, which overflows float64 above "
            f"~354 -- silently, as a nan gradient. Use chunk_size <= "
            f"{int(300 // float(dA_clamp))} for this dA_clamp, or lower dA_clamp."
        )

    dA = torch.einsum('bld,dn->bldn', delta, A).clamp(min=-float(dA_clamp))
    dB_u = torch.einsum('bld,bld,bln->bldn', delta, u, B)

    # pad the length up to a whole number of chunks. dA = 0 on the pad
    pad = (K - L % K) % K
    if pad:
        dA = F.pad(dA, (0, 0, 0, 0, 0, pad))
        dB_u = F.pad(dB_u, (0, 0, 0, 0, 0, pad))
    Lp = L + pad
    nc = Lp // K

    work_dtype = torch.float64
    in_dtype = dB_u.dtype
    dA_c = dA.to(work_dtype).reshape(b, nc, K, d, n)
    dB_c = dB_u.to(work_dtype).reshape(b, nc, K, d, n)

    # inclusive cumulative sum within each chunk (<= 0)
    S = torch.cumsum(dA_c, dim=2)
    # shifted version: Sh_0 = 0, Sh_t/Sh_{t-1} ratio = exp(dA_t)
    Sh = S - dA_c[:, :, :1]
    Q = torch.exp(Sh)

    # solution within the chunk, assuming zero initial state
    part = Q * torch.cumsum(dB_c / Q, dim=2)

    # ---- propagate state across chunks  ----
    decay_chunk = torch.exp(S[:, :, -1])          # [b, nc, d, n]
    part_end = part[:, :, -1]                     # [b, nc, d, n]

    if last_state is not None:
        h = last_state.to(work_dtype)
        if h.shape[0] != b:                       # batch size changed
            h = torch.zeros(b, d, n, dtype=work_dtype, device=dB_u.device)
    else:
        h = torch.zeros(b, d, n, dtype=work_dtype, device=dB_u.device)

    h_init = []
    for c in range(nc):
        h_init.append(h)
        h = decay_chunk[:, c] * h + part_end[:, c]
    h_init = torch.stack(h_init, dim=1)           # [b, nc, d, n]

    # contribution of each chunk's incoming state
    x = part + torch.exp(S) * h_init.unsqueeze(2)

    x = x.reshape(b, Lp, d, n)[:, :L].to(in_dtype)
    new_state = h.to(in_dtype)

    y = torch.einsum('bldn,bln->bld', x, C)
    return y + u * D, new_state


class MambaBlock(nn.Module):
    def __init__(self,
                 model_input_dims,
                 model_internal_dim,
                 conv_kernel_size,
                 delta_t_rank,
                 model_states,
                 conv_use_bias=True,
                 dense_use_bias=True,
                 scan_chunk_size=32,
                 dA_clamp=8.0,
                 dt_min=1e-4,
                 dt_max=1e-1,
                 dt_init_floor=1e-6,
                 dt_scale=1.0):
        super(MambaBlock, self).__init__()
        self.scan_chunk_size = scan_chunk_size
        self.dA_clamp = dA_clamp
        self.model_input_dims = model_input_dims
        self.model_internal_dim = model_internal_dim
        self.conv_kernel_size = conv_kernel_size
        self.delta_t_rank = delta_t_rank
        self.model_states = model_states

        self.in_projection = nn.Linear(model_input_dims, model_internal_dim * 2, bias=False)

        # depthwise causal conv: pad left by k-1 then trim the tail
        self.conv1d = nn.Conv1d(in_channels=model_internal_dim,
                                out_channels=model_internal_dim,
                                kernel_size=conv_kernel_size,
                                groups=model_internal_dim,
                                padding=conv_kernel_size - 1,
                                bias=conv_use_bias)

        self.x_projection = nn.Linear(model_internal_dim,
                                      delta_t_rank + model_states * 2, bias=False)
        self.delta_t_projection = nn.Linear(delta_t_rank, model_internal_dim, bias=True)

        A = torch.arange(1, model_states + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(model_internal_dim, 1)  # [d_in, n]
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(model_internal_dim))

        self.out_projection = nn.Linear(model_internal_dim, model_input_dims,
                                        bias=dense_use_bias)

        self._init_dt(model_internal_dim, delta_t_rank,
                      dt_min, dt_max, dt_init_floor, dt_scale)

        self.state = None  # [batch, d_in, n], allocated lazily

    def _init_dt(self, d_inner, dt_rank, dt_min, dt_max, dt_init_floor, dt_scale):
        dt_init_std = dt_rank ** -0.5 * dt_scale
        with torch.no_grad():
            self.delta_t_projection.weight.uniform_(-dt_init_std, dt_init_std)

            dt = torch.exp(
                torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # inverse of softplus: x = dt + log(1 - exp(-dt))
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.delta_t_projection.bias.copy_(inv_dt)

        self.delta_t_projection.bias._no_reinit = True

    def effective_time_constants(self, sample_rate=None):
        with torch.no_grad():
            dt = F.softplus(self.delta_t_projection.bias)      # [d_inner]
            A = torch.exp(self.A_log)                          # [d_inner, n], >0
            decay = dt.unsqueeze(-1) * A                       # |dA| per step
            half_life = math.log(2.0) / decay.clamp(min=1e-12)
        return half_life / sample_rate if sample_rate else half_life

    def reset_state(self):
        self.state = None

    def detach_state(self):
        if self.state is not None:
            self.state = self.state.clone().detach()

    def ssm(self, x):
        A = -torch.exp(self.A_log)          # [d_in, n]
        D = self.D
        x_dbl = self.x_projection(x)        # [b, l, dt_rank + 2n]
        delta, B, C = torch.split(
            x_dbl,
            [self.delta_t_rank, self.model_states, self.model_states],
            dim=-1)

        delta = F.softplus(self.delta_t_projection(delta))

        last_state = self.state
        if (last_state is not None) and (last_state.shape[0] != x.shape[0]):
            # batch size changed
            last_state = None

        y, new_state = selective_scan(x, delta, A, B, C, D,
                                      last_state=last_state,
                                      chunk_size=self.scan_chunk_size,
                                      dA_clamp=self.dA_clamp)
        self.state = new_state
        return y

    def forward(self, x):
        # x: [batch, length, model_input_dims]
        L = x.shape[1]
        x_and_res = self.in_projection(x)
        x, res = torch.split(x_and_res,
                             [self.model_internal_dim, self.model_internal_dim],
                             dim=-1)

        x = x.transpose(1, 2)               # [b, d_in, l]
        x = self.conv1d(x)[..., :L]         # causal trim
        x = x.transpose(1, 2)               # [b, l, d_in]

        x = F.silu(x)                       # swish
        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_projection(y)


class MambaModel(nn.Module):
    """Selective-state-space compressor model.

    Args:
        nparams (int): number of conditioning parameters.
        window (int): samples of input context per timestep (64 in the paper).
        n_fft (int): rfft length used for the spectral features (255 ->
            128 bins, matching the TF dataset).
        model_states (int): SSM state dimension N.
        stateful (bool): carry SSM / GRU state across chunks.
    """

    def __init__(self,
                 nparams=1,
                 window=64,
                 n_fft=255,
                 feat_dims=2,
                 model_states=8,
                 projection_expand_factor=2,
                 model_input_dims=2,
                 conv_kernel_size=4,
                 scan_chunk_size=32,
                 dA_clamp=8.0,
                 dt_min=1e-4,
                 dt_max=1e-1,
                 **kwargs
              ):
        super(MambaModel, self).__init__()
        self.nparams = nparams
        self.window = window
        self.n_fft = n_fft
        self.nbins = n_fft // 2 + 1
        self.feat_dims = feat_dims

        self.model_internal_dim = int(projection_expand_factor * model_input_dims)
        self.delta_t_rank = math.ceil(model_input_dims / 2)

        self.in_dense = nn.Linear(window, model_input_dims)

        self.mamba1 = MambaBlock(model_states=model_states,
                               model_internal_dim=self.model_internal_dim,
                               model_input_dims=model_input_dims,
                               conv_kernel_size=conv_kernel_size,
                               delta_t_rank=self.delta_t_rank,
                               scan_chunk_size=scan_chunk_size,
                               dA_clamp=dA_clamp,
                               dt_min=dt_min, dt_max=dt_max)
        self.dense1 = nn.Linear(model_input_dims, model_input_dims)

        self.feat_proj = nn.Linear(self.nbins, feat_dims)

        self.film = FiLM(in_size=model_input_dims, cond_dim=nparams + feat_dims)
        self.tfilm = TemporalFiLM(in_size=model_input_dims, cond_dim=feat_dims,
                                  )

        self.mamba2 = MambaBlock(model_states=model_states,
                                 model_internal_dim=self.model_internal_dim,
                                 model_input_dims=model_input_dims,
                                 conv_kernel_size=conv_kernel_size,
                                 delta_t_rank=self.delta_t_rank,
                                 scan_chunk_size=scan_chunk_size,
                                 dA_clamp=dA_clamp,
                                 dt_min=dt_min, dt_max=dt_max)

        self.dense2 = nn.Linear(model_input_dims, model_input_dims)
        self.out_dense = nn.Linear(model_input_dims, 1)

    def effective_time_constants(self, sample_rate=None):

        return {"mamba1": self.mamba1.effective_time_constants(sample_rate),
                "mamba2": self.mamba2.effective_time_constants(sample_rate)}

    def reset_states(self):
        self.mamba1.reset_state()
        self.mamba2.reset_state()
        self.tfilm.reset_state()

    def detach_states(self):
        self.mamba1.detach_state()
        self.mamba2.detach_state()
        self.tfilm.detach_state()

    def get_input_length(self, target_length: int):
        return

    def compute_features(self, x_win):
        """|rfft| of each window."""
        spec = torch.fft.rfft(x_win, n=self.n_fft, dim=-1)
        return spec.abs()

    def forward(self, x_win, p=None, feats=None):
        # x_win : [batch, length, window]
        # p     : [batch, nparams] (broadcast over length) or [batch, length, nparams]
        # feats : optional precomputed [batch, length, nbins]
        B, L, _ = x_win.shape

        x = self.in_dense(x_win)
        x = self.mamba1(x)
        x = F.gelu(self.dense1(x))

        if feats is None:
            feats = self.compute_features(x_win)
        feats = self.feat_proj(feats)                    # [B, L, feat_dims]

        if self.nparams > 0 and p is not None:
            if p.dim() == 2:
                p = p.unsqueeze(1).expand(B, L, p.shape[-1])
            elif p.dim() == 3 and p.shape[1] == 1:
                p = p.expand(B, L, p.shape[-1])
            cond = torch.cat([p, feats], dim=-1)
        else:
            cond = feats

        x = self.film(x, cond)
        x = self.tfilm(x, feats)

        x = self.mamba2(x)
        x = F.gelu(self.dense2(x))
        x = self.out_dense(x)

        # predict a gain applied to the most recent input sample
        return x_win[..., -1:] * x