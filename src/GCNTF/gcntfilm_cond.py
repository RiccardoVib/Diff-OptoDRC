"""
GCNTF-TFiLM: code adapted from https://github.com/mcomunita/gcn-tfilm/
gcn-tfilm -- Comunità et al., ICASSP 2023 (config "GCNTF3")
"""
import torch


class TFiLM(torch.nn.Module):

    def __init__(self, nchannels, block_size=128, stateful=False):
        super(TFiLM, self).__init__()
        self.nchannels = nchannels
        self.block_size = block_size
        self.num_layers = 1
        self.stateful = stateful
        self.hidden_state = None  # (hidden_state, cell_state)

        self.maxpool = torch.nn.MaxPool1d(kernel_size=block_size,
                                          stride=None,
                                          padding=0,
                                          dilation=1,
                                          return_indices=False,
                                          ceil_mode=False)

        self.lstm = torch.nn.LSTM(input_size=nchannels,
                                  hidden_size=nchannels,
                                  num_layers=self.num_layers,
                                  batch_first=False,
                                  bidirectional=False)

    def forward(self, x):
        # x = [batch, channels, length]
        x_shape = x.shape
        nsteps = x_shape[-1] // self.block_size

        if nsteps * self.block_size != x_shape[-1]:
            raise ValueError(
                f"TFiLM: sequence length {x_shape[-1]} is not a multiple of "
                f"block_size {self.block_size}. The wrapper's "
                f"get_input_length() is responsible for rounding the total "
                f"input length up to a multiple of tfilm_block_size."
            )

        # downsample
        x_down = self.maxpool(x)

        # shape for LSTM (length, batch, channels)
        x_down = x_down.permute(2, 0, 1)

        # modulation sequence
        if self.hidden_state is None:  # state was reset
            h0 = torch.zeros(self.num_layers, x.size(0), self.nchannels,
                             device=x.device, dtype=x.dtype)
            c0 = torch.zeros(self.num_layers, x.size(0), self.nchannels,
                             device=x.device, dtype=x.dtype)

        x_norm, hidden_state = self.lstm(x_down, (h0, c0))

        if self.stateful:
            self.hidden_state = hidden_state.detach()

        # put shape back (batch, channels, length)
        x_norm = x_norm.permute(1, 2, 0)

        # reshape input and modulation sequence into blocks
        x_in = torch.reshape(x, shape=(-1, self.nchannels, nsteps, self.block_size))
        x_norm = torch.reshape(x_norm, shape=(-1, self.nchannels, nsteps, 1))

        x_out = x_norm * x_in

        return torch.reshape(x_out, shape=x_shape)

    def detach_state(self):
        if self.hidden_state is None:
            return
        if self.hidden_state.__class__ == tuple:
            self.hidden_state = tuple([h.clone().detach() for h in self.hidden_state])
        else:
            self.hidden_state = self.hidden_state.clone().detach()

    def reset_state(self):
        self.hidden_state = None


class FiLM(torch.nn.Module):
    """Feature-wise Linear Modulation from a global conditioning vector.
    """

    def __init__(self, num_features, cond_dim, use_bn=False):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False) if use_bn else None
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

        torch.nn.init.zeros_(self.adaptor.bias)
        with torch.no_grad():
            self.adaptor.bias[:num_features].fill_(1.0)  # gamma -> 1
            self.adaptor.weight.mul_(0.1)  # gentle initial modulation

    def forward(self, x, cond):
        # x    : [batch, channels, length]
        # cond : [batch, cond_dim]
        gb = self.adaptor(cond)
        g, b = torch.chunk(gb, 2, dim=-1)
        g = g.unsqueeze(-1)  # [batch, channels, 1] -> broadcasts over length
        b = b.unsqueeze(-1)

        if self.bn is not None:
            x = self.bn(x)

        return (x * g) + b


class GatedConv1d(torch.nn.Module):
    """Gated causal conv layer: Conv1D -> gated act -> TFiLM -> FiLM -> mix + residual."""

    def __init__(self,
                 in_ch,
                 out_ch,
                 dilation,
                 kernel_size,
                 tfilm_block_size,
                 cond_size=0,
                 stateful=False,
                 film_batchnorm=False):
        super(GatedConv1d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilation = dilation
        self.stateful = stateful
        self.kernal_size = kernel_size
        self.tfilm_block_size = tfilm_block_size
        self.cond_size = cond_size

        self.conv = torch.nn.Conv1d(in_channels=in_ch,
                                    out_channels=out_ch * 2,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=0,
                                    dilation=dilation)

        self.tfilm = TFiLM(nchannels=out_ch,
                           block_size=tfilm_block_size, stateful=stateful)

        self.film = FiLM(out_ch, cond_size, use_bn=film_batchnorm) if cond_size > 0 else None

        self.mix = torch.nn.Conv1d(in_channels=out_ch,
                                   out_channels=out_ch,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

    def forward(self, x, cond=None):
        residual = x

        # dilated conv
        y = self.conv(x)

        # gated activation
        z = torch.tanh(y[:, :self.out_ch, :]) * \
            torch.sigmoid(y[:, self.out_ch:, :])

        # zero pad on the left side, so that z is the same length as x
        z = torch.cat((torch.zeros(residual.shape[0],
                                   self.out_ch,
                                   residual.shape[2] - z.shape[2],
                                   device=x.device, dtype=x.dtype),
                       z),
                      dim=2)

        # time-varying modulation (from the signal)
        z = self.tfilm(z)

        # parameter modulation (from the control values)
        if self.film is not None and cond is not None:
            z = self.film(z, cond)

        x = self.mix(z) + residual

        return x, z


class GCNBlock(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 nlayers,
                 kernel_size,
                 stateful,
                 dilation_growth,
                 tfilm_block_size,
                 cond_size=0,
                 film_batchnorm=False):
        super(GCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nlayers = nlayers
        self.stateful = stateful
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.tfilm_block_size = tfilm_block_size

        dilations = [dilation_growth ** l for l in range(nlayers)]

        self.layers = torch.nn.ModuleList()
        for d in dilations:
            self.layers.append(GatedConv1d(in_ch=in_ch,
                                           out_ch=out_ch,
                                           dilation=d,
                                           kernel_size=kernel_size,
                                           tfilm_block_size=tfilm_block_size,
                                           cond_size=cond_size,
                                           stateful=stateful,
                                           film_batchnorm=film_batchnorm))
            in_ch = out_ch

    def forward(self, x, cond=None):
        # [batch, channels, length]
        z = torch.empty([x.shape[0], self.nlayers * self.out_ch, x.shape[2]],
                        device=x.device, dtype=x.dtype)

        for n, layer in enumerate(self.layers):
            x, zn = layer(x, cond)
            z[:, n * self.out_ch: (n + 1) * self.out_ch, :] = zn

        return x, z


class GCNTFCond(torch.nn.Module):
    """GCNTF-TFiLM with optional FiLM conditioning on control parameters.
    """

    def __init__(self,
                 nparams=0,
                 nblocks=2,
                 nlayers=9,
                 nchannels=8,
                 kernel_size=3,
                 stateful=False,
                 dilation_growth=2,
                 tfilm_block_size=128,
                 cond_size=32,
                 film_batchnorm=False,
                 ninputs=1,
                 **kwargs):
        super(GCNTFCond, self).__init__()
        self.nparams = nparams
        self.nblocks = nblocks
        self.stateful = stateful
        self.nlayers = nlayers
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.tfilm_block_size = tfilm_block_size
        self.cond_size = cond_size if nparams > 0 else 0
        self.ninputs = ninputs

        if nparams > 0:
            self.condnet = torch.nn.Sequential(
                torch.nn.Linear(nparams, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, cond_size),
                torch.nn.ReLU()
            )
        else:
            self.condnet = None

        self.blocks = torch.nn.ModuleList()
        for b in range(nblocks):
            self.blocks.append(GCNBlock(in_ch=ninputs if b == 0 else nchannels,
                                        out_ch=nchannels,
                                        nlayers=nlayers,
                                        kernel_size=kernel_size,
                                        dilation_growth=dilation_growth,
                                        tfilm_block_size=tfilm_block_size,
                                        cond_size=self.cond_size,
                                        stateful=self.stateful,
                                        film_batchnorm=film_batchnorm))

        # output mixing layer
        self.blocks.append(
            torch.nn.Conv1d(in_channels=nchannels * nlayers * nblocks,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0))

    def forward(self, x, p=None):
        # x.shape = [length, batch, channels]
        x = x.permute(1, 2, 0)  # -> [batch, channels, length]

        # global conditioning embedding, shared by every FiLM layer
        cond = None
        if self.condnet is not None and p is not None:
            if p.dim() > 2:
                p = p.reshape(p.shape[0], -1)  # tolerate [B, 1, nparams]
            cond = self.condnet(p)

        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]],
                        device=x.device, dtype=x.dtype)

        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x, cond)
            z[:,
            n * self.nchannels * self.nlayers:
            (n + 1) * self.nchannels * self.nlayers,
            :] = zn

        # back to [length, batch, channels]
        return self.blocks[-1](z).permute(2, 0, 1)

    def detach_states(self):
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.detach_state()

    def reset_states(self):
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.reset_state()

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1, self.nblocks * self.nlayers):
            dilation = self.dilation_growth ** (n % self.nlayers)
            rf = rf + ((self.kernel_size - 1) * dilation)
        return rf
