# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato


import os
import torch
import h5py
from torch.utils.data import Dataset, Sampler
from scipy.signal import butter, filtfilt
import torch.nn.functional as F

def log_normalize(v, v_min, v_max):
    """Maps v in [v_min, v_max] → [0, 1] in log space."""
    return torch.log(v / v_min) / torch.log(torch.tensor(v_max / v_min))

def log_denormalize(z_norm, v_min, v_max):
    return v_min * (v_max / v_min) ** z_norm

def highpass(data, cutoff, sample_rate, order=2):
    """Apply high-pass Butterworth filter."""
    b, a = butter(order, cutoff, btype='highpass', fs=sample_rate)
    filtered = filtfilt(b, a, data)  # filtfilt takes b, a, x
    return filtered

class DataGeneratorHDF5(Dataset):
    """
    Dataset for compressor modeling using HDF5 files.
    Data layout in HDF5: (n_recordings, n_time, channels)

    Args:
        h5path (str): Path to the HDF5 file.
        mini_batch_size (int): Number of target time samples per chunk.
        buffer (int): Number of warm-up / context samples prepended to X and Z.
        hop_size (int | None): Step between consecutive chunk starts.
                               Defaults to mini_batch_size (non-overlapping).
                               Set to a smaller value to get overlapping segments.
        return_mem_y (bool): also return the *target's* history, the same slice
                             on `y` that `mem` takes on `x`. Needed only for
                             teacher forcing a feedback model: the detector is
                             driven by the gain the device applied, which needs
                             y over the warm-up as well as over the chunk.
                             Off by default because it changes the length of the
                             returned tuple and doubles the history read.
    """

    def __init__(self, h5path: str,
                 mini_batch_size: int = 2048,
                 buffer: int = 48000 * 5,
                 hop_size: int | None = None,
                 return_mem_y: bool = False,
                 mem_mode: str = "env",  # "env" | "stft"
                 env_frame_size: int = 480,
                 env_hop_size: int = 480,
                 stft_n_fft: int = 1024,
                 stft_hop: int = 256,
                 stft_win_length: int = 1024,
                 stft_bins: int | None = 128,
                 eps: float = 1e-8
                 ):

        self.h5path = h5path
        self.mini_batch_size = mini_batch_size
        self.buffer = buffer
        self.hop_size = hop_size if hop_size is not None else mini_batch_size
        self.return_mem_y = return_mem_y
        self.file = None
        self.mem_mode = mem_mode
        self.env_frame_size = env_frame_size
        self.env_hop_size = env_hop_size
        self.stft_n_fft = stft_n_fft
        self.stft_hop = stft_hop
        self.stft_win_length = stft_win_length
        self.stft_bins = stft_bins
        self.eps = eps
        self.pad_start = True

        if self.hop_size <= 0:
            raise ValueError(f"hop_size must be > 0, got {self.hop_size}")
        if self.hop_size > mini_batch_size:
            raise ValueError(
                f"hop_size ({self.hop_size}) > mini_batch_size ({mini_batch_size}); "
                "this would leave gaps between segments."
            )

        # Read only metadata at init — no signal data loaded into RAM
        with h5py.File(h5path, 'r') as f:
            self.n_recordings = int(f.attrs['n_recordings'])
            self.n_time = int(f.attrs['n_time'])
            self.cond_dim = int(f.attrs.get('cond_dim', f['z'].shape[-1]))
            self.n_channels = int(f['x'].shape[-1])

        self.chunk_start_offset = 0 if self.pad_start else buffer
        # Truncate to multiple of seqlen for clean chunking
        usable_time = self.n_time - self.chunk_start_offset
        self.n_chunks = max(0, (usable_time - mini_batch_size) // self.hop_size + 1)
        # Absolute time indices where chunks start and end (within the full signal)
        # First valid sample index is context_length-1 (to have enough history)

        self.example_to_indices = {
            rec: [rec * self.n_chunks + chunk for chunk in range(self.n_chunks)]
            for rec in range(self.n_recordings)
        }

        overlap_pct = (1.0 - self.hop_size / mini_batch_size) * 100

        print(f"[DataGeneratorHDF5] {os.path.basename(h5path)}")
        print(
            f"  n_recordings={self.n_recordings}, n_time={self.n_time}, "
            f"mini_batch_size={mini_batch_size}, hop_size={self.hop_size} "
            f"({overlap_pct:.0f}% overlap), "
            f"buffer={buffer}, n_chunks={self.n_chunks}"
        )

    def _open(self):
        """Open HDF5 file lazily — each worker opens its own handle."""
        if self.file is None:
            self.file = h5py.File(self.h5path, 'r')

    def _read_history(self, rec_idx, t0, key="x"):
        """The `buffer` samples of `key` preceding t0, padded at the front where
        they run off the start of the recording.
        """
        start = t0 - self.buffer

        if start >= 0:
            return torch.from_numpy(
                self.file[key][rec_idx, start:t0, :].copy()).float()

        n_real = max(t0, 0)
        n_pad = self.buffer - n_real

        if n_real > 0:
            hist = torch.from_numpy(
                self.file[key][rec_idx, 0:t0, :].copy()).float()
        else:
            hist = torch.zeros(0, self.n_channels)

        pad = torch.zeros(n_pad, self.n_channels)

        return torch.cat([pad, hist], dim=0)

    def _compute_env(self, x_hist):
        # x_hist: [T, 1]
        x = x_hist.squeeze(-1).unsqueeze(0)  # [1, T]
        frames = x.unfold(-1, self.env_frame_size, self.env_hop_size)  # [1, N, F]
        rms = (frames.pow(2).mean(dim=-1) + self.eps).sqrt()
        peak = frames.abs().amax(dim=-1)
        crest = peak / (rms + self.eps)

        mem = torch.stack([
            torch.log(rms + self.eps),
            peak,
            torch.log(crest + self.eps),
        ], dim=-1).squeeze(0)  # [N, 3]
        return mem

    def _compute_stft(self, x_hist):
        # x_hist: [T, 1]
        x = x_hist.squeeze(-1)
        window = torch.hann_window(self.stft_win_length, dtype=x.dtype, device=x.device)
        X = torch.stft(
            x,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_win_length,
            window=window,
            center=False,
            return_complex=True,
        )  # [F, N]
        mag = torch.log(X.abs() + self.eps).transpose(0, 1)  # [N, F]

        if self.stft_bins is not None and mag.shape[-1] != self.stft_bins:
            mag = F.interpolate(
                mag.transpose(0, 1).unsqueeze(0),
                size=self.stft_bins,
                mode="linear",
                align_corners=False
            ).squeeze(0).transpose(0, 1)
        return mag


    def __len__(self):
        return self.n_recordings * self.n_chunks

    def __getitem__(self, idx):
        self._open()

        rec_idx = idx // self.n_chunks
        chunk_idx = idx % self.n_chunks
        reset = (chunk_idx == 0)

        t0 = self.chunk_start_offset + chunk_idx * self.hop_size
        t1 = t0 + self.mini_batch_size

        x_hist = self._read_history(rec_idx, t0)
        x_cur  = torch.from_numpy(self.file["x"][rec_idx, t0:t1, :].copy()).float()
        y      = torch.from_numpy(self.file["y"][rec_idx, t0:t1, :].copy()).float()
        z_seq  = torch.from_numpy(self.file["z"][rec_idx, t0:t1, :].copy()).float()

        z_seq = log_normalize(z_seq + 1.0, v_min=1.0, v_max=2.0)
        z_seq = z_seq[-1]
        if self.mem_mode == "env":
            mem = self._compute_env(x_hist)
        elif self.mem_mode == "stft":
            mem = self._compute_stft(x_hist)
        elif self.mem_mode == "":
            mem = x_hist
        else:
            raise ValueError(f"Unsupported mem_mode: {self.mem_mode}")

        if self.return_mem_y:
            mem_y = self._read_history(rec_idx, t0, key="y")
            return x_cur, y, mem, z_seq, reset, mem_y

        return x_cur, y, mem, z_seq, reset

    def __del__(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass