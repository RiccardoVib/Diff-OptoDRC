# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_time_last(*tensors):
    """Validate the [B, C, L] / [B, L] convention.
    """
    out = []
    for x in tensors:
        if x.dim() == 3 and x.shape[-1] == 1 and x.shape[1] != 1:
            raise ValueError(
                f"expected [B, C, L] with time last, got {tuple(x.shape)} "
                "-- this looks like [B, L, 1]; transpose(1, 2) it."
            )
        if x.dim() > 3:
            raise ValueError(f"expected [B, C, L] or [B, L], got {tuple(x.shape)}")
        out.append(x)
    return out[0] if len(out) == 1 else tuple(out)

class MultiScaleTemporalDynamicLoss(nn.Module):
    def __init__(self, frame_lengths=[256, 512, 1024], hop_length=256):
        """
        Multi-scale temporal dynamic loss for capturing dynamics at different time scales
        """
        super(MultiScaleTemporalDynamicLoss, self).__init__()
        self.frame_lengths = frame_lengths
        self.hop_length = hop_length

    def compute_dynamics_at_scale(self, x, frame_length):

        frames = x.unfold(dimension=-1, size=frame_length, step=self.hop_length)

        # RMS energy per frame
        rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-8)

        # First-order derivative (dynamics)
        dynamics = torch.diff(rms, dim=-1)

        return dynamics

    def forward(self, predictions, targets):
        total_loss = 0.0

        predictions, targets = _check_time_last(predictions, targets)

        # Compute loss at multiple time scales
        for frame_length in self.frame_lengths:
            dynamics_pred = self.compute_dynamics_at_scale(predictions, frame_length)
            dynamics_target = self.compute_dynamics_at_scale(targets, frame_length)

            total_loss += F.l1_loss(dynamics_pred, dynamics_target)

        return total_loss / len(self.frame_lengths)

class DCLoss(nn.Module):
    """
    DC Offset Loss for audio processing neural networks.
    Penalizes DC bias in the prediction error or output signal.
    """

    def __init__(self, mode='error', reduction='mean'):
        """
        Args:
            mode: 'error' to penalize DC in (target - prediction),
                  'output' to penalize DC in prediction directly
            reduction: 'mean' or 'sum' for batch reduction
        """
        super(DCLoss, self).__init__()
        self.mode = mode
        self.reduction = reduction

    def forward(self, predictions, targets=None):
        """
        Args:
            prediction: Model output tensor of shape (batch, time) or (batch, channels, time)
            target: Target tensor (required if mode='error')

        Returns:
            DC loss scalar
        """
        if len(predictions.shape) == 3:
            predictions = predictions.squeeze(-1)
            targets = targets.squeeze(-1)

        # Compute mean along time dimension
        dc_component = torch.mean(targets - predictions, dim=-1)
        # Square the DC component
        dc_loss = dc_component ** 2

        # Reduction over batch (and channels if present)
        return dc_loss.mean()




class SpectralFluxLoss(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512, win_length=None):
        super(SpectralFluxLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.register_buffer('window', torch.hann_window(self.win_length))

    def compute_spectral_flux(self, x):
        # Compute STFT
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )

        # Compute magnitude spectrogram
        mag = torch.abs(stft)  # [batch, freq_bins, time_frames]

        # Compute spectral flux (difference between consecutive frames)
        flux = torch.diff(mag, dim=-1)  # difference along time axis

        return flux

    def forward(self, pred, target):
    
        if len(pred.shape) == 3:
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)
            
        # Compute spectral flux for both signals
        flux_pred = self.compute_spectral_flux(pred)
        flux_target = self.compute_spectral_flux(target)

        # Compute loss
        loss = F.l1_loss(flux_pred, flux_target)

        return loss

class CustomLoss(nn.Module):
    """
    Error-to-Signal Ratio Loss
    Normalized squared error between prediction and target
    """

    def __init__(self, frame_lengths=[256, 512, 1024], hop_length=256):
        super().__init__()

        self.multiDynamicLoss = MultiScaleTemporalDynamicLoss(frame_lengths=frame_lengths, hop_length=hop_length)
        self.dc = DCLoss()

    def forward(self, targets, predictions):
        """
        Args:
            targets: target coefficients List[batch, channels, time] or [batch, time]
            predictions: predicted coefficients List[batch, channels, time] or [batch, time]
        Returns:
            loss value
        """

        if len(predictions.shape) == 3:
            predictions = predictions.squeeze(-1)
            targets = targets.squeeze(-1)

        mse_loss = F.mse_loss(targets, predictions)
        multi_dynamic_loss = self.multiDynamicLoss(targets, predictions)
        dc_loss = self.dc(targets, predictions)

        return mse_loss + multi_dynamic_loss + dc_loss

class MelSTFTLoss(nn.Module):
    """
    Mel-scaled STFT Loss
    """

    def __init__(self, sample_rate=44100, fft_size=2048, hop_size=512,
                 win_length=2048, n_mels=128, f_min=0.0, f_max=None,
                 window="hann", w_sc=1.0, w_log_mag=1.0, epsilon=1e-8):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.epsilon = epsilon

        # Register window buffer
        self.register_buffer('window', self._get_window(window, win_length))

        # Create mel filterbank
        self.register_buffer('mel_basis', self._create_mel_filterbank())

    def _get_window(self, window_type, win_length):
        if window_type == "hann":
            return torch.hann_window(win_length)
        elif window_type == "hamming":
            return torch.hamming_window(win_length)
        else:
            raise ValueError(f"Unknown window type: {window_type}")

    def _create_mel_filterbank(self):
        """Create mel filterbank matrix"""
        n_fft = self.fft_size
        n_freqs = n_fft // 2 + 1

        # Convert to mel scale
        mel_min = 2595.0 * torch.log10(torch.tensor(1.0 + self.f_min / 700.0))
        mel_max = 2595.0 * torch.log10(torch.tensor(1.0 + self.f_max / 700.0))

        # Create mel points
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        # Convert to FFT bin indices
        bin_points = torch.floor((n_fft + 1) * hz_points / self.sample_rate).long()

        # Create filterbank
        filterbank = torch.zeros(self.n_mels, n_freqs)
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising slope
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)

            # Falling slope
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def _mel_spectrogram(self, x):
        """Compute mel spectrogram"""
        # x: [batch, channels, time] or [batch, time]
        if x.dim() == 3:
            batch, channels, time = x.shape
            x = x.reshape(batch * channels, time)

        # Compute STFT
        stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            normalized=False
        )
        magnitude = torch.abs(stft)  # [batch*channels, freq, time]

        # Apply mel filterbank
        mel_spec = torch.matmul(self.mel_basis, magnitude)
        return mel_spec

    def forward(self, pred, target):
        """
        Args:
            pred: predicted signal [batch, channels, time] or [batch, time]
            target: target signal [batch, channels, time] or [batch, time]
        """
        pred_mel = self._mel_spectrogram(pred)
        target_mel = self._mel_spectrogram(target)

        # Spectral convergence
        sc_loss = torch.norm(target_mel - pred_mel, p="fro") / (torch.norm(target_mel, p="fro") + self.epsilon)

        # Log magnitude loss
        log_mag_loss = F.l1_loss(torch.log(target_mel + self.epsilon), torch.log(pred_mel + self.epsilon))

        total_loss = self.w_sc * sc_loss + self.w_log_mag * log_mag_loss
        return total_loss
