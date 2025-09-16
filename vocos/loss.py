from typing import List, Tuple

import torch
import torchaudio
from torch import nn
import numpy as np

from vocos.modules import safe_log


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output


def anti_wrapping_function(x):
    # Apply STE to allow gradients to flow through torch.round
    return torch.abs(x - _RoundSTE.apply(x / (2 * np.pi)) * 2 * np.pi)


class PhaseLoss(nn.Module):
    """
    Computes phase spectrum loss, including instantaneous phase, group delay, and phase time derivative losses.
    """

    def __init__(self, n_fft: int):
        super().__init__()
        self.n_fft = n_fft

    def forward(
        self, phase_r: torch.Tensor, phase_g: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            phase_r (Tensor): Ground truth phase spectrum of shape (B, F, T).
            phase_g (Tensor): Predicted phase spectrum of shape (B, F, T).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: IP_loss, GD_loss, PTD_loss
        """
        frames = phase_r.shape[2]
        freq_bins = self.n_fft // 2 + 1

        # Instantaneous Phase Loss
        ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))

        # Group Delay Loss
        GD_matrix = (
            torch.triu(torch.ones(freq_bins, freq_bins), diagonal=1)
            - torch.triu(torch.ones(freq_bins, freq_bins), diagonal=2)
            - torch.eye(freq_bins)
        )
        GD_matrix = GD_matrix.to(phase_g.device)

        GD_r = torch.matmul(phase_r.permute(0, 2, 1), GD_matrix)
        GD_g = torch.matmul(phase_g.permute(0, 2, 1), GD_matrix)
        gd_loss = torch.mean(anti_wrapping_function(GD_r - GD_g))

        # Phase Time Derivative Loss
        PTD_matrix = (
            torch.triu(torch.ones(frames, frames), diagonal=1)
            - torch.triu(torch.ones(frames, frames), diagonal=2)
            - torch.eye(frames)
        )
        PTD_matrix = PTD_matrix.to(phase_g.device)

        PTD_r = torch.matmul(phase_r, PTD_matrix)
        PTD_g = torch.matmul(phase_g, PTD_matrix)
        ptd_loss = torch.mean(anti_wrapping_function(PTD_r - PTD_g))

        return ip_loss, gd_loss, ptd_loss

class MelSpecReconstructionLoss(nn.Module):
    """
    L1 distance between the mel-scaled magnitude spectrograms of the ground truth sample and the generated sample
    """

    def __init__(
        self, sample_rate: int = 24000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 100,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=True, power=1,
        )

    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        mel_hat = safe_log(self.mel_spec(y_hat))
        mel = safe_log(self.mel_spec(y))

        loss = torch.nn.functional.l1_loss(mel, mel_hat)

        return loss


class GeneratorLoss(nn.Module):
    """
    Generator Loss module. Calculates the loss for the generator based on discriminator outputs.
    """

    def forward(self, disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            disc_outputs (List[Tensor]): List of discriminator outputs.

        Returns:
            Tuple[Tensor, List[Tensor]]: Tuple containing the total loss and a list of loss values from
                                         the sub-discriminators
        """
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """
    Discriminator Loss module. Calculates the loss for the discriminator based on real and generated outputs.
    """

    def forward(
        self, disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            disc_real_outputs (List[Tensor]): List of discriminator outputs for real samples.
            disc_generated_outputs (List[Tensor]): List of discriminator outputs for generated samples.

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: A tuple containing the total loss, a list of loss values from
                                                       the sub-discriminators for real outputs, and a list of
                                                       loss values for generated outputs.
        """
        loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)

        return loss, r_losses, g_losses


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss module. Calculates the feature matching loss between feature maps of the sub-discriminators.
    """

    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            fmap_r (List[List[Tensor]]): List of feature maps from real samples.
            fmap_g (List[List[Tensor]]): List of feature maps from generated samples.

        Returns:
            Tensor: The calculated feature matching loss.
        """
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss
