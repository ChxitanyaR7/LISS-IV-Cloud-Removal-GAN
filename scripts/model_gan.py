"""
Adapted Pix2Pix Conditional GAN (cGAN) Architecture for LISS-IV Cloud Removal.

Based on Isola et al. (CVPR 2017) adapted for 4-channel multispectral remote sensing:
1. Generator: 4-Channel U-Net 256 with Skip Connections
   - Input:  (B, 4, 256, 256) [B2 Green, B3 Red, B4 NIR, Cloud_Mask] in [-1, 1]
   - Output: (B, 3, 256, 256) [B2 Green, B3 Red, B4 NIR] restored ground in [-1, 1]
2. Discriminator: 70x70 Markovian PatchGAN
   - Input:  (B, 7, 256, 256) [Guidance_Tensor (4ch) + Candidate_Ground (3ch)]
   - Output: (B, 1, 30, 30) Real vs Fake decision matrix
3. Remote Sensing Loss Functions:
   - Spectral Angle Mapper (SAM) Loss: Preserves radiometric vector angle across bands.
   - NDVI Consistency Loss: Preserves Near-Infrared agricultural vegetation index.

Usage:
    python scripts/model_gan.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ==============================================================================
# 1. Building Blocks for U-Net & PatchGAN
# ==============================================================================
class DownBlock(nn.Module):
    """Conv2d -> [InstanceNorm2d] -> LeakyReLU (Downsampling block)."""
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """ConvTranspose2d -> InstanceNorm2d -> [Dropout] -> ReLU (Upsampling block)."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        # Skip connection concatenation along channel dimension
        return torch.cat([x, skip], dim=1)


# ==============================================================================
# 2. 4-Channel U-Net 256 Generator
# ==============================================================================
class UNetGenerator(nn.Module):
    """
    Pix2Pix U-Net 256 Generator adapted for 4-channel input:
    [B2 Green, B3 Red, B4 NIR, Cloud_Mask].
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, base_filters: int = 64):
        super().__init__()
        f = base_filters

        # Encoder (Downsampling path: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)
        self.d1 = DownBlock(in_channels, f, normalize=False)  # 256 -> 128
        self.d2 = DownBlock(f, f * 2)                         # 128 -> 64
        self.d3 = DownBlock(f * 2, f * 4)                     # 64 -> 32
        self.d4 = DownBlock(f * 4, f * 8)                     # 32 -> 16
        self.d5 = DownBlock(f * 8, f * 8)                     # 16 -> 8
        self.d6 = DownBlock(f * 8, f * 8)                     # 8 -> 4
        self.d7 = DownBlock(f * 8, f * 8)                     # 4 -> 2
        self.d8 = DownBlock(f * 8, f * 8, normalize=False)    # 2 -> 1 (Bottleneck)

        # Decoder (Upsampling path with skip connections)
        self.u1 = UpBlock(f * 8, f * 8, dropout=0.5)          # 1 -> 2
        self.u2 = UpBlock(f * 16, f * 8, dropout=0.5)         # 2 -> 4
        self.u3 = UpBlock(f * 16, f * 8, dropout=0.5)         # 4 -> 8
        self.u4 = UpBlock(f * 16, f * 8)                      # 8 -> 16
        self.u5 = UpBlock(f * 16, f * 4)                      # 16 -> 32
        self.u6 = UpBlock(f * 8, f * 2)                       # 32 -> 64
        self.u7 = UpBlock(f * 4, f)                           # 64 -> 128

        # Final reconstruction layer (128 -> 256)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1] matching normalized satellite tensors
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass with skip activations saved
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)
        e5 = self.d5(e4)
        e6 = self.d6(e5)
        e7 = self.d7(e6)
        e8 = self.d8(e7)

        # Decoder forward pass with skip connections
        u1 = self.u1(e8, e7)
        u2 = self.u2(u1, e6)
        u3 = self.u3(u2, e5)
        u4 = self.u4(u3, e4)
        u5 = self.u5(u4, e3)
        u6 = self.u6(u5, e2)
        u7 = self.u7(u6, e1)

        return self.final(u7)


# ==============================================================================
# 3. 70x70 Markovian PatchGAN Discriminator
# ==============================================================================
class PatchGANDiscriminator(nn.Module):
    """
    Pix2Pix 70x70 PatchGAN Discriminator.
    Evaluates overlapping local 70x70 image patches for realistic sensor texture.
    Input: (B, 4 + 3 = 7, 256, 256) [Conditioning Guidance (4ch) + Candidate (3ch)].
    Output: (B, 1, 30, 30) Real vs Fake decision matrix.
    """
    def __init__(self, in_channels: int = 7, base_filters: int = 64):
        super().__init__()
        f = base_filters

        self.net = nn.Sequential(
            # C64: 256 -> 128
            nn.Conv2d(in_channels, f, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # C128: 128 -> 64
            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # C256: 64 -> 32
            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # C512: 32 -> 31 (stride 1)
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(f * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Output 1-channel prediction map: 31 -> 30
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, conditioning: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        # Concatenate conditioning guidance and candidate patch along channels
        x = torch.cat([conditioning, candidate], dim=1)
        return self.net(x)


# ==============================================================================
# 4. Remote Sensing Spectral Loss Functions
# ==============================================================================
class SpectralAngleMapperLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss for Multispectral Satellite Imagery.
    Computes the average angle between true and generated pixel spectral vectors:
        SAM(y, y_hat) = arccos( (y . y_hat) / (|y|_2 * |y_hat|_2 + eps) )
    Penalizes spectral distortion and inter-band color shifts in Near-Infrared.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Scale tensors from [-1, 1] back to [0, 1] for physical angle calculation
        p = (pred + 1.0) * 0.5
        t = (target + 1.0) * 0.5

        # Dot product across spectral channels (dim=1)
        dot_product = torch.sum(p * t, dim=1)

        # Vector magnitudes
        norm_p = torch.sqrt(torch.sum(p ** 2, dim=1) + self.eps)
        norm_t = torch.sqrt(torch.sum(t ** 2, dim=1) + self.eps)

        cos_theta = dot_product / (norm_p * norm_t)
        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)

        angle = torch.acos(cos_theta)  # In radians
        return torch.mean(angle)


class NDVIConsistencyLoss(nn.Module):
    """
    Normalized Difference Vegetation Index (NDVI) Loss.
    Ensures that agricultural vegetation health is preserved after cloud removal:
        NDVI = (NIR - Red) / (NIR + Red + eps)
        L_NDVI = L1_loss(NDVI(pred), NDVI(target))
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Channel 1 is Red (B3), Channel 2 is NIR (B4)
        p = (pred + 1.0) * 0.5
        t = (target + 1.0) * 0.5

        pred_red, pred_nir = p[:, 1:2, :, :], p[:, 2:3, :, :]
        tgt_red, tgt_nir = t[:, 1:2, :, :], t[:, 2:3, :, :]

        ndvi_pred = (pred_nir - pred_red) / (pred_nir + pred_red + self.eps)
        ndvi_tgt = (tgt_nir - tgt_red) / (tgt_nir + tgt_red + self.eps)

        return F.l1_loss(ndvi_pred, ndvi_tgt)


# ==============================================================================
# 5. Verification Test Routine
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Testing Adapted Pix2Pix cGAN Architecture for LISS-IV")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # 1. Test Generator
    gen = UNetGenerator(in_channels=4, out_channels=3).to(device)
    dummy_input = torch.randn(2, 4, 256, 256, device=device)
    gen_output = gen(dummy_input)

    print(f"[OK] Generator Forward Pass Successful!")
    print(f"     Input Shape : {dummy_input.shape} (Batch: 2, Channels: 4 [B2, B3, B4, Mask], 256x256)")
    print(f"     Output Shape: {gen_output.shape} (Batch: 2, Channels: 3 [B2, B3, B4], 256x256)")
    print(f"     Output Range: [{gen_output.min().item():.2f}, {gen_output.max().item():.2f}] (Tanh: [-1, 1])")

    # 2. Test Discriminator
    disc = PatchGANDiscriminator(in_channels=7).to(device)
    dummy_target = torch.randn(2, 3, 256, 256, device=device)
    disc_output = disc(dummy_input, dummy_target)

    print(f"\n[OK] PatchGAN Discriminator Forward Pass Successful!")
    print(f"     Concatenated Input : 4 + 3 = 7 channels")
    print(f"     Decision Map Shape : {disc_output.shape} (70x70 PatchGAN 30x30 grid)")

    # 3. Test Loss Functions
    sam_loss_fn = SpectralAngleMapperLoss()
    ndvi_loss_fn = NDVIConsistencyLoss()

    sam_val = sam_loss_fn(gen_output, dummy_target)
    ndvi_val = ndvi_loss_fn(gen_output, dummy_target)

    print(f"\n[OK] Remote Sensing Loss Functions Successful!")
    print(f"     Spectral Angle Mapper (SAM) Loss : {sam_val.item():.4f} rad ({sam_val.item() * 57.2958:.2f} deg)")
    print(f"     NDVI Consistency Loss             : {ndvi_val.item():.4f}")

    # Param count
    gen_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    disc_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"\n[OK] Model Parameters:")
    print(f"     Generator Parameter Count    : {gen_params / 1e6:.2f} Million")
    print(f"     Discriminator Parameter Count: {disc_params / 1e6:.2f} Million")
    print("=" * 60)
