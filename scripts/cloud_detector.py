"""
Cloud Detection, Segmentation & Parameter Estimation Module for LISS-IV Imagery.

Provides:
1. Physical Spectral Cloud Detector (rule-based multi-index detector for LISS-IV B2, B3, B4)
2. Deep Learning Cloud Segmentation U-Net (PyTorch model for continuous cloud mask prediction)
3. Cloud Parameter Extraction (Cloud Coverage Ratio, Opacity, Thick vs Thin cloud breakdown)
4. GAN Guidance Tensor generation (4-channel input: [B2, B3, B4, Mask] for downstream Inpainting GAN)

Usage:
    python scripts/cloud_detector.py --preview --num_samples 4
    python scripts/cloud_detector.py --input Data/Cloudy_Patches/patch_00000.npy
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import cv2
import matplotlib.pyplot as plt

# PyTorch import handled gracefully
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object


# ==============================================================================
# 1. Physical Spectral Cloud Detector (Rule-Based Multi-Index)
# ==============================================================================
class SpectralCloudDetector:
    """
    Multispectral physical cloud detection algorithm tailored for LISS-IV sensor bands:
        - Band 2 (Green: 0.52 - 0.59 um)
        - Band 3 (Red:   0.62 - 0.68 um)
        - Band 4 (NIR:   0.77 - 0.86 um)

    Uses a fusion of:
      1. Overall Luminance / Brightness
      2. Spectral Whiteness / Flatness Index (Mie scattering property)
      3. Anti-Vegetation Filter (NDVI suppression to distinguish bright vegetation from cloud)
      4. Morphological spatial consistency
    """

    def __init__(
        self,
        brightness_threshold: float = 0.40,
        whiteness_tolerance: float = 0.22,
        ndvi_max_cloud: float = 0.20,
        blur_kernel_size: int = 7
    ):
        self.b_thresh = brightness_threshold
        self.w_tol = whiteness_tolerance
        self.ndvi_max = ndvi_max_cloud
        self.ksize = blur_kernel_size

    def predict_mask(self, patch: np.ndarray) -> np.ndarray:
        """
        Computes a continuous cloud density mask in range [0, 1].

        Args:
            patch: np.ndarray of shape (3, H, W) in range [0, 1].
                   Channels: [B2 Green, B3 Red, B4 NIR].

        Returns:
            np.ndarray of shape (H, W) with float32 values in [0, 1].
        """
        b2 = patch[0]
        b3 = patch[1]
        b4 = patch[2]

        # 1. Luminance (Mean across bands)
        mean_intensity = (b2 + b3 + b4) / 3.0

        # 2. Spectral Flatness / Whiteness Index
        # Clouds scatter all visible/NIR wavelengths evenly -> low channel variance
        spread = (np.abs(b2 - mean_intensity) + np.abs(b3 - mean_intensity) + np.abs(b4 - mean_intensity)) / 3.0
        whiteness = 1.0 - np.clip(spread / (self.w_tol + 1e-6), 0.0, 1.0)

        # 3. Normalized Difference Vegetation Index (NDVI)
        # Clouds have low NDVI (~ -0.1 to +0.2). Vegetation has high NDVI (> 0.35)
        ndvi = (b4 - b3) / (b4 + b3 + 1e-6)
        ndvi_penalty = np.clip((ndvi - self.ndvi_max) / 0.3, 0.0, 1.0)
        cloud_ndvi_score = 1.0 - ndvi_penalty

        # 4. Composite Probability Map
        brightness_score = np.clip((mean_intensity - self.b_thresh) / (1.0 - self.b_thresh + 1e-6), 0.0, 1.0)
        raw_prob = brightness_score * (0.6 * whiteness + 0.4 * cloud_ndvi_score)

        # Boost dense clouds where all 3 bands are high
        dense_mask = (b2 > 0.65) & (b3 > 0.65) & (b4 > 0.70)
        raw_prob[dense_mask] = np.maximum(raw_prob[dense_mask], 0.85)

        # 5. Spatial Smoothing to remove salt-and-pepper noise
        if self.ksize > 1:
            raw_prob = cv2.GaussianBlur(raw_prob, (self.ksize, self.ksize), 0)

        return np.clip(raw_prob, 0.0, 1.0).astype(np.float32)


# ==============================================================================
# 2. Deep Learning Cloud Segmentation Architecture (U-Net)
# ==============================================================================
if TORCH_AVAILABLE:
    class DoubleConv(nn.Module):
        """(Conv2d -> BatchNorm -> LeakyReLU) * 2"""
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class CloudUNet(nn.Module):
        """
        Lightweight U-Net for LISS-IV Cloud Detection and Continuous Mask Prediction.
        Input:  (B, 3, H, W) [B2, B3, B4] satellite patch.
        Output: (B, 1, H, W) continuous cloud density / opacity map in [0, 1].
        """
        def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 32):
            super().__init__()
            f = base_filters
            # Encoder
            self.inc = DoubleConv(in_channels, f)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))

            # Bottleneck
            self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))

            # Decoder with Skip Connections
            self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
            self.conv_up1 = DoubleConv(f * 16, f * 8)

            self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
            self.conv_up2 = DoubleConv(f * 8, f * 4)

            self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
            self.conv_up3 = DoubleConv(f * 4, f * 2)

            self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            self.conv_up4 = DoubleConv(f * 2, f)

            # Output Head: Sigmoid gives continuous opacity [0, 1]
            self.outc = nn.Sequential(
                nn.Conv2d(f, out_channels, kernel_size=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.bot(x4)

            d1 = self.conv_up1(torch.cat([self.up1(x5), x4], dim=1))
            d2 = self.conv_up2(torch.cat([self.up2(d1), x3], dim=1))
            d3 = self.conv_up3(torch.cat([self.up3(d2), x2], dim=1))
            d4 = self.conv_up4(torch.cat([self.up4(d3), x1], dim=1))

            return self.outc(d4)

    class CombinedBCEDiceLoss(nn.Module):
        """Combines Binary Cross Entropy and Soft Dice Loss for robust cloud mask training."""
        def __init__(self, dice_weight: float = 0.5, smooth: float = 1e-5):
            super().__init__()
            self.dice_weight = dice_weight
            self.smooth = smooth
            self.bce = nn.BCELoss()

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            bce_loss = self.bce(pred, target)
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            intersection = (pred_flat * target_flat).sum()
            dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
            return (1.0 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss
else:
    CloudUNet = None
    CombinedBCEDiceLoss = None


# ==============================================================================
# 3. Cloud Parameter Extraction for GAN Guidance
# ==============================================================================
def extract_cloud_parameters(
    cloudy_patch: np.ndarray,
    cloud_mask: np.ndarray,
    detection_threshold: float = 0.20,
    thick_threshold: float = 0.65
) -> Dict[str, Any]:
    """
    Extracts geometric and photometric cloud parameters from the detection mask.
    These parameters guide the downstream GAN on whether to inpaint, dehaze, or preserve.

    Args:
        cloudy_patch: np.ndarray of shape (3, H, W) in [0, 1].
        cloud_mask: np.ndarray of shape (H, W) in [0, 1].
        detection_threshold: Opacity threshold to consider a pixel cloudy.
        thick_threshold: Opacity threshold above which ground is completely obscured.

    Returns:
        Dictionary of parameters + 4-channel GAN guidance tensor.
    """
    total_pixels = cloud_mask.size
    is_cloudy_pixel = cloud_mask >= detection_threshold
    is_thick_pixel = cloud_mask >= thick_threshold
    is_thin_pixel = is_cloudy_pixel & (~is_thick_pixel)

    cloud_coverage = float(is_cloudy_pixel.mean())
    thick_fraction = float(is_thick_pixel.mean())
    thin_fraction = float(is_thin_pixel.mean())

    if is_cloudy_pixel.any():
        mean_cloud_opacity = float(cloud_mask[is_cloudy_pixel].mean())
    else:
        mean_cloud_opacity = 0.0

    # Binary mask for hard inpainting boundary
    binary_mask = (cloud_mask >= detection_threshold).astype(np.float32)

    # 4-Channel GAN Guidance Tensor: [B2, B3, B4, Cloud_Mask]
    # Shape: (4, H, W). Feeds directly into Inpainting GAN Generator
    mask_channel = np.expand_dims(cloud_mask, axis=0)  # (1, H, W)
    guidance_tensor = np.concatenate([cloudy_patch, mask_channel], axis=0).astype(np.float32)

    # Classification label for conditional GAN
    if cloud_coverage < 0.05:
        category = "clear"
    elif thick_fraction > 0.20:
        category = "heavy_cloud"
    elif thin_fraction > 0.15:
        category = "thin_haze"
    else:
        category = "mixed_cloud"

    return {
        "cloud_coverage_percent": round(cloud_coverage * 100.0, 2),
        "mean_cloud_opacity": round(mean_cloud_opacity, 3),
        "thick_cloud_fraction": round(thick_fraction * 100.0, 2),
        "thin_cloud_fraction": round(thin_fraction * 100.0, 2),
        "category": category,
        "is_cloudy": bool(cloud_coverage >= 0.05),
        "binary_mask": binary_mask,
        "guidance_tensor": guidance_tensor,
    }


# ==============================================================================
# 4. Visualization & Preview
# ==============================================================================
def to_fcc(patch: np.ndarray) -> np.ndarray:
    """Converts 3-band [B2, B3, B4] patch into False Color Composite (B4 NIR, B3 Red, B2 Green)."""
    return np.clip(np.stack([patch[2], patch[1], patch[0]], axis=-1), 0.0, 1.0)


def preview_detection(
    cloudy_patch: np.ndarray,
    cloud_mask: np.ndarray,
    clear_patch: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    show_plot: bool = True
):
    """
    Renders a multi-panel visualization showing:
      1. Cloudy Input (FCC)
      2. Detected Cloud Mask (Heatmap)
      3. Cloud Mask Overlay on Satellite Image
      4. Ground Truth Clear Patch (if available) or Inpainting Target Guidance
    """
    params = extract_cloud_parameters(cloudy_patch, cloud_mask)
    num_cols = 4 if clear_patch is not None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

    # Panel 1: Cloudy Input (FCC)
    cloudy_fcc = to_fcc(cloudy_patch)
    axes[0].imshow(cloudy_fcc)
    axes[0].set_title("1. Cloudy Input\n(FCC: NIR, Red, Green)")
    axes[0].axis("off")

    # Panel 2: Cloud Mask Heatmap
    im_mask = axes[1].imshow(cloud_mask, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"2. Detected Cloud Mask\nCoverage: {params['cloud_coverage_percent']}% | Opacity: {params['mean_cloud_opacity']}"
    )
    axes[1].axis("off")
    plt.colorbar(im_mask, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Masked Overlay
    overlay = cloudy_fcc.copy()
    overlay[cloud_mask >= 0.20, 0] = np.clip(overlay[cloud_mask >= 0.20, 0] + 0.3, 0, 1)  # highlight red
    axes[2].imshow(overlay)
    axes[2].set_title(f"3. Mask Overlay ({params['category']})\nThick: {params['thick_cloud_fraction']}% | Thin: {params['thin_cloud_fraction']}%")
    axes[2].axis("off")

    # Panel 4: Ground Truth (if provided)
    if clear_patch is not None:
        axes[3].imshow(to_fcc(clear_patch))
        axes[3].set_title("4. Ground Truth Clear\n(Target for GAN Training)")
        axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[OK] Saved detection preview figure to: {save_path.resolve()}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "Data"
    cloudy_dir = data_dir / "Cloudy_Patches"
    clear_dir = data_dir / "Clear_Patches"

    parser = argparse.ArgumentParser(description="Cloud detection and parameter extraction for LISS-IV imagery.")
    parser.add_argument("--input", type=Path, default=None, help="Path to a specific cloudy .npy patch.")
    parser.add_argument("--preview", action="store_true", help="Preview detection on sample patches.")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of preview samples.")
    parser.add_argument("--save_dir", type=Path, default=data_dir / "Preview_Images", help="Preview output folder.")
    parser.add_argument("--no_show", action="store_true", help="Do not open matplotlib interactive window.")

    args = parser.parse_args()
    detector = SpectralCloudDetector()

    if args.input:
        if not args.input.exists():
            print(f"[!] Input file not found: {args.input}")
            return
        cloudy_patch = np.load(args.input)
        mask = detector.predict_mask(cloudy_patch)
        params = extract_cloud_parameters(cloudy_patch, mask)
        print(f"--- Analysis for: {args.input.name} ---")
        for k, v in params.items():
            if k not in ["binary_mask", "guidance_tensor"]:
                print(f"  {k:25s}: {v}")
        out_file = args.save_dir / f"detection_{args.input.stem}.png"
        preview_detection(cloudy_patch, mask, save_path=out_file, show_plot=not args.no_show)
        return

    if args.preview or True:
        cloudy_files = sorted(list(cloudy_dir.glob("*.npy")))
        clear_files = sorted(list(clear_dir.glob("*.npy")))

        if not cloudy_files:
            print("[!] No cloudy patches found. Run patch generation first.")
            return

        total = min(len(cloudy_files), len(clear_files))
        num = min(args.num_samples, total)
        indices = np.random.choice(total, size=num, replace=False)

        print(f"[*] Running Cloud Detector on {num} random sample patches...")
        for i, idx in enumerate(indices):
            c_patch = np.load(clear_files[idx])
            cl_patch = np.load(cloudy_files[idx])

            mask = detector.predict_mask(cl_patch)
            params = extract_cloud_parameters(cl_patch, mask)

            print(f"\n[Patch #{idx:05d} - {cloudy_files[idx].name}]")
            print(f"  Coverage: {params['cloud_coverage_percent']}% | Opacity: {params['mean_cloud_opacity']}")
            print(f"  Thick: {params['thick_cloud_fraction']}% | Thin: {params['thin_cloud_fraction']}% | Category: {params['category']}")
            print(f"  Guidance Tensor Shape: {params['guidance_tensor'].shape} (Ready for Inpainting GAN)")

            out_file = args.save_dir / f"cloud_detection_sample_{i:02d}.png"
            preview_detection(cl_patch, mask, clear_patch=c_patch, save_path=out_file, show_plot=not args.no_show)


if __name__ == "__main__":
    main()
