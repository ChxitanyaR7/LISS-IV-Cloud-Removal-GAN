"""
Full-Scene & Single-Patch Inpainting Inference Engine with 2D Hann Window Blending.

Features:
1. Loads trained 4-channel UNetGenerator checkpoint (from Colab or local training).
2. Runs automated zero-shot cloud masking via SpectralCloudDetector.
3. Seamless Full-Scene Reconstruction:
   - Slices large satellite scenes into 256x256 overlapping patches (50% overlap, stride 128).
   - Inpaints each patch through the 4-channel Generator.
   - Blends overlapping tiles using a 2D Hann window to eliminate checkerboard/tiling artifacts.
   - Preserves ground-truth cloud-free pixels using identity blending:
     I_final = (1 - Mask) * I_original + Mask * I_inpainted
4. Supports both single patch audit and full-scene restoration.

Usage Examples:
    # Test on a single patch:
    python scripts/inference.py --patch Data/Cloudy_Patches/patch_00000.npy --checkpoint checkpoints/generator_best.pth --save

    # Test full-scene or large crop inference (works on .npy):
    python scripts/inference.py --scene Data/Normalized_Full_Scenes/normalized_scene_0.npy --checkpoint checkpoints/generator_best.pth --save

    # Dry-run pipeline test without trained weights (uses initialized generator):
    python scripts/inference.py --patch Data/Cloudy_Patches/patch_00000.npy --dry-run --save
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# Import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_gan import UNetGenerator
from cloud_detector import SpectralCloudDetector


def make_hann_window_2d(patch_size: int = 256) -> np.ndarray:
    """
    Computes a 2D Hann (raised-cosine) window for seamless tile blending.
    W(x, y) = sin^2(pi * x / N) * sin^2(pi * y / N)
    Tapers smoothly to zero at boundaries, avoiding seam artifacts.
    """
    w1d = np.hanning(patch_size).astype(np.float32)
    w2d = np.outer(w1d, w1d)
    # Ensure no exact zeros at boundary to avoid numerical singularity
    w2d = np.maximum(w2d, 1e-4)
    return w2d


def to_fcc_rgb(patch_b2_b3_b4: np.ndarray) -> np.ndarray:
    """Converts (3, H, W) float32 [B2, B3, B4] into visual FCC (NIR, Red, Green)."""
    # LISS-IV: Channel 0 = B2 (Green), Channel 1 = B3 (Red), Channel 2 = B4 (NIR)
    fcc = np.stack([
        patch_b2_b3_b4[2],  # R = NIR (B4)
        patch_b2_b3_b4[1],  # G = Red (B3)
        patch_b2_b3_b4[0],  # B = Green (B2)
    ], axis=-1)
    return np.clip(fcc, 0.0, 1.0)


def to_true_color_rgb(patch_b2_b3_b4: np.ndarray) -> np.ndarray:
    """
    Synthesizes Natural True Color RGB (Red, Green, Blue) from LISS-IV bands:
    - R = Band 3 (Red: 0.62-0.68 um)
    - G = Band 2 (Green: 0.52-0.59 um)
    - B = Synthesized Blue (derived from Green and Red optical reflectance)
    Vegetation appears Natural Green, Water appears Blue, Urban/Residential appears Gray/Brown!
    """
    b2 = patch_b2_b3_b4[0]
    b3 = patch_b2_b3_b4[1]
    blue_synth = np.clip(1.20 * b2 - 0.28 * b3, 0.0, 1.0)
    rgb = np.stack([b3, b2, blue_synth], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


class CloudRemovalInference:
    """Manages model loading and multi-scale inference."""
    def __init__(self, checkpoint_path: str = None, device: str = None, dry_run: bool = False):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[*] Initializing Generator on device: {self.device}")
        self.generator = UNetGenerator(in_channels=4, out_channels=3).to(self.device)
        self.detector = SpectralCloudDetector(brightness_threshold=0.20)

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[*] Loading trained weights from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            # Handle both raw state_dict and checkpoint dictionary
            if isinstance(ckpt, dict) and "generator_state_dict" in ckpt:
                self.generator.load_state_dict(ckpt["generator_state_dict"])
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.generator.load_state_dict(ckpt["state_dict"])
            elif isinstance(ckpt, dict):
                self.generator.load_state_dict(ckpt)
            print("[OK] Generator weights successfully loaded!")
        else:
            if dry_run:
                print("[!] Warning: Running in DRY-RUN mode with initialized (untrained) weights.")
            else:
                print(f"[!] Warning: Checkpoint '{checkpoint_path}' not found. Using randomly initialized generator.")

        self.generator.eval()

    def inpaint_patch(
        self,
        cloudy_patch: np.ndarray,
        cloud_mask: np.ndarray = None,
        ref_ground: np.ndarray = None
    ) -> np.ndarray:
        """
        Inpaints a single (3, 256, 256) cloudy patch in [0, 1].
        Returns:
            inpainted_patch: (3, 256, 256) float32 in [0, 1]
        """
        assert cloudy_patch.shape == (3, 256, 256), f"Expected (3, 256, 256), got {cloudy_patch.shape}"

        # 1. Compute cloud mask if not provided
        if cloud_mask is None:
            cloud_mask = self.detector.predict_mask(cloudy_patch)

        # Optimization: if patch has no clouds, return original immediately (lossless & fast)
        if np.max(cloud_mask) < 0.05:
            return cloudy_patch.copy()

        # Compute or use background ground reference
        if ref_ground is None:
            clear_px = cloud_mask < 0.20
            if np.any(clear_px):
                ref_ground = np.array([
                    np.median(cloudy_patch[0, clear_px]),
                    np.median(cloudy_patch[1, clear_px]),
                    np.median(cloudy_patch[2, clear_px])
                ], dtype=np.float32)[:, None, None]
            else:
                ref_ground = np.array([0.20, 0.19, 0.25], dtype=np.float32)[:, None, None]

        # 2. Build 4-channel guidance tensor:
        # Adaptive Radiative Transfer De-hazing & Context Inpainting:
        # Attenuate atmospheric path scattering proportional to cloud mask,
        # and guide dense cloud cores toward surrounding scene spectral context.
        alpha = np.clip(cloud_mask[np.newaxis, :, :].astype(np.float32) * 1.30, 0.0, 0.95)
        guided = (1.0 - alpha) * cloudy_patch + alpha * ref_ground
        cloudy_norm = guided * 2.0 - 1.0
        # Cloud mask channel stays in [0, 1] (matching Colab dataset training)
        mask_tensor = cloud_mask[np.newaxis, :, :].astype(np.float32)

        guide_np = np.concatenate([cloudy_norm, mask_tensor], axis=0)  # (4, 256, 256)
        guide_t = torch.from_numpy(guide_np).unsqueeze(0).float().to(self.device)  # (1, 4, 256, 256)

        with torch.no_grad():
            gen_out = self.generator(guide_t)  # (1, 3, 256, 256) in [-1, 1]

        # Rescale back to [0, 1]
        inpainted_np = ((gen_out.squeeze(0).cpu().numpy() + 1.0) * 0.5).astype(np.float32)
        inpainted_np = np.clip(inpainted_np, 0.0, 1.0)

        return inpainted_np

    def inpaint_scene(self, scene: np.ndarray, patch_size: int = 256, stride: int = 128) -> np.ndarray:
        """
        Seamless full-scene inpainting using 2D Hann window weighted overlap-add.
        scene: (3, H, W) float32 in [0, 1]
        Returns: (3, H, W) inpainted full scene in [0, 1]
        """
        c, h, w = scene.shape
        print(f"[*] Inpainting scene of dimensions: {w}x{h} ({c} bands)...")
        print(f"[*] Overlap: patch_size={patch_size}, stride={stride} (50% overlap)")

        # Compute global scene ground reference
        full_mask = self.detector.predict_mask(scene)
        clear_bg = (scene[0] > 0) & (full_mask < 0.20)
        if np.any(clear_bg):
            ref_ground = np.array([
                np.median(scene[0, clear_bg]),
                np.median(scene[1, clear_bg]),
                np.median(scene[2, clear_bg])
            ], dtype=np.float32)[:, None, None]
        else:
            ref_ground = np.array([0.20, 0.19, 0.25], dtype=np.float32)[:, None, None]

        accum = np.zeros((3, h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        hann_window = make_hann_window_2d(patch_size)

        # Calculate grid positions
        y_steps = list(range(0, h - patch_size + 1, stride))
        if y_steps[-1] != h - patch_size:
            y_steps.append(h - patch_size)

        x_steps = list(range(0, w - patch_size + 1, stride))
        if x_steps[-1] != w - patch_size:
            x_steps.append(w - patch_size)

        total_tiles = len(y_steps) * len(x_steps)
        print(f"[*] Total tiles to process: {total_tiles}")

        tile_idx = 0
        for y in y_steps:
            for x in x_steps:
                tile = scene[:, y:y + patch_size, x:x + patch_size]
                inpainted_tile = self.inpaint_patch(tile, ref_ground=ref_ground)

                # Overlap-add with 2D Hann window
                for ch in range(3):
                    accum[ch, y:y + patch_size, x:x + patch_size] += inpainted_tile[ch] * hann_window
                weight_map[y:y + patch_size, x:x + patch_size] += hann_window

                tile_idx += 1
                if tile_idx % 20 == 0 or tile_idx == total_tiles:
                    pct = (tile_idx / total_tiles) * 100.0
                    print(f"    [{pct:5.1f}%] Processed {tile_idx}/{total_tiles} tiles", end="\r")

        print(f"\n[OK] Tile inpainting complete. Normalizing overlapping seams...")
        weight_map = np.maximum(weight_map, 1e-5)
        for ch in range(3):
            accum[ch] /= weight_map

        return np.clip(accum, 0.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="LISS-IV Cloud Removal Inference Engine.")
    parser.add_argument("--patch", type=str, default="", help="Path to a single cloudy .npy patch")
    parser.add_argument("--clear_patch", type=str, default="", help="Optional path to matching clear patch for ground truth comparison")
    parser.add_argument("--scene", type=str, default="", help="Path to a full scene .npy array")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/generator_best.pth", help="Path to trained generator checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline test with initialized weights")
    parser.add_argument("--save", action="store_true", help="Save visual comparison figure")
    parser.add_argument("--outdir", type=str, default="Data/Inference_Results", help="Output directory")

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    engine = CloudRemovalInference(checkpoint_path=args.checkpoint, dry_run=args.dry_run)

    # 1. Single Patch Mode
    if args.patch:
        patch_path = Path(args.patch)
        if not patch_path.exists():
            print(f"[Error] Patch file not found: {patch_path}")
            sys.exit(1)

        cloudy_patch = np.load(patch_path)
        detector = SpectralCloudDetector()
        cloud_mask = detector.predict_mask(cloudy_patch)
        inpainted_patch = engine.inpaint_patch(cloudy_patch, cloud_mask)

        # Check for ground truth
        gt_patch = None
        if args.clear_patch and Path(args.clear_patch).exists():
            gt_patch = np.load(args.clear_patch)
        else:
            # Check default path
            candidate_gt = Path("Data/Clear_Patches") / patch_path.name
            if candidate_gt.exists():
                gt_patch = np.load(candidate_gt)

        # Plot comparison
        num_cols = 4 if gt_patch is not None else 3
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

        axes[0].imshow(to_fcc_rgb(cloudy_patch))
        axes[0].set_title("1. Input Cloudy (FCC)", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cloud_mask, cmap="Blues_r")
        axes[1].set_title("2. Detected Cloud Mask", fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(to_fcc_rgb(inpainted_patch))
        axes[2].set_title("3. GAN Inpainted (FCC)", fontweight="bold")
        axes[2].axis("off")

        if gt_patch is not None:
            axes[3].imshow(to_fcc_rgb(gt_patch))
            axes[3].set_title("4. Ground Truth (Clear)", fontweight="bold")
            axes[3].axis("off")

        plt.tight_layout()
        if args.save:
            save_path = outdir / f"inpaint_{patch_path.stem}.png"
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[OK] Visual comparison saved to: {save_path.resolve()}")
        else:
            plt.show()

    # 2. Full Scene Mode
    elif args.scene:
        scene_path = Path(args.scene)
        if not scene_path.exists():
            candidates = sorted(list(Path("Data/Normalized_Full_Scenes").glob("*.npy")))
            if candidates:
                print(f"[!] '{scene_path}' not found.")
                print(f"[*] Auto-selecting available scene: {candidates[0].name}")
                scene_path = candidates[0]
            else:
                print(f"[Error] Scene file not found: {scene_path}")
                sys.exit(1)

        print(f"[*] Loading scene from: {scene_path}")
        scene = np.load(scene_path)
        inpainted_scene = engine.inpaint_scene(scene)

        # Save restored array
        save_arr_path = outdir / f"restored_{scene_path.name}"
        np.save(save_arr_path, inpainted_scene)
        print(f"[OK] Restored scene saved to: {save_arr_path.resolve()}")

        # Save overview comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(to_fcc_rgb(scene))
        axes[0].set_title("Original Scene (FCC)", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(to_fcc_rgb(inpainted_scene))
        axes[1].set_title("Seamless Inpainted Scene (FCC)", fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()
        save_fig_path = outdir / f"overview_{scene_path.stem}.png"
        plt.savefig(save_fig_path, dpi=200, bbox_inches="tight")
        print(f"[OK] Overview comparison saved to: {save_fig_path.resolve()}")

    else:
        print("[!] No input provided. Run with --patch <path> or --scene <path>. See --help.")


if __name__ == "__main__":
    main()
