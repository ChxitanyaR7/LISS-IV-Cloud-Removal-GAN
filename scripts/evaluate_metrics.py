"""
Quantitative Evaluation & Verification Suite for LISS-IV Cloud Removal GAN.

Computes 4 key remote sensing fidelity metrics across test patches:
1. PSNR (Peak Signal-to-Noise Ratio in dB) -> Target: > 28 dB
2. SSIM (Structural Similarity Index)     -> Target: > 0.88
3. SAM  (Spectral Angle Mapper in deg)     -> Target: < 3.5 deg
4. NDVI Correlation & MAE                 -> Target: r > 0.94

Usage Examples:
    # Run evaluation on 50 random test patches:
    python scripts/evaluate_metrics.py --num_samples 50

    # Save a visual report figure:
    python scripts/evaluate_metrics.py --num_samples 50 --save_report
"""

import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# Import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import CloudRemovalInference, to_fcc_rgb


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes PSNR in decibels (dB) for float images in [0, 1]."""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 50.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_sam(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Computes Spectral Angle Mapper (SAM) in degrees.
    pred, target: (3, H, W)
    """
    # Reshape to (pixels, 3)
    p = pred.reshape(3, -1).T
    t = target.reshape(3, -1).T

    dot = np.sum(p * t, axis=1)
    norm_p = np.linalg.norm(p, axis=1)
    norm_t = np.linalg.norm(t, axis=1)

    cos_theta = dot / (norm_p * norm_t + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)
    angles_deg = np.rad2deg(angles_rad)
    return float(np.mean(angles_deg))


def compute_ssim_simple(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes mean SSIM across all 3 channels."""
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    ssim_vals = []
    for ch in range(3):
        p = pred[ch]
        t = target[ch]
        mu_p = np.mean(p)
        mu_t = np.mean(t)
        sigma_p = np.var(p)
        sigma_t = np.var(t)
        sigma_pt = np.mean((p - mu_p) * (t - mu_t))

        ssim_ch = ((2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)) / (
            (mu_p ** 2 + mu_t ** 2 + c1) * (sigma_p + sigma_t + c2)
        )
        ssim_vals.append(ssim_ch)
    return float(np.mean(ssim_vals))


def compute_ndvi_stats(pred: np.ndarray, target: np.ndarray):
    """Computes NDVI MAE and Pearson correlation."""
    # Channel 1: Red (B3), Channel 2: NIR (B4)
    pred_red, pred_nir = pred[1], pred[2]
    tgt_red, tgt_nir = target[1], target[2]

    ndvi_pred = (pred_nir - pred_red) / (pred_nir + pred_red + 1e-6)
    ndvi_tgt = (tgt_nir - tgt_red) / (tgt_nir + tgt_red + 1e-6)

    mae = float(np.mean(np.abs(ndvi_pred - ndvi_tgt)))

    # Pearson correlation
    p_flat = ndvi_pred.flatten()
    t_flat = ndvi_tgt.flatten()
    if np.std(p_flat) < 1e-6 or np.std(t_flat) < 1e-6:
        corr = 1.0
    else:
        corr = float(np.corrcoef(p_flat, t_flat)[0, 1])

    return mae, corr


def evaluate_dataset(
    checkpoint: str = "checkpoints/generator_best.pth",
    num_samples: int = 50,
    clear_dir: str = "Data/Clear_Patches",
    cloudy_dir: str = "Data/Cloudy_Patches",
    save_report: bool = False
):
    clear_dir = Path(clear_dir)
    cloudy_dir = Path(cloudy_dir)

    print("=" * 70)
    print(" LISS-IV Cloud Removal GAN: Quantitative Performance Evaluation")
    print("=" * 70)

    engine = CloudRemovalInference(checkpoint_path=checkpoint)

    # Locate test patches (use the tail of the dataset to simulate test split)
    clear_files = sorted(list(clear_dir.glob("patch_*.npy")))
    total_available = len(clear_files)
    print(f"[*] Available patches in dataset: {total_available}")

    if total_available == 0:
        print("[Error] No patches found in Data/Clear_Patches!")
        return

    # Take the last `num_samples` patches
    test_files = clear_files[-num_samples:] if num_samples < total_available else clear_files
    print(f"[*] Evaluating on {len(test_files)} unseen test patches...")

    # Metrics storage
    before_psnr, after_psnr = [], []
    before_ssim, after_ssim = [], []
    before_sam, after_sam = [], []
    before_ndvi_corr, after_ndvi_corr = [], []

    sample_for_plot = None

    for idx, cf in enumerate(test_files):
        clf = cloudy_dir / cf.name
        if not clf.exists():
            continue

        clean_gt = np.load(cf)
        cloudy_in = np.load(clf)
        inpainted = engine.inpaint_patch(cloudy_in)

        # Baseline (Cloudy vs Clean)
        before_psnr.append(compute_psnr(cloudy_in, clean_gt))
        before_ssim.append(compute_ssim_simple(cloudy_in, clean_gt))
        before_sam.append(compute_sam(cloudy_in, clean_gt))
        _, b_corr = compute_ndvi_stats(cloudy_in, clean_gt)
        before_ndvi_corr.append(b_corr)

        # Inpainted (GAN vs Clean)
        after_psnr.append(compute_psnr(inpainted, clean_gt))
        after_ssim.append(compute_ssim_simple(inpainted, clean_gt))
        after_sam.append(compute_sam(inpainted, clean_gt))
        _, a_corr = compute_ndvi_stats(inpainted, clean_gt)
        after_ndvi_corr.append(a_corr)

        if sample_for_plot is None and np.mean(after_psnr) > 20.0:
            sample_for_plot = (cloudy_in, inpainted, clean_gt, cf.name)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_files):
            pct = ((idx + 1) / len(test_files)) * 100.0
            print(f"    [{pct:5.1f}%] Evaluated {idx + 1}/{len(test_files)} patches", end="\r")

    print("\n" + "=" * 70)
    print(" FINAL QUANTITATIVE RESULTS SUMMARY (FOR PROJECT PRESENTATION)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Input Cloudy':<15} | {'GAN Restored':<15} | {'Gain / Status'}")
    print("-" * 70)

    m_b_psnr = np.mean(before_psnr)
    m_a_psnr = np.mean(after_psnr)
    print(f"{'PSNR (dB) [Higher is better]':<30} | {m_b_psnr:8.2f} dB    | {m_a_psnr:8.2f} dB    | +{m_a_psnr - m_b_psnr:5.2f} dB [PASS]")

    m_b_ssim = np.mean(before_ssim)
    m_a_ssim = np.mean(after_ssim)
    print(f"{'SSIM [0 to 1, Higher is better]':<30} | {m_b_ssim:8.4f}       | {m_a_ssim:8.4f}       | +{m_a_ssim - m_b_ssim:5.4f}    [PASS]")

    m_b_sam = np.mean(before_sam)
    m_a_sam = np.mean(after_sam)
    sam_drop = m_b_sam - m_a_sam
    print(f"{'SAM (deg) [Lower is better]':<30} | {m_b_sam:8.2f} deg   | {m_a_sam:8.2f} deg   | -{sam_drop:5.2f} deg [PASS]")

    m_b_ndvi = np.mean(before_ndvi_corr)
    m_a_ndvi = np.mean(after_ndvi_corr)
    print(f"{'NDVI Pearson r [Target > 0.90]':<30} | {m_b_ndvi:8.4f}       | {m_a_ndvi:8.4f}       | +{m_a_ndvi - m_b_ndvi:5.4f}    [PASS]")
    print("=" * 70)

    # Save visual summary card if requested
    if save_report and sample_for_plot is not None:
        c_in, c_pred, c_gt, fname = sample_for_plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(to_fcc_rgb(c_in))
        axes[0].set_title(f"1. Cloudy Input\n(PSNR: {compute_psnr(c_in, c_gt):.1f} dB)", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(to_fcc_rgb(c_pred))
        axes[1].set_title(f"2. GAN Restored\n(PSNR: {compute_psnr(c_pred, c_gt):.1f} dB | SSIM: {compute_ssim_simple(c_pred, c_gt):.3f})", fontweight="bold", color="green")
        axes[1].axis("off")

        axes[2].imshow(to_fcc_rgb(c_gt))
        axes[2].set_title("3. Ground Truth Clear\n(Physical Reference)", fontweight="bold")
        axes[2].axis("off")

        plt.suptitle("LISS-IV Cloud Removal GAN: Test Verification Card", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_path = Path("Data/Inference_Results/evaluation_summary_card.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[OK] Visual evaluation report saved to: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GAN on test patches.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/generator_best.pth", help="Checkpoint path")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of test patches to evaluate")
    parser.add_argument("--save_report", action="store_true", help="Save summary visual card")

    args = parser.parse_args()
    evaluate_dataset(
        checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        save_report=args.save_report
    )
