
"""
Dataset Inspection & Preview Utility for LISS-IV Cloud Removal GAN.

Visualizes paired Clear vs Synthetic Cloudy patches in standard Remote Sensing
False Color Composite (FCC: Red=NIR Band 4, Green=Red Band 3, Blue=Green Band 2).

Usage:
    python scripts/preview_patches.py
    python scripts/preview_patches.py --num_samples 6 --save
    python scripts/preview_patches.py --random --num_samples 8
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def to_fcc(patch: np.ndarray) -> np.ndarray:
    """
    Converts 3-band LISS-IV patch [B2, B3, B4] of shape (3, H, W)
    into standard False Color Composite (FCC: B4, B3, B2) of shape (H, W, 3).
    """
    b2, b3, b4 = patch[0], patch[1], patch[2]
    fcc = np.stack([b4, b3, b2], axis=-1)
    return np.clip(fcc, 0.0, 1.0)


def preview_dataset(
    clear_dir: Path,
    cloudy_dir: Path,
    num_samples: int = 4,
    random_sample: bool = False,
    save_dir: Path = None,
    show_plot: bool = True
):
    clear_files = sorted(list(clear_dir.glob("*.npy")))
    cloudy_files = sorted(list(cloudy_dir.glob("*.npy")))

    print(f"[*] Found {len(clear_files)} clear patches in: {clear_dir.resolve()}")
    print(f"[*] Found {len(cloudy_files)} cloudy patches in: {cloudy_dir.resolve()}")

    if not clear_files or not cloudy_files:
        print("[!] Dataset is empty. Run `python scripts/generate_patches.py` first.")
        return

    total = min(len(clear_files), len(cloudy_files))
    if random_sample:
        indices = np.random.choice(total, size=min(num_samples, total), replace=False)
    else:
        indices = list(range(min(num_samples, total)))

    # Inspect first patch statistics
    sample_p = np.load(clear_files[indices[0]])
    print("\n" + "=" * 50)
    print(" SAMPLE PATCH STATISTICAL SUMMARY")
    print(f" Array Shape : {sample_p.shape} (Channels: 3, Height: {sample_p.shape[1]}, Width: {sample_p.shape[2]})")
    bands = ["B2 (Green)", "B3 (Red)  ", "B4 (NIR)  "]
    for i, bname in enumerate(bands):
        b = sample_p[i]
        print(f" {bname} -> Min: {b.min():.4f} | Max: {b.max():.4f} | Mean: {b.mean():.4f} | Std: {b.std():.4f}")
    print("=" * 50 + "\n")

    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, patch_idx in enumerate(indices):
        c_patch = np.load(clear_files[patch_idx])
        cl_patch = np.load(cloudy_files[patch_idx])

        clear_fcc = to_fcc(c_patch)
        cloudy_fcc = to_fcc(cl_patch)

        axes[row_idx, 0].imshow(clear_fcc)
        axes[row_idx, 0].set_title(f"Clear Ground Truth [{clear_files[patch_idx].name}]\n(FCC: NIR, Red, Green)")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(cloudy_fcc)
        axes[row_idx, 1].set_title(f"Synthetic Cloudy Input [{cloudy_files[patch_idx].name}]\n(Multi-Octave Simulated)")
        axes[row_idx, 1].axis("off")

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "dataset_preview_composite.png"
        plt.savefig(out_path, dpi=150)
        print(f"[OK] Saved preview composite figure to: {out_path.resolve()}")

    if show_plot:
        plt.show()


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "Data"

    parser = argparse.ArgumentParser(description="Inspect and preview LISS-IV paired patch dataset.")
    parser.add_argument("--clear_dir", type=Path, default=data_dir / "Clear_Patches", help="Path to clear patches.")
    parser.add_argument("--cloudy_dir", type=Path, default=data_dir / "Cloudy_Patches", help="Path to cloudy patches.")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of patch pairs to preview (default: 4).")
    parser.add_argument("--random", action="store_true", help="Select random samples instead of first N.")
    parser.add_argument("--save", action="store_true", help="Save figure to Data/Preview_Images/.")
    parser.add_argument("--save_dir", type=Path, default=data_dir / "Preview_Images", help="Preview save directory.")
    parser.add_argument("--no_show", action="store_true", help="Do not display interactive matplotlib window.")

    args = parser.parse_args()

    preview_dataset(
        clear_dir=args.clear_dir,
        cloudy_dir=args.cloudy_dir,
        num_samples=args.num_samples,
        random_sample=args.random,
        save_dir=args.save_dir if args.save else None,
        show_plot=not args.no_show
    )


if __name__ == "__main__":
    main()
