"""
Automated Patch Generation Pipeline for LISS-IV Cloud Removal GAN.

Unzips raw LISS-IV scene archives, extracts Band 2 (Green), Band 3 (Red), and Band 4 (NIR),
applies percentile-based normalization, filters valid ground patches, applies realistic
synthetic cloud masks, and saves paired patches as .npy files.

Usage:
    python scripts/generate_patches.py
    python scripts/generate_patches.py --raw_dir Data/Raw_Zips --patch_size 256 --stride 128
"""

import os
import sys
import gc
import shutil
import zipfile
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

# Allow very large satellite imagery (e.g., 18000 x 17000 px) in PIL
Image.MAX_IMAGE_PIXELS = None

# Support direct script execution and relative imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_simulator import add_synthetic_clouds


def enhance_band(path: Path, scale: int = 1) -> np.ndarray:
    """
    Reads a single GeoTIFF band, applies 2%-98% linear percentile stretch,
    and normalizes values to [0.0, 1.0]. Zero-valued border pixels are preserved.
    Supports both rasterio and PIL for robust multi-platform execution.
    """
    band = None
    try:
        import rasterio
        from rasterio.enums import Resampling
        with rasterio.open(path) as src:
            band = src.read(
                1,
                out_shape=(src.height // scale, src.width // scale),
                resampling=Resampling.bilinear
            ).astype(np.float32)
    except Exception:
        # Fallback to PIL for systems where rasterio DLLs are restricted
        with Image.open(path) as img:
            if scale > 1:
                new_size = (img.width // scale, img.height // scale)
                img = img.resize(new_size, Image.BILINEAR)
            band = np.array(img, dtype=np.float32)

    valid_mask = band > 0
    if not np.any(valid_mask):
        return band

    p2 = np.percentile(band[valid_mask], 2)
    p98 = np.percentile(band[valid_mask], 98)

    if p98 > p2:
        band = (band - p2) / (p98 - p2)

    return np.clip(band, 0.0, 1.0).astype(np.float32)


def locate_bands(search_dir: Path) -> Dict[str, Path]:
    """Finds B2, B3, and B4 GeoTIFF files within an unzipped scene folder."""
    tif_files = list(search_dir.rglob("*.tif")) + list(search_dir.rglob("*.tiff"))
    band_paths: Dict[str, Path] = {}

    for tif in tif_files:
        name = tif.name.upper()
        if "BAND2" in name or "_B2" in name or "BAND_2" in name:
            band_paths["B2"] = tif
        elif "BAND3" in name or "_B3" in name or "BAND_3" in name:
            band_paths["B3"] = tif
        elif "BAND4" in name or "_B4" in name or "BAND_4" in name:
            band_paths["B4"] = tif

    return band_paths


def process_scenes(
    raw_zips_dir: Path,
    temp_extract_dir: Path,
    clear_dir: Path,
    cloudy_dir: Path,
    patch_size: int = 256,
    stride: int = 128,
    valid_threshold: float = 0.90,
    min_std: float = 0.05,
    min_mean: float = 0.15,
    scale: int = 1,
    max_patches: Optional[int] = None
) -> Tuple[int, int]:
    """
    Processes all ZIP files in `raw_zips_dir` and saves paired patches.

    Returns:
        Tuple of (total_valid_pairs_saved, total_patches_skipped)
    """
    clear_dir.mkdir(parents=True, exist_ok=True)
    cloudy_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(list(raw_zips_dir.glob("*.zip")))
    if not zip_files:
        zip_files = sorted(list(raw_zips_dir.rglob("*.zip")))

    if not zip_files:
        print(f"[!] No .zip archives found in: {raw_zips_dir.resolve()}")
        return 0, 0

    print(f"[*] Found {len(zip_files)} scene ZIP archive(s) to process.")

    # Determine next starting file index so we don't overwrite existing patches
    existing_clears = list(clear_dir.glob("patch_*.npy"))
    total_valid = len(existing_clears)
    total_skipped = 0

    print(f"[*] Existing patches detected: {total_valid}. Starting next patch index at: patch_{total_valid:05d}.npy")

    for idx, zip_file in enumerate(zip_files, 1):
        print(f"\n--- [{idx}/{len(zip_files)}] Unpacking: {zip_file.name} ---")

        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        temp_extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        band_paths = locate_bands(temp_extract_dir)
        if not ("B2" in band_paths and "B3" in band_paths and "B4" in band_paths):
            print(f" [!] Missing required bands in {zip_file.name}. Found: {list(band_paths.keys())}. Skipping scene.")
            continue

        print(" [+] Loading and normalizing bands (B2: Green, B3: Red, B4: NIR)...")
        b2 = enhance_band(band_paths["B2"], scale=scale)
        b3 = enhance_band(band_paths["B3"], scale=scale)
        b4 = enhance_band(band_paths["B4"], scale=scale)

        h, w = b2.shape
        scene_valid = 0
        scene_skipped = 0

        print(f" [+] Scene Dimensions: {w} x {h}. Extracting {patch_size}x{patch_size} patches (stride: {stride})...")

        for row in range(0, h - patch_size + 1, stride):
            for col in range(0, w - patch_size + 1, stride):
                p_b2 = b2[row : row + patch_size, col : col + patch_size]
                p_b3 = b3[row : row + patch_size, col : col + patch_size]
                p_b4 = b4[row : row + patch_size, col : col + patch_size]

                # Border check: all 3 bands must have non-zero signal
                valid_mask = (p_b2 > 0) & (p_b3 > 0) & (p_b4 > 0)
                if valid_mask.mean() < valid_threshold:
                    scene_skipped += 1
                    continue

                clear_patch = np.stack([p_b2, p_b3, p_b4], axis=0).astype(np.float32)

                # Quality filter: skip uniform or overly dark patches
                if clear_patch[:, valid_mask].std() < min_std or clear_patch[:, valid_mask].mean() < min_mean:
                    scene_skipped += 1
                    continue

                # Add multi-octave synthetic cloud simulation
                cloudy_patch = add_synthetic_clouds(clear_patch)

                # Sanity check
                if not (np.isfinite(clear_patch).all() and np.isfinite(cloudy_patch).all()):
                    scene_skipped += 1
                    continue

                filename = f"patch_{total_valid:05d}.npy"
                np.save(clear_dir / filename, clear_patch)
                np.save(cloudy_dir / filename, cloudy_patch)

                total_valid += 1
                scene_valid += 1

                if max_patches and total_valid >= max_patches:
                    print(f"[*] Reached target limit of {max_patches} patches.")
                    break

            if max_patches and total_valid >= max_patches:
                break

        print(f" [OK] Scene Complete: Saved {scene_valid} valid patches (Filtered out {scene_skipped} invalid).")
        total_skipped += scene_skipped

        del b2, b3, b4
        gc.collect()

    # Clean temporary working files
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    print("\n" + "=" * 60)
    print(f" Pipeline Finished! Total patches in dataset: {total_valid}")
    print(f" Total patches skipped due to quality filters: {total_skipped}")
    print("=" * 60)
    return total_valid, total_skipped


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "Data"

    parser = argparse.ArgumentParser(description="Extract paired LISS-IV clear/cloudy patches for GAN training.")
    parser.add_argument("--raw_dir", type=Path, default=data_dir / "Raw_Zips", help="Path to raw scene ZIP files.")
    parser.add_argument("--clear_dir", type=Path, default=data_dir / "Clear_Patches", help="Destination for clear .npy patches.")
    parser.add_argument("--cloudy_dir", type=Path, default=data_dir / "Cloudy_Patches", help="Destination for cloudy .npy patches.")
    parser.add_argument("--temp_dir", type=Path, default=data_dir / "temp_extract", help="Temporary unzipping folder.")
    parser.add_argument("--patch_size", type=int, default=256, help="Square patch dimension (default: 256).")
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride (default: 128).")
    parser.add_argument("--valid_threshold", type=float, default=0.90, help="Min fraction of non-zero valid pixels.")
    parser.add_argument("--min_std", type=float, default=0.05, help="Min standard deviation to avoid flat patches.")
    parser.add_argument("--min_mean", type=float, default=0.15, help="Min mean brightness threshold.")
    parser.add_argument("--max_patches", type=int, default=None, help="Stop after saving this many total patches.")

    args = parser.parse_args()

    process_scenes(
        raw_zips_dir=args.raw_dir,
        temp_extract_dir=args.temp_dir,
        clear_dir=args.clear_dir,
        cloudy_dir=args.cloudy_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        valid_threshold=args.valid_threshold,
        min_std=args.min_std,
        min_mean=args.min_mean,
        max_patches=args.max_patches
    )


if __name__ == "__main__":
    main()
