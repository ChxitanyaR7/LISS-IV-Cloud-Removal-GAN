"""
Dataset Packaging Utility for Google Colab / Google Drive Upload.

Packages paired LISS-IV patches (Clear and Cloudy) into a single compressed .zip file.
Uploading one .zip file to Google Drive is 100x faster than uploading tens of
thousands of individual loose .npy files.

Usage Examples:
    # Package a quick subset of 5,000 paired patches for fast training:
    python scripts/package_dataset.py --num_pairs 5000

    # Package all available patches (58,000+):
    python scripts/package_dataset.py --all

    # Specify custom output path:
    python scripts/package_dataset.py --num_pairs 10000 --output Data/liss_iv_10k.zip
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path


def package_dataset(
    clear_dir: Path,
    cloudy_dir: Path,
    output_zip: Path,
    num_pairs: int = 0,
    compress: bool = True
) -> None:
    """Packages paired clear and cloudy patches into a clean zip archive."""
    clear_dir = Path(clear_dir)
    cloudy_dir = Path(cloudy_dir)
    output_zip = Path(output_zip)

    if not clear_dir.exists() or not cloudy_dir.exists():
        print(f"[Error] Directory not found: {clear_dir} or {cloudy_dir}")
        sys.exit(1)

    print("=" * 60)
    print(" LISS-IV Dataset Packager for Google Colab Upload")
    print("=" * 60)
    print(f"[*] Scanning {clear_dir} ...")

    clear_files = sorted(list(clear_dir.glob("patch_*.npy")))
    cloudy_files = sorted(list(cloudy_dir.glob("patch_*.npy")))

    clear_set = {f.name for f in clear_files}
    cloudy_set = {f.name for f in cloudy_files}
    matched_names = sorted(list(clear_set.intersection(cloudy_set)))

    total_matched = len(matched_names)
    print(f"[*] Found {total_matched} verified matching (Clear, Cloudy) patch pairs.")

    if total_matched == 0:
        print("[Error] No matching patch pairs found!")
        sys.exit(1)

    if num_pairs > 0 and num_pairs < total_matched:
        selected_names = matched_names[:num_pairs]
        print(f"[*] Packaging selected subset: {num_pairs} pairs")
    else:
        selected_names = matched_names
        print(f"[*] Packaging ALL {total_matched} patch pairs")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED

    total_files = len(selected_names) * 2
    print(f"[*] Target zip archive: {output_zip.resolve()}")
    print(f"[*] Total files to archive: {total_files}")
    print("[*] Archiving files (please wait)...")

    count = 0
    with zipfile.ZipFile(output_zip, "w", compression=compression) as zipf:
        for idx, name in enumerate(selected_names):
            clear_path = clear_dir / name
            cloudy_path = cloudy_dir / name

            # Archive with directory structure: Data/Clear_Patches/patch_*.npy
            zipf.write(clear_path, arcname=f"Data/Clear_Patches/{name}")
            zipf.write(cloudy_path, arcname=f"Data/Cloudy_Patches/{name}")

            count += 2
            if (idx + 1) % 500 == 0 or (idx + 1) == len(selected_names):
                pct = ((idx + 1) / len(selected_names)) * 100.0
                print(f"    [{pct:5.1f}%] Packaged {idx + 1} / {len(selected_names)} pairs ({count} files)")

    file_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print("[OK] Packaging Complete!")
    print(f"     Archive Location : {output_zip.resolve()}")
    print(f"     Archive Size     : {file_size_mb:.2f} MB ({file_size_mb / 1024:.2f} GB)")
    print(f"     Pairs Packaged   : {len(selected_names)}")
    print("=" * 60)
    print("\n[>>] Next Steps for Google Colab:")
    print("1. Upload this zip file to your Google Drive (e.g. into a folder named 'LISS-IV-Dataset').")
    print("2. Open 'notebooks/LISS_IV_Cloud_Removal_Colab.ipynb' in Google Colab.")
    print("3. In Step 2 of the notebook, run:")
    print("      !unzip -q /content/drive/MyDrive/LISS-IV-Dataset/" + output_zip.name + " -d /content/")
    print("4. Begin GPU training!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package paired patches into zip for Colab upload.")
    parser.add_argument("--clear_dir", type=str, default="Data/Clear_Patches", help="Clear patches directory")
    parser.add_argument("--cloudy_dir", type=str, default="Data/Cloudy_Patches", help="Cloudy patches directory")
    parser.add_argument("--output", type=str, default="Data/liss_iv_patches.zip", help="Output zip file path")
    parser.add_argument("--num_pairs", type=int, default=0, help="Number of pairs to package (0 = all)")
    parser.add_argument("--all", action="store_true", help="Package all available pairs")

    args = parser.parse_args()

    num = 0 if args.all else args.num_pairs
    package_dataset(
        clear_dir=Path(args.clear_dir),
        cloudy_dir=Path(args.cloudy_dir),
        output_zip=Path(args.output),
        num_pairs=num
    )
