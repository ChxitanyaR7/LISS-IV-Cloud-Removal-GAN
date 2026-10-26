import os
import sys
from pathlib import Path
import numpy as np

clear_dir = Path("Data/Clear_Patches")

print("=" * 80)
print("TASK 2: SPATIAL INDEPENDENCE & OVERLAP AUDIT")
print("=" * 80)

# Check consecutive horizontal overlap around boundary
print("\n--- 1. Boundary Patch Horizontal Overlaps ---")
pairs_to_check = [
    (9498, 9499),
    (9499, 9500),
    (9500, 9501),
    (9501, 9502),
    (9598, 9599)
]

for p1_idx, p2_idx in pairs_to_check:
    p1 = np.load(clear_dir / f"patch_{p1_idx:05d}.npy")
    p2 = np.load(clear_dir / f"patch_{p2_idx:05d}.npy")
    
    # Horizontal shift by 128: p1 right half vs p2 left half
    diff_h = np.max(np.abs(p1[:, :, 128:] - p2[:, :, :128]))
    # Vertical shift by 128: p1 bottom half vs p2 top half
    diff_v = np.max(np.abs(p1[:, 128:, :] - p2[:, :128, :]))
    
    print(f"Patches {p1_idx:05d} -> {p2_idx:05d}:")
    print(f"  Horizontal overlap diff (p1 right vs p2 left): {diff_h:.6f} {'[EXACT OVERLAP]' if diff_h == 0 else ''}")
    print(f"  Vertical overlap diff   (p1 bottom vs p2 top): {diff_v:.6f} {'[EXACT OVERLAP]' if diff_v == 0 else ''}")

# Now let's trace consecutive horizontal sequences to reconstruct the raster row structure!
print("\n--- 2. Tracing Raster Rows around 9500 ---")
# Start from patch 9400 to 9650 and check where horizontal continuity breaks (end of a raster row)
row_breaks = []
for idx in range(9400, 9650):
    pA = np.load(clear_dir / f"patch_{idx:05d}.npy")
    pB = np.load(clear_dir / f"patch_{idx+1:05d}.npy")
    diff_h = np.max(np.abs(pA[:, :, 128:] - pB[:, :, :128]))
    if diff_h > 1e-5:
        row_breaks.append(idx)
        print(f"Row break detected at patch_{idx:05d} -> patch_{idx+1:05d} (diff={diff_h:.4f})")

print(f"Total row breaks between 9400 and 9650: {len(row_breaks)}")

# Let's find vertical matches!
# For a patch in row R, its top half matches the bottom half of some patch in row R-1.
print("\n--- 3. Testing Vertical Overlap with Row Above ---")
p9500 = np.load(clear_dir / "patch_09500.npy")
found_vertical_match = False
for train_idx in range(9300, 9500):
    p_train = np.load(clear_dir / f"patch_{train_idx:05d}.npy")
    diff_v = np.max(np.abs(p_train[:, 128:, :] - p9500[:, :128, :]))
    if diff_v == 0.0:
        print(f"[FOUND!] Patch 9500 top half exactly matches Patch {train_idx:05d} bottom half! (Row above)")
        found_vertical_match = True
        step_to_row_above = 9500 - train_idx
        print(f"Raster row width (patches per row): {step_to_row_above}")
        break

if not found_vertical_match:
    print("Direct vertical match not found in 9300-9499 window, searching broader range...")
    for train_idx in range(9000, 9500):
        p_train = np.load(clear_dir / f"patch_{train_idx:05d}.npy")
        diff_v = np.max(np.abs(p_train[:, 128:, :] - p9500[:, :128, :]))
        if diff_v == 0.0:
            print(f"[FOUND!] Patch 9500 top half matches Patch {train_idx:05d} bottom half! (step={9500 - train_idx})")
            found_vertical_match = True
            break
