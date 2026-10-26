import os
import sys
from pathlib import Path
import numpy as np

clear_dir = Path("Data/Clear_Patches")

val_indices = list(range(9500, 9600))
# To optimize search, training patches that can possibly overlap are within a few rows above 9500,
# specifically from index 9200 to 9499. Let's load them or search across 9000 to 9499.
train_indices = list(range(9200, 9500))

print(f"Loading {len(train_indices)} training patches (indices {train_indices[0]} to {train_indices[-1]})...")
train_patches = {idx: np.load(clear_dir / f"patch_{idx:05d}.npy") for idx in train_indices}

print(f"Loading {len(val_indices)} validation patches (indices {val_indices[0]} to {val_indices[-1]})...")
val_patches = {idx: np.load(clear_dir / f"patch_{idx:05d}.npy") for idx in val_indices}

# We test 4 types of spatial overlap on the 128-stride raster grid:
# 1. Horizontal: left half of V matches right half of T (overlap = 128 x 256 = 32768 px, 50%)
# 2. Vertical: top half of V matches bottom half of T (overlap = 256 x 128 = 32768 px, 50%)
# 3. Diagonal 1: top-left of V matches bottom-right of T (overlap = 128 x 128 = 16384 px, 25%)
# 4. Diagonal 2: top-right of V matches bottom-left of T (overlap = 128 x 128 = 16384 px, 25%)

overlapping_val_patches = {}
overlapping_pairs = []

for v_idx in val_indices:
    pV = val_patches[v_idx]
    v_overlaps = []
    
    for t_idx in train_indices:
        pT = train_patches[t_idx]
        
        # Check Horizontal: pT[:, :, 128:] == pV[:, :, :128]
        if np.max(np.abs(pT[:, :, 128:] - pV[:, :, :128])) < 1e-5:
            v_overlaps.append((t_idx, "Horizontal (T right = V left)", 32768, 0.50))
            overlapping_pairs.append((t_idx, v_idx, "Horizontal", 32768))
            
        # Check Vertical: pT[:, 128:, :] == pV[:, :128, :]
        if np.max(np.abs(pT[:, 128:, :] - pV[:, :128, :])) < 1e-5:
            v_overlaps.append((t_idx, "Vertical (T bottom = V top)", 32768, 0.50))
            overlapping_pairs.append((t_idx, v_idx, "Vertical", 32768))
            
        # Check Diagonal 1: pT[:, 128:, 128:] == pV[:, :128, :128]
        if np.max(np.abs(pT[:, 128:, 128:] - pV[:, :128, :128])) < 1e-5:
            v_overlaps.append((t_idx, "Diagonal 1 (T bot-right = V top-left)", 16384, 0.25))
            overlapping_pairs.append((t_idx, v_idx, "Diagonal 1", 16384))
            
        # Check Diagonal 2: pT[:, 128:, :128] == pV[:, :128, 128:]
        if np.max(np.abs(pT[:, 128:, :128] - pV[:, :128, 128:])) < 1e-5:
            v_overlaps.append((t_idx, "Diagonal 2 (T bot-left = V top-right)", 16384, 0.25))
            overlapping_pairs.append((t_idx, v_idx, "Diagonal 2", 16384))
            
    if v_overlaps:
        overlapping_val_patches[v_idx] = v_overlaps

total_val_count = len(val_indices)
overlapping_val_count = len(overlapping_val_patches)
pct_overlapping = (overlapping_val_count / total_val_count) * 100.0

print("\n" + "=" * 80)
print("SPATIAL OVERLAP ANALYSIS RESULTS:")
print("=" * 80)
print(f"Total Validation Patches Evaluated (9500-9599): {total_val_count}")
print(f"Validation Patches with Direct Training Overlap : {overlapping_val_count} ({pct_overlapping:.1f}%)")
print(f"Total Overlapping (Train, Val) Pairs Found     : {len(overlapping_pairs)}")

print("\nSample Overlapping Validation Patches:")
for v_idx in list(overlapping_val_patches.keys())[:10]:
    overlaps = overlapping_val_patches[v_idx]
    details = ", ".join([f"T_{t[0]} ({t[1]}, {t[3]*100:.0f}%)" for t in overlaps])
    print(f"  Val Patch {v_idx:05d}: overlaps with {len(overlaps)} train patches -> {details}")

print(f"\nChecking the 3 specific audit failure patches:")
for p_idx in [9525, 9535, 9559]:
    if p_idx in overlapping_val_patches:
        overlaps = overlapping_val_patches[p_idx]
        details = ", ".join([f"T_{t[0]} ({t[1]}, {t[3]*100:.0f}%)" for t in overlaps])
        print(f"  Patch {p_idx:05d}: DIRECT OVERLAP with {details}")
    else:
        print(f"  Patch {p_idx:05d}: No direct 1-step overlap in tested window")

# Also check patch 9599
if 9599 in overlapping_val_patches:
    overlaps = overlapping_val_patches[9599]
    details = ", ".join([f"T_{t[0]} ({t[1]}, {t[3]*100:.0f}%)" for t in overlaps])
    print(f"  Patch 09599: DIRECT OVERLAP with {details}")
else:
    print(f"  Patch 09599: No direct 1-step overlap in tested window")
