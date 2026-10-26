import csv
from pathlib import Path
import numpy as np

csv_path = Path("results/evaluation_metrics.csv")
assert csv_path.exists(), f"{csv_path} does not exist!"

with open(csv_path, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows in CSV: {len(rows)}")

# Columns to parse as float
float_cols = [
    "PSNR_dB", "SSIM", "SAM_degrees", "NDVI_MAE", "NDVI_Pearson_Correlation",
    "valid_pixel_count", "cloud_pixel_count",
    "PSNR_cloudy_dB", "PSNR_gain_dB",
    "SSIM_cloudy", "SSIM_gain",
    "SAM_cloudy_degrees", "SAM_reduction_degrees",
    "NDVI_MAE_cloudy", "NDVI_MAE_reduction",
    "NDVI_Pearson_cloudy", "NDVI_Pearson_gain",
    "PSNR_cloud_region_dB", "SAM_cloud_region_degrees"
]

data = {col: [] for col in float_cols}
image_names = []

nan_counts = {col: 0 for col in float_cols}
inf_counts = {col: 0 for col in float_cols}

for r in rows:
    image_names.append(r["image_name"])
    for col in float_cols:
        val_str = r[col].strip()
        if val_str == "" or val_str.lower() == "nan":
            nan_counts[col] += 1
            val = np.nan
        else:
            val = float(val_str)
            if np.isnan(val):
                nan_counts[col] += 1
            elif np.isinf(val):
                inf_counts[col] += 1
        data[col].append(val)

# Convert to numpy arrays
for col in float_cols:
    data[col] = np.array(data[col], dtype=np.float64)

print("\n" + "=" * 80)
print("SANITY CHECKS: NaN / INF / CLIPPING / VALUE CHECKS")
print("=" * 80)
print(f"Any NaN values: {any(c > 0 for c in nan_counts.values())} -> {nan_counts}")
print(f"Any Inf values: {any(c > 0 for c in inf_counts.values())} -> {inf_counts}")

# Check value bounds
print(f"SSIM Restored in [0, 1]: {np.all((data['SSIM'] >= 0.0) & (data['SSIM'] <= 1.0))}")
print(f"SSIM Cloudy in [0, 1]  : {np.all((data['SSIM_cloudy'] >= 0.0) & (data['SSIM_cloudy'] <= 1.0))}")
print(f"SAM Restored >= 0 deg  : {np.all(data['SAM_degrees'] >= 0.0)}")
print(f"SAM Cloudy >= 0 deg    : {np.all(data['SAM_cloudy_degrees'] >= 0.0)}")
print(f"NDVI Pearson r in [-1, 1]: {np.all((data['NDVI_Pearson_Correlation'] >= -1.0) & (data['NDVI_Pearson_Correlation'] <= 1.0))}")
print(f"NDVI Pearson Cloudy in [-1, 1]: {np.all((data['NDVI_Pearson_cloudy'] >= -1.0) & (data['NDVI_Pearson_cloudy'] <= 1.0))}")

# Check difference column consistency:
# PSNR_gain_dB == PSNR_dB - PSNR_cloudy_dB
psnr_diff_mismatch = np.max(np.abs(data["PSNR_gain_dB"] - (data["PSNR_dB"] - data["PSNR_cloudy_dB"])))
ssim_diff_mismatch = np.max(np.abs(data["SSIM_gain"] - (data["SSIM"] - data["SSIM_cloudy"])))
sam_diff_mismatch = np.max(np.abs(data["SAM_reduction_degrees"] - (data["SAM_cloudy_degrees"] - data["SAM_degrees"])))
ndvi_mae_diff_mismatch = np.max(np.abs(data["NDVI_MAE_reduction"] - (data["NDVI_MAE_cloudy"] - data["NDVI_MAE"])))
ndvi_r_diff_mismatch = np.max(np.abs(data["NDVI_Pearson_gain"] - (data["NDVI_Pearson_Correlation"] - data["NDVI_Pearson_cloudy"])))

print(f"\nInternal Column Difference Consistency Check:")
print(f"  Max PSNR diff error    : {psnr_diff_mismatch:.6e}")
print(f"  Max SSIM diff error    : {ssim_diff_mismatch:.6e}")
print(f"  Max SAM diff error     : {sam_diff_mismatch:.6e}")
print(f"  Max NDVI MAE diff error: {ndvi_mae_diff_mismatch:.6e}")
print(f"  Max NDVI r diff error  : {ndvi_r_diff_mismatch:.6e}")

print("\n" + "=" * 80)
print("INDEPENDENT RECOMPUTATION OF METRIC SUMMARIES (N = 100)")
print("=" * 80)

def calc_stats(arr):
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) # sample std
    pop_std = float(np.std(arr, ddof=0))
    median = float(np.median(arr))
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q75 - q25
    return {
        "mean": mean,
        "sample_std": std,
        "pop_std": pop_std,
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": iqr
    }

metrics_def = [
    {
        "name": "PSNR (dB)",
        "cloudy": "PSNR_cloudy_dB",
        "restored": "PSNR_dB",
        "gain": "PSNR_gain_dB",
        "win_rule": lambda g: g > 0.0,
        "win_dir": "Higher is better (Restored > Cloudy)"
    },
    {
        "name": "SSIM (Gaussian sigma=1.5)",
        "cloudy": "SSIM_cloudy",
        "restored": "SSIM",
        "gain": "SSIM_gain",
        "win_rule": lambda g: g > 0.0,
        "win_dir": "Higher is better (Restored > Cloudy)"
    },
    {
        "name": "SAM (degrees)",
        "cloudy": "SAM_cloudy_degrees",
        "restored": "SAM_degrees",
        "gain": "SAM_reduction_degrees",
        "win_rule": lambda g: g > 0.0, # SAM reduction > 0 means restored < cloudy
        "win_dir": "Lower is better (Restored < Cloudy)"
    },
    {
        "name": "NDVI MAE",
        "cloudy": "NDVI_MAE_cloudy",
        "restored": "NDVI_MAE",
        "gain": "NDVI_MAE_reduction",
        "win_rule": lambda g: g > 0.0, # MAE reduction > 0 means restored < cloudy
        "win_dir": "Lower is better (Restored < Cloudy)"
    },
    {
        "name": "NDVI Pearson r",
        "cloudy": "NDVI_Pearson_cloudy",
        "restored": "NDVI_Pearson_Correlation",
        "gain": "NDVI_Pearson_gain",
        "win_rule": lambda g: g > 0.0,
        "win_dir": "Higher is better (Restored > Cloudy)"
    }
]

for m in metrics_def:
    c_stats = calc_stats(data[m["cloudy"]])
    r_stats = calc_stats(data[m["restored"]])
    g_stats = calc_stats(data[m["gain"]])
    wins = int(np.sum(m["win_rule"](data[m["gain"]])))
    win_rate = (wins / len(data[m["gain"]])) * 100.0
    
    print(f"\n>>> Metric: {m['name']} <<<")
    print(f"    Win Direction  : {m['win_dir']}")
    print(f"    Cloudy Baseline: Mean={c_stats['mean']:.4f} +/- {c_stats['pop_std']:.4f} (sample_std={c_stats['sample_std']:.4f}) | Median={c_stats['median']:.4f} [IQR={c_stats['iqr']:.4f}, Q1={c_stats['q25']:.4f}, Q3={c_stats['q75']:.4f}]")
    print(f"    GAN Restored   : Mean={r_stats['mean']:.4f} +/- {r_stats['pop_std']:.4f} (sample_std={r_stats['sample_std']:.4f}) | Median={r_stats['median']:.4f} [IQR={r_stats['iqr']:.4f}, Q1={r_stats['q25']:.4f}, Q3={r_stats['q75']:.4f}]")
    print(f"    Net Gain/Reduct: Mean={g_stats['mean']:.4f} +/- {g_stats['pop_std']:.4f} (sample_std={g_stats['sample_std']:.4f}) | Median={g_stats['median']:.4f} [IQR={g_stats['iqr']:.4f}, Q1={g_stats['q25']:.4f}, Q3={g_stats['q75']:.4f}]")
    print(f"    Win Rate       : {wins} / {len(data[m['gain']])} patches ({win_rate:.1f}%)")

print("\n" + "=" * 80)
print("WORST FIVE PSNR CHANGES (AND IMPACT ANALYSIS)")
print("=" * 80)
# Sort by PSNR_gain_dB ascending
sorted_indices = np.argsort(data["PSNR_gain_dB"])
worst_5_indices = sorted_indices[:5]

total_sum_gain = float(np.sum(data["PSNR_gain_dB"]))
n_total = len(data["PSNR_gain_dB"])
mean_gain_all = total_sum_gain / n_total

worst_5_loss_sum = 0.0

for rank, idx in enumerate(worst_5_indices, 1):
    name = image_names[idx]
    c_psnr = data["PSNR_cloudy_dB"][idx]
    r_psnr = data["PSNR_dB"][idx]
    gain = data["PSNR_gain_dB"][idx]
    c_ssim = data["SSIM_cloudy"][idx]
    r_ssim = data["SSIM"][idx]
    c_sam = data["SAM_cloudy_degrees"][idx]
    r_sam = data["SAM_degrees"][idx]
    c_pix = int(data["cloud_pixel_count"][idx])
    worst_5_loss_sum += gain
    print(f"{rank}. {name}:")
    print(f"   PSNR: {c_psnr:.2f} -> {r_psnr:.2f} dB (Change: {gain:+.2f} dB)")
    print(f"   SSIM: {c_ssim:.4f} -> {r_ssim:.4f} (Change: {r_ssim - c_ssim:+.4f})")
    print(f"   SAM : {c_sam:.2f} deg -> {r_sam:.2f} deg (Change: {r_sam - c_sam:+.2f} deg)")
    print(f"   Cloud Affected Pixels: {c_pix} / 65536 ({c_pix/65536*100:.1f}%)")

remaining_indices = sorted_indices[5:]
mean_gain_remaining = float(np.mean(data["PSNR_gain_dB"][remaining_indices]))
median_gain_all = float(np.median(data["PSNR_gain_dB"]))
median_gain_remaining = float(np.median(data["PSNR_gain_dB"][remaining_indices]))

print(f"\n--- Aggregate Statistics Impact ---")
print(f"Mean PSNR gain (All 100 patches)       : {mean_gain_all:+.2f} dB")
print(f"Mean PSNR gain (Without 5 worst patches): {mean_gain_remaining:+.2f} dB (Net impact of worst 5: {mean_gain_all - mean_gain_remaining:+.2f} dB)")
print(f"Median PSNR gain (All 100 patches)     : {median_gain_all:+.2f} dB")
print(f"Median PSNR gain (Without 5 worst)     : {median_gain_remaining:+.2f} dB")
