"""
Post-Evaluation Results Aggregation & Statistical Verification Script.

Inspects results/evaluation_metrics.csv and recalculates exact:
- Mean and Standard Deviation for all metrics
- Baseline (Cloudy) vs Restored (GAN) metrics
- Net Gains and Reductions
- Win Rates (% of patches improved)
- Best and worst outlier patches
"""

import csv
import numpy as np
from pathlib import Path


def analyze_csv(csv_path: str = "results/evaluation_metrics.csv"):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"[*] Loaded {total} rows from: {path.resolve()}\n")

    # Extract columns
    psnr_rest = np.array([float(r["PSNR_dB"]) for r in rows])
    psnr_cld = np.array([float(r["PSNR_cloudy_dB"]) for r in rows])
    psnr_gain = np.array([float(r["PSNR_gain_dB"]) for r in rows])

    ssim_rest = np.array([float(r["SSIM"]) for r in rows])
    ssim_cld = np.array([float(r["SSIM_cloudy"]) for r in rows])
    ssim_gain = np.array([float(r["SSIM_gain"]) for r in rows])

    sam_rest = np.array([float(r["SAM_degrees"]) for r in rows])
    sam_cld = np.array([float(r["SAM_cloudy_degrees"]) for r in rows])
    sam_red = np.array([float(r["SAM_reduction_degrees"]) for r in rows])

    ndvi_mae_rest = np.array([float(r["NDVI_MAE"]) for r in rows])
    ndvi_mae_cld = np.array([float(r["NDVI_MAE_cloudy"]) for r in rows])
    ndvi_mae_red = np.array([float(r["NDVI_MAE_reduction"]) for r in rows])

    ndvi_r_rest = np.array([float(r["NDVI_Pearson_Correlation"]) for r in rows])
    ndvi_r_cld = np.array([float(r["NDVI_Pearson_cloudy"]) for r in rows])
    ndvi_r_gain = np.array([float(r["NDVI_Pearson_gain"]) for r in rows])

    # Print Verification
    print("=" * 115)
    print(" 1. EXACT RECALCULATED PARAMETRIC METRICS (MEAN +/- STD)")
    print("=" * 115)
    print(f"{'Metric':<25} | {'Cloudy Input (Mean +/- Std)':<25} | {'GAN Restored (Mean +/- Std)':<25} | {'Net Mean Gain':<15} | {'Win Rate'}")
    print("-" * 115)

    def row_str(name, c_arr, r_arr, g_arr, unit=""):
        c_m, c_s = np.mean(c_arr), np.std(c_arr)
        r_m, r_s = np.mean(r_arr), np.std(r_arr)
        g_m = np.mean(g_arr)
        win_count = np.sum(g_arr > 0)
        win_pct = (win_count / total) * 100.0
        return f"{name:<25} | {c_m:8.4f} +/- {c_s:7.4f} {unit:<4} | {r_m:8.4f} +/- {r_s:7.4f} {unit:<4} | {g_m:+10.4f} {unit:<4} | {win_count}/{total} ({win_pct:5.1f}%)"

    print(row_str("PSNR (dB)", psnr_cld, psnr_rest, psnr_gain, "dB"))
    print(row_str("SSIM (Gauss sigma=1.5)", ssim_cld, ssim_rest, ssim_gain))
    print(row_str("SAM (deg)", sam_cld, sam_rest, sam_red, "deg"))
    print(row_str("NDVI MAE", ndvi_mae_cld, ndvi_mae_rest, ndvi_mae_red))
    print(row_str("NDVI Pearson r", ndvi_r_cld, ndvi_r_rest, ndvi_r_gain))
    print("=" * 115)

    print("\n" + "=" * 115)
    print(" 2. ROBUST NON-PARAMETRIC METRICS (MEDIAN [INTERQUARTILE RANGE IQR])")
    print("=" * 115)
    print(f"{'Metric':<25} | {'Cloudy Median [IQR]':<25} | {'Restored Median [IQR]':<25} | {'Median Gain [IQR]'}")
    print("-" * 115)

    def row_med_str(name, c_arr, r_arr, g_arr, unit=""):
        c_med, c_iqr = np.median(c_arr), np.percentile(c_arr, 75) - np.percentile(c_arr, 25)
        r_med, r_iqr = np.median(r_arr), np.percentile(r_arr, 75) - np.percentile(r_arr, 25)
        g_med, g_iqr = np.median(g_arr), np.percentile(g_arr, 75) - np.percentile(g_arr, 25)
        return f"{name:<25} | {c_med:8.4f} [{c_iqr:7.4f}] {unit:<4} | {r_med:8.4f} [{r_iqr:7.4f}] {unit:<4} | {g_med:+10.4f} [{g_iqr:7.4f}] {unit:<4}"

    print(row_med_str("PSNR (dB)", psnr_cld, psnr_rest, psnr_gain, "dB"))
    print(row_med_str("SSIM (Gauss sigma=1.5)", ssim_cld, ssim_rest, ssim_gain))
    print(row_med_str("SAM (deg)", sam_cld, sam_rest, sam_red, "deg"))
    print(row_med_str("NDVI MAE", ndvi_mae_cld, ndvi_mae_rest, ndvi_mae_red))
    print(row_med_str("NDVI Pearson r", ndvi_r_cld, ndvi_r_rest, ndvi_r_gain))
    print("=" * 115)

    # Worst 5 PSNR changes
    print("\n--- WORST 5 PSNR CHANGES (Dominating Variance) ---")
    sorted_psnr = sorted(rows, key=lambda r: float(r["PSNR_gain_dB"]))
    for idx, r in enumerate(sorted_psnr[:5], 1):
        print(f"  #{idx} {r['image_name']}: Cloudy {float(r['PSNR_cloudy_dB']):5.2f} dB -> Restored {float(r['PSNR_dB']):5.2f} dB (Change: {float(r['PSNR_gain_dB']):+6.2f} dB)")


if __name__ == "__main__":
    analyze_csv()
