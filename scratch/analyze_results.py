import csv
import numpy as np

with open('results/evaluation_metrics.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total = len(rows)
psnr_gains = [float(r['PSNR_gain_dB']) for r in rows]
ssim_gains = [float(r['SSIM_gain']) for r in rows]
sam_reds = [float(r['SAM_reduction_degrees']) for r in rows]
ndvi_mae_reds = [float(r['NDVI_MAE_reduction']) for r in rows]
ndvi_r_gains = [float(r['NDVI_Pearson_gain']) for r in rows]

print(f"Total Patches: {total}")
print(f"PSNR:     {sum(x > 0 for x in psnr_gains)}/{total} improved ({sum(x > 0 for x in psnr_gains)/total*100:.1f}%), Mean Gain: {np.mean(psnr_gains):+.2f} dB")
print(f"SSIM:     {sum(x > 0 for x in ssim_gains)}/{total} improved ({sum(x > 0 for x in ssim_gains)/total*100:.1f}%), Mean Gain: {np.mean(ssim_gains):+.4f}")
print(f"SAM:      {sum(x > 0 for x in sam_reds)}/{total} improved ({sum(x > 0 for x in sam_reds)/total*100:.1f}%), Mean Error Drop: {np.mean(sam_reds):+.2f} deg")
print(f"NDVI MAE: {sum(x > 0 for x in ndvi_mae_reds)}/{total} improved ({sum(x > 0 for x in ndvi_mae_reds)/total*100:.1f}%), Mean Drop: {np.mean(ndvi_mae_reds):+.4f}")
print(f"NDVI r:   {sum(x > 0 for x in ndvi_r_gains)}/{total} improved ({sum(x > 0 for x in ndvi_r_gains)/total*100:.1f}%), Mean Gain: {np.mean(ndvi_r_gains):+.4f}")

# Top SSIM drops
sorted_ssim = sorted(rows, key=lambda r: float(r['SSIM_gain']))
print("\n--- Top 3 SSIM Drops ---")
for r in sorted_ssim[:3]:
    print(f"{r['image_name']}: SSIM {r['SSIM_cloudy']} -> {r['SSIM']} (Gain: {r['SSIM_gain']}), PSNR: {r['PSNR_cloudy_dB']} -> {r['PSNR_dB']}")

# Top SSIM gains
sorted_ssim_gain = sorted(rows, key=lambda r: float(r['SSIM_gain']), reverse=True)
print("\n--- Top 3 SSIM Gains ---")
for r in sorted_ssim_gain[:3]:
    print(f"{r['image_name']}: SSIM {r['SSIM_cloudy']} -> {r['SSIM']} (Gain: {r['SSIM_gain']}), PSNR: {r['PSNR_cloudy_dB']} -> {r['PSNR_dB']}")

# Top NDVI r drops
sorted_ndvi = sorted(rows, key=lambda r: float(r['NDVI_Pearson_gain']))
print("\n--- Top 3 NDVI Pearson r Drops ---")
for r in sorted_ndvi[:3]:
    print(f"{r['image_name']}: NDVI r {r['NDVI_Pearson_cloudy']} -> {r['NDVI_Pearson_Correlation']} (Gain: {r['NDVI_Pearson_gain']})")

# Top NDVI r gains
sorted_ndvi_gain = sorted(rows, key=lambda r: float(r['NDVI_Pearson_gain']), reverse=True)
print("\n--- Top 3 NDVI Pearson r Gains ---")
for r in sorted_ndvi_gain[:3]:
    print(f"{r['image_name']}: NDVI r {r['NDVI_Pearson_cloudy']} -> {r['NDVI_Pearson_Correlation']} (Gain: {r['NDVI_Pearson_gain']})")
