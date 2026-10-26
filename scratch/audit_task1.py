import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

sys.path.insert(0, str(Path("scripts").resolve()))
from inference import CloudRemovalInference, to_fcc_rgb
from cloud_detector import SpectralCloudDetector

def compute_psnr(pred, target):
    return float(skimage_psnr(target, pred, data_range=1.0))

engine = CloudRemovalInference("checkpoints/generator_best.pth", device="cpu")
patches = ["patch_09525.npy", "patch_09559.npy", "patch_09535.npy"]

clear_dir = Path("Data/Clear_Patches")
cloudy_dir = Path("Data/Cloudy_Patches")
out_dir = Path("results/pre_model_audit")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TASK 1: INFERENCE PIPELINE TRACE & COMPARISON")
print("=" * 80)

for p_name in patches:
    c_path = clear_dir / p_name
    cl_path = cloudy_dir / p_name
    
    clean_gt = np.load(c_path).astype(np.float32)
    cloudy_in = np.load(cl_path).astype(np.float32)
    
    # 1. Cloudy input PSNR
    psnr_cloudy = compute_psnr(cloudy_in, clean_gt)
    
    # 2. Cloud mask from detector
    cloud_mask = engine.detector.predict_mask(cloudy_in)
    
    # 3. Final restored output as currently produced by engine.inpaint_patch
    final_restored = engine.inpaint_patch(cloudy_in, cloud_mask=cloud_mask)
    psnr_final = compute_psnr(final_restored, clean_gt)
    
    # 4. Raw generator output WITHOUT guided heuristic (i.e. pure cloudy input normalized [-1, 1] + mask)
    # This matches the training data format in dataset_loader.py
    cloudy_norm_raw = cloudy_in * 2.0 - 1.0
    mask_tensor = cloud_mask[np.newaxis, :, :].astype(np.float32)
    guide_raw_np = np.concatenate([cloudy_norm_raw, mask_tensor], axis=0)
    guide_raw_t = torch.from_numpy(guide_raw_np).unsqueeze(0).float().to(engine.device)
    with torch.no_grad():
        gen_raw_out = engine.generator(guide_raw_t)
    raw_gen_np = np.clip((gen_raw_out.squeeze(0).cpu().numpy() + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    psnr_raw_gen = compute_psnr(raw_gen_np, clean_gt)
    
    # 5. Generator output WITH guided heuristic but before any potential blending
    # (Notice in inpaint_patch, this is what inpaint_patch returns!)
    # Let's verify whether final_restored == gen_with_guidance
    alpha = np.clip(cloud_mask[np.newaxis, :, :].astype(np.float32) * 1.30, 0.0, 0.95)
    clear_px = cloud_mask < 0.20
    if np.any(clear_px):
        ref_ground = np.array([
            np.median(cloudy_in[0, clear_px]),
            np.median(cloudy_in[1, clear_px]),
            np.median(cloudy_in[2, clear_px])
        ], dtype=np.float32)[:, None, None]
    else:
        ref_ground = np.array([0.20, 0.19, 0.25], dtype=np.float32)[:, None, None]
    guided = (1.0 - alpha) * cloudy_in + alpha * ref_ground
    guided_norm = guided * 2.0 - 1.0
    guide_guided_np = np.concatenate([guided_norm, mask_tensor], axis=0)
    guide_guided_t = torch.from_numpy(guide_guided_np).unsqueeze(0).float().to(engine.device)
    with torch.no_grad():
        gen_guided_out = engine.generator(guide_guided_t)
    guided_gen_np = np.clip((gen_guided_out.squeeze(0).cpu().numpy() + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    psnr_guided_gen = compute_psnr(guided_gen_np, clean_gt)
    
    # 6. Optional postprocessing: Identity blending with mask: (1 - mask) * cloudy + mask * gen
    blended_final = (1.0 - cloud_mask[np.newaxis, :, :]) * cloudy_in + cloud_mask[np.newaxis, :, :] * final_restored
    psnr_blended = compute_psnr(blended_final, clean_gt)
    
    # 7. Ground-truth difference proxy for cloud corruption:
    # Difference between cloudy and clear: diff = |cloudy - clean|
    diff = np.abs(cloudy_in - clean_gt)
    mean_diff = np.mean(diff, axis=0) # (256, 256)
    # Pixels where synthetic cloud actually altered signal (> 0.01)
    actual_corrupted_proxy = mean_diff > 0.01
    corrupted_count = int(np.sum(actual_corrupted_proxy))
    
    # Predicted cloud mask thresholded at 0.15
    pred_cloud_binary = cloud_mask >= 0.15
    pred_count = int(np.sum(pred_cloud_binary))
    
    intersection = np.sum(pred_cloud_binary & actual_corrupted_proxy)
    union = np.sum(pred_cloud_binary | actual_corrupted_proxy)
    iou = float(intersection / (union + 1e-6))
    precision = float(intersection / (pred_count + 1e-6))
    recall = float(intersection / (corrupted_count + 1e-6))
    
    print(f"\n--- Patch: {p_name} ---")
    print(f"Clean GT Mean: {np.mean(clean_gt, axis=(1,2))} Std: {np.std(clean_gt, axis=(1,2))}")
    print(f"Cloudy In Mean: {np.mean(cloudy_in, axis=(1,2))} Std: {np.std(cloudy_in, axis=(1,2))}")
    print(f"Predicted Mask Max: {np.max(cloud_mask):.4f}, Mean: {np.mean(cloud_mask):.4f}, Pixels >= 0.15: {pred_count}")
    print(f"Actual Corrupted (|cloudy - clean| > 0.01) Pixels: {corrupted_count}, Mean diff: {np.mean(mean_diff):.4f}")
    print(f"Mask Overlap with Corrupted Region (Proxy): IoU={iou:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    print(f"PSNR Comparison:")
    print(f"  1. Cloudy Input                  : {psnr_cloudy:.2f} dB")
    print(f"  2. Raw Generator (no guidance)   : {psnr_raw_gen:.2f} dB (Change vs Cloudy: {psnr_raw_gen - psnr_cloudy:+.2f} dB)")
    print(f"  3. Generator with Guided Inpaint : {psnr_guided_gen:.2f} dB (Change vs Cloudy: {psnr_guided_gen - psnr_cloudy:+.2f} dB)")
    print(f"  4. Final Restored (engine output): {psnr_final:.2f} dB (Change vs Cloudy: {psnr_final - psnr_cloudy:+.2f} dB)")
    print(f"  5. Blended Restored (postprocess): {psnr_blended:.2f} dB (Change vs Cloudy: {psnr_blended - psnr_cloudy:+.2f} dB)")
    
    # Visual comparison saving
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(to_fcc_rgb(clean_gt))
    axes[0].set_title(f"1. Clear Ground Truth\n(Reference)", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(to_fcc_rgb(cloudy_in))
    axes[1].set_title(f"2. Cloudy Input\nPSNR: {psnr_cloudy:.2f} dB", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    im_mask = axes[2].imshow(cloud_mask, cmap="Blues_r", vmin=0, vmax=1)
    axes[2].set_title(f"3. Predicted Mask\nMax={np.max(cloud_mask):.2f}, Mean={np.mean(cloud_mask):.2f}", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im_mask, ax=axes[2], fraction=0.046, pad=0.04)
    
    axes[3].imshow(to_fcc_rgb(raw_gen_np))
    axes[3].set_title(f"4. Raw Gen (No Guid)\nPSNR: {psnr_raw_gen:.2f} dB ({psnr_raw_gen - psnr_cloudy:+.2f})", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    
    axes[4].imshow(to_fcc_rgb(final_restored))
    axes[4].set_title(f"5. Final Restored\nPSNR: {psnr_final:.2f} dB ({psnr_final - psnr_cloudy:+.2f})", fontsize=11, fontweight="bold")
    axes[4].axis("off")
    
    plt.tight_layout()
    save_path = out_dir / f"pipeline_trace_{Path(p_name).stem}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved visual comparison to: {save_path.resolve()}")
