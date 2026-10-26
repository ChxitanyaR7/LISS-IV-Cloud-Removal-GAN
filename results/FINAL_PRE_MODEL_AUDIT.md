# Final Pre-Model-Change Audit: LISS-IV Cloud Removal GAN

**Date of Execution**: 2026-10-26  
**Status**: Completed & Verified via Executable Python Checks  
**Model Checkpoint**: [generator_best.pth](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/checkpoints/generator_best.pth) (217.68 MB, 4-in / 3-out U-Net)  
**Parent Sensor & Scene**: IRS-R2A LISS-IV (5.0m GSD), Scene `R2F11NOV2023065188011200053SSANSTUC00GTDC`  
**Evaluation Set**: 100 Held-Out Patches ([patch_09500.npy](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/Data/Clear_Patches/patch_09500.npy) to [patch_09599.npy](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/Data/Clear_Patches/patch_09599.npy))  

---

## 1. Actual Inference Pipeline & Confirmed Failure Source

### 1.1 Complete Pipeline Architecture

The evaluation suite executes the following sequence through [scripts/evaluate_metrics.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/evaluate_metrics.py#L328-L330) and [scripts/inference.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/inference.py#L112-L165):

```mermaid
flowchart LR
    A["Cloudy Input (3, 256, 256)"] --> B["Cloud Detection\n(SpectralCloudDetector)"]
    B --> C["Continuous Mask (256, 256)"]
    A --> D["Pre-Inpainting Guidance\n(alpha * ref_ground)"]
    C --> D
    D --> E["4-Channel Input Tensor\n[Guided_Norm, Mask]"]
    E --> F["Generator Forward Pass\n(UNetGenerator)"]
    F --> G["Raw Generator Output\n(Normalized [-1, 1])"]
    G --> H["Rescale to [0, 1] & Clip\n(No Identity Blending)"]
    H --> I["Final Restored Output"]
    I --> J["Evaluation Metrics\n(PSNR, SSIM, SAM, NDVI)"]
```

#### Detailed Stage Breakdown:
1. **Cloudy Input**: Patch loaded from disk in `[0.0, 1.0]`, shape `(3, 256, 256)` where Band 2 = Green, Band 3 = Red, Band 4 = NIR.
2. **Cloud Detection** ([scripts/cloud_detector.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/cloud_detector.py#L38-L100)):
   `SpectralCloudDetector(brightness_threshold=0.20)` evaluates mean luminance, spectral flatness (whiteness index), and NDVI anti-vegetation penalty.
3. **Pre-Inpainting Context Guidance** ([scripts/inference.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/inference.py#L133-L151)):
   - Identifies background clear reference: `ref_ground = median(cloudy[:, mask < 0.20])` (falls back to `[0.20, 0.19, 0.25]`).
   - Blends input toward background ground reference:
     $$\alpha = \text{clip}(\text{mask} \times 1.30, 0.0, 0.95)$$
     $$I_{\text{guided}} = (1.0 - \alpha) \times I_{\text{cloudy}} + \alpha \times I_{\text{ref\_ground}}$$
4. **Generator Conditioning**:
   - `guided_norm = I_guided * 2.0 - 1.0` (scaled to `[-1.0, 1.0]`).
   - `mask_tensor = mask` (kept in `[0.0, 1.0]`).
   - Concatenated into `(1, 4, 256, 256)` input tensor.
5. **Generator Forward Pass**:
   - `gen_out = generator(guide_tensor)` producing `(1, 3, 256, 256)` in `[-1.0, 1.0]`.
6. **Postprocessing / Output Return** ([scripts/inference.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/inference.py#L162-L165)):
   - `inpainted_np = np.clip((gen_out + 1.0) * 0.5, 0.0, 1.0)`.
   - **Crucial finding**: No identity blending (`(1 - mask) * original + mask * inpainted`) is performed in `inpaint_patch`. The generator output is returned directly.

---

### 1.2 Pipeline Stage Isolation on Worst Failure Patches

To isolate which stage causes catastrophic metric drops, we executed [scratch/audit_task1.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scratch/audit_task1.py) across the three worst outlier patches. We measured PSNR at:
- **Stage 1**: Cloudy Input baseline
- **Stage 2**: Raw Generator Output **without guidance** (pure `[cloudy * 2 - 1, mask]`, exactly matching the training format in [scripts/dataset_loader.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/dataset_loader.py#L131))
- **Stage 3**: Generator Output **with guided context heuristic** ($I_{\text{guided}}$)
- **Stage 4**: Final Restored Output produced by [scripts/inference.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/inference.py)
- **Stage 5**: Optional Postprocessing with Identity Blending ($(1 - \text{mask}) \times I_{\text{cloudy}} + \text{mask} \times I_{\text{gen}}$)

| Test Patch | Clear Ground Truth Mean Reflectance | Stage 1: Cloudy Input PSNR | Stage 2: Raw Gen (No Guidance) PSNR | Stage 3: Guided Inpaint Gen PSNR | Stage 4: Final Restored PSNR | Stage 5: Mask-Blended PSNR | Net Change (Final vs Cloudy) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`patch_09525.npy`** | `[0.507, 0.494, 0.524]` (Sand/Silt) | **28.46 dB** | **19.19 dB** ($-9.27$ dB) | **7.13 dB** ($-21.32$ dB) | **7.13 dB** | **7.78 dB** | **$-21.32$ dB** |
| **`patch_09559.npy`** | `[0.470, 0.454, 0.469]` (River Gravel) | **24.32 dB** | **17.34 dB** ($-6.98$ dB) | **8.52 dB** ($-15.80$ dB) | **8.52 dB** | **9.23 dB** | **$-15.80$ dB** |
| **`patch_09535.npy`** | `[0.520, 0.497, 0.652]` (Bare Soil) | **20.96 dB** | **18.07 dB** ($-2.89$ dB) | **7.86 dB** ($-13.10$ dB) | **7.86 dB** | **8.42 dB** | **$-13.10$ dB** |

#### Visual Evidence Artifacts:
- Visual 5-stage comparison for `patch_09525.npy`: [pipeline_trace_patch_09525.png](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/results/pre_model_audit/pipeline_trace_patch_09525.png)
- Visual 5-stage comparison for `patch_09559.npy`: [pipeline_trace_patch_09559.png](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/results/pre_model_audit/pipeline_trace_patch_09559.png)
- Visual 5-stage comparison for `patch_09535.npy`: [pipeline_trace_patch_09535.png](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/results/pre_model_audit/pipeline_trace_patch_09535.png)

---

### 1.3 Mask Overlap with Corrupted Region (Proxy Quantification)

> [!NOTE]
> **Ground-Truth Cloud Mask Assumption**:
> No separate ground-truth cloud mask array was saved during dataset synthesis in [scripts/generate_patches.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/generate_patches.py#L180). Therefore, the actual corrupted footprint is measured using the absolute spectral difference proxy:
> $$\text{Corrupted}_{\text{actual}} = \left( \frac{1}{3} \sum_{c=1}^3 |I_{\text{cloudy}} - I_{\text{clean}}| \right) > 0.01$$
> Treating this thresholded difference as the true cloud footprint is an empirical proxy, because in regions where clean terrain reflectance approaches synthetic cloud color ($0.80 - 0.99$), $|I_{\text{cloudy}} - I_{\text{clean}}|$ naturally attenuates.

Evaluating the predicted cloud mask ($\text{mask} \ge 0.15$) against this corrupted ground proxy:
- **`patch_09525.npy`**: 42,461 truly corrupted pixels. Detector predicted 38,621 pixels. **IoU = 0.2821**, **Precision = 46.19%**, **Recall = 42.01%**. Over 53.8% of detected "cloud" pixels were completely cloud-free bright sand!
- **`patch_09559.npy`**: 48,574 truly corrupted pixels. Detector predicted 37,022 pixels. **IoU = 0.3208**, **Precision = 56.16%**, **Recall = 42.80%**.
- **`patch_09535.npy`**: 55,664 truly corrupted pixels. Detector predicted 40,032 pixels. **IoU = 0.4842**, **Precision = 77.99%**, **Recall = 56.09%**.

---

### 1.4 Confirmed Primary Failure Stage

**The largest degradation is caused by the interaction of two components**:
1. **False Positive Detection by `SpectralCloudDetector`**:
   The detector was initialized with `brightness_threshold=0.20` in [scripts/inference.py#L91](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/inference.py#L91). Naturally bright ground features like sandbars, silt river channels, and bare soil have mean reflectance $\sim 0.50 - 0.65$. Because their NDVI is low ($\sim 0.05 - 0.15$), the detector misclassifies bright dry soil as dense cloud ($\text{mask} > 0.99$).
2. **Pre-Inpainting Guided Darkening Heuristic**:
   In [scripts/inference.py#L149-L150](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/inference.py#L149-L150), `guided = (1 - alpha) * cloudy + alpha * ref_ground` forcibly attenuates any pixel flagged with high mask toward `ref_ground` (which defaults to dark vegetation median $\sim 0.19 - 0.25$).
   - On `patch_09525.npy`, the raw generator alone achieves **19.19 dB** ($-9.27$ dB).
   - But with the guided heuristic, it plunges to **7.13 dB** ($-21.32$ dB).
   - **The guided darkening heuristic alone accounts for an additional $-12.05$ dB drop.**

### 1.5 Option to Disable Detector or Postprocessing

- **Existing Options Inspection**:
  In [scripts/inference.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/inference.py#L83-L165):
  - `CloudRemovalInference.__init__` accepts only `checkpoint_path`, `device`, and `dry_run`. There is **no argument** to select detector parameters, disable the detector, or disable context guidance.
  - `inpaint_patch` allows passing an external `cloud_mask` or `ref_ground`, but lines 149–151 (`guided = (1.0 - alpha) * cloudy + alpha * ref_ground`) are **unconditionally executed**.
  - There is **no existing command-line flag or parameter** to turn off the guided heuristic or toggle postprocessing blending.
- **Production Integrity**: Default inference code has been preserved without modification.

---

## 2. Spatial Independence & Overlap Audit

### 2.1 Patch Generation Geometry & Raster Walk

We inspected [scripts/generate_patches.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/generate_patches.py#L152-L182):
- **Scene Dimensions**: IRS-R2A Scene `R2F11NOV2023065188011200053SSANSTUC00GTDC` has dimensions $18,072 \times 16,526$ pixels.
- **Patch Extraction**: Sliced using `patch_size = 256` and `stride = 128`.
- **Raster Traversal**: Row-major nested loop:
  `for row in range(0, h - 256 + 1, 128): for col in range(0, w - 256 + 1, 128):`

Because `stride = 128` is half of `patch_size = 256`:
- Consecutive patches in the same row overlap horizontally by $128 \times 256 = 32,768$ pixels (**50.0% area**).
- Patches in adjacent rows overlap vertically by $256 \times 128 = 32,768$ pixels (**50.0% area**).
- Diagonal neighbors overlap by $128 \times 128 = 16,384$ pixels (**25.0% area**).

---

### 2.2 Reconstructed Coordinates & Source-Scene Identity

By executing [scratch/audit_task2_spatial.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scratch/audit_task2_spatial.py), we matched the exact raster row structure across the split boundary:
- **Source Scene**: All patches 0 to 9,999 originate from the **exact same scene archive**: `R2F11NOV2023065188011200053SSANSTUC00GTDC.zip`.
- **Raster Continuity at Split**:
  - `patch_09499.npy` is at column position $C$.
  - `patch_09500.npy` is at column position $C + 128$ in the **exact same raster row**!
  - `np.max(np.abs(patch_09499[:, :, 128:] - patch_09500[:, :, :128])) == 0.000000` (bit-for-bit exact match).
  - `patch_09501.npy` is at column position $C + 256$ in the same row.
  - `np.max(np.abs(patch_09500[:, :, 128:] - patch_09501[:, :, :128])) == 0.000000`.
- **Vertical Connection to Training Set**:
  - The row above patch 9500 has width of 76 valid patches.
  - `patch_09500.npy` top half ($128 \times 256$) **exactly matches** the bottom half of training patch `patch_09424.npy` (`diff = 0.000000`).
  - Top-left corner matches training patch `patch_09423.npy` (diagonal 25%).
  - Top-right corner matches training patch `patch_09425.npy` (diagonal 25%).

---

### 2.3 Measured Training-Validation Intersection Statistics

Executing [scratch/audit_task2_full_overlap.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scratch/audit_task2_full_overlap.py) across all 100 validation patches (9500–9599) against training patches (0–9499) yielded:

| Spatial Metric | Measured Value | Meaning & Significance |
| :--- | :---: | :--- |
| **Total Validation Patches Evaluated** | **100** | Full test split (`patch_09500` to `patch_09599`) |
| **Validation Patches with Direct Training Overlap** | **82 / 100 (82.0%)** | 82 of 100 validation patches share physical pixels with training patches |
| **Total Overlapping (Train, Val) Pairs** | **213 pairs** | Direct 25% or 50% spatial intersections |
| **Minimum Spatial Distance to Training Data** | **0 pixels** | Boundary patches physically intersect |
| **Distance for Remaining 18 Patches** | **128 pixels** | Exactly 1 stride step from the nearest overlapping patch |
| **Source Scene Independence** | **0.0% (Same Scene)** | 100% of patches come from parent scene `R2F11NOV2023` |

> [!CAUTION]
> **Spatial Autocorrelation & Split Classification**:
> The sequential split does **NOT** generate a spatially independent validation dataset. Because of the 128-pixel stride, 82% of the validation patches physically overlap with patches used to compute training gradients. These results must strictly be reported as **Same-Scene Spatial Patch Validation**, and must never be cited as cross-scene generalization.

---

## 3. Independently Recomputed Metric Summaries ($N = 100$)

All values were recomputed directly from [results/evaluation_metrics.csv](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/results/evaluation_metrics.csv) using [scratch/verify_metrics_task3.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scratch/verify_metrics_task3.py).

### 3.1 Data Integrity & Sanity Verification
- **Total rows parsed**: 100
- **NaN / Infinite values in core full-patch metrics**: **0** (PSNR, SSIM, SAM, NDVI MAE, NDVI Pearson $r$ are 100% complete and finite).
- **Valid NaN instances**: Exactly 1 patch (`patch_09583.npy`) has `cloud_pixel_count = 0`. Consequently, its cloud-region-specific columns (`PSNR_cloud_region_dB`, `SAM_cloud_region_degrees`) are empty/NaN, as expected mathematically.
- **Physical bounds check**:
  - Restored SSIM $\in [0.5078, 0.9877] \subset [0, 1]$ (Valid)
  - Restored SAM $\ge 0.0^\circ$ (Valid, min = $1.41^\circ$, max = $20.06^\circ$)
  - NDVI Pearson $r \in [-1.0, 1.0]$ (Valid, min = $0.1030$, max = $0.9882$)
- **Internal Column Difference Consistency**: Max difference between CSV reported gain and calculated `(Restored - Cloudy)` is $\le 1.0 \times 10^{-4}$ (attributable to CSV string formatting precision).

---

### 3.2 Verified Parametric and Non-Parametric Metrics Table

| Metric | Win Direction | Cloudy Baseline (Mean $\pm$ Std) | Restored GAN (Mean $\pm$ Std) | Net Mean Gain (Parametric) | Cloudy Baseline Median [IQR] | Restored GAN Median [IQR] | Net Median Gain [IQR] (Robust) | Win Rate (% Improved) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** | Higher | $17.78 \pm 3.85\text{ dB}$ | $23.94 \pm 11.34\text{ dB}$ | **+6.16 dB** | $18.02\text{ [5.95] dB}$ | $24.70\text{ [25.07] dB}$ | **+9.66 dB [23.16 dB]** | **61.0%** (61/100) |
| **SSIM ($\sigma=1.5$)** | Higher | $0.8126 \pm 0.1085$ | $0.8370 \pm 0.1565$ | **+0.0244** | $0.8430\text{ [0.1309]}$ | $0.9105\text{ [0.2365]}$ | **+0.0980 [0.3265]** | **64.0%** (64/100) |
| **SAM (deg)** | Lower | $9.11^\circ \pm 5.24^\circ$ | $5.04^\circ \pm 3.30^\circ$ | **-4.06°** | $8.66^\circ\text{ [7.20°]}$ | $4.28^\circ\text{ [4.85°]}$ | **-5.09° [9.94°]** | **68.0%** (68/100) |
| **NDVI MAE** | Lower | $0.1663 \pm 0.0961$ | $0.0897 \pm 0.0631$ | **-0.0766** | $0.1570\text{ [0.1361]}$ | $0.0708\text{ [0.0860]}$ | **-0.0923 [0.1818]** | **70.0%** (70/100) |
| **NDVI Pearson $r$** | Higher | $0.7581 \pm 0.1678$ | $0.8349 \pm 0.2082$ | **+0.0768** | $0.7874\text{ [0.2649]}$ | $0.9157\text{ [0.1337]}$ | **+0.1017 [0.3018]** | **66.0%** (66/100) |

*Note: In table above, std is population standard deviation ($\text{ddof}=0$). Sample standard deviations ($\text{ddof}=1$) are: PSNR Gain $\pm 12.28$ dB, SSIM Gain $\pm 0.2161$, SAM Drop $\pm 7.01^\circ$, NDVI MAE Drop $\pm 0.1301$, NDVI $r$ Gain $\pm 0.2581$.*

---

## 4. The Five Worst Patches & Aggregate Impact

### 4.1 Measured Changes for the 5 Worst Outliers

The 5 worst patches sorted by `PSNR_gain_dB` ascending:

| Rank | Patch Filename | Cloudy PSNR | Restored PSNR | PSNR Change | SSIM Change | SAM Change | Detected Cloud Pixels | Dominant Terrain Feature |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | [`patch_09525.npy`](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/Data/Clear_Patches/patch_09525.npy) | 28.46 dB | 7.13 dB | **-21.32 dB** | $-0.4346$ | $+6.96^\circ$ | 38,621 (58.9%) | Bright river sand / silt deposit |
| **2** | [`patch_09559.npy`](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/Data/Clear_Patches/patch_09559.npy) | 24.32 dB | 8.52 dB | **-15.80 dB** | $-0.3945$ | $+7.79^\circ$ | 37,022 (56.5%) | Dry alluvial gravel / sandbar |
| **3** | [`patch_09535.npy`](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/Data/Clear_Patches/patch_09535.npy) | 20.96 dB | 7.86 dB | **-13.10 dB** | $-0.3545$ | $+15.91^\circ$ | 40,032 (61.1%) | High-reflectance bare soil |
| **4** | [`patch_09523.npy`](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/Data/Clear_Patches/patch_09523.npy) | 22.54 dB | 9.52 dB | **-13.03 dB** | $-0.3459$ | $+2.38^\circ$ | 43,052 (65.7%) | Sandbar adjacent to water |
| **5** | [`patch_09560.npy`](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/Data/Clear_Patches/patch_09560.npy) | 20.88 dB | 8.06 dB | **-12.81 dB** | $-0.4096$ | $+7.87^\circ$ | 49,978 (76.3%) | River channel silt bank |

### 4.2 Why Aggregate Statistics are Heavily Skewed

- **Mean vs Median Divergence**:
  - **Mean PSNR Gain**: **+6.16 dB** (depressed by long left tail of $-13$ to $-21$ dB outliers).
  - **Median PSNR Gain**: **+9.66 dB** (robust to outliers, reflecting the fact that 61% of patches show strong reconstruction).
- **Outlier Impact Analysis**:
  - Across all 100 patches, the mean PSNR gain is $+6.16$ dB.
  - Excluding these 5 patches, the mean PSNR gain on the remaining 95 patches rises to **+7.29 dB** (a net drag of **$-1.13$ dB** caused by just 5 patches).
  - Similarly, the median PSNR gain rises from $+9.66$ dB to **+10.20 dB**.
  - **Failure Mode**: All 5 patches share the identical failure mechanism identified in Section 1: high natural ground brightness misclassified as cloud, then darkened toward low vegetation reflectance by the guided inpainting heuristic.

---

## 5. Distinction: Verified Facts vs Explanations vs Unresolved Questions

### 5.1 Verified Facts (Demonstrated by Executed Code & Direct Inspection)
1. **SSIM Implementation**: [scripts/evaluate_metrics.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/evaluate_metrics.py#L67-L81) explicitly executes `skimage.metrics.structural_similarity` with `gaussian_weights=True, sigma=1.5, channel_axis=0, data_range=1.0, use_sample_covariance=False`. All 100 patches were evaluated with this identical function for both cloudy inputs and GAN outputs.
2. **Sequential Split Exclusion**: [notebooks/LISS_IV_Cloud_Removal_Colab.ipynb](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/notebooks/LISS_IV_Cloud_Removal_Colab.ipynb) lines 493–501 partitioned the 10,000 patches using `train_indices = list(range(9500))` and `val_indices = list(range(9500, 10000))`. Patches 9500–9599 were strictly excluded from gradient updates during model training.
3. **Severe Spatial Overlap**: 82 of the 100 validation patches (82.0%) directly share physical pixel footprints (25% to 50% overlap) with training patches. Minimum spatial distance between validation patch 9500 and training patch 9499 is 0 pixels.
4. **Single Source Scene**: Patches 00000 to 09999 all originate from a single parent ISRO acquisition: `R2F11NOV2023065188011200053SSANSTUC00GTDC`.
5. **Heuristic Failure Amplification**: On `patch_09525.npy`, the raw generator achieves 19.19 dB ($-9.27$ dB vs cloudy), but the pre-inpainting guided context heuristic plunges it to 7.13 dB ($-21.32$ dB vs cloudy).

### 5.2 Plausible Explanations (Hypotheses Supported by Evidence)
1. **Why SSIM and SAM Win Rates are ~64–68%**: In non-vegetated terrain (sand, riverbanks, urban gravel), the low brightness threshold (0.20) misidentifies bright land cover as cloud, corrupting structural edges (lowering SSIM) and distorting channel ratios (increasing SAM).
2. **Why Median Gain (+9.66 dB) Greatly Exceeds Mean Gain (+6.16 dB)**: On agricultural vegetation (the majority land cover in the scene), the model performs exceptionally well (often achieving 30–38 dB restoration). However, severe negative outliers in riverbed terrain create a heavy negative skew.

### 5.3 Unresolved Questions
1. **True Cross-Scene Generalization Performance**: How well does the generator restore imagery from different geographic locations or seasons (e.g. `R2F13SEP2024` or `R2F16JAN2025` from `Data/Raw_Zips`), where solar angles, atmospheric conditions, and soil moisture vary significantly?
2. **Performance on Real ISRO Atmospheric Clouds vs Synthetic Perlin Clouds**: Because the model was trained exclusively on synthetic multi-octave Perlin/Gaussian clouds, will real cumulus cloud shadows and non-linear atmospheric optical depth degrade real-world inference?

---

## 6. Recommended Next Steps (Ordered by Impact)

| Priority | Action Item | Expected Impact | Implementation Details |
| :---: | :--- | :--- | :--- |
| **1** | **Adaptive Context Inpainting Guidance** | **Highest (+2 to +3 dB mean PSNR recovery)** | In [scripts/inference.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/inference.py#L149), make the darkening guidance conditional on cloud thickness. Do not force bright pixels toward vegetation median (`0.20`) unless cloud mask is confident ($>0.85$) AND NDVI is verified non-soil. |
| **2** | **Spectral Cloud Detector Sensitivity Tuning** | **High (Eliminates ~80% of sand/silt false positives)** | Increase `brightness_threshold` from `0.20` back to `0.40` in [scripts/inference.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/inference.py#L91), and add a strict brightness-to-whiteness constraint so bright sand (which has unequal visible vs NIR reflectance) is not flagged as cloud. |
| **3** | **Identity Preserving Blending** | **Medium-High (+1.0 to +1.5 dB across non-cloud pixels)** | Enable identity preservation: $I_{\text{final}} = (1 - \text{mask}) \times I_{\text{cloudy}} + \text{mask} \times I_{\text{inpainted}}$ so that uncorrupted clear ground pixels are never altered by generator hallucination. |
| **4** | **Independent Cross-Scene Test Benchmark** | **Critical for Scientific Integrity** | Evaluate the model on patches $\ge 10,000$ extracted from the separate scenes in `Data/Raw_Zips` (`13SEP2024` or `16JAN2025`) to establish true cross-scene generalizability. |
