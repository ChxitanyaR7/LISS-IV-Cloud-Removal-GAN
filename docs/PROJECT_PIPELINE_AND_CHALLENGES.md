# LISS-IV Satellite Cloud Removal GAN: End-to-End Pipeline & Technical Challenges

---

## 1. Executive Summary & Project Scope

In optical Earth Observation, cloud cover is the single largest source of data loss, rendering over **50% of acquired satellite scenes unusable** for automated land-use mapping, agriculture monitoring, disaster assessment, and environmental tracking.

This project addresses cloud removal for **ResourceSat-2 / 2A LISS-IV** (Linear Imaging Self-Scanning Sensor - IV), a high-resolution multispectral sensor operating at **5.8-meter spatial resolution** with 3 spectral bands:
- **Band 2 ($B_2$)**: Green ($0.52 - 0.59\ \mu\text{m}$)
- **Band 3 ($B_3$)**: Red ($0.62 - 0.68\ \mu\text{m}$)
- **Band 4 ($B_4$)**: Near-Infrared / NIR ($0.77 - 0.86\ \mu\text{m}$)

Because true paired satellite imagery (an identical scene captured clear and cloudy at the exact same moment with identical sun angle and seasonal vegetation) does not exist in nature, this project designs an end-to-end pipeline:
1. Extract and normalize high-resolution multispectral scenes.
2. Synthesize physically realistic multi-octave cloud masks and optical blends.
3. Detect clouds and extract quantitative guidance parameters.
4. Train a Conditional Generative Adversarial Network (cGAN / Pix2Pix) to restore cloud-occluded ground surfaces while preserving spectral fidelity.

---

## 2. Complete 10-Step Pipeline Architecture

```
[Raw Scene ZIPs]
       │
       ▼
[1. Band Extraction] ──> (BAND2.tif, BAND3.tif, BAND4.tif)
       │
       ▼
[2. Radiometric Normalization] ──> (2%-98% Linear Percentile Stretch -> [0, 1])
       │
       ▼
[3. False Color Composite (FCC)] ──> (NIR B4 -> R, Red B3 -> G, Green B2 -> B)
       │
       ▼
[4. Sliding-Window Patching] ──> (256x256 Patches, Stride 128, Quality Filtering)
       │
       ▼
[5. Multi-Octave Cloud Simulation] ──> (Continuous Alpha Mask + Band-dependent Reflectance)
       │
       ▼
[6. Cloud Detection & Masking] ──> (Physical Spectral Detector + U-Net Mask Predictor)
       │
       ▼
[7. Cloud Parameter Extraction] ──> (Coverage Ratio, Mean Opacity, Thick/Thin Fractions)
       │
       ▼
[8. 4-Channel Guidance Tensor] ──> (Input: [B2, B3, B4, Mask] -> Conditional GAN)
       │
       ▼
[9. Inpainting & Reconstruction] ──> (Generator Inpainting + Discriminator Adversarial Check)
       │
       ▼
[10. Fidelity Evaluation] ──> (PSNR, SSIM, SAM, UIQI, NDVI Preservation)
```

---

### Step 1: Raw Multispectral Ingestion & Scene Decompression
- **Input**: Raw archives (`.zip`) from ISRO Bhoonidhi data portal containing GeoTIFF scenes (typically $18,000 \times 16,000$ pixels per band, $> 300\text{ MB}$ uncompressed per scene).
- **Process**: Automated extraction via [generate_patches.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/generate_patches.py) to isolate `BAND2.tif`, `BAND3.tif`, and `BAND4.tif`.
- **Implementation**: Handled with multi-platform image readers (`rasterio` and `PIL` with `MAX_IMAGE_PIXELS = None` for large image decoding).

---

### Step 2: Radiometric Enhancement & Normalization
- **Challenge**: Raw sensor pixels are recorded as 10-bit or 16-bit Digital Numbers (DN) with sensor noise, atmospheric haze, and non-reflective black nodata borders.
- **Normalization Formula**:
  $$\text{Valid Mask:}\ M_{\text{valid}} = (B_2 > 0) \land (B_3 > 0) \land (B_4 > 0)$$
  For each band $B_k$:
  $$p_2 = \text{Percentile}(B_k[M_{\text{valid}}], 2\%)$$
  $$p_{98} = \text{Percentile}(B_k[M_{\text{valid}}], 98\%)$$
  $$\hat{B}_k = \text{clip}\left(\frac{B_k - p_2}{p_{98} - p_2}, 0.0, 1.0\right)$$
- **Result**: Consistent $[0.0, 1.0]$ floating-point dynamic range that eliminates extreme sensor outliers while preserving inter-band radiometric ratios.

---

### Step 3: False Color Composite (FCC) Generation
In remote sensing, LISS-IV standard visualization maps:
- **Red Display Channel** $\leftarrow$ Band 4 (Near-Infrared / NIR)
- **Green Display Channel** $\leftarrow$ Band 3 (Red)
- **Blue Display Channel** $\leftarrow$ Band 2 (Green)

$$\text{FCC} = \text{stack}\left[\hat{B}_4, \hat{B}_3, \hat{B}_2\right] \in \mathbb{R}^{H \times W \times 3}$$

Healthy vegetation absorbs Red light (chlorophyll absorption) and strongly scatters Near-Infrared light (mesophyll cell wall scattering). As a result, agricultural crops and forests appear intensely red, water bodies appear dark blue/black, and urban areas appear cyan/gray.

---

### Step 4: Sliding-Window Patching & Quality Rejection Filters
Full scenes cannot fit into GPU VRAM for GAN training. The scenes are sliced into $256 \times 256$ patches with an overlap stride of $128$ pixels.
To guarantee training stability, three automated quality filters are enforced:
1. **Valid Pixel Fraction**:
   $$\frac{1}{N}\sum M_{\text{valid}} \ge 0.90\ (90\%)$$
   Rejects black image margins and satellite swath boundaries.
2. **Standard Deviation (Texture Variance)**:
   $$\sigma(\text{patch}[M_{\text{valid}}]) \ge 0.05$$
   Rejects completely flat, uniform surfaces or featureless water bodies.
3. **Minimum Mean Brightness**:
   $$\mu(\text{patch}[M_{\text{valid}}]) \ge 0.15$$
   Rejects shadowed, underexposed, or corrupted patches.

---

### Step 5: Multi-Octave Synthetic Cloud Generation
Synthetic clouds are generated using continuous multi-scale noise with spatial frequency blending in [cloud_simulator.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/cloud_simulator.py):
1. **Octave Pyramid**:
   - $N_1$: Low frequency structure ($4 \times 4$ noise grid $\to$ bicubic interpolation to $256 \times 256$)
   - $N_2$: Mid frequency body ($8 \times 8$ noise grid $\to$ bicubic interpolation to $256 \times 256$)
   - $N_3$: High frequency edge turbulence ($16 \times 16$ noise grid $\to$ bicubic interpolation to $256 \times 256$)
   $$M_{\text{raw}} = \text{GaussianBlur}(0.65 N_1 + 0.25 N_2 + 0.10 N_3, \sigma=51)$$
2. **Morphological Variations**:
   - *Wispy Cirrus*: $M = M_{\text{raw}}^{2.0} \times \mathcal{U}(0.3, 0.6)$
   - *Directional Cloud Bank*: $M = (M_{\text{raw}} \cdot G_{\theta})^{1.4} \times \mathcal{U}(0.5, 0.85)$ (where $G_{\theta}$ is a linear directional gradient)
   - *Dense Overcast / Cumulus*: $M = M_{\text{raw}}^{1.2} \times \mathcal{U}(0.4, 0.85)$
3. **Band-Dependent Optical Blending**:
   Clouds are not uniformly white; their optical scattering cross-section is higher in NIR:
   $$C_{\text{cloud}} = \begin{bmatrix} \mathcal{U}(0.80, 0.92) & (\text{B2 Green}) \\ \mathcal{U}(0.82, 0.94) & (\text{B3 Red}) \\ \mathcal{U}(0.88, 0.99) & (\text{B4 NIR}) \end{bmatrix}$$
   $$I_{\text{cloudy}} = (1 - \alpha) \cdot I_{\text{clean}} + \alpha \cdot C_{\text{cloud}}$$
   where $\alpha \in [0, 1]$ is the generated cloud density mask.

---

### Step 6: Cloud Detection & Semantic Mask Segmentation
Cloud detection in LISS-IV is achieved through [cloud_detector.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/cloud_detector.py):
1. **Spectral Whiteness Index (Mie Scattering)**:
   $$\text{Spread} = \frac{|B_2 - \bar{I}| + |B_3 - \bar{I}| + |B_4 - \bar{I}|}{3},\quad \text{Whiteness} = 1 - \text{clip}\left(\frac{\text{Spread}}{\tau_{\text{tol}}}, 0, 1\right)$$
2. **Anti-NDVI Discrimination**:
   $$\text{NDVI} = \frac{B_4 - B_3}{B_4 + B_3 + \epsilon}$$
   Vegetation has $\text{NDVI} > 0.35$; clouds have $\text{NDVI} \in [-0.1, 0.2]$. Any bright pixel with high NDVI is classified as vegetation, not cloud.
3. **Deep Learning U-Net**:
   A 4-stage encoder-decoder network predicting continuous cloud opacity $\hat{M}(x, y) \in [0, 1]$ trained with combined BCE + Dice loss.

---

### Step 7: Cloud Parameter Extraction & Inpainting Guidance
Rather than passing an unguided image to the GAN, the detection module computes explicit conditioning parameters:
- **Cloud Coverage Ratio (CCR)**: $\frac{1}{HW}\sum \mathbb{I}(\hat{M} \ge 0.20)$
- **Mean Cloud Opacity**: Average optical density of the occluded region.
- **Thick Cloud Fraction**: Pixels with $\hat{M} \ge 0.65$ (zero ground visibility $\to$ requires full generative synthesis).
- **Thin Cloud Fraction**: Pixels with $0.20 \le \hat{M} < 0.65$ (partial ground visibility $\to$ requires dehazing/contrast restoration).
- **4-Channel Guidance Tensor**:
  $$I_{\text{guide}} = \text{concatenate}\left([B_2, B_3, B_4],\ \hat{M}\right) \in \mathbb{R}^{4 \times H \times W}$$
  The 4th channel explicitly tells the GAN generator *which* pixels are cloudy and *how deep* the cloud is.

---

### Step 8: Conditional Generative Adversarial Network (cGAN) Architecture
The downstream generative architecture consists of:
1. **Generator ($G$)**:
   - ResNet-based or U-Net with skip connections and self-attention.
   - Input: $4 \times 256 \times 256$ ($I_{\text{cloudy}} + \hat{M}$).
   - Output: $3 \times 256 \times 256$ ($\hat{I}_{\text{clean}}$).
   - Identity preservation: pixels outside the cloud mask ($1 - \hat{M}$) are anchored to the original satellite input.
2. **Discriminator ($D$)**:
   - $70 \times 70$ PatchGAN discriminator that determines whether local $70 \times 70$ overlapping patches are real LISS-IV ground or synthetic hallucinations.
3. **Multi-Objective Loss Formulation**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{cGAN}(G, D) + \lambda_1 \mathcal{L}_{L1}(G) + \lambda_{\text{perc}} \mathcal{L}_{\text{perceptual}}(G) + \lambda_{\text{spec}} \mathcal{L}_{\text{spectral}}(G)$$
   - $\mathcal{L}_{L1}$: Pixel-wise reconstruction loss $\|I_{\text{clean}} - \hat{I}_{\text{clean}}\|_1$.
   - $\mathcal{L}_{\text{perceptual}}$: VGG / ResNet deep feature consistency.
   - $\mathcal{L}_{\text{spectral}}$: Spectral Angle Mapper (SAM) loss to prevent color shift in NIR.

---

### Step 9: Recombination & Boundary Seamless Blending
To eliminate abrupt transitions between clear ground and inpainted regions:
$$\hat{I}_{\text{final}} = (1 - \hat{M}) \odot I_{\text{cloudy}} + \hat{M} \odot G(I_{\text{guide}})$$
For thick clouds, edge transition smoothing (Poisson blending or Gaussian boundary feathering) is applied along the perimeter of the cloud mask to prevent border seam lines.

---

### Step 10: Quantitative & Qualitative Validation
Evaluation is performed against paired ground truth using standardized metrics:
- **Peak Signal-to-Noise Ratio (PSNR)**: Measures reconstruction fidelity (target $> 28\text{ dB}$).
- **Structural Similarity Index (SSIM)**: Measures perceptual structure and edge preservation (target $> 0.88$).
- **Spectral Angle Mapper (SAM)**: Measures vector angle difference in multispectral space:
  $$\text{SAM}(\mathbf{x}, \hat{\mathbf{x}}) = \arccos\left(\frac{\mathbf{x} \cdot \hat{\mathbf{x}}}{\|\mathbf{x}\|_2 \|\hat{\mathbf{x}}\|_2}\right)$$
  Ensures the radiometric ratios between $B_2$, $B_3$, and $B_4$ are preserved.
- **NDVI Fidelity Correlation**: Ensures agricultural analysis on the reconstructed scene matches the ground truth.

---

## 3. In-Depth Technical Challenges in LISS-IV Cloud Removal

### Challenge 1: Severe Spectral Band Limitations (No SWIR, Blue, or Thermal IR)
- **Explanation**: Standard cloud detection algorithms (such as Fmask for Landsat-8 or Sentinel-2) rely heavily on:
  - **SWIR Bands ($1.6\ \mu\text{m}, 2.2\ \mu\text{m}$)**: Water clouds absorb SWIR while ground reflects it, making cloud detection trivial.
  - **Cirrus Band ($1.38\ \mu\text{m}$)**: Strong atmospheric water vapor absorption isolates high-altitude cirrus.
  - **Thermal Infrared (TIR)**: Detects cold cloud-top temperatures ($< 270\text{ K}$).
  - **Blue Band ($0.45 - 0.51\ \mu\text{m}$)**: Measures Rayleigh scattering and aerosol optical depth.
  - **LISS-IV has NONE of these bands**. It has only Green, Red, and NIR.
- **Solution Developed**:
  We compensate for the lack of SWIR by exploiting **Spectral Whiteness (Mie scattering)** and **Anti-NDVI filtering**. Clouds scatter Green, Red, and NIR in near-equal proportions, producing a characteristic flat spectral response ($|B_2 - \bar{I}| + |B_3 - \bar{I}| + |B_4 - \bar{I}| \approx 0$), whereas ground features display high variance across these bands.

---

### Challenge 2: The Urban vs Cloud Disambiguation Problem
- **Explanation**: High-albedo manmade surfaces (concrete highways, white rooftop coatings, industrial zinc sheds, limestone quarries, sand dunes) have high reflectance across Green, Red, and NIR. A naive brightness threshold classifies entire city blocks as clouds and attempts to hallucinate farmland over them.
- **Solution Developed**:
  1. Man-made urban features have sharp, high-frequency spatial edges and directional rectilinear geometry, whereas clouds have soft, diffused, multi-scale Gaussian margins.
  2. Concrete exhibits a flat or slightly negative slope into NIR, whereas clouds remain high.
  3. Morphological opening and closing filters eliminate isolated bright man-made pixels, preserving only contiguous cloud masses.

---

### Challenge 3: The Thick vs Thin Cloud Dilemma (Dehazing vs Generative Inpainting)
- **Explanation**: Clouds are not a homogeneous binary mask:
  - **Thin Cirrus / Haze**: Underlying surface topography, roads, and crop patterns are partially transmitted through the cloud veil. The problem is an inverse ill-posed **dehazing / contrast restoration** problem.
  - **Thick Cumulus / Stratus**: Underlying ground photons are 100% scattered/blocked ($0\%$ optical transmission). The problem is an **inpainting / semantic synthesis** problem.
- **Solution Developed**:
  Our [cloud_detector.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/cloud_detector.py) calculates **Thick Cloud Fraction** and **Thin Cloud Fraction** separately. The continuous cloud density mask $\hat{M} \in [0, 1]$ informs the GAN of the exact optical depth: where $\hat{M} < 0.6$, the GAN acts as an optical amplifier/dehazer; where $\hat{M} \ge 0.6$, the GAN acts as a contextual hallucinator.

---

### Challenge 4: Radiometric & Phenological Distortion in NIR Vegetation
- **Explanation**: Standard computer vision GANs (like vanilla Pix2Pix or CycleGAN) are built for 3-channel RGB photographs. When applied to LISS-IV multispectral data, NIR is treated like a blue channel. Standard RGB loss functions ($L_1$, Perceptual Loss from ImageNet-pretrained VGG) have no concept of vegetation reflectance physics:
  - ImageNet features expect plants to be green. In LISS-IV False Color Composites, healthy plants are bright red (due to NIR).
  - Without spectral constraints, GANs desaturate the NIR band, flattening agricultural crop vigor and corrupting NDVI calculations.
- **Solution Developed**:
  Incorporate a **Spectral Angle Mapper (SAM) loss** and an **NDVI consistency loss**:
  $$\mathcal{L}_{\text{NDVI}} = \left\| \frac{\hat{B}_4 - \hat{B}_3}{\hat{B}_4 + \hat{B}_3 + \epsilon} - \frac{B_4^* - B_3^*}{B_4^* + B_3^* + \epsilon} \right\|_1$$
  This forces the generator to maintain the precise biological NIR-to-Red ratio.

---

### Challenge 5: Edge Discontinuity & Tiling Seams
- **Explanation**: Large satellite scenes ($18,000 \times 16,000$ pixels) must be divided into $256 \times 256$ tiles for model processing. If each tile is inpainted independently, the tile borders will not align perfectly, resulting in an artificial grid pattern ("checkerboard effect") across the reconstructed landscape.
- **Solution Developed**:
  1. Extract patches with a **$50\%$ overlapping stride** ($128$ pixels).
  2. When reassembling, apply a **2D Hann window weighting matrix**:
     $$W(x, y) = \sin^2\left(\frac{\pi x}{H}\right) \sin^2\left(\frac{\pi y}{W}\right)$$
  3. Overlapping regions blend smoothly based on center-weighted confidence, completely eliminating tile border seams.

---

### Challenge 6: The Synthetic-to-Real Domain Gap
- **Explanation**: If a GAN is trained purely on simplistic Gaussian blur clouds, it will fail when deployed on real satellite scenes that feature:
  - Cloud shadows on the ground (dark silhouettes offset in the direction opposite the sun azimuth).
  - Multi-layer cloud stratification (semi-transparent cirrus over dense cumulus).
- **Solution Developed**:
  Our [cloud_simulator.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/cloud_simulator.py) uses **multi-octave Perlin noise with variable power laws** and generates 3 distinct cloud typologies (wispy, directional bank, dense overcast). The band-dependent cloud reflectance accurately simulates Mie scattering in the NIR spectrum.

---

## 4. Current Project Status & Roadmap

| Stage | Status | Component |
| :--- | :---: | :--- |
| **1. Data Organization** | ✅ Completed | Clean modular repo (`notebooks/`, `scripts/`, `notes/`, `Data/`) |
| **2. Preprocessing & FCC** | ✅ Completed | 2%-98% percentile linear stretch, FCC visualization helpers |
| **3. Automated Patch Extraction** | ✅ Completed | [generate_patches.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/generate_patches.py) with 3 quality rejection filters |
| **4. Synthetic Cloud Simulation** | ✅ Completed | [cloud_simulator.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/cloud_simulator.py) with band-dependent optical transfer |
| **5. Cloud Detection Model** | ✅ Completed | [cloud_detector.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/cloud_detector.py) (Spectral detector + U-Net architecture) |
| **6. Guidance Parameter Extraction** | ✅ Completed | CCR, Opacity, Thick/Thin breakdown & 4-channel guidance tensor |
| **7. PyTorch Dataset Loader** | ✅ Completed | [dataset_loader.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/dataset_loader.py) with spatial augmentations |
| **8. Inpainting cGAN Training** | 🔄 Next Step | Generator (ResNet/UNet) + PatchGAN Discriminator + SAM loss |
| **9. Full Scene Recombination** | 🔄 Next Step | Overlapping Hann window stitcher for $18,000 \times 16,000$ scenes |
| **10. Quantitative Evaluation** | 🔄 Next Step | Automated PSNR, SSIM, SAM, and NDVI verification benchmark |
