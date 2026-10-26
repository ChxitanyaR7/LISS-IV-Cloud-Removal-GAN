# 📊 Evaluation Metrics Guide: Explained in Simple Plain English

> **For Your Major Project Viva & Final Presentation**  
> If an examiner asks: *"What metrics did you use to evaluate your model, and what do they mean?"* — use this simple guide.

---

## The Big Picture (Why Satellite Metrics are Different)

In standard computer vision (like Photoshop or Instagram filters), an image only needs to **"look pleasing to the human eye"**.

In **ISRO satellite remote sensing**, looking nice is not enough. Scientists and government agencies use satellite data for:
1. **Measuring crop health & drought** (Agriculture)
2. **Detecting flood water levels** (Disaster Management)
3. **Urban planning and road mapping** (Infrastructure)

Therefore, we evaluate our Cloud Removal GAN using **4 complementary metrics**:
* **2 Spatial Metrics**: **PSNR** (Noise level) and **SSIM** (Edge & shape fidelity)
* **2 Remote Sensing Physics Metrics**: **SAM** (Color/spectral purity) and **NDVI** (Agricultural usefulness)

---

## 1. PSNR (Peak Signal-to-Noise Ratio)

* **What it measures**: Numerical cleanliness and reconstruction error.
* **Unit**: Decibels (**dB**).
* **Direction**: **HIGHER is better ($\uparrow$)**.

### Plain English Explanation:
Think of a phone call. If there is a lot of static and buzzing on the line, the signal-to-noise ratio is low. If the voice is crystal clear with zero background hiss, the ratio is high.
In satellite images, **clouds and blur are the "noise"**, and the **real ground surface is the "signal"**.

* **Under 15 dB**: Heavy cloud occlusion, blurry, lots of error.
* **20 – 25 dB**: Good operational restoration.
* **Above 28 – 35 dB**: Pristine, near-perfect ground recovery.
* **Our Result**: Our GAN achieves up to **+18 dB to +23 dB improvement** on occluded test patches!

---

## 2. SSIM (Structural Similarity Index Measure)

* **What it measures**: Human visual structure, shapes, and boundaries.
* **Range**: **$0.0$ to $1.0$** (or 0% to 100%).
* **Direction**: **HIGHER is better ($\uparrow$)** ($1.0$ = identical).

### Plain English Explanation:
PSNR only compares individual pixel numbers one-by-one. But humans see **patterns, borders, and textures**. 
SSIM looks at local windows to check:
1. Are the **edges of riverbanks** straight and sharp?
2. Are **agricultural field grids** preserved?
3. Did the GAN blur or smudge the landscape?

An SSIM above **0.85 – 0.90** proves that fine spatial details (roads, water bodies, field borders) are preserved without distortion.

---

## 3. SAM (Spectral Angle Mapper) — *The Remote Sensing Super-Metric*

* **What it measures**: Radiometric and color purity across sensor bands.
* **Unit**: Degrees (**° angle**).
* **Direction**: **LOWER is better ($\downarrow$)** ($0^\circ$ = identical spectral signature).

### Plain English Explanation:
*(This is the metric that will wow the professors because most CS students don't know it!)*

Every satellite pixel has 3 reflectance values: `[Band 2 Green, Band 3 Red, Band 4 NIR]`.
Imagine those 3 numbers as a **3D arrow pointing in space** (a vector):
* If the real ground arrow points in direction A, and the GAN predicted arrow points in direction B, **SAM is the angle between those two arrows**.

If a cloud covers the ground, it turns all bands white, causing a large spectral angle error ($> 15^\circ$).
When our GAN removes the cloud, the spectral angle drops down toward $0^\circ$, proving that the GAN **didn't just paint random colors—it restored the true optical spectral signature**.
* **Our Result**: Cloud cover creates an error of **$15.97^\circ$**, which our GAN cuts down to **$9.16^\circ$** (a **$-6.82^\circ$ reduction in spectral error**).

---

## 4. NDVI Fidelity (Normalized Difference Vegetation Index)

* **Formula**: $\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} = \frac{B_4 - B_3}{B_4 + B_3}$
* **What it measures**: Agricultural validity and vegetation health.
* **Direction**: High Pearson Correlation ($r \to 1.0$) and Low Mean Absolute Error ($\text{MAE} \to 0$).

### Plain English Explanation:
NDVI is the official formula used by ISRO and agricultural ministries worldwide to monitor crops. Healthy green crops reflect high NIR and absorb Red, producing high NDVI ($> 0.5$). Clouds have near-zero NDVI.

If an AI hallucinates or invents fake terrain, the NDVI numbers will be corrupted, making the image useless for farmers.
By measuring NDVI correlation before and after inpainting, we prove that our restored image is **scientifically reliable for downstream agricultural and forestry applications**.

---

## Summary Slide Table (Ready to Copy into Presentation)

| Metric | Full Name | Unit / Range | What it Measures | Desired Trend |
| :--- | :--- | :--- | :--- | :--- |
| **PSNR** | Peak Signal-to-Noise Ratio | Decibels (dB) | Pixel-level reconstruction fidelity & noise reduction | **Higher ($\uparrow$)** |
| **SSIM** | Structural Similarity Index | $0.0 - 1.0$ | Edge preservation, field borders & spatial texture | **Higher ($\uparrow$)** |
| **SAM** | Spectral Angle Mapper | Degrees (°) | Spectral/color purity across LISS-IV VNIR bands | **Lower ($\downarrow$)** |
| **NDVI** | Normalized Difference Vegetation Index | $-1.0 - +1.0$ | Downstream scientific utility for agriculture/forestry | **High Correlation** |

---

## Your 20-Second Viva Answer (Practice Saying This Out Loud):

> *"To evaluate our LISS-IV Cloud Removal GAN, we did not rely on visual inspection alone. We used four quantitative metrics: **PSNR** and **SSIM** to measure reconstruction cleanliness and structural edge preservation; and two specialized remote sensing metrics—**SAM (Spectral Angle Mapper)** to verify that multispectral band angles are not distorted, and **NDVI fidelity** to confirm that the restored imagery remains scientifically usable for agricultural crop monitoring."*
