# LISS-IV Cloud Removal GAN

Automated dataset generation pipeline, multi-octave synthetic cloud simulation, and cGAN (Pix2Pix) training framework for ResourceSat-2 / 2A **LISS-IV multispectral satellite imagery**.

---

## 📁 Repository Structure at a Glance

The codebase is organized into dedicated, focused directories so you can easily locate what you need:

```
LISS-IV-Cloud-Removal-GAN/
│
├── notebooks/                         # 📓 Interactive Jupyter Notebooks
│   ├── 01_explore_and_inspect.ipynb   # Step 1: Band reading, statistics & FCC composite visualization
│   ├── 02_patch_generation_pipeline.ipynb # Step 2: Automated patch extraction & synthetic cloud generation
│   ├── 03_preview_and_verify.ipynb    # Step 3: Dataset inspection, histogram comparisons & PyTorch loader test
│   └── Reading-tiff.ipynb             # Preserved original experimental notebook
│
├── scripts/                           # ⚙️ Standalone Python Scripts (CLI & Modules)
│   ├── generate_patches.py            # End-to-end pipeline CLI: ZIPs -> Normalized paired patches
│   ├── cloud_simulator.py             # Realistic multi-octave cloud mask generation & blending
│   ├── preview_patches.py             # Quick CLI tool to view False Color Composite (FCC) samples
│   └── dataset_loader.py              # PyTorch Dataset & DataLoader class for GAN training
│
├── notes/                             # 📝 Research Notes & Experimental Logs
│   ├── scratch_cells.txt              # Preserved Colab/interactive experiment cells
│   └── README.md                      # Guide explaining the experimental cells
│
├── Data/                              # 🛰️ Datasets (Ignored by Git)
│   ├── Raw_Zips/                      # Raw LISS-IV scene archives (.zip)
│   ├── Clear_Patches/                 # 256x256 ground truth clear patches (.npy, shape: 3, 256, 256)
│   ├── Cloudy_Patches/                # 256x256 paired synthetic cloudy patches (.npy, shape: 3, 256, 256)
│   ├── Normalized_Full_Scenes/        # Normalized full scenes and overview FCC PNGs
│   ├── Preview_Images/                # Generated preview figures
│   └── temp_extract/                  # Temporary working directory for unzipping
│
├── .gitignore                         # Configured to ignore data arrays, caches & checkpoints
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation and quickstart guide
```

---

## 🔍 Where to Find What You Need

| Task | Where to Go | Command / Notebook |
| :--- | :--- | :--- |
| **Inspect raw GeoTIFF bands & FCC** | `notebooks/` | [01_explore_and_inspect.ipynb](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/notebooks/01_explore_and_inspect.ipynb) |
| **Run patch extraction interactively** | `notebooks/` | [02_patch_generation_pipeline.ipynb](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/notebooks/02_patch_generation_pipeline.ipynb) |
| **Visually verify generated patches** | `notebooks/` | [03_preview_and_verify.ipynb](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/notebooks/03_preview_and_verify.ipynb) |
| **Run automated extraction from Terminal** | `scripts/` | `python scripts/generate_patches.py` |
| **Quickly inspect patches from Terminal** | `scripts/` | `python scripts/preview_patches.py --num_samples 4 --save` |
| **Modify or test cloud simulation logic** | `scripts/` | [cloud_simulator.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/cloud_simulator.py) |
| **Load patches into PyTorch for GANs** | `scripts/` | [dataset_loader.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/dataset_loader.py) |
| **Original notebook archive** | `notebooks/` | [Reading-tiff.ipynb](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/notebooks/Reading-tiff.ipynb) |

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Activate your Python environment and install dependencies:

```bash
# Using existing .venv
.\.venv\Scripts\activate

# Or install dependencies
pip install -r requirements.txt
```

### 2. Generate Paired Patches (CLI)
To run the automated extraction on all ZIP archives in `Data/Raw_Zips/`:

```bash
python scripts/generate_patches.py --patch_size 256 --stride 128
```

Available options:
- `--raw_dir`: Path to raw ZIP archives (default: `Data/Raw_Zips`)
- `--clear_dir`: Output path for clear ground truth patches (default: `Data/Clear_Patches`)
- `--cloudy_dir`: Output path for synthetic cloudy patches (default: `Data/Cloudy_Patches`)
- `--patch_size`: Patch dimension in pixels (default: `256`)
- `--stride`: Sliding window step (default: `128`)
- `--valid_threshold`: Non-zero pixel coverage threshold (default: `0.90`)
- `--min_std`: Minimum standard deviation to reject uniform/flat patches (default: `0.05`)
- `--min_mean`: Minimum mean brightness to reject dark borders (default: `0.15`)
- `--max_patches`: Optional cap on total patches to extract

### 3. Preview & Verify Patches
Inspect the generated patch pairs in False Color Composite (FCC: B4 NIR, B3 Red, B2 Green):

```bash
# Display 4 samples interactively and save composite image
python scripts/preview_patches.py --num_samples 4 --save

# Randomly select 6 patch pairs
python scripts/preview_patches.py --num_samples 6 --random --save
```

Preview images are saved to `Data/Preview_Images/`.

### 4. Load Dataset in PyTorch for GAN Training
Import the ready-to-use dataset and dataloaders in your model training scripts:

```python
from scripts.dataset_loader import get_dataloaders

# Automatically loads from Data/Clear_Patches and Data/Cloudy_Patches
train_loader, val_loader = get_dataloaders(batch_size=16, val_ratio=0.1)

for batch_idx, (cloudy, clear) in enumerate(train_loader):
    # cloudy: Tensor of shape (B, 3, 256, 256) scaled to [-1, 1]
    # clear:  Tensor of shape (B, 3, 256, 256) scaled to [-1, 1]
    ...
```

---

## 🛰️ Sensor Specifications & False Color Composite (FCC)

ResourceSat LISS-IV sensor bands:
- **Band 2 (B2)**: Green (0.52 - 0.59 µm)
- **Band 3 (B3)**: Red (0.62 - 0.68 µm)
- **Band 4 (B4)**: Near-Infrared / NIR (0.77 - 0.86 µm)

In standard Remote Sensing False Color Composite (FCC):
- **Red Display Channel** = Band 4 (NIR)
- **Green Display Channel** = Band 3 (Red)
- **Blue Display Channel** = Band 2 (Green)

Healthy vegetation strongly reflects NIR radiation and appears vibrant red in FCC images.
