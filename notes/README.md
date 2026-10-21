# Research & Development Notes

This directory contains research scratchpads, interactive session dumps, and notes from experimental trials.

---

### Files:
- **`scratch_cells.txt`**: Raw code cells preserved from initial Google Colab and interactive Jupyter notebook explorations.
  - *Cells 50-52*: Initial raw band statistics and spatial window checks.
  - *Cells 53-54*: Scene-level vs patch-level normalization experiments and sliding window extraction.
  - *Cells 55-57*: False Color Composite (FCC) visualization and percentile stretch tests.

---

### Current Clean Equivalents:
All useful experimental logic from `scratch_cells.txt` has been promoted into production modules:
- Band normalization & patch extraction $\rightarrow$ [generate_patches.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/generate_patches.py)
- Cloud generation $\rightarrow$ [cloud_simulator.py](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/scripts/cloud_simulator.py)
- Interactive inspection & visualization $\rightarrow$ [notebooks/](file:///c:/Projects/Major-project/LISS-IV-Cloud-Removal-GAN/notebooks)
