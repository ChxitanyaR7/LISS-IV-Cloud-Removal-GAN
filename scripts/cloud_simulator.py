"""
Synthetic Cloud Simulation Module for LISS-IV Satellite Imagery.

Generates multi-octave Perlin/Gaussian-like continuous cloud masks and blends them
into 3-band satellite patches (B2: Green, B3: Red, B4: NIR) with realistic,
band-dependent optical transmission and reflectance.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def generate_realistic_cloud_mask(
    height: int,
    width: int,
    cloud_type: Optional[str] = None
) -> np.ndarray:
    """
    Generates a realistic continuous multi-octave cloud mask in range [0, 1].

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        cloud_type: One of 'wispy', 'bank', 'overcast', or None (random choice).

    Returns:
        np.ndarray of shape (height, width) with float32 values in [0, 1].
    """
    # Octave 1: Low frequency large cloud structure (4x4)
    n1 = np.random.uniform(0, 1, (4, 4)).astype(np.float32)
    m1 = cv2.resize(n1, (width, height), interpolation=cv2.INTER_CUBIC)

    # Octave 2: Medium frequency cloud bodies (8x8)
    n2 = np.random.uniform(0, 1, (8, 8)).astype(np.float32)
    m2 = cv2.resize(n2, (width, height), interpolation=cv2.INTER_CUBIC)

    # Octave 3: High frequency edge detail (16x16)
    n3 = np.random.uniform(0, 1, (16, 16)).astype(np.float32)
    m3 = cv2.resize(n3, (width, height), interpolation=cv2.INTER_CUBIC)

    # Multi-octave weighted blend
    raw_mask = 0.65 * m1 + 0.25 * m2 + 0.10 * m3
    raw_mask = cv2.GaussianBlur(raw_mask, (51, 51), 0)

    # Normalize base noise to [0, 1]
    r_min, r_max = raw_mask.min(), raw_mask.max()
    if r_max > r_min:
        raw_mask = (raw_mask - r_min) / (r_max - r_min)

    if cloud_type is None:
        cloud_type = np.random.choice(["wispy", "bank", "overcast"])

    if cloud_type == "wispy":
        # Cirrus / thin clouds: power curve attenuates thin haze
        mask = np.power(raw_mask, 2.0) * np.random.uniform(0.3, 0.6)
    elif cloud_type == "bank":
        # Directional cloud front / bank
        angle = np.random.uniform(0, 2 * np.pi)
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)
        gradient = np.cos(angle) * xx + np.sin(angle) * yy
        g_min, g_max = gradient.min(), gradient.max()
        gradient = (gradient - g_min) / (g_max - g_min + 1e-6)
        mask = raw_mask * gradient
        mask = np.power(mask, 1.4) * np.random.uniform(0.5, 0.85)
    else:  # 'overcast'
        # Thick cumulus coverage
        mask = np.power(raw_mask, 1.2) * np.random.uniform(0.4, 0.85)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def add_synthetic_clouds(
    clean_patch: np.ndarray,
    cloud_type: Optional[str] = None,
    custom_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Applies synthetic clouds to a clean 3-band satellite patch.
    Uses band-dependent cloud color reflectance:
        B2 (Green): ~0.80 - 0.92
        B3 (Red):   ~0.82 - 0.94
        B4 (NIR):   ~0.88 - 0.99 (Brightest in NIR)

    Args:
        clean_patch: np.ndarray of shape (3, H, W) in range [0, 1].
                     Channels: index 0 -> B2, 1 -> B3, 2 -> B4.
        cloud_type: Optional cloud geometry ('wispy', 'bank', 'overcast').
        custom_mask: Optional 2D mask (H, W). If None, generated automatically.

    Returns:
        np.ndarray of shape (3, H, W) with cloudy satellite patch.
    """
    c, h, w = clean_patch.shape
    if custom_mask is None:
        cloud_mask = generate_realistic_cloud_mask(h, w, cloud_type=cloud_type)
    else:
        cloud_mask = custom_mask

    alpha = np.expand_dims(cloud_mask, axis=0)  # Shape: (1, H, W)

    # Cloud reflectance profile across LISS-IV bands
    cloud_color = np.array([
        np.random.uniform(0.80, 0.92),  # B2 Green
        np.random.uniform(0.82, 0.94),  # B3 Red
        np.random.uniform(0.88, 0.99),  # B4 NIR
    ], dtype=np.float32)[:, None, None]  # Shape: (3, 1, 1)

    # Standard radiative transfer alpha blend: I_cloudy = (1 - alpha) * I_clean + alpha * I_cloud
    cloudy_patch = (1.0 - alpha) * clean_patch + alpha * cloud_color
    return np.clip(cloudy_patch, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("[Cloud Simulator] Generating demo cloud masks for all types...")
    types = ["wispy", "bank", "overcast"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for idx, ctype in enumerate(types):
        m = generate_realistic_cloud_mask(256, 256, cloud_type=ctype)
        axes[idx].imshow(m, cmap="gray", vmin=0, vmax=1)
        axes[idx].set_title(f"Type: {ctype.capitalize()}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()
