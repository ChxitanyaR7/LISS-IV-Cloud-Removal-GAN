"""
PyTorch & NumPy Dataset and DataLoader for Paired LISS-IV Cloud Removal GAN Training.

Loads paired (Cloudy, Clear) .npy satellite patches, provides data augmentations
(random flips, rotations), scales tensors into [-1, 1] for standard cGANs / Pix2Pix,
and supports 4-channel guidance tensors [B2, B3, B4, Mask] for mask-conditioned GANs.

Works seamlessly with PyTorch (when installed) and provides pure NumPy fallbacks.

Usage:
    from dataset_loader import LISSPairDataset, get_dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=16)
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Union, Any
import numpy as np

# Safe PyTorch import with robust class fallbacks for static analyzers
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

    class Dataset:  # type: ignore[no-redef]
        """Fallback Dataset class when PyTorch is not installed."""
        def __len__(self) -> int:
            return 0

        def __getitem__(self, idx: int) -> Any:
            raise NotImplementedError

    class DataLoader:  # type: ignore[no-redef]
        """Fallback DataLoader class when PyTorch is not installed."""
        def __init__(self, dataset: Any, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, **kwargs: Any):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers


class LISSPairDataset(Dataset):
    """
    Paired Satellite Patch Dataset for Cloud Removal cGANs.
    Each item contains:
        cloudy_input: Tensor or ndarray of shape (3, H, W) or (4, H, W) with guidance mask
        clear_target: Tensor or ndarray of shape (3, H, W) [B2, B3, B4] (Ground Truth Target)
    """

    def __init__(
        self,
        clear_dir: Union[str, Path],
        cloudy_dir: Union[str, Path],
        indices: Optional[List[int]] = None,
        normalize_tanh: bool = True,
        augment: bool = False,
        return_guidance: bool = False
    ):
        self.clear_dir = Path(clear_dir)
        self.cloudy_dir = Path(cloudy_dir)
        self.normalize_tanh = normalize_tanh
        self.augment = augment
        self.return_guidance = return_guidance

        all_clears = sorted(list(self.clear_dir.glob("*.npy")))
        all_cloudies = sorted(list(self.cloudy_dir.glob("*.npy")))

        if not all_clears:
            raise FileNotFoundError(f"No .npy patches found in: {self.clear_dir.resolve()}")
        if not all_cloudies:
            raise FileNotFoundError(f"No .npy patches found in: {self.cloudy_dir.resolve()}")

        assert len(all_clears) == len(all_cloudies), (
            f"Mismatch: found {len(all_clears)} clear patches and {len(all_cloudies)} cloudy patches!"
        )

        if indices is not None:
            self.clear_files = [all_clears[i] for i in indices]
            self.cloudy_files = [all_cloudies[i] for i in indices]
        else:
            self.clear_files = all_clears
            self.cloudy_files = all_cloudies

    def __len__(self) -> int:
        return len(self.clear_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        clear = np.load(self.clear_files[idx]).astype(np.float32)    # (3, H, W) in [0, 1]
        cloudy = np.load(self.cloudy_files[idx]).astype(np.float32)  # (3, H, W) in [0, 1]

        # Data Augmentations (applied identically to both paired patches)
        if self.augment:
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                clear = np.flip(clear, axis=2).copy()
                cloudy = np.flip(cloudy, axis=2).copy()

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                clear = np.flip(clear, axis=1).copy()
                cloudy = np.flip(cloudy, axis=1).copy()

            # Random 90-degree Rotation
            k = np.random.randint(0, 4)
            if k > 0:
                clear = np.rot90(clear, k=k, axes=(1, 2)).copy()
                cloudy = np.rot90(cloudy, k=k, axes=(1, 2)).copy()

        # Optional 4-channel guidance: estimate cloud mask and concatenate
        if self.return_guidance:
            # Simple physical spectral whiteness proxy for cloud mask
            mean_intensity = (cloudy[0] + cloudy[1] + cloudy[2]) / 3.0
            spread = (np.abs(cloudy[0] - mean_intensity) + np.abs(cloudy[1] - mean_intensity) + np.abs(cloudy[2] - mean_intensity)) / 3.0
            whiteness = 1.0 - np.clip(spread / 0.22, 0.0, 1.0)
            brightness = np.clip((mean_intensity - 0.40) / 0.60, 0.0, 1.0)
            cloud_mask = np.clip(brightness * whiteness, 0.0, 1.0)
            mask_channel = np.expand_dims(cloud_mask, axis=0)  # (1, H, W)
            cloudy_input = np.concatenate([cloudy, mask_channel], axis=0)  # (4, H, W)
        else:
            cloudy_input = cloudy

        # Scale from [0, 1] to [-1, 1] for tanh generator outputs
        if self.normalize_tanh:
            clear = (clear * 2.0) - 1.0
            if self.return_guidance:
                cloudy_input[:3] = (cloudy_input[:3] * 2.0) - 1.0
                # Mask channel stays in [0, 1]
            else:
                cloudy_input = (cloudy_input * 2.0) - 1.0

        if TORCH_AVAILABLE and torch is not None:
            return torch.from_numpy(cloudy_input), torch.from_numpy(clear)
        return cloudy_input, clear


def get_dataloaders(
    clear_dir: Optional[Union[str, Path]] = None,
    cloudy_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    return_guidance: bool = False
) -> Tuple[Any, Any]:
    """
    Creates train and validation DataLoaders with reproducible split.
    Works with PyTorch when available, or returns pure NumPy generators.
    """
    root = Path(__file__).resolve().parent.parent
    if clear_dir is None:
        clear_dir = root / "Data" / "Clear_Patches"
    if cloudy_dir is None:
        cloudy_dir = root / "Data" / "Cloudy_Patches"

    clear_dir = Path(clear_dir)
    cloudy_dir = Path(cloudy_dir)

    all_clears = list(clear_dir.glob("*.npy"))
    total = len(all_clears)
    if total == 0:
        raise FileNotFoundError(f"No patches found in {clear_dir.resolve()}. Run patch generation first.")

    np.random.seed(seed)
    shuffled_indices = np.random.permutation(total).tolist()
    split = int(total * (1.0 - val_ratio))

    train_indices = shuffled_indices[:split]
    val_indices = shuffled_indices[split:]

    train_ds = LISSPairDataset(clear_dir, cloudy_dir, indices=train_indices, augment=True, return_guidance=return_guidance)
    val_ds = LISSPairDataset(clear_dir, cloudy_dir, indices=val_indices, augment=False, return_guidance=return_guidance)

    if TORCH_AVAILABLE and DataLoader is not None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"[*] PyTorch DataLoader initialized: {len(train_ds)} train samples, {len(val_ds)} validation samples.")
        return train_loader, val_loader
    else:
        print(f"[*] NumPy Dataset initialized: {len(train_ds)} train samples, {len(val_ds)} validation samples.")
        return train_ds, val_ds


def get_batch_numpy(
    clear_dir: Optional[Union[str, Path]] = None,
    cloudy_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    return_guidance: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick zero-dependency helper to grab a single batch of (cloudy, clear) arrays.
    """
    root = Path(__file__).resolve().parent.parent
    c_dir = Path(clear_dir) if clear_dir else root / "Data" / "Clear_Patches"
    cl_dir = Path(cloudy_dir) if cloudy_dir else root / "Data" / "Cloudy_Patches"

    ds = LISSPairDataset(c_dir, cl_dir, augment=False, return_guidance=return_guidance)
    indices = np.random.choice(len(ds), size=min(batch_size, len(ds)), replace=False)

    batch_in = []
    batch_out = []
    for idx in indices:
        inp, tgt = ds[idx]
        batch_in.append(inp.numpy() if hasattr(inp, "numpy") else inp)
        batch_out.append(tgt.numpy() if hasattr(tgt, "numpy") else tgt)

    return np.stack(batch_in, axis=0), np.stack(batch_out, axis=0)


if __name__ == "__main__":
    print(f"[*] Testing dataset_loader.py (PyTorch Available: {TORCH_AVAILABLE})")
    root = Path(__file__).resolve().parent.parent
    clear_p = root / "Data" / "Clear_Patches"
    cloudy_p = root / "Data" / "Cloudy_Patches"

    if clear_p.exists() and any(clear_p.glob("*.npy")):
        ds = LISSPairDataset(clear_p, cloudy_p, normalize_tanh=True, return_guidance=True)
        cloudy, clear = ds[0]
        c_shape = cloudy.shape
        cl_shape = clear.shape
        print(f"[OK] LISSPairDataset item loaded successfully!")
        print(f"     Cloudy Guidance Tensor: shape={c_shape} (4 channels: [B2, B3, B4, Mask])")
        print(f"     Clear Target Tensor   : shape={cl_shape} (3 channels: [B2, B3, B4])")

        # Test single batch extraction
        b_in, b_out = get_batch_numpy(clear_p, cloudy_p, batch_size=4, return_guidance=True)
        print(f"[OK] Batch test: Input Batch={b_in.shape}, Target Batch={b_out.shape}")
    else:
        print("[!] Clear_Patches directory is empty. Generate patches first.")
