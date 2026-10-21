"""
PyTorch Dataset and DataLoader for Paired LISS-IV Cloud Removal GAN Training.

Loads paired (Cloudy, Clear) .npy satellite patches, provides data augmentations
(random flips, rotations), and scales tensors into [-1, 1] for standard cGANs / Pix2Pix.

Usage:
    from dataset_loader import LISSPairDataset, get_dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=16)
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


class LISSPairDataset(Dataset):
    """
    Paired Satellite Patch Dataset for Cloud Removal cGANs.
    Each item contains:
        cloudy_tensor: FloatTensor of shape (3, H, W) [B2, B3, B4] (Generator Input)
        clear_tensor:  FloatTensor of shape (3, H, W) [B2, B3, B4] (Ground Truth Target)
    """

    def __init__(
        self,
        clear_dir: Path,
        cloudy_dir: Path,
        indices: Optional[List[int]] = None,
        normalize_tanh: bool = True,
        augment: bool = False
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use LISSPairDataset. Please install torch: pip install torch")

        self.clear_dir = Path(clear_dir)
        self.cloudy_dir = Path(cloudy_dir)
        self.normalize_tanh = normalize_tanh
        self.augment = augment

        all_clears = sorted(list(self.clear_dir.glob("*.npy")))
        all_cloudies = sorted(list(self.cloudy_dir.glob("*.npy")))

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

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
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

        # Scale from [0, 1] to [-1, 1] for tanh generator outputs
        if self.normalize_tanh:
            clear = (clear * 2.0) - 1.0
            cloudy = (cloudy * 2.0) - 1.0

        return torch.from_numpy(cloudy), torch.from_numpy(clear)


def get_dataloaders(
    clear_dir: Optional[Path] = None,
    cloudy_dir: Optional[Path] = None,
    batch_size: int = 16,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple["DataLoader", "DataLoader"]:
    """
    Creates train and validation DataLoaders with reproducible split.
    """
    root = Path(__file__).resolve().parent.parent
    if clear_dir is None:
        clear_dir = root / "Data" / "Clear_Patches"
    if cloudy_dir is None:
        cloudy_dir = root / "Data" / "Cloudy_Patches"

    total = len(list(clear_dir.glob("*.npy")))
    if total == 0:
        raise FileNotFoundError(f"No patches found in {clear_dir.resolve()}. Run patch generation first.")

    np.random.seed(seed)
    shuffled_indices = np.random.permutation(total).tolist()
    split = int(total * (1.0 - val_ratio))

    train_indices = shuffled_indices[:split]
    val_indices = shuffled_indices[split:]

    train_ds = LISSPairDataset(clear_dir, cloudy_dir, indices=train_indices, augment=True)
    val_ds = LISSPairDataset(clear_dir, cloudy_dir, indices=val_indices, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[*] DataLoader initialized: {len(train_ds)} train samples, {len(val_ds)} validation samples.")
    return train_loader, val_loader


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("[!] PyTorch is not installed in the current environment. To install: pip install torch")
    else:
        root = Path(__file__).resolve().parent.parent
        clear_p = root / "Data" / "Clear_Patches"
        cloudy_p = root / "Data" / "Cloudy_Patches"

        if clear_p.exists() and any(clear_p.glob("*.npy")):
            ds = LISSPairDataset(clear_p, cloudy_p, normalize_tanh=True)
            cloudy, clear = ds[0]
            print(f"[OK] LISSPairDataset test successful!")
            print(f"    Cloudy shape: {cloudy.shape}, Range: [{cloudy.min():.2f}, {cloudy.max():.2f}]")
            print(f"    Clear shape:  {clear.shape}, Range: [{clear.min():.2f}, {clear.max():.2f}]")
        else:
            print("[!] Clear_Patches directory is empty. Generate patches first.")
