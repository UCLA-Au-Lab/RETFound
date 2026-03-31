import numpy as np
import torch
from PIL import Image


def preprocess_bscan(bscan: np.ndarray) -> np.ndarray:
    """
    Convert a single 2D B-scan to a (224, 224, 3) float32 array ready for
    the model.

    Accepts:
        (C, H, W)    – channel-first (C=1 or C=3)
        (H, W)       – grayscale
        (H, W, 3)    – channel-last RGB

    Returns:
        float32 array of shape (224, 224, 3), per-channel z-score normalised
    """
    arr = bscan.astype(np.float32)

    # channel-first → channel-last
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]  # (H, W, 1) → (H, W)

    if arr.ndim == 2:
        pil = Image.fromarray(
            np.clip(arr / arr.max() * 255, 0, 255).astype(np.uint8)
            if arr.max() > 0 else arr.astype(np.uint8),
            mode="L",
        ).convert("RGB")
    else:
        pil = Image.fromarray(
            np.clip(arr / arr.max() * 255, 0, 255).astype(np.uint8)
            if arr.max() > 0 else arr.astype(np.uint8)
        )

    pil = pil.resize((224, 224), Image.BICUBIC)
    img = np.array(pil).astype(np.float64) / 255.0

    for c in range(3):
        std = img[..., c].std()
        if std > 0:
            img[..., c] = (img[..., c] - img[..., c].mean()) / std
        else:
            img[..., c] = img[..., c] - img[..., c].mean()

    return img.astype(np.float32)


def _default_transform(bscan: np.ndarray) -> torch.Tensor:
    """preprocess_bscan → (3, 224, 224) float32 tensor."""
    return torch.from_numpy(preprocess_bscan(bscan)).permute(2, 0, 1)


class UCLA_b_scans(torch.utils.data.Dataset):
    def __init__(
        self,
        groups,
        oct_key="volume",
        emb_key="RETFound_mae_natureOCT",
        out_dims="d e",
        transforms=_default_transform,
    ):
        self.groups = groups
        self.oct_key = oct_key
        self.emb_key = emb_key
        self.out_dims = out_dims
        self.transforms = transforms

        self.num_bscans = np.zeros(len(groups), dtype=int)
        self.bscans = []

        for i, g in enumerate(self.groups):
            if self.oct_key not in g:
                raise KeyError(f"'{self.oct_key}' not found in group {g.name!r}")
            self.num_bscans[i] = g[self.oct_key].shape[1]
            if self.out_dims == "d e":
                g.require_array(
                    self.emb_key,
                    shape=(int(self.num_bscans[i]), 1024),
                    dtype=np.float32,
                    chunks=(1, 1024),
                    overwrite=False,
                )
            elif self.out_dims == "e":
                g.require_array(
                    self.emb_key,
                    shape=(1024,),
                    dtype=np.float32,
                    chunks=(1024,),
                    overwrite=False,
                )
            else:
                raise ValueError(f"Unsupported out_dims: {self.out_dims!r}")

            for b in range(self.num_bscans[i]):
                self.bscans.append((g, b))

    def __len__(self):
        return len(self.bscans)

    def __getitem__(self, idx):
        group, b = self.bscans[idx]

        b_scan = np.array(group[self.oct_key][:, b])  # materialise (C, H, W)

        if self.transforms is not None:
            b_scan = self.transforms(b_scan)

        return b_scan, group[self.emb_key], b
