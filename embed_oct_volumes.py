"""
embed_oct_volumes.py

Embed 3D OCT volumes (stored as zarr) using RETFound.

Each volume is processed by extracting 2D B-scans along the depth axis,
embedding each B-scan with RETFound_mae_natureOCT, then mean-pooling
across slices to produce a single 1024-dim vector per volume.

Usage
-----
    python embed_oct_volumes.py \
        --zarr_paths /data/vol1.zarr /data/vol2.zarr \
        --output embeddings.npz \
        [--zarr_key 0]           # zarr array key / path inside the store
        [--depth_axis 0]         # which axis is the B-scan stack
        [--batch_size 32]
        [--save_slice_embeddings] # also save per-slice (D, 1024) arrays
        [--device cuda]

Outputs
-------
    embeddings.npz  – numpy archive with keys:
        "volume_embeddings"  : float32 (N, 1024)  – one row per volume
        "volume_names"       : object  (N,)        – zarr path strings
        "slice_embeddings"   : float32 (N, D, 1024) (only with --save_slice_embeddings;
                               zero-padded to max depth if volumes differ in depth)
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import zarr
from functools import partial
from huggingface_hub import hf_hub_download
from PIL import Image

import models_vit as models


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_retfound_oct(device: torch.device) -> nn.Module:
    """Download and load RETFound_mae_natureOCT from HuggingFace."""
    chkpt = hf_hub_download(
        repo_id="YukunZhou/RETFound_mae_natureOCT",
        filename="RETFound_mae_natureOCT.pth",
    )
    model = models.RETFound_mae(
        img_size=224,
        num_classes=0,       # no classification head needed
        drop_path_rate=0,
        global_pool=True,
    )
    checkpoint = torch.load(chkpt, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(f"Loaded checkpoint: {msg}")
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_bscan(bscan: np.ndarray) -> np.ndarray:
    """
    Convert a single 2D B-scan to a (224, 224, 3) float32 array ready for
    the model.

    Accepts:
        (H, W)       – grayscale
        (H, W, 1)    – grayscale with channel dim
        (H, W, 3)    – already RGB / 3-channel

    Returns:
        float32 array of shape (224, 224, 3), per-channel z-score normalised
    """
    # --- normalise to uint8 range for PIL ---
    arr = bscan.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]

    if arr.ndim == 2:
        # grayscale → replicate to 3 channels
        pil = Image.fromarray(
            np.clip(arr / arr.max() * 255, 0, 255).astype(np.uint8)
            if arr.max() > 0 else arr.astype(np.uint8),
            mode="L",
        ).convert("RGB")
    else:
        # already 3-channel
        pil = Image.fromarray(
            np.clip(arr / arr.max() * 255, 0, 255).astype(np.uint8)
            if arr.max() > 0 else arr.astype(np.uint8)
        )

    pil = pil.resize((224, 224), Image.BICUBIC)
    img = np.array(pil).astype(np.float64) / 255.0  # (224, 224, 3) in [0, 1]

    # per-channel z-score (match latent_feature.ipynb)
    for c in range(3):
        std = img[..., c].std()
        if std > 0:
            img[..., c] = (img[..., c] - img[..., c].mean()) / std
        else:
            img[..., c] = img[..., c] - img[..., c].mean()

    return img.astype(np.float32)


def bscans_to_tensor(bscans: list[np.ndarray]) -> torch.Tensor:
    """Stack preprocessed (224,224,3) arrays into (B, 3, 224, 224) tensor."""
    arr = np.stack(bscans, axis=0)                  # (B, 224, 224, 3)
    tensor = torch.from_numpy(arr)
    tensor = tensor.permute(0, 3, 1, 2)             # (B, 3, 224, 224)
    return tensor


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_slices(
    model: nn.Module,
    slices: list[np.ndarray],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run forward_features on a list of preprocessed B-scans in batches.

    Returns
    -------
    float32 array of shape (len(slices), 1024)
    """
    all_embeddings = []

    for start in range(0, len(slices), batch_size):
        batch = slices[start : start + batch_size]
        x = bscans_to_tensor(batch).to(device)      # (B, 3, 224, 224)

        latent = model.forward_features(x.float())  # (B, 1, 1024) with global_pool
        latent = latent.squeeze(1)                   # (B, 1024)

        all_embeddings.append(latent.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)   # (D, 1024)


def embed_volume(
    model: nn.Module,
    vol: np.ndarray,
    depth_axis: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed a single 3D OCT volume.

    Returns
    -------
    volume_emb  : float32 (1024,)   – mean-pooled over slices
    slice_embs  : float32 (D, 1024) – per-slice embeddings
    """
    # Move depth axis to front so we can iterate
    if depth_axis != 0:
        vol = np.moveaxis(vol, depth_axis, 0)        # (D, H, W) or (D, H, W, C)

    slices_preprocessed = [preprocess_bscan(vol[i]) for i in range(vol.shape[0])]

    slice_embs = embed_slices(model, slices_preprocessed, batch_size, device)
    volume_emb = slice_embs.mean(axis=0)

    return volume_emb, slice_embs


# ---------------------------------------------------------------------------
# Zarr-group-level API
# ---------------------------------------------------------------------------

def embed_zarr_groups(
    groups: list[zarr.Group],
    model: nn.Module,
    oct_key: str,
    emb_key: str,
    depth_axis: int = 0,
    batch_size: int = 32,
    device: torch.device | None = None,
    overwrite: bool = False,
) -> None:
    """
    Embed OCT volumes stored inside zarr groups and write results back in-place.

    For each group, reads the array at ``oct_key``, embeds it with RETFound
    (mean-pooled across B-scans), and saves a float32 array of shape (1024,)
    under ``emb_key`` in the same group.

    Parameters
    ----------
    groups : list[zarr.Group]
        Open zarr groups, each containing an OCT volume at ``oct_key``.
        Groups must be opened with write access (mode "r+" or "a").
    model : nn.Module
        Loaded RETFound model (e.g. from ``load_retfound_oct``).
    oct_key : str
        Key of the OCT volume array inside each group, e.g. ``"oct"``.
    emb_key : str
        Key under which the (1024,) embedding will be written, e.g. ``"oct_emb"``.
    depth_axis : int
        Axis of the OCT array that indexes B-scans (default 0).
    batch_size : int
        B-scans per forward pass (default 32).
    device : torch.device, optional
        Defaults to CUDA if available, otherwise CPU.
    overwrite : bool
        If False (default), skip groups that already have ``emb_key``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, group in enumerate(groups):
        if emb_key in group and not overwrite:
            print(f"[{i}] skipping – '{emb_key}' already exists")
            continue

        if oct_key not in group:
            raise KeyError(f"[{i}] '{oct_key}' not found in group {group.name!r}")

        vol = np.array(group[oct_key])
        volume_emb, _ = embed_volume(model, vol, depth_axis, batch_size, device)

        group[emb_key] = volume_emb  # writes (1024,) float32 array
        print(f"[{i}] wrote {emb_key!r} {volume_emb.shape} → {group.name!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Embed 3D OCT zarr volumes with RETFound")
    p.add_argument("--zarr_paths", nargs="+", required=True,
                   help="Paths to zarr stores (one per volume, or a single store)")
    p.add_argument("--output", default="embeddings.npz",
                   help="Output .npz file path")
    p.add_argument("--zarr_key", default=None,
                   help="Key / path inside each zarr store (e.g. '0' or 'volume'). "
                        "If None, the root array is used.")
    p.add_argument("--depth_axis", type=int, default=0,
                   help="Axis index corresponding to the B-scan stack (default: 0)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Number of B-scans to process per forward pass")
    p.add_argument("--save_slice_embeddings", action="store_true",
                   help="Also save per-slice embeddings (zero-padded to max depth)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def open_zarr_array(path: str, key: str | None) -> np.ndarray:
    store = zarr.open(path, mode="r")
    if key is not None:
        arr = store[key]
    elif isinstance(store, zarr.Array):
        arr = store
    else:
        # group with a single array at root level — take first child
        keys = list(store.keys())
        if len(keys) == 1:
            arr = store[keys[0]]
        else:
            raise ValueError(
                f"zarr store at '{path}' is a Group with keys {keys}. "
                "Specify --zarr_key to pick one."
            )
    return np.array(arr)


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("Loading RETFound_mae_natureOCT …")
    model = load_retfound_oct(device)

    volume_embeddings = []
    slice_embeddings_list = []
    volume_names = []

    for path in args.zarr_paths:
        print(f"Processing {path} …")
        vol = open_zarr_array(path, args.zarr_key)
        print(f"  Volume shape: {vol.shape}  dtype: {vol.dtype}")

        vol_emb, slice_embs = embed_volume(
            model, vol, args.depth_axis, args.batch_size, device
        )
        print(f"  → slice embeddings: {slice_embs.shape}  volume embedding: {vol_emb.shape}")

        volume_embeddings.append(vol_emb)
        slice_embeddings_list.append(slice_embs)
        volume_names.append(path)

    volume_embeddings = np.stack(volume_embeddings, axis=0)  # (N, 1024)

    save_dict = {
        "volume_embeddings": volume_embeddings,
        "volume_names": np.array(volume_names, dtype=object),
    }

    if args.save_slice_embeddings:
        max_depth = max(s.shape[0] for s in slice_embeddings_list)
        padded = np.zeros((len(slice_embeddings_list), max_depth, 1024), dtype=np.float32)
        for i, s in enumerate(slice_embeddings_list):
            padded[i, : s.shape[0]] = s
        save_dict["slice_embeddings"] = padded

    np.savez(args.output, **save_dict)
    print(f"\nSaved embeddings to {args.output}")
    print(f"  volume_embeddings shape: {volume_embeddings.shape}")


if __name__ == "__main__":
    main()
