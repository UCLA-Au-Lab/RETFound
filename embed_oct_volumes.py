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
        [--in_dims "c d h w"]    # axis names of the OCT array
        [--out_dims "e"]         # "e" → (1024,) per volume; "d e" → (D, 1024) per volume
        [--batch_size 32]
        [--device cuda]

Outputs
-------
    embeddings.npz  – numpy archive with keys:
        "embeddings"   : float32 – shape (N, 1024) for out_dims="e",
                                         (N, D, 1024) for out_dims="d e"
        "volume_names" : object  (N,)   – zarr path strings
        "out_dims"     : str            – the out_dims used
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import zarr
from einops import rearrange
from functools import partial
from huggingface_hub import hf_hub_download
from tqdm import tqdm

import models_vit as models
from ucla_dataset import preprocess_bscan, UCLA_b_scans


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
# Embedding
# ---------------------------------------------------------------------------

class _BscanDataset(torch.utils.data.Dataset):
    """Internal dataset for embed_volume: wraps a single (D, *) volume."""

    def __init__(self, vol_d_first: np.ndarray) -> None:
        self.vol = vol_d_first

    def __len__(self) -> int:
        return self.vol.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = preprocess_bscan(self.vol[idx])            # (224, 224, 3)
        return torch.from_numpy(arr).permute(2, 0, 1)   # (3, 224, 224)


def _collate_bscans(batch):
    """Collate that stacks tensors but keeps zarr arrays as a plain list."""
    b_scans = torch.stack([item[0] for item in batch])
    emb_arrays = [item[1] for item in batch]
    b_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return b_scans, emb_arrays, b_indices


def embed_volume(
    model: nn.Module,
    vol: np.ndarray,
    in_dims: str,
    out_dims: str,
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
) -> np.ndarray:
    """
    Embed a single 3D OCT volume.

    Parameters
    ----------
    vol : np.ndarray
        Volume array whose axes match ``in_dims``.
    in_dims : str
        Einops-style axis names for ``vol``, e.g. ``"c d h w"``.
        Must contain ``'d'`` for the B-scan axis.
    out_dims : str
        Desired output shape:

        - ``"d e"`` → per-slice embeddings, shape ``(D, 1024)``
        - ``"e"``   → mean-pooled volume embedding, shape ``(1024,)``
    num_workers : int
        DataLoader worker processes for parallel B-scan preprocessing (default 4).

    Returns
    -------
    np.ndarray with shape matching ``out_dims`` (e = 1024).
    """
    in_axes = in_dims.split()
    out_axes = out_dims.split()

    if "d" not in in_axes:
        raise ValueError(f"'d' (B-scan axis) not found in in_dims='{in_dims}'")
    if "e" not in out_axes:
        raise ValueError(f"'e' (embedding axis) not found in out_dims='{out_dims}'")
    if set(out_axes) - {"d", "e"}:
        raise ValueError(f"out_dims='{out_dims}' contains unknown axes; only 'd' and 'e' are supported")

    # Put d first so we can iterate B-scans: "c d h w" -> "d c h w"
    remaining = [a for a in in_axes if a != "d"]
    vol_d_first = rearrange(vol, f'{in_dims} -> d {" ".join(remaining)}')

    loader = torch.utils.data.DataLoader(
        _BscanDataset(vol_d_first),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            latent = model.forward_features(x.float())
            all_embeddings.append(latent.squeeze(1).cpu().float().numpy())

    slice_embs = np.concatenate(all_embeddings, axis=0)  # (D, 1024)

    if out_axes == ["e"]:
        return slice_embs.mean(axis=0)   # (1024,)
    else:  # ["d", "e"]
        return slice_embs                # (D, 1024)


# ---------------------------------------------------------------------------
# Zarr-group-level API
# ---------------------------------------------------------------------------

def embed_zarr_groups(
    groups: list[zarr.Group],
    model: nn.Module,
    oct_key: str,
    emb_key: str,
    in_dims: str = "c d h w",
    out_dims: str = "d e",
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device | None = None,
    overwrite: bool = False,
) -> None:
    """
    Embed OCT volumes stored inside zarr groups and write results back in-place.

    For ``out_dims="d e"``, uses ``UCLA_b_scans`` to batch B-scans across all
    volumes simultaneously, writing each slice embedding back as it is produced.

    For ``out_dims="e"``, processes each volume independently and writes the
    mean-pooled ``(1024,)`` embedding when the volume is complete.

    Parameters
    ----------
    groups : list[zarr.Group]
        Open zarr groups, each containing an OCT volume at ``oct_key``.
        Must be opened with write access (mode ``"r+"`` or ``"a"``).
    model : nn.Module
        Loaded RETFound model (e.g. from ``load_retfound_oct``).
    oct_key : str
        Key of the OCT volume array inside each group, e.g. ``"oct"``.
        Array must have shape ``(C, D, H, W)``.
    emb_key : str
        Key under which the embedding array will be written, e.g. ``"oct_emb"``.
    in_dims : str
        Einops-style axis names for the OCT array (default ``"c d h w"``).
        Only used for ``out_dims="e"``; ``UCLA_b_scans`` always assumes axis 1 is D.
    out_dims : str
        - ``"d e"`` → per-slice ``(D, 1024)`` written incrementally  *(default)*
        - ``"e"``   → mean-pooled ``(1024,)`` written per volume
    batch_size : int
        B-scans per forward pass (default 32).
    num_workers : int
        DataLoader worker processes for B-scan preprocessing (default 4).
    device : torch.device, optional
        Defaults to CUDA if available, otherwise CPU.
    overwrite : bool
        If False (default), skip groups that already have ``emb_key``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pending = [g for g in groups if overwrite or emb_key not in g]
    skipped = len(groups) - len(pending)
    if skipped:
        print(f"Skipping {skipped} group(s) where '{emb_key}' already exists")
    if not pending:
        return

    if out_dims == "d e":
        dataset = UCLA_b_scans(pending, oct_key=oct_key, emb_key=emb_key, out_dims=out_dims)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate_bscans,
        )
        total_bscans = sum(dataset.num_bscans)
        with torch.no_grad():
            with tqdm(total=total_bscans, unit="b-scan", desc="Embedding") as pbar:
                for b_scans, emb_arrays, b_indices in loader:
                    x = b_scans.to(device)
                    latent = model.forward_features(x.float()).squeeze(1)  # (B, 1024)
                    latent_np = latent.cpu().float().numpy()
                    for emb, arr, b in zip(latent_np, emb_arrays, b_indices.tolist()):
                        arr[b] = emb
                    pbar.update(len(b_indices))

    else:  # "e"
        for group in tqdm(pending, unit="volume", desc="Embedding"):
            vol = np.array(group[oct_key])
            emb = embed_volume(model, vol, in_dims, out_dims, batch_size, device, num_workers)
            group[emb_key] = emb


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
    p.add_argument("--in_dims", default="c d h w",
                   help="Einops-style axis names for each OCT array (default: 'c d h w')")
    p.add_argument("--out_dims", default="e",
                   help="Output shape: 'e' for mean-pooled (1024,) or 'd e' for per-slice (D, 1024) "
                        "(default: 'e')")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Number of B-scans to process per forward pass")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker processes for B-scan preprocessing (default: 4)")
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

    embeddings = []
    volume_names = []

    for path in args.zarr_paths:
        print(f"Processing {path} …")
        vol = open_zarr_array(path, args.zarr_key)
        print(f"  Volume shape: {vol.shape}  dtype: {vol.dtype}")

        emb = embed_volume(model, vol, args.in_dims, args.out_dims, args.batch_size, device, args.num_workers)
        print(f"  → embedding: {emb.shape}")

        embeddings.append(emb)
        volume_names.append(path)

    save_dict = {
        "embeddings": np.array(embeddings),
        "volume_names": np.array(volume_names, dtype=object),
        "out_dims": args.out_dims,
    }

    np.savez(args.output, **save_dict)
    print(f"\nSaved embeddings to {args.output}")
    print(f"  embeddings shape: {np.array(embeddings).shape}")


if __name__ == "__main__":
    main()
