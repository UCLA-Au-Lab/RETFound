import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from embed_oct_volumes import load_retfound_oct, embed_zarr_groups
import zarr
import torch
from pathlib import Path
import pickle


def main():

    oct_key = "volume"
    store_path = Path("/phi/hdd_drive0/UCLA_MP/volumetric/arrs.zarr")

    model = load_retfound_oct(torch.device("cuda"))
    
    if Path("groups.pkl").exists():
        print(f"Store path groups.pkl already exists. Loading existing groups...")
        with open("groups.pkl", "rb") as f:
            groups = pickle.load(f)
    else:

        groups = [zarr.open_group(p.parent.parent, mode="a") for p in store_path.glob(f"**/{oct_key}/zarr.json")]

        with open("groups.pkl", "wb") as f:
            pickle.dump(groups, f)
    
    embed_zarr_groups(
        groups=groups,
        model=model,
        oct_key=oct_key,
        emb_key="RETFound_mae_natureOCT",
        in_dims="c d h w",
        out_dims="d e",
        batch_size=64,
    )
    return groups


if __name__ == "__main__":
    main()
