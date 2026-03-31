import torch
import numpy as np

class UCLA_b_scans(torch.utils.data.Dataset):
    def __init__(self, groups, oct_key="volume", emb_key="RETFound_mae_natureOCT", transforms=None):
        self.groups = groups
        self.oct_key = oct_key
        self.emb_key = emb_key

        self.num_bscans = np.zeros(len(groups), dtype=int)
        self.transforms = transforms

        self.bscans = []

        for i, g in enumerate(self.groups):
            if self.oct_key not in g:
                raise KeyError(f"'{self.oct_key}' not found in group {g.name!r}")
            self.num_bscans[i] = g[self.oct_key].shape[1]
            if self.out_dims == "d e":
                g.require_array(self.emb_key, shape=(int(self.num_bscans[i]), 1024), dtype=np.float32, chunks=(1, 1024), overwrite=False)
            elif self.out_dims == "e":
                g.require_array(self.emb_key, shape=(1024,), dtype=np.float32, chunks=(1024,), overwrite=False)
            else:
                raise ValueError(f"Unsupported out_dims: {self.out_dims!r}")
            
            for b in range(self.num_bscans[i]):
                self.bscans.append((g, b))


    def __len__(self):
        return len(self.bscans)

    def __getitem__(self, idx):
        
        group, b = self.bscans[idx]

        b_scan = group[self.oct_key][:, b]

        if self.transforms is not None:
            b_scan = self.transforms(b_scan)

        return b_scan, group[self.emb_key], b
        