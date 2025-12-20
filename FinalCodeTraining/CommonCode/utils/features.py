import re, json, pandas as pd
from config import COMBINED_CSV_NAME
import torch
import torch.nn as nn

_res_pat  = re.compile(r"^feat_(\d+)_resnet$")
_vgg_pat  = re.compile(r"^feat_(\d+)_vgg$")
_bare_pat = re.compile(r"^feat_(\d+)$")  # treat as VGG extras

def harmonize_vgg_cols(df: pd.DataFrame) -> pd.DataFrame:
    if any(_vgg_pat.match(c) for c in df.columns):
        ren = {c: f"{c}_vgg" for c in df.columns if _bare_pat.match(c)}
        if ren: df = df.rename(columns=ren)
    return df

def pick_ordered_feature_cols(df, use_resnet=True, use_vgg=True):
    res_cols, vgg_cols = [], []
    if use_resnet:
        pairs = [(int(m.group(1)), c) for c in df.columns if (m:=_res_pat.match(c))]
        res_cols = [c for _,c in sorted(pairs)]
    if use_vgg:
        pairs = []
        for c in df.columns:
            m = _vgg_pat.match(c) or _bare_pat.match(c)
            if m: pairs.append((int(m.group(1)), c))
        idx_to_col = {}
        for idx, col in sorted(pairs):
            if idx not in idx_to_col or col.endswith("_vgg"):
                idx_to_col[idx] = col
        vgg_cols = [idx_to_col[i] for i in sorted(idx_to_col.keys())]
    return res_cols + vgg_cols

def pick_present_aus(df, au_master):
    return [c for c in au_master if c in df.columns]

def save_feature_cols(cols, path):
    with open(path, "w") as f: json.dump(cols, f)

def load_feature_cols(path):
    with open(path) as f: return json.load(f)



class Standardize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std",  torch.clamp(std, min=1e-6))
    def forward(self, x):  # (B,D)
        return (x - self.mean) / self.std

@torch.no_grad()
def compute_train_mean_std(loader, device):
    cnt = 0
    mean = None
    m2 = None
    for x, _ in loader:
        x = x.to(device).float()  # (B,D)
        n  = x.shape[0]
        bm = x.mean(dim=0)
        bM2 = ((x - bm)**2).sum(dim=0)  # batch sum of squares about its mean
        if mean is None:
            mean = bm
            m2   = bM2
            cnt  = n
        else:
            delta = bm - mean
            new_cnt = cnt + n
            mean = mean + delta * (n / new_cnt)
            m2 = m2 + bM2 + (delta**2) * (cnt * n / new_cnt)
            cnt = new_cnt
    var = m2 / max(1, (cnt - 1))
    std = torch.sqrt(var + 1e-6)
    return mean.detach(), std.detach()


class L2SubsetNorm(nn.Module):
    """L2-normalize selected feature subsets per sample. idx_groups: list[list[int]]"""
    def __init__(self, idx_groups, eps=1e-6):
        super().__init__()
        # store as buffers for device movement
        self.register_buffer("_dummy", torch.tensor(0.))  # for device
        self.idx_groups = [torch.as_tensor(g, dtype=torch.long) for g in idx_groups if len(g)]
        self.eps = eps

    def forward(self, x):  # x: (B,D)
        dev = x.device
        for g in self.idx_groups:
            g = g.to(dev)
            if g.numel() == 0: 
                continue
            chunk = x.index_select(1, g)                  # (B, |g|)
            denom = chunk.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)
            x[:, g] = chunk / denom
        return x


def build_preprocessor(norm_mode, feature_cols, device,
                       mean=None, std=None, keep_au_c_raw=True):
    """
    Returns a nn.Module that applies the selected normalization(s).
    - norm_mode: 'none' | 'l2' | 'zscore' | 'zscore+l2'
    - If zscore is requested, mean/std must be provided (compute on TRAIN).
    """
    steps = []

    # z-score (dimension-wise) first, if requested
    if norm_mode in ("zscore", "zscore+l2"):
        assert mean is not None and std is not None, "Provide mean/std for z-score."
        if keep_au_c_raw:
            auc_idx = [i for i, n in enumerate(feature_cols) if n.endswith("_c")]
            if auc_idx:
                mean = mean.clone(); std = std.clone()
                mean[auc_idx] = 0.0
                std[auc_idx]  = 1.0
        # Standardize class is already defined above in this file
        steps.append(Standardize(mean, std))

    # then L2 on CNN subsets, if requested
    if norm_mode in ("l2", "zscore+l2"):
        res_idx = [i for i, n in enumerate(feature_cols) if "_resnet" in n]
        vgg_idx = [i for i, n in enumerate(feature_cols)  if "_vgg" in n]
        steps.append(L2SubsetNorm([res_idx, vgg_idx]))

    if not steps:
        return nn.Identity().to(device)
    return nn.Sequential(*steps).to(device)

