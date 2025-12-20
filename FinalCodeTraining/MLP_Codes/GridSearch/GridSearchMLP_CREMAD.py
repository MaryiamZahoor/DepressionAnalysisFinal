# grid_search_cv.py
# K-fold CV on TRAIN+VAL, pick best config, retrain on TRAIN+VAL, final eval on TEST
# Uses: per-video .npy caches, fast DataLoader (workers+pin+prefetch),
# non_blocking=True copies, feature selection over a master feature list,
# VIDEO-LEVEL validation accuracy per fold + per-config aggregates,
# and PER-EPOCH TIMINGS (train/val data+compute times saved to epoch_times.json).

import os
import json
import math
from typing import List, Tuple, Optional
import gc
import time  # <- timing

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# ---- project imports ----
import config as CFG
from models.TwoLayerMLP import FrameClassifier
from utils.features import save_feature_cols, Standardize, harmonize_vgg_cols
from data.datasets import build_au_master  # used to force-in AU names

# ======================
#      USER CONSTANTS
# ======================

# CV & training
K_FOLDS        = 10
MAX_EPOCHS     = CFG.EPOCHS
ES_PATIENCE    = 15
ES_MONITOR     = "val_acc"           # "val_acc" or "val_loss"
SKIP_FIRST_N   = CFG.SKIP_FRAME
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataloader perf
CPU_COUNT = os.cpu_count() or 4
NUM_WORKERS = min(8, max(0, CPU_COUNT - 2))  # set 0 for quick debugging
PIN_MEMORY  = torch.cuda.is_available()
PREFETCH_FACTOR = 4  # used only when NUM_WORKERS > 0

def _loader_kws():
    base = dict(
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    if NUM_WORKERS > 0:
        base.update(dict(
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True,
        ))
    return base

def _safe_collate(batch):
    xs, ys = zip(*batch)
    X = torch.stack(xs, 0)                      # pinning happens in main process
    y = torch.tensor(ys, dtype=torch.long)
    return X, y

def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _current_lr(optim):
    return optim.param_groups[0]['lr'] if optim.param_groups else float('nan')

# ---------- Paths / lists ----------
SPLIT_PATH     = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/"
TRAIN_LIST     = os.path.join(SPLIT_PATH, "train_videos_full.txt")
VAL_LIST       = os.path.join(SPLIT_PATH, "val_videos_full.txt")
TEST_LIST      = os.path.join(SPLIT_PATH, "test_videos_full.txt")
BEST_MODEL_TRAINED_final="/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/cremad_GridSearch_unscaled_MLP/bestModel_std/"
select="cv_mean_acc"    #cv_mean_acc    cv_mean_video_acc
total_epochs=100
Model_name= str(total_epochs)+"_epochs_"+ select.split("_")[2]
SCALER_PATH_FINAL=os.path.join( BEST_MODEL_TRAINED_final,Model_name, "scaler.pt")
os.makedirs(os.path.dirname(SCALER_PATH_FINAL), exist_ok=True)
FEATCOLS_FINAL= os.path.join(BEST_MODEL_TRAINED_final,Model_name, "feature_cols.json")
os.makedirs(os.path.dirname(FEATCOLS_FINAL), exist_ok=True)
BEST_WEIGHTS_PATH_FINAL=os.path.join( BEST_MODEL_TRAINED_final,Model_name, "best.pt")
os.makedirs(os.path.dirname(BEST_WEIGHTS_PATH_FINAL), exist_ok=True)
INCLUDE_LIST = None
EXCLUDE_LIST = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/exclude_videos.txt"

# ---------- Labels / columns ----------
LABEL_COL          = getattr(CFG, "SPLIT_LABEL_COL", "Actual_Emotion")
COMBINED_CSV_NAME  = getattr(CFG, "COMBINED_CSV_NAME", "affwild_resnet_au_vgg_with_gt.csv")
EMOTION_TO_IDX     = getattr(CFG, "emotion_to_idx")
IDX_TO_EMO         = {v: k for k, v in EMOTION_TO_IDX.items()}

# ---------- Artifacts layout ----------
ART_DIR_TAG = "cremad_GridSearch_unscaled_MLP"
PROJECT_DIR = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/"
ART_DIR_SUB = os.path.join(PROJECT_DIR, "artifacts", ART_DIR_TAG)
os.makedirs(ART_DIR_SUB, exist_ok=True)

GRID_OUT_DIR = os.path.join(ART_DIR_SUB, "grid_cv")   # tables, caches, master features
os.makedirs(GRID_OUT_DIR, exist_ok=True)

CONFIGS_DIR = os.path.join(ART_DIR_SUB, "configs")    # per-config parent; folds inside
os.makedirs(CONFIGS_DIR, exist_ok=True)

IDS_LABELS_CACHE     = os.path.join(GRID_OUT_DIR, "ids_labels.json")
MASTER_FEATURES_JSON = os.path.join(GRID_OUT_DIR, "master_feature_cols.json")
MASTER_SCAN_LIMIT    = 10  # how many CSVs to peek at to build the master

# Standardization
DO_STANDARDIZE = True             # False = raw features; True = z-score on TRAIN only
KEEP_AU_C_RAW  = True                # *_c columns left unscaled if DO_STANDARDIZE

# Search space
ARCH_GRID = [
    {"hidden_dim": 256,  "hidden_dim2": None},
    {"hidden_dim": 512,  "hidden_dim2": None},
    {"hidden_dim": 1024, "hidden_dim2": None},
    {"hidden_dim": 1024, "hidden_dim2": 512},
    {"hidden_dim": 512,  "hidden_dim2": 256},
]
OPTIMIZERS     = ["adam"]
LRS            = [1e-4]
WEIGHT_DECAY   = [1e-5]
DROPOUTS       = [0.5]
BATCH_SIZES    = [512]

# Feature-set variants
FEATURE_SETS = [
    {"name":"VGG",                 "use_vgg":True,  "use_resnet":False, "use_au_c":False, "use_au_r":False},
    {"name":"RESNET",              "use_vgg":False, "use_resnet":True,  "use_au_c":False, "use_au_r":False},
    {"name":"VGG+RESNET",          "use_vgg":True,  "use_resnet":True,  "use_au_c":False, "use_au_r":False},
    {"name":"VGG+AU",              "use_vgg":True,  "use_resnet":False, "use_au_c":True,  "use_au_r":True },
    {"name":"RESNET+AU",           "use_vgg":False, "use_resnet":True,  "use_au_c":True,  "use_au_r":True },
    {"name":"VGG+RESNET+AU",       "use_vgg":True,  "use_resnet":True,  "use_au_c":True,  "use_au_r":True },
]

# Resume/Skip behavior
RESUME_FOLDS   = True   # resume fold from last_ckpt.pt when available
SKIP_DONE_FOLDS= True   # skip fold if metrics.json says done:true

# ======================
#        HELPERS
# ======================

def _require_file(path, desc):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")
    return path

def _read_ids(list_path: str) -> List[str]:
    with open(list_path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def _apply_include_exclude(ids: List[str]) -> List[str]:
    s = set(ids)
    if INCLUDE_LIST and os.path.isfile(INCLUDE_LIST):
        with open(INCLUDE_LIST) as f:
            inc = {ln.strip() for ln in f if ln.strip()}
        s &= inc
    if EXCLUDE_LIST and os.path.isfile(EXCLUDE_LIST):
        with open(EXCLUDE_LIST) as f:
            exc = {ln.strip() for ln in f if ln.strip()}
        s -= exc
    return sorted(s)

def _scan_labels(video_ids: List[str]) -> Tuple[List[str], np.ndarray]:
    """Return (ids, y_idx) by reading mode label from each video's CSV."""
    ids_out, y_out = [], []
    for (i,vid) in enumerate(video_ids):
        csvp = os.path.join(CFG.OUTPUT_DIR, vid, COMBINED_CSV_NAME)
        if not os.path.isfile(csvp):
            continue
        try:
            s = pd.read_csv(csvp, usecols=[LABEL_COL])[LABEL_COL]
        except Exception:
            continue
        s = s.dropna().astype(str).str.upper()
        if s.empty:
            continue
        lab = s.mode().iat[0]
        if lab not in EMOTION_TO_IDX:
            continue
        ids_out.append(vid)
        y_out.append(EMOTION_TO_IDX[lab])
        if i%50==0:
            print("done: ", i)
    if not ids_out:
        raise RuntimeError("No labeled videos found in provided lists.")
    return ids_out, np.array(y_out, dtype=int)

# ======================
#   MASTER FEATURE LIST + PER-VIDEO CACHES
# ======================

MASTER_FEATURE_COLS: Optional[List[str]] = None

def _vid_csv_path(vid: str) -> str:
    return os.path.join(CFG.OUTPUT_DIR, vid, COMBINED_CSV_NAME)

def _cache_paths(vid: str):
    cdir = os.path.join(CFG.OUTPUT_DIR, vid, "cache")
    return (
        os.path.join(cdir, "X.npy"),
        os.path.join(cdir, "y.npy"),
        os.path.join(cdir, "meta.json"),
    )

def _append_unique(dst_list, names_iterable):
    seen = set(dst_list)
    for n in names_iterable:
        if n not in seen:
            dst_list.append(n); seen.add(n)

def get_master_feature_cols(ids_all: List[str], n_scan=MASTER_SCAN_LIMIT) -> List[str]:
    if os.path.isfile(MASTER_FEATURES_JSON):
        with open(MASTER_FEATURES_JSON, "r") as f:
            cols = json.load(f)
        print(f"[features] Loaded master list ({len(cols)}) → {MASTER_FEATURES_JSON}")
        return cols

    picked = 0
    master: List[str] = []
    for vid in ids_all:
        csvp = _vid_csv_path(vid)
        if not os.path.isfile(csvp):
            continue
        try:
            df = pd.read_csv(csvp, nrows=1)
            df = harmonize_vgg_cols(df)
            cnn_cols = [c for c in df.columns if c.endswith("_vgg") or c.endswith("_resnet")]
            _append_unique(master, cnn_cols)
            picked += 1
            if picked >= n_scan:
                break
        except Exception as e:
            print(f"[features] warn: header scan failed for {csvp}: {e}")

    # include AUs in canonical order
    au_all = build_au_master(True, True)  # both _c and _r variants
    _append_unique(master, list(au_all))

    with open(MASTER_FEATURES_JSON, "w") as f:
        json.dump(master, f, indent=2)
    print(f"[features] Saved master list ({len(master)}) → {MASTER_FEATURES_JSON}")
    return master

def build_video_cache_master(vid: str, master_feature_cols: List[str],
                             label_col: str, skip_first_n: int) -> None:
    """Create X.npy and y.npy in the master column order (once per video)."""
    xnp, ynp, meta = _cache_paths(vid)
    os.makedirs(os.path.dirname(xnp), exist_ok=True)
    if os.path.isfile(xnp) and os.path.isfile(ynp) and os.path.isfile(meta):
        return

    csvp = _vid_csv_path(vid)
    if not os.path.isfile(csvp):
        np.save(xnp, np.empty((0, len(master_feature_cols)), dtype=np.float32))
        np.save(ynp, np.empty((0,), dtype=np.int64))
        with open(meta, "w") as f: json.dump({"feature_cols": master_feature_cols}, f)
        return

    df = pd.read_csv(csvp)
    df = harmonize_vgg_cols(df)
    if skip_first_n > 0:
        df = df.iloc[skip_first_n:].reset_index(drop=True)

    # labels (map to idx, drop NaNs)
    y = (df[label_col].astype(str).str.upper()
         .map(EMOTION_TO_IDX).dropna().astype(np.int64))
    idx = y.index

    # numeric matrix strictly in master order; missing/non-numeric -> zeros
    series_list = []
    for col in master_feature_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            series_list.append(df.loc[idx, col])
        else:
            series_list.append(pd.Series(0.0, index=idx))
    X = (pd.concat(series_list, axis=1)
           .replace([np.inf, -np.inf], np.nan)
           .fillna(0.0)
           .astype("float32")
           .to_numpy(copy=True))  # contiguous, writable

    np.save(xnp, X)
    np.save(ynp, y.to_numpy(copy=True))
    with open(meta, "w") as f:
        json.dump({"feature_cols": master_feature_cols}, f)

# ======================
#   CACHED DATASET (select feature subset per config)
# ======================
class CachedFrameDatasetMaster(torch.utils.data.Dataset):
    """
    Concatenates frames from per-video X.npy / y.npy stored in master feature order.
    For each config, we select a column index subset (VGG/RESNET/AU) on the fly.
    """
    def __init__(self, video_ids: List[str],
                 master_feature_cols: List[str],
                 use_vgg: bool, use_resnet: bool, use_au_c: bool, use_au_r: bool,
                 skip_first_n: int):
        self.vids = list(video_ids)
        self.master_cols = list(master_feature_cols)

        # Ensure per-video caches exist
        for i, vid in enumerate(self.vids):
            build_video_cache_master(vid, self.master_cols, LABEL_COL, skip_first_n)
            if (i+1) % 200 == 0:
                print(f"[cache] built {i+1}/{len(self.vids)} videos")

        # Build feature index for this config
        want = []
        for i, name in enumerate(self.master_cols):
            if name.endswith("_vgg"):    want.append(i) if use_vgg    else None
            if name.endswith("_resnet"): want.append(i) if use_resnet else None
            if name.endswith("_c"):      want.append(i) if use_au_c   else None
            if name.endswith("_r"):      want.append(i) if use_au_r   else None
        self.col_idx = np.asarray(sorted(set(want)), dtype=np.int64)
        if self.col_idx.size == 0:
            raise ValueError("No columns selected by this feature combo.")

        self.feature_cols = [self.master_cols[i] for i in self.col_idx.tolist()]
        self.input_dim = len(self.feature_cols)

        # Load memmaps and global ranges
        self.arrs, self.ranges, total = [], [], 0
        for vid in self.vids:
            xnp, ynp, _ = _cache_paths(vid)
            if not (os.path.isfile(xnp) and os.path.isfile(ynp)):
                continue
            Xm = np.load(xnp, mmap_mode="r")  # (N_frames, D_master)
            ym = np.load(ynp, mmap_mode="r")  # (N_frames,)
            n = len(ym)
            if n == 0:
                continue
            v_idx = len(self.arrs)
            self.arrs.append((Xm, ym))
            self.ranges.append((v_idx, total, n))
            total += n
        self.total = total

    def __len__(self):
        return self.total

    def close(self):
        """Close np.memmap file descriptors and drop references."""
        try:
            for Xm, ym in getattr(self, "arrs", []):
                for arr in (Xm, ym):
                    if isinstance(arr, np.memmap):
                        try:
                            arr._mmap.close()
                        except Exception:
                            base = getattr(arr, "base", None)
                            try:
                                if hasattr(base, "close"):
                                    base.close()
                            except Exception:
                                pass
            self.arrs.clear(); self.ranges.clear()
        except Exception:
            pass
        gc.collect()

    def __del__(self):
        try: self.close()
        except Exception: pass

    def __getitem__(self, i):
        # binary search over video ranges
        lo, hi = 0, len(self.ranges) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            v_idx, start, n = self.ranges[mid]
            if i < start: hi = mid - 1
            elif i >= start + n: lo = mid + 1
            else:
                Xm, ym = self.arrs[v_idx]
                off = i - start
                feat = np.asarray(Xm[off, self.col_idx], dtype=np.float32)
                x = torch.from_numpy(feat.copy())  # contiguous, writable
                y = torch.tensor(int(ym[off]), dtype=torch.long)
                return x, y
        raise IndexError(i)

# ======================
#   BUILD DATALOADERS FROM CACHED DATA
# ======================
def _make_dataset(list_ids: List[str],
                  feature_cols_lock=None,   # not used; kept for compatibility
                  use_vgg=True, use_resnet=True, use_au_c=True, use_au_r=True):
    if MASTER_FEATURE_COLS is None:
        raise RuntimeError("MASTER_FEATURE_COLS not initialized; call in main().")
    return CachedFrameDatasetMaster(
        video_ids=list_ids,
        master_feature_cols=MASTER_FEATURE_COLS,
        use_vgg=use_vgg, use_resnet=use_resnet, use_au_c=use_au_c, use_au_r=use_au_r,
        skip_first_n=SKIP_FIRST_N,
    )

@torch.no_grad()
def _compute_mean_std_per_feature(loader: DataLoader, device, feature_names: List[str]):
    """Per-feature mean/std on TRAIN ONLY frames (z-score). AU_c kept raw if requested."""
    D = len(feature_names)
    n_total = 0
    s1 = torch.zeros(D, device=device, dtype=torch.float64)
    s2 = torch.zeros(D, device=device, dtype=torch.float64)

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True).float()
        n = xb.shape[0]
        n_total += n
        s1 += xb.sum(dim=0).double()
        s2 += (xb.double().pow(2)).sum(dim=0)

    mean = s1 / max(1, n_total)
    var  = (s2 / max(1, n_total)) - mean.pow(2)
    var  = torch.clamp(var, min=1e-12)
    std  = torch.sqrt(var).float()
    mean = mean.float()

    if DO_STANDARDIZE and KEEP_AU_C_RAW:
        auc_idx = [i for i, n in enumerate(feature_names) if n.endswith("_c")]
        if auc_idx:
            idx = torch.tensor(auc_idx, dtype=torch.long, device=device)
            mean.index_fill_(0, idx, 0.0)
            std.index_fill_(0, idx, 1.0)

    return mean.detach().cpu(), std.detach().cpu()

def _build_optimizer(opt_name: str, params, lr: float, weight_decay: float):
    if opt_name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

def _eval_loader(model, loader, device, scaler):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = scaler(xb.to(device, non_blocking=True).float())
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss   = ce(logits, yb)
            loss_sum += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            total   += yb.numel()
    return (loss_sum / max(1, len(loader))), (correct / max(1, total))

@torch.no_grad()
def _eval_loader_timed(model, loader, device, scaler):
    """Same as _eval_loader, but returns timing breakdown."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    data_time = 0.0
    compute_time = 0.0
    _sync_cuda(); prev_end = time.time()
    for xb, yb in loader:
        t_after_load = time.time()
        data_time += (t_after_load - prev_end)

        _sync_cuda(); t0 = time.time()
        xb = scaler(xb.to(device, non_blocking=True).float())
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss   = ce(logits, yb)
        _sync_cuda(); t1 = time.time()
        compute_time += (t1 - t0)

        loss_sum += loss.item()
        correct  += (logits.argmax(1) == yb).sum().item()
        total    += yb.numel()
        prev_end = t1
    return (loss_sum / max(1, len(loader))), (correct / max(1, total)), {
        "data_s": data_time,
        "compute_s": compute_time,
        "epoch_s": data_time + compute_time,
    }

def _config_tag(cfg_run: dict) -> str:
    parts = []
    fb = []
    if cfg_run["use_resnet"]: fb.append("RES")
    if cfg_run["use_vgg"]:    fb.append("VGG")
    if cfg_run["use_au_c"]:   fb.append("AUc")
    if cfg_run["use_au_r"]:   fb.append("AUr")
    if not fb: fb = ["RAW"]
    parts.append("-".join(fb))
    h2 = cfg_run["hidden_dim2"] if cfg_run["hidden_dim2"] not in (None, 0) else "none"
    parts.append(f"H1{cfg_run['hidden_dim']}_H2{h2}")
    parts.append(f"{cfg_run['optimizer']}_lr{cfg_run['lr']}_wd{cfg_run['weight_decay']}")
    parts.append(f"do{cfg_run['dropout']}_bs{cfg_run['batch_size']}")
    return "__".join(parts)

def _fold_dir_for(cfg_run: dict, fold_id: int) -> str:
    cfg_dir = os.path.join(CONFIGS_DIR, _config_tag(cfg_run))
    os.makedirs(cfg_dir, exist_ok=True)
    fdir = os.path.join(cfg_dir, f"fold_{fold_id}")
    os.makedirs(fdir, exist_ok=True)
    return fdir

def _save_ckpt(path, epoch, model, optimizer, scheduler, best_metric, no_improve, best_state_path=None):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "no_improve": no_improve,
        "best_state_path": best_state_path,
    }
    torch.save(ckpt, path)

def _load_ckpt(path, device):
    return torch.load(path, map_location=device)

def _select_indices_for_cfg(master_cols: List[str], use_vgg, use_resnet, use_au_c, use_au_r):
    idx = []
    for i, c in enumerate(master_cols):
        if c.endswith("_vgg")    and use_vgg:    idx.append(i)
        if c.endswith("_resnet") and use_resnet: idx.append(i)
        if c.endswith("_c")      and use_au_c:   idx.append(i)
        if c.endswith("_r")      and use_au_r:   idx.append(i)
    return np.asarray(sorted(set(idx)), dtype=np.int64)

@torch.no_grad()
def _video_acc_on_videos(model, vids: List[str], sel_idx: np.ndarray, scaler: nn.Module):
    """
    Mean-softmax per video on cached X.npy; GT is per-video majority of cached y.npy.
    """
    correct, total = 0, 0
    model.eval()
    for vid in vids:
        xnp, ynp, _ = _cache_paths(vid)
        if not (os.path.isfile(xnp) and os.path.isfile(ynp)):
            continue
        Xm = np.load(xnp, mmap_mode="r")
        yv = np.load(ynp, mmap_mode="r")
        if Xm.shape[0] == 0:
            continue

        probs_acc = None
        bs = 16384
        for i in range(0, Xm.shape[0], bs):
            chunk_np = np.array(Xm[i:i+bs][:, sel_idx], copy=True)
            xb = torch.from_numpy(chunk_np).float().to(DEVICE, non_blocking=True)
            xb = scaler(xb)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).sum(dim=0)
            probs_acc = probs if probs_acc is None else (probs_acc + probs)
        pred = int(torch.argmax(probs_acc).item())
        gt = int(np.bincount(yv).argmax())
        correct += int(pred == gt)
        total += 1
    return float(correct / total) if total else float("nan")

def _train_one_fold(train_ids, val_ids, cfg_run, fold_tag, fold_dir):
    """
    Train on (train_ids) and validate on (val_ids) with resume support.
    fold_dir contains: last_ckpt.pt, best_state.pt, metrics.json (with frame & video acc)
    and epoch_times.json (per-epoch timing breakdown).
    """
    # early skip if already finished and allowed
    metrics_path = os.path.join(fold_dir, "metrics.json")
    epoch_times_path = os.path.join(fold_dir, "epoch_times.json")
    epoch_logs = []
    if os.path.isfile(epoch_times_path):
        try:
            epoch_logs = json.load(open(epoch_times_path, "r"))
        except Exception:
            epoch_logs = []

    if SKIP_DONE_FOLDS and os.path.isfile(metrics_path):
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            if m.get("done", False):
                print(f"[{fold_tag}] found metrics.json (done). Skipping training.")
                return {"val_acc": float(m["val_acc"]), "val_loss": float(m["val_loss"]),
                        "video_val_acc": float(m.get("video_val_acc", float("nan")))}
        except Exception:
            pass

    # ---- Datasets (cache-based) ----
    ds_train = _make_dataset(train_ids,
                             feature_cols_lock=None,
                             use_vgg=cfg_run["use_vgg"],
                             use_resnet=cfg_run["use_resnet"],
                             use_au_c=cfg_run["use_au_c"],
                             use_au_r=cfg_run["use_au_r"])
    ds_val   = _make_dataset(val_ids,
                             feature_cols_lock=None,
                             use_vgg=cfg_run["use_vgg"],
                             use_resnet=cfg_run["use_resnet"],
                             use_au_c=cfg_run["use_au_c"],
                             use_au_r=cfg_run["use_au_r"])

    train_loader = DataLoader(ds_train, batch_size=cfg_run["batch_size"], shuffle=True,  drop_last=False,
                              collate_fn=_safe_collate, **_loader_kws())
    val_loader   = DataLoader(ds_val,   batch_size=cfg_run["batch_size"], shuffle=False, drop_last=False,
                              collate_fn=_safe_collate, **_loader_kws())

    # ---- Scaler (fit on train only) ----
    if DO_STANDARDIZE:
        tmp_loader = DataLoader(ds_train, batch_size=4096, shuffle=False,
                                collate_fn=_safe_collate, **_loader_kws())
        mean, std = _compute_mean_std_per_feature(tmp_loader, DEVICE, ds_train.feature_cols)
        scaler = Standardize(mean, std).to(DEVICE)
    else:
        scaler = nn.Identity()

    # ---- Model / Optim / Scheduler ----
    num_classes = len(EMOTION_TO_IDX)
    model = FrameClassifier(
        input_dim=ds_train.input_dim,
        hidden_dim=cfg_run["hidden_dim"],
        hidden_dim2=cfg_run["hidden_dim2"],
        dropout=cfg_run["dropout"],
        num_classes=num_classes,
    ).to(DEVICE)

    optim = _build_optimizer(cfg_run["optimizer"], model.parameters(),
                             lr=cfg_run["lr"], weight_decay=cfg_run["weight_decay"])
    ce = nn.CrossEntropyLoss()
    plateau_mode = "max" if ES_MONITOR == "val_acc" else "min"
    scheduler = ReduceLROnPlateau(optim, mode=plateau_mode, factor=0.1, patience=5, min_lr=1e-6)

    # ---- Resume? ----
    last_ckpt = os.path.join(fold_dir, "last_ckpt.pt")
    best_path = os.path.join(fold_dir, "best_state.pt")
    start_epoch = 1
    best_metric = -math.inf if ES_MONITOR == "val_acc" else math.inf
    no_improve  = 0

    if RESUME_FOLDS and os.path.isfile(last_ckpt):
        print(f"[{fold_tag}] resuming from {last_ckpt}")
        ckpt = _load_ckpt(last_ckpt, DEVICE)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        if ckpt.get("sched") is not None:
            scheduler.load_state_dict(ckpt["sched"])
        best_metric = ckpt.get("best_metric", best_metric)
        no_improve  = int(ckpt.get("no_improve", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    # ---- Train with early stopping on val metric ----
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        # TRAIN loop timings
        model.train()
        run_loss = 0.0
        tr_data_time = 0.0
        tr_compute_time = 0.0

        _sync_cuda(); prev_end = time.time()
        for xb, yb in train_loader:
            t_after_load = time.time()
            tr_data_time += (t_after_load - prev_end)

            _sync_cuda(); t0 = time.time()
            xb = scaler(xb.to(DEVICE, non_blocking=True).float())
            yb = yb.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            optim.step()
            _sync_cuda(); t1 = time.time()
            tr_compute_time += (t1 - t0)

            run_loss += loss.item()
            prev_end = t1

        # VAL loop timings
        val_loss, val_acc, val_timing = _eval_loader_timed(model, val_loader, DEVICE, scaler)
        sched_value = val_acc if plateau_mode == "max" else val_loss
        scheduler.step(sched_value)

        monitor   = val_acc if ES_MONITOR == "val_acc" else val_loss
        improved  = (monitor > best_metric) if ES_MONITOR == "val_acc" else (monitor < best_metric)
        if improved:
            best_metric = monitor
            no_improve  = 0
            torch.save(model.state_dict(), best_path)  # persist best
        else:
            no_improve += 1

        tr_epoch_time = tr_data_time + tr_compute_time
        print(f"[{fold_tag}] ep {epoch:03d} | tr_loss {run_loss/max(1,len(train_loader)):.4f} "
              f"| va_loss {val_loss:.4f} | va_acc {val_acc:.4f} "
              f"| best_{ES_MONITOR} {best_metric:.4f} | lr {_current_lr(optim):.2e} "
              f"| ⏱ train (data {tr_data_time:.2f}s, comp {tr_compute_time:.2f}s, total {tr_epoch_time:.2f}s) "
              f"| val (data {val_timing['data_s']:.2f}s, comp {val_timing['compute_s']:.2f}s, total {val_timing['epoch_s']:.2f}s)")

        # persist last checkpoint and epoch timings
        _save_ckpt(last_ckpt, epoch, model, optim, scheduler, best_metric, no_improve, best_state_path=best_path)

        epoch_logs.append({
            "epoch": int(epoch),
            "train_data_s": float(tr_data_time),
            "train_compute_s": float(tr_compute_time),
            "train_total_s": float(tr_epoch_time),
            "val_data_s": float(val_timing["data_s"]),
            "val_compute_s": float(val_timing["compute_s"]),
            "val_total_s": float(val_timing["epoch_s"]),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(_current_lr(optim)),
        })
        try:
            with open(epoch_times_path, "w") as f:
                json.dump(epoch_logs, f, indent=2)
        except Exception:
            pass

        if no_improve >= ES_PATIENCE:
            break

    # final eval with best weights (frame-level)
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    # also capture a last val timing snapshot
    val_loss, val_acc, val_timing = _eval_loader_timed(model, val_loader, DEVICE, scaler)

    # compute VIDEO-LEVEL accuracy on the validation videos for this fold (timed)
    sel_idx = _select_indices_for_cfg(MASTER_FEATURE_COLS,
                                      cfg_run["use_vgg"], cfg_run["use_resnet"],
                                      cfg_run["use_au_c"], cfg_run["use_au_r"])
    t0 = time.time(); _sync_cuda()
    video_val_acc = _video_acc_on_videos(model, val_ids, sel_idx, scaler)
    _sync_cuda(); video_eval_s = time.time() - t0

    # write metrics.json so we can skip next time (now includes video_val_acc + timing summary)
    try:
        with open(metrics_path, "w") as f:
            json.dump({
                "done": True,
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "video_val_acc": float(video_val_acc),
                "timing_summary": {
                    "last_epoch_train_s": float(epoch_logs[-1]["train_total_s"]) if epoch_logs else None,
                    "last_epoch_val_s": float(epoch_logs[-1]["val_total_s"]) if epoch_logs else None,
                    "video_eval_s": float(video_eval_s),
                }
            }, f, indent=2)
    except Exception:
        pass

    # cleanup memmaps
    if hasattr(ds_train, "close"): ds_train.close()
    if hasattr(ds_val, "close"): ds_val.close()
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    return {"val_acc": float(val_acc), "val_loss": float(val_loss), "video_val_acc": float(video_val_acc)}

def _config_iter():
    for feat in FEATURE_SETS:
        for arch in ARCH_GRID:
            for opt in OPTIMIZERS:
                for lr in LRS:
                    for wd in WEIGHT_DECAY:
                        for dr in DROPOUTS:
                            for bs in BATCH_SIZES:
                                yield {
                                    "feature_set": feat["name"],
                                    "use_vgg": feat["use_vgg"],
                                    "use_resnet": feat["use_resnet"],
                                    "use_au_c": feat["use_au_c"],
                                    "use_au_r": feat["use_au_r"],
                                    "hidden_dim": arch["hidden_dim"],
                                    "hidden_dim2": arch["hidden_dim2"],
                                    "optimizer": opt,
                                    "lr": lr,
                                    "weight_decay": wd,
                                    "dropout": dr,
                                    "batch_size": bs,
                                }

# ======================
#   TEST EVALUATION (cache-based)
# ======================
def _evaluate_on_test(best_cfg, state_dict, master_cols, scaler_mean, scaler_std):
    """
    Evaluate on TEST (frame- & video-level) using cached X.npy / y.npy.
    """
    test_list = _require_file(TEST_LIST, "TEST_LIST")
    vids = _read_ids(test_list)

    # ensure caches exist for all test vids
    for v in vids:
        build_video_cache_master(v, master_cols, LABEL_COL, SKIP_FIRST_N)

    sel_idx = _select_indices_for_cfg(master_cols,
                                      best_cfg["use_vgg"], best_cfg["use_resnet"],
                                      best_cfg["use_au_c"], best_cfg["use_au_r"])

    # scaler
    if DO_STANDARDIZE:
        scaler = Standardize(scaler_mean, scaler_std).to(DEVICE)
    else:
        scaler = nn.Identity()

    # model
    model = FrameClassifier(
        input_dim=len(sel_idx),
        hidden_dim=best_cfg["hidden_dim"],
        hidden_dim2=best_cfg["hidden_dim2"],
        dropout=best_cfg["dropout"],
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # frame-level over all frames
    y_true, y_pred = [], []
    with torch.no_grad():
        for vid in vids:
            xnp, ynp, _ = _cache_paths(vid)
            if not (os.path.isfile(xnp) and os.path.isfile(ynp)):
                continue
            Xm = np.load(xnp, mmap_mode="r")
            yv = np.load(ynp, mmap_mode="r")
            if Xm.shape[0] == 0:
                continue
            bs = 16384
            for i in range(0, Xm.shape[0], bs):
                chunk_np = np.array(Xm[i:i+bs][:, sel_idx], copy=True)
                xb = torch.from_numpy(chunk_np).float().to(DEVICE, non_blocking=True)
                xb = scaler(xb)
                logits = model(xb)
                y_pred.append(logits.argmax(1).cpu().numpy())
                y_true.append(yv[i:i+bs].copy())

    if y_true:
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        classes = [IDX_TO_EMO[i] for i in range(len(IDX_TO_EMO))]
        print("\n[TEST][FRAME] report:")
        print(classification_report(y_true, y_pred, target_names=classes, digits=3))
        print("[TEST][FRAME] Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    else:
        print("\n[TEST][FRAME] No frames evaluated.")

    # video-level (mean softmax per video)
    y_true_v, y_pred_v = [], []
    with torch.no_grad():
        for vid in vids:
            xnp, ynp, _ = _cache_paths(vid)
            if not (os.path.isfile(xnp) and os.path.isfile(ynp)):
                continue
            Xm = np.load(xnp, mmap_mode="r")
            yv = np.load(ynp, mmap_mode="r")
            if Xm.shape[0] == 0:
                continue
            Xs = np.array(Xm[:, sel_idx], copy=True)
            X  = torch.from_numpy(Xs).float().to(DEVICE, non_blocking=True)
            X  = scaler(X)
            probs = torch.softmax(model(X), dim=1).mean(dim=0)
            y_pred_v.append(int(probs.argmax().item()))
            y_true_v.append(int(np.bincount(yv).argmax()))

    y_true_v = np.array(y_true_v); y_pred_v = np.array(y_pred_v)
    if y_true_v.size:
        classes = [IDX_TO_EMO[i] for i in range(len(IDX_TO_EMO))]
        print("\n[TEST][VIDEO] report:")
        print(classification_report(y_true_v, y_pred_v, target_names=classes, digits=3))
        print("[TEST][VIDEO] Confusion Matrix:\n", confusion_matrix(y_true_v, y_pred_v))
        cm = confusion_matrix(y_true_v, y_pred_v)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format=".0f")
        plt.title("Confusion Matrix (Video Level)")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        # --- SAVE IT ---
        plt.savefig("confusion_matrix_video.png", dpi=300, bbox_inches="tight")
        
    else:
        print("\n[TEST][VIDEO] No videos evaluated (empty after filtering).")

def get_ids_and_labels() -> Tuple[List[str], np.ndarray]:
    """
    Load (ids, y) from IDS_LABELS_CACHE if present, otherwise build and cache.
    """
    if os.path.isfile(IDS_LABELS_CACHE):
        with open(IDS_LABELS_CACHE, "r") as f:
            data = json.load(f)
        ids = list(data.get("ids", []))
        y   = np.array(data.get("labels", []), dtype=np.int64)
        if len(ids) == len(y) and len(ids) > 0:
            print(f"[ids] loaded {len(ids)} ids from cache → {IDS_LABELS_CACHE}")
            return ids, y
        else:
            print(f"[ids] cache invalid or empty, rebuilding…")

    train_ids = _read_ids(_require_file(TRAIN_LIST, "TRAIN_LIST"))
    val_ids   = _read_ids(_require_file(VAL_LIST,   "VAL_LIST"))
    all_ids   = sorted(set(train_ids) | set(val_ids))
    all_ids   = _apply_include_exclude(all_ids)
    ids, y    = _scan_labels(all_ids)

    os.makedirs(os.path.dirname(IDS_LABELS_CACHE), exist_ok=True)
    with open(IDS_LABELS_CACHE, "w") as f:
        json.dump({"ids": ids, "labels": y.astype(int).tolist()}, f, indent=2)
    print(f"[ids] saved {len(ids)} ids → {IDS_LABELS_CACHE}")
    return ids, y

# ======================
#   MAIN
# ======================
def main():
    global MASTER_FEATURE_COLS
    # 1) Load TRAIN & VAL IDs+labels (cached)
    ids, y = get_ids_and_labels()
    print(f"[cv] videos in TRAIN+VAL union: {len(ids)}")

    # 1b) Build master feature list once, then build per-video caches for all ids
    MASTER_FEATURE_COLS = get_master_feature_cols(ids)
    for i, vid in enumerate(ids, 1):
        build_video_cache_master(vid, MASTER_FEATURE_COLS, LABEL_COL, SKIP_FIRST_N)

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=CFG.SEED)
    n_folds = skf.get_n_splits()
    fold_cols = [f"fold_{i}" for i in range(1, n_folds + 1)]

    # 2) Build the grid
    grid = list(_config_iter())
    print(f"[grid] total configurations: {len(grid)}")

    results = []
    summary_rows_frame, summary_rows_video = [], []
    run_idx = 0

    # 3) Sweep
    for cfg_run in grid:
        run_idx += 1
        tag = (f"{cfg_run['feature_set']} | H1={cfg_run['hidden_dim']} "
               f"H2={cfg_run['hidden_dim2']} | opt={cfg_run['optimizer']} "
               f"lr={cfg_run['lr']} wd={cfg_run['weight_decay']} "
               f"do={cfg_run['dropout']} bs={cfg_run['batch_size']}")
        print("\n" + "="*100)
        print(f"[run {run_idx}/{len(grid)}] {tag}")
        print("="*100)

        fold_accs, fold_losses, fold_video_accs = [], [], []
        for fold_id, (tr_idx, va_idx) in enumerate(skf.split(ids, y), start=1):
            tr_ids = [ids[i] for i in tr_idx]
            va_ids = [ids[i] for i in va_idx]

            fold_dir = _fold_dir_for(cfg_run, fold_id)
            info = _train_one_fold(tr_ids, va_ids, cfg_run, fold_tag=f"fold{fold_id}", fold_dir=fold_dir)
            fold_accs.append(info["val_acc"])
            fold_losses.append(info["val_loss"])
            fold_video_accs.append(info["video_val_acc"])
            print(f"[cv][{tag}][fold {fold_id}/{n_folds}] "
                  f"frame_val_acc={fold_accs[-1]:.4f} | video_val_acc={fold_video_accs[-1]:.4f}")

        # record per-config rows (frame)
        row_f = {"config": tag}
        for i, acc in enumerate(fold_accs, start=1):
            row_f[f"fold_{i}"] = acc
        row_f["avg"] = float(np.mean(fold_accs))
        summary_rows_frame.append(row_f)

        # record per-config rows (video)
        row_v = {"config": tag}
        for i, accv in enumerate(fold_video_accs, start=1):
            row_v[f"fold_{i}"] = accv
        row_v["avg"] = float(np.mean(fold_video_accs))
        summary_rows_video.append(row_v)

        mean_acc = float(np.mean(fold_accs))
        std_acc  = float(np.std(fold_accs))
        mean_v   = float(np.mean(fold_video_accs))
        std_v    = float(np.std(fold_video_accs))
        mean_loss= float(np.mean(fold_losses))
        results.append({
            **cfg_run,
            "cv_mean_acc": mean_acc,
            "cv_std_acc": std_acc,
            "cv_mean_video_acc": mean_v,
            "cv_std_video_acc": std_v,
            "cv_mean_loss": mean_loss,
            "tag": tag,
        })
        print(f"[cv][{tag}] -> FRAME acc: {mean_acc:.4f} ± {std_acc:.4f} | "
              f"VIDEO acc: {mean_v:.4f} ± {std_v:.4f} | loss: {mean_loss:.4f}")

        # write per-config aggregate metrics.json in config dir
        cfg_dir = os.path.join(CONFIGS_DIR, _config_tag(cfg_run))
        with open(os.path.join(cfg_dir, "metrics.json"), "w") as f:
            json.dump({
                "k_folds": int(n_folds),
                "folds_completed": int(len(fold_accs)),
                "frame_val_acc_mean": mean_acc,
                "frame_val_acc_std": std_acc,
                "video_val_acc_mean": mean_v,
                "video_val_acc_std": std_v,
                "done": (len(fold_accs) == n_folds),
            }, f, indent=2)

    # 4) Save per-config CSVs (to GRID_OUT_DIR)
    cols = ["config"] + fold_cols + ["avg"]
    # frame CSV
    for r in summary_rows_frame:
        for c in fold_cols:
            r.setdefault(c, np.nan)
    df_frame = pd.DataFrame(summary_rows_frame)[cols].sort_values("avg", ascending=False)
    out_csv_frame = os.path.join(GRID_OUT_DIR, "cv_results_by_config.csv")
    df_frame.to_csv(out_csv_frame, index=False)
    print(f"\n[done] Wrote per-config FRAME CV table -> {out_csv_frame}")
    print(df_frame.to_string(index=False))
    # video CSV
    for r in summary_rows_video:
        for c in fold_cols:
            r.setdefault(c, np.nan)
    df_video = pd.DataFrame(summary_rows_video)[cols].sort_values("avg", ascending=False)
    out_csv_video = os.path.join(GRID_OUT_DIR, "cv_results_by_config_video.csv")
    df_video.to_csv(out_csv_video, index=False)
    print(f"\n[done] Wrote per-config VIDEO CV table -> {out_csv_video}")
    print(df_video.to_string(index=False))

    # 5) Pick best (by frame mean acc; change to cv_mean_video_acc if you prefer)
   
    results_sorted = sorted(results, key=lambda d: d[select], reverse=True)
    best = results_sorted[0]
    print("\n" + "#"*100)
    print("[BEST CONFIG]")
    print(json.dumps(best, indent=2))
    print("#"*100 + "\n")
    print(SCALER_PATH_FINAL)  
    # 6) Retrain best config on full TRAIN+VAL (with early stopping) and eval on TEST
    best_cfg = best.copy()
    
    print("[final] Retraining best config on full TRAIN+VAL…")

    ds_full_train = _make_dataset(
      ids,
      feature_cols_lock=None,
      use_vgg=best_cfg["use_vgg"],
      use_resnet=best_cfg["use_resnet"],
      use_au_c=best_cfg["use_au_c"],
      use_au_r=best_cfg["use_au_r"],
    )
    loader_full = DataLoader(
      ds_full_train,
      batch_size=best_cfg["batch_size"],
      shuffle=True,
      drop_last=False,
      collate_fn=_safe_collate,
      **_loader_kws(),
    )

    # Scaler on full TRAIN+VAL
    if DO_STANDARDIZE:
      tmp_loader = DataLoader(
        ds_full_train, batch_size=4096, shuffle=False,
        collate_fn=_safe_collate, **_loader_kws()
      )
      mean, std = _compute_mean_std_per_feature(tmp_loader, DEVICE, ds_full_train.feature_cols)
      scaler = Standardize(mean, std).to(DEVICE)
    else:
      mean = torch.zeros(ds_full_train.input_dim)
      std  = torch.ones(ds_full_train.input_dim)
      scaler = nn.Identity()

    # Model / optim / sched
    model = FrameClassifier(
      input_dim=ds_full_train.input_dim,
      hidden_dim=best_cfg["hidden_dim"],
      hidden_dim2=best_cfg["hidden_dim2"],
      dropout=best_cfg["dropout"],
      num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)

    optim = _build_optimizer(best_cfg["optimizer"], model.parameters(),
                         lr=best_cfg["lr"], weight_decay=best_cfg["weight_decay"])
    ce = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optim, mode="min", factor=0.1, patience=5, min_lr=1e-6)

    # ---- Early stopping on train loss ----
    FINAL_ES_PATIENCE   = ES_PATIENCE          # or set a custom int, e.g., 20
    FINAL_ES_MIN_DELTA  = 1e-5                 # required improvement in loss
    best_tr_loss        = float("inf")
    no_improve          = 0
    best_state          = None

    for epoch in range(1, total_epochs + 1):
      model.train()
      run_loss, correct, total = 0.0, 0, 0

      for xb, yb in loader_full:
        xb = scaler(xb.to(DEVICE, non_blocking=True).float())
        yb = yb.to(DEVICE, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = ce(logits, yb)
        loss.backward()
        optim.step()

        run_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total   += yb.numel()

      avg_tr = run_loss / max(1, len(loader_full))
      train_acc = correct / max(1, total)
      scheduler.step(avg_tr)

      improved = (best_tr_loss - avg_tr) > FINAL_ES_MIN_DELTA
      if improved:
        best_tr_loss = avg_tr
        no_improve = 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
      else:
        no_improve += 1

      print(f"[final-fit] ep {epoch:03d} | train_loss {avg_tr:.4f} | "
          f"train_acc {train_acc:.4f} | best_tr_loss {best_tr_loss:.4f} | "
          f"no_improve {no_improve}/{FINAL_ES_PATIENCE} | lr {optim.param_groups[0]['lr']:.2e}")

      if no_improve >= FINAL_ES_PATIENCE:
        print(f"[final-fit] Early stopping at epoch {epoch} (no improvement for {FINAL_ES_PATIENCE} epochs).")
        break

    # restore best weights before saving/evaluating
    if best_state is not None:
      model.load_state_dict(best_state)

    # Save final artifacts to your standard paths
    
    os.makedirs(os.path.dirname(SCALER_PATH_FINAL), exist_ok=True)
    torch.save({"mean": mean, "std": std}, SCALER_PATH_FINAL)
    os.makedirs(os.path.dirname(FEATCOLS_FINAL), exist_ok=True)
    save_feature_cols(ds_full_train.feature_cols, FEATCOLS_FINAL)
    os.makedirs(os.path.dirname(BEST_WEIGHTS_PATH_FINAL), exist_ok=True)
    torch.save(model.state_dict(),BEST_WEIGHTS_PATH_FINAL)
    
    print(f"[save] scaler -> {CFG.SCALER_PATH}")
    print(f"[save] feature_cols -> {CFG.FEATCOLS_JSON}")
    print(f"[save] weights -> {CFG.BEST_WEIGHTS}")

    # Evaluate on TEST via caches
    _evaluate_on_test(best_cfg, model.state_dict(),
                      master_cols=MASTER_FEATURE_COLS,
                      scaler_mean=mean, scaler_std=std)

    if hasattr(ds_full_train, "close"): ds_full_train.close()
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()

