import os, sys, json, math, time, random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# --- clean env for torch import (avoids LD_LIBRARY_PATH conflicts) ---
if os.environ.get("LD_LIBRARY_PATH") and not os.environ.get("_LDLIBPATH_CLEANED"):
    env = dict(os.environ); env.pop("LD_LIBRARY_PATH", None); env["_LDLIBPATH_CLEANED"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)
os.environ.pop("LD_LIBRARY_PATH", None)

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# ---- project imports ----
import config as CFG
from utils.features import (
    pick_ordered_feature_cols, pick_present_aus, harmonize_vgg_cols,
    Standardize
)
from models.CNN_RNNmodel import TemporalFFRNN
from data.seq_dataset import build_au_master  # only for AU list

# ======================
#   DEVICE / LOADER OPTS
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_COUNT = os.cpu_count() or 4
NUM_WORKERS = min(8, max(2, CPU_COUNT - 1))   # tuned for single GPU box
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

# ======================
#   RAVDESS CONSTANTS
# ======================
# Split lists (RAVDESS)
SPLIT_PATH  = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/data/"
TRAIN_LIST  = os.path.join(SPLIT_PATH, "train_videos_RAV.txt")
VAL_LIST    = os.path.join(SPLIT_PATH, "val_videos_RAV.txt")
TEST_LIST   = os.path.join(SPLIT_PATH, "test_videos_RAV.txt")

# Where to cache the master feature list (under artifacts data dir is fine too)
MASTER_FEATURES_JSON = os.path.join(SPLIT_PATH, "master_feature_cols_RAVDESS.json")
MASTER_SCAN_LIMIT = 10   # scan first N videos we find

# I/O
INCLUDE_LIST = None
EXCLUDE_LIST = None  # set a path if you want to exclude some video IDs

# Feature CSV per video
LABEL_COL          = "emotion"
SKIP_FIRST_N       = getattr(CFG, "SKIP_FRAME", 0)
COMBINED_CSV_NAME  = "au_resnet_vgg_with_gt.csv"
OUTPUT_DIR         = "/media/root918/OS/[REDACTED]Project/copiedFilesRAVDESS/"

# RAVDESS 8-class mapping
EMOTION_TO_IDX = {
    "neutral":   0,
    "calm":      1,
    "happy":     2,
    "sad":       3,
    "anger":     4,
    "fearful":   5,
    "disgust":   6,
    "surprise": 7,
}
#IDX_TO_EMO = {v:k for k,v in EMOTION_TO_IDX.items()}
IDX_TO_EMO = {v:k for k,v in EMOTION_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_EMO[i] for i in range(len(IDX_TO_EMO))]

# Artifacts
ART_DIR_TAG   = "ravdess_GridSearch_unscaled_RNN/ravdess_GridSearch_unscaled_RNN"
PROJECT_DIR   = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/"
ART_DIR_SUB   = os.path.join(PROJECT_DIR, "artifacts", ART_DIR_TAG)
os.makedirs(ART_DIR_SUB, exist_ok=True)
GRID_OUT_DIR  = os.path.join(ART_DIR_SUB, "grid_config_RAV")
os.makedirs(GRID_OUT_DIR, exist_ok=True)
FINAL_DIR     = os.path.join(ART_DIR_SUB, "final_retrains")
os.makedirs(FINAL_DIR, exist_ok=True)
# ======================
#   TRAINING HYPERS / GRID
# ======================
EPOCHS = 300
SEED   = getattr(CFG, "SEED", 42)

DO_STANDARDIZE = False
KEEP_AU_C_RAW  = True

FF_ARCH_GRID = [
    #{"hidden_dim": 256,  "hidden_dim2": None},
    {"hidden_dim": 512,  "hidden_dim2": None},
    {"hidden_dim": 1024, "hidden_dim2": None},
    {"hidden_dim": 1024, "hidden_dim2": 512},
    {"hidden_dim": 512,  "hidden_dim2": 256},
]

OPTIMIZERS   = ["adam"]
LRS          = [1e-4]
WEIGHT_DECAY = [1e-5]
DROPOUTS     = [0.5]
BATCH_SIZES  = [512]  # reduce a bit vs 512 for RNN sequences (increase if you have VRAM)
CLIP_NORM    = 1.0
ES_MONITOR   = "val_acc"      # "val_acc" or "val_loss"
ES_PATIENCE  = 15
PLATEAU_PATIENCE = 5
PLATEAU_FACTOR  = 0.1
MIN_LR          = 1e-6
FINAL_PATIENCE_ESTOP=15

FEATURE_SETS = [
   # {"name":"VGG",           "use_vgg":True,  "use_resnet":False, "use_au_c":False, "use_au_r":False},
   # {"name":"RESNET",        "use_vgg":False, "use_resnet":True,  "use_au_c":False, "use_au_r":False},
    {"name":"VGG+RESNET",    "use_vgg":True,  "use_resnet":True,  "use_au_c":False, "use_au_r":False},
    {"name":"VGG+AU",        "use_vgg":True,  "use_resnet":False, "use_au_c":True,  "use_au_r":True },
    {"name":"RESNET+AU",     "use_vgg":False, "use_resnet":True,  "use_au_c":True,  "use_au_r":True },
    {"name":"VGG+RESNET+AU", "use_vgg":True,  "use_resnet":True,  "use_au_c":True,  "use_au_r":True}
]

RNN_TYPES = [ "gru", "lstm"]
RNN_ARCH_GRID = [
    {"layers": 1, "hidden": 128},
    {"layers": 1, "hidden": 256},
    {"layers": 2, "hidden": 128},
    {"layers": 2, "hidden": 256},
]

SEQ_LENGTHS = [10, 30]
def strides_for(T: int) -> List[int]:
    return sorted({T, max(1, T // 2)})

RESUME_CONFIGS    = True
SKIP_DONE_CONFIGS = True

# ======================
#       TIMING HELPERS
# ======================
def _log_time(operation: str, t0: float):
    dt = time.time() - t0
    print(f"[TIMING] {operation}: {dt:.2f}s")
    return dt

class _T:
    def __init__(self, name): self.name=name; self.t0=None
    def __enter__(self): self.t0=time.time(); return self
    def __exit__(self, *exc): _log_time(self.name, self.t0)

def _time(operation: str): return _T(operation)

# ======================
#   TYPE COERCION HELPERS
# ======================
def _none_if_nan(x):
    return None if (isinstance(x, float) and math.isnan(x)) else x

def _py_int(x):
    if x is None: return None
    return int(x)

def _py_float(x):
    if x is None: return None
    return float(x)

def _py_bool(x):
    return bool(x)

# ======================
#       IO HELPERS
# ======================
def _require_file(path, desc):
    if not os.path.isfile(path): raise FileNotFoundError(f"Missing {desc}: {path}")
    return path

def _read_list(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def _apply_include_exclude(ids: List[str], inc: str|None, exc: str|None) -> List[str]:
    s = set(ids)
    if inc and os.path.isfile(inc): s &= set(_read_list(inc))
    if exc and os.path.isfile(exc): s -= set(_read_list(exc))
    return sorted(s)

def _vid_csv_path(vid: str) -> str:
    return os.path.join(OUTPUT_DIR, vid, COMBINED_CSV_NAME)

def _cache_dir(vid: str) -> str:
    # RAVDESS-specific cache folder to avoid mixing with other experiments
    return os.path.join(OUTPUT_DIR, vid, "cache_RAVDESS")

def _cache_paths(vid: str):
    cdir = _cache_dir(vid)
    return (
        os.path.join(cdir, "X.npy"),
        os.path.join(cdir, "y.npy"),
        os.path.join(cdir, "meta.json"),
    )

# ======================
#   MASTER FEATURE LIST
# ======================
def _append_unique(dst_list, names_iterable):
    seen = set(dst_list)
    for n in names_iterable:
        if n not in seen:
            dst_list.append(n)
            seen.add(n)

def _discover_csvs_for_scan(ids, limit):
    out, picked = [], 0
    for vid in ids:
        csvp = os.path.join(OUTPUT_DIR, vid, COMBINED_CSV_NAME)
        if os.path.isfile(csvp):
            out.append(csvp)
            picked += 1
            if picked >= limit:
                break
    return out

def get_master_feature_cols(n_scan=MASTER_SCAN_LIMIT):
    """
    Load master_feature_cols.json if present, else build it by scanning up to `n_scan`
    CSVs from train+val (in that order). We:
      - harmonize VGG names,
      - include present VGG/RESNET features,
      - include present AUs,
      - FORCE-IN all AUs from build_au_master(True, True) so AUs never go missing.
    """
    if os.path.isfile(MASTER_FEATURES_JSON):
        with open(MASTER_FEATURES_JSON, "r") as f:
            cols = json.load(f)
        print(f"[features] Loaded master feature list ({len(cols)}) -> {MASTER_FEATURES_JSON}")
        return cols

    train_ids = _read_list(_require_file(TRAIN_LIST, "TRAIN_LIST"))
    val_ids   = _read_list(_require_file(VAL_LIST,   "VAL_LIST"))
    if INCLUDE_LIST or EXCLUDE_LIST:
        train_ids = _apply_include_exclude(train_ids, INCLUDE_LIST, EXCLUDE_LIST)
        val_ids   = _apply_include_exclude(val_ids,   INCLUDE_LIST, EXCLUDE_LIST)

    scan_paths = _discover_csvs_for_scan(train_ids + val_ids, n_scan)
    if not scan_paths:
        raise RuntimeError("Could not find any CSVs to scan for master feature list.")
    au_master_all = build_au_master(True, True)
    master = []

    for csvp in scan_paths:
        try:
            df = pd.read_csv(csvp, nrows=1)
            df = harmonize_vgg_cols(df)
            feats = pick_ordered_feature_cols(df, use_vgg=True, use_resnet=True)
            _append_unique(master, feats)
            aus_present = pick_present_aus(df, au_master_all)
            _append_unique(master, aus_present)
        except Exception as e:
            print(f"[features] warn: failed header scan for {csvp}: {e}")

    _append_unique(master, list(au_master_all))

    os.makedirs(os.path.dirname(MASTER_FEATURES_JSON), exist_ok=True)
    with open(MASTER_FEATURES_JSON, "w") as f:
        json.dump(master, f, indent=2)
    print(f"[features] Saved master feature list ({len(master)}) from {len(scan_paths)} csv(s) -> {MASTER_FEATURES_JSON}")
    return master

# ======================
#   CACHE BUILDER + STREAMING DATASET
# ======================
DEBUG_PRINT_LABELS_FOR_ALL = True  # set True to print for every video

def build_video_cache(vid: str, master_feature_cols: List[str]) -> bool:
    """
    Build per-video X/y cache from the combined CSV, with rich debug prints.
    Returns True if (re)built, False if cache already existed.
    """
    xnp, ynp, meta = _cache_paths(vid)
    os.makedirs(os.path.dirname(xnp), exist_ok=True)
    if os.path.isfile(xnp) and os.path.isfile(ynp) and os.path.isfile(meta):
        return False

    csvp = _vid_csv_path(vid)
    if not os.path.isfile(csvp):
        print(f"[cache][warn] {vid}: CSV not found → {csvp}")
        # Save empty artifacts so dataset can skip it gracefully
        np.save(xnp, np.empty((0, len(master_feature_cols)), dtype=np.float32))
        np.save(ynp, np.empty((0,), dtype=np.int64))
        with open(meta, "w") as f:
            json.dump({"feature_cols": master_feature_cols}, f)
        return True

    df = pd.read_csv(csvp)
    df = harmonize_vgg_cols(df)
    if SKIP_FIRST_N > 0:
        df = df.iloc[SKIP_FIRST_N:].reset_index(drop=True)

    # --- Raw labels (pre-mapping) ---
    raw = df[LABEL_COL].astype(str)
    raw_stripped = raw.str.strip()
    raw_lower = raw_stripped.str.lower()

    # Normalize mapping keys once
    mapping_keys = set(k.lower() for k in EMOTION_TO_IDX.keys())

    # Identify which rows will map and which won't
    unknown_rows = raw_lower[~raw_lower.isin(mapping_keys)]

    # Map labels -> indices
    emo2idx_lower = {k.lower(): v for k, v in EMOTION_TO_IDX.items()}
    y_map = raw_lower.map(emo2idx_lower)
    y = y_map.dropna().astype(np.int64)
    idx = y.index

    # --- Build X (only at indices that have valid labels) ---
    if len(idx) > 0:
        X = (df.loc[idx, master_feature_cols]
               .replace([np.inf, -np.inf], np.nan)
               .fillna(0.0)
               .astype("float32")
               .to_numpy(copy=False))
        y_np = y.to_numpy(copy=False)
    else:
        X = np.empty((0, len(master_feature_cols)), dtype=np.float32)
        y_np = np.empty((0,), dtype=np.int64)

    # ========== DEBUG PRINTS ==========
    def _print_label_debug():
        raw_counts = raw_stripped.value_counts(dropna=False)
        print(f"[cache][labels-raw]     {vid}: {dict(raw_counts)}")
        if len(unknown_rows) > 0:
            u_counts = unknown_rows.value_counts(dropna=False)
            u_preview = list(u_counts.index[:10])
            print(f"[cache][labels-unknown]{vid}: {dict(u_counts)}")
            print(f"[cache][unknown-keys]  {vid}: first10={u_preview}")
        if len(y) > 0:
            mapped_names = [IDX_TO_EMO[i] for i in y.values]
            mapped_counts = pd.Series(mapped_names).value_counts()
            print(f"[cache][labels-mapped] {vid}: {dict(mapped_counts)}")
        else:
            print(f"[cache][labels-mapped] {vid}: {{}} (no mapped labels)")

    if DEBUG_PRINT_LABELS_FOR_ALL or X.shape[0] == 0:
        _print_label_debug()

    np.save(xnp, X)
    np.save(ynp, y_np)
    with open(meta, "w") as f:
        json.dump({"feature_cols": master_feature_cols}, f)

    return True

class StreamingSequenceDataset(Dataset):
    """
    Builds overlapping sequences from per-video caches on the fly.
    Window label = mode over REAL frames only (unpadded portion).
    """
    def __init__(self, video_ids: List[str], cfg_run: dict, master_feature_cols: List[str]):
        self.vids = list(video_ids)
        self.T    = int(cfg_run["seq_len"])
        self.S    = int(cfg_run["stride"])

        for (i,vid) in enumerate(self.vids):
            try:
                _ = build_video_cache(vid, master_feature_cols)
                if (i % 50) == 0:
                    print(f"[cache] built: ", i)
            except Exception as e:
                print(f"[cache] skip {vid}: {e}")

        # feature selection indices
        want = []
        for i, name in enumerate(master_feature_cols):
            if name.endswith("_vgg")    and cfg_run["use_vgg"]:    want.append(i)
            if name.endswith("_resnet") and cfg_run["use_resnet"]: want.append(i)
            if name.endswith("_c")      and cfg_run["use_au_c"]:   want.append(i)
            if name.endswith("_r")      and cfg_run["use_au_r"]:   want.append(i)
        self.col_idx = np.asarray(sorted(set(want)), dtype=np.int64)
        if self.col_idx.size == 0:
            raise ValueError("No feature columns selected for this config.")

        self._feature_cols = [master_feature_cols[i] for i in self.col_idx.tolist()]
        self._input_dim = len(self._feature_cols)

        # (vid, start) index
        self.index = []
        self._lengths = {}
        for vid in self.vids:
            X_path, _, _ = _cache_paths(vid)
            if not os.path.isfile(X_path): 
                continue
            X_m = np.load(X_path, mmap_mode="r")
            N = len(X_m)
            self._lengths[vid] = N
            if N >= self.T:
                for s in range(0, N - self.T + 1, self.S):
                    self.index.append((vid, s))
                # add tail start if not aligned
                tail_start = ((N - self.T + self.S - 1) // self.S) * self.S
                if tail_start < N and tail_start > (N - self.T):
                    self.index.append((vid, tail_start))
            else:
                self.index.append((vid, 0))

        # small LRU of memmaps
        self._arrays = {}
        print(f"[stream-ds] feats={len(self.col_idx)} | sequences={len(self.index)}")

    def __len__(self): return len(self.index)
    @property
    def feature_cols(self): return self._feature_cols
    @property
    def input_dim(self): return self._input_dim

    def _get_arrays(self, vid):
        arrs = self._arrays.get(vid)
        if arrs is None:
            Xp, yp, _ = _cache_paths(vid)
            X_m = np.load(Xp, mmap_mode="r")
            y_m = np.load(yp, mmap_mode="r")
            arrs = (X_m, y_m)
            if len(self._arrays) > 8: self._arrays.clear()
            self._arrays[vid] = arrs
        return arrs

    def __getitem__(self, idx):
        vid, start = self.index[idx]
        X_m, y_m = self._get_arrays(vid)
        N = self._lengths[vid]
        end = start + self.T

        if end <= N:
            win = X_m[start:end, :][:, self.col_idx]
            lab_real = y_m[start:end]
        else:
            sel_all = X_m[:, self.col_idx]
            win = np.empty((self.T, sel_all.shape[1]), dtype=sel_all.dtype)
            k = max(0, N - 1)
            n_real = max(0, N - start)
            if n_real > 0:
                win[:n_real] = sel_all[start:N, :]
            win[n_real:] = sel_all[k]
            lab_real = y_m[start:N]  # only real labels

        vals, counts = np.unique(lab_real, return_counts=True)
        seq_label = int(vals[np.argmax(counts)])
        return torch.from_numpy(win), torch.tensor(seq_label, dtype=torch.long)

# ======================
#     TRAIN/EVAL UTILS
# ======================
def _make_loader(ds, batch_size: int, shuffle: bool):
    batch_size = int(batch_size)  # ensure plain Python int
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(ds, **kwargs)

@torch.no_grad()
def compute_mean_std_from_cache(video_ids, master_feature_cols, col_idx: np.ndarray,
                                device: torch.device, batch_size: int = 262144):
    S = None
    SS = None
    N = 0
    for vid in video_ids:
        Xp, _, _ = _cache_paths(vid)
        if not os.path.isfile(Xp): continue
        X = np.load(Xp, mmap_mode="r")
        if X.shape[0] == 0: continue
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i+batch_size, :][:, col_idx]
            xb = torch.from_numpy(np.ascontiguousarray(xb)).to(device).float()
            sum_b   = xb.sum(dim=0)
            sumsq_b = (xb * xb).sum(dim=0)
            n_b     = xb.shape[0]
            if S is None:
                S, SS, N = sum_b, sumsq_b, n_b
            else:
                S  = S  + sum_b
                SS = SS + sumsq_b
                N  = N  + n_b
    if S is None or N == 0:
        D = len(col_idx)
        return torch.zeros(D, device=device), torch.ones(D, device=device)
    mean = S / N
    var  = (SS / N) - (mean * mean)
    var  = torch.clamp(var, min=1e-8)
    std  = torch.sqrt(var)
    return mean, std

class L2Normalize(torch.nn.Module):
    """Per-row L2 normalization."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return x / (x.norm(dim=1, keepdim=True).clamp_min(self.eps))

def build_preprocessor(norm_mode: str,
                       feature_cols: list[str],
                       device: torch.device,
                       mean: torch.Tensor | None = None,
                       std: torch.Tensor | None = None,
                       keep_au_c_raw: bool = True) -> torch.nn.Module:
    """
    - 'none'        -> Identity()
    - 'l2'          -> L2Normalize()
    - 'zscore'      -> Standardize(mean,std)               [+ optional AU_c override]
    - 'zscore+l2'   -> Standardize(mean,std) -> L2Normalize()
    """
    mode = str(norm_mode).lower()
    if mode == "none":
        return torch.nn.Identity().to(device)
    if mode == "l2":
        return L2Normalize().to(device)

    assert mean is not None and std is not None, "mean/std are required for z-score modes"
    if mean.dim() != 1: mean = mean.view(-1)
    if std.dim()  != 1: std  = std.view(-1)
    D = len(feature_cols)
    if mean.numel() != D or std.numel() != D:
        raise ValueError(f"[preproc] mean/std length ({mean.numel()},{std.numel()}) != feature dim ({D}).")
    mean = mean.clone(); std = std.clone()
    if keep_au_c_raw and feature_cols:
        auc_idx = [i for i, n in enumerate(feature_cols) if n.endswith('_c')]
        if auc_idx:
            mean[auc_idx] = 0.0
            std[auc_idx]  = 1.0
    z = Standardize(mean, std).to(device)
    if mode == "zscore": return z
    if mode == "zscore+l2": return torch.nn.Sequential(z, L2Normalize()).to(device)
    raise ValueError(f"Unknown norm_mode: {norm_mode}")

@torch.no_grad()
def _eval_epoch_seq(model, loader, device, preproc) -> Tuple[float, float, float, int]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, run_loss = 0, 0, 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True)
        B, L, D = xb.shape
        xb = xb.view(B*L, D); xb = preproc(xb); xb = xb.view(B, L, D)
        logits = model(xb)
        loss = ce(logits, yb)
        run_loss += loss.item()
        preds = logits.argmax(1)
        correct  += (preds == yb).sum().item()
        total    += yb.numel()
    seq_acc = correct / max(1, total)
    avg_loss = run_loss / max(1, len(loader))
    frame_acc_approx = seq_acc
    return avg_loss, seq_acc, frame_acc_approx, total

@torch.no_grad()
def _collect_seq_preds(model, loader, device, preproc) -> Tuple[np.ndarray, np.ndarray]:
    """Collect sequence-level preds/labels (for confusion matrix / report)."""
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        B, L, D = xb.shape
        xb = xb.view(B*L, D); xb = preproc(xb); xb = xb.view(B, L, D)
        logits = model(xb)
        preds = logits.argmax(1).detach().cpu().numpy()
        ys.append(yb.numpy())
        ps.append(preds)
    if ys:
        return np.concatenate(ys), np.concatenate(ps)
    return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

# ---------- Frame & Video aggregators (probability-based) ----------
@torch.no_grad()
def _collect_frame_preds_stream(model, video_ids, cfg_run, master_feature_cols, preproc):
    # feature indices once
    want = []
    for i, name in enumerate(master_feature_cols):
        if name.endswith("_vgg")    and cfg_run["use_vgg"]:    want.append(i)
        if name.endswith("_resnet") and cfg_run["use_resnet"]: want.append(i)
        if name.endswith("_c")      and cfg_run["use_au_c"]:   want.append(i)
        if name.endswith("_r")      and cfg_run["use_au_r"]:   want.append(i)
    col_idx = np.asarray(sorted(set(want)), dtype=np.int64)
    C = len(EMOTION_TO_IDX)

    y_true_all, y_pred_all = [], []

    for vid in video_ids:
        Xp, yp, _ = _cache_paths(vid)
        if not (os.path.isfile(Xp) and os.path.isfile(yp)):
            continue
        X = np.load(Xp, mmap_mode="r")  # (N, D_all)
        y = np.load(yp, mmap_mode="r")  # (N,)
        N = len(X)
        if N == 0:
            continue

        frame_prob_sum = np.zeros((N, C), dtype=np.float64)
        frame_count    = np.zeros(N, dtype=np.int32)

        T, S = cfg_run["seq_len"], cfg_run["stride"]
        seqs = []
        if N >= T:
            for s in range(0, N - T + 1, S):
                seqs.append((s, s+T))
            # tail (if any)
            tail_start = ((N - T + S - 1) // S) * S
            if tail_start < N and tail_start > (N - T):
                seqs.append((tail_start, tail_start + T))
        else:
            seqs.append((0, T))

        bs = int(cfg_run["batch_size"])
        for i in range(0, len(seqs), bs):
            chunk = seqs[i:i+bs]
            starts, real_lens, batch = [], [], []
            for (a, b) in chunk:
                starts.append(a)
                rl = min(T, max(0, N - a))  # real (unpadded) frames
                real_lens.append(rl)
                if b <= N:
                    win = X[a:b, :][:, col_idx]
                else:
                    sel = X[:, col_idx]
                    win = np.empty((T, sel.shape[1]), dtype=sel.dtype)
                    k = max(0, N - 1)
                    if rl > 0: win[:rl] = sel[a:N]
                    win[rl:] = sel[k]   # replicate-pad
                batch.append(torch.from_numpy(win).float())

            xb = torch.stack(batch, dim=0).to(DEVICE, non_blocking=True)  # (B,T,Dsel)
            B, L, D = xb.shape
            xb = xb.view(B*L, D); xb = preproc(xb); xb = xb.view(B, L, D)
            probs = torch.softmax(model(xb), dim=1).detach().cpu().numpy()  # (B,C)

            for j, (a, rl) in enumerate(zip(starts, real_lens)):
                if rl <= 0: 
                    continue
                frame_prob_sum[a:a+rl] += probs[j]    # add window prob to frames it covers
                frame_count[a:a+rl]    += 1

        counts   = np.clip(frame_count, 1, None)[:, None]
        avg_prob = frame_prob_sum / counts
        pred_f   = np.argmax(avg_prob, axis=1)

        y_true_all.append(y.copy())
        y_pred_all.append(pred_f)

    if y_true_all:
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
    else:
        y_true_all = np.array([], dtype=np.int64)
        y_pred_all = np.array([], dtype=np.int64)
    return y_true_all, y_pred_all

@torch.no_grad()
def _collect_video_preds_from_frames(model, video_ids, cfg_run, master_feature_cols, preproc):
    want = []
    for i, name in enumerate(master_feature_cols):
        if name.endswith("_vgg")    and cfg_run["use_vgg"]:    want.append(i)
        if name.endswith("_resnet") and cfg_run["use_resnet"]: want.append(i)
        if name.endswith("_c")      and cfg_run["use_au_c"]:   want.append(i)
        if name.endswith("_r")      and cfg_run["use_au_r"]:   want.append(i)
    col_idx = np.asarray(sorted(set(want)), dtype=np.int64)
    C = len(EMOTION_TO_IDX)

    y_true_v, y_pred_v = [], []

    for vid in video_ids:
        Xp, yp, _ = _cache_paths(vid)
        if not (os.path.isfile(Xp) and os.path.isfile(yp)):
            continue
        X = np.load(Xp, mmap_mode="r")
        y = np.load(yp, mmap_mode="r")
        N = len(X)
        if N == 0:
            continue

        frame_prob_sum = np.zeros((N, C), dtype=np.float64)
        frame_count    = np.zeros(N, dtype=np.int32)

        T, S = cfg_run["seq_len"], cfg_run["stride"]
        seqs = []
        if N >= T:
            for s in range(0, N - T + 1, S):
                seqs.append((s, s+T))
            tail_start = ((N - T + S - 1) // S) * S
            if tail_start < N and tail_start > (N - T):
                seqs.append((tail_start, tail_start + T))
        else:
            seqs.append((0, T))

        bs = int(cfg_run["batch_size"])
        for i in range(0, len(seqs), bs):
            chunk = seqs[i:i+bs]
            starts, real_lens, batch = [], [], []
            for (a, b) in chunk:
                starts.append(a)
                rl = min(T, max(0, N - a))
                real_lens.append(rl)
                if b <= N:
                    win = X[a:b, :][:, col_idx]
                else:
                    sel = X[:, col_idx]
                    win = np.empty((T, sel.shape[1]), dtype=sel.dtype)
                    k = max(0, N - 1)
                    if rl > 0: win[:rl] = sel[a:N]
                    win[rl:] = sel[k]
                batch.append(torch.from_numpy(win).float())

            xb = torch.stack(batch, dim=0).to(DEVICE, non_blocking=True)
            B, L, D = xb.shape
            xb = xb.view(B*L, D); xb = preproc(xb); xb = xb.view(B, L, D)
            probs = torch.softmax(model(xb), dim=1).detach().cpu().numpy()  # (B,C)

            for j, (a, rl) in enumerate(zip(starts, real_lens)):
                if rl <= 0: 
                    continue
                frame_prob_sum[a:a+rl] += probs[j]
                frame_count[a:a+rl]    += 1

        counts   = np.clip(frame_count, 1, None)[:, None]
        avg_prob_frames = frame_prob_sum / counts           # (N, C)
        vid_prob = avg_prob_frames.mean(axis=0)             # (C,)
        pred = int(np.argmax(vid_prob))

        vals, counts_gt = np.unique(y, return_counts=True)
        y_mode = int(vals[np.argmax(counts_gt)])

        y_true_v.append(y_mode)
        y_pred_v.append(pred)

    return np.array(y_true_v, dtype=np.int64), np.array(y_pred_v, dtype=np.int64)

@torch.no_grad()
def _eval_video_level_from_frames(model, video_ids, cfg_run, master_feature_cols, preproc) -> float:
    y_true, y_pred = _collect_video_preds_from_frames(model, video_ids, cfg_run, master_feature_cols, preproc)
    return float((y_true == y_pred).mean()) if y_true.size else float("nan")

# ======================
#       GRID: TRAIN 1 CONFIG
# ======================
def _build_optimizer(name: str, params, lr: float, wd: float):
    name = name.lower()
    if name == "adam": return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":  return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")

def _config_tag(cfg_run: dict) -> str:
    parts, fb = [], []
    if cfg_run["use_resnet"]: fb.append("RES")
    if cfg_run["use_vgg"]:    fb.append("VGG")
    if cfg_run["use_au_c"]:   fb.append("AUc")
    if cfg_run["use_au_r"]:   fb.append("AUr")
    if not fb: fb = ["RAW"]
    parts.append("-".join(fb))
    parts.append(f"FF({cfg_run['ff_hidden']},{cfg_run['ff_hidden2']})")
    parts.append(f"RNN{cfg_run['rnn_type'].upper()}_h{cfg_run['rnn_hidden']}_l{cfg_run['rnn_layers']}_bi{int(cfg_run['bidirectional'])}")
    parts.append(f"T{cfg_run['seq_len']}_S{cfg_run['stride']}")
    parts.append(f"{cfg_run['optimizer']}_lr{cfg_run['lr']}_wd{cfg_run['weight_decay']}")
    parts.append(f"do{cfg_run['dropout']}_bs{cfg_run['batch_size']}")
    return "__".join(parts)

def _cfg_dir_for(cfg_run: dict) -> str:
    d = os.path.join(GRID_OUT_DIR, _config_tag(cfg_run))
    os.makedirs(d, exist_ok=True)
    return d

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

def _iter_grid():
    for feat in FEATURE_SETS:
        for ff in FF_ARCH_GRID:
            for rnn_type in RNN_TYPES:
                for rnn in RNN_ARCH_GRID:
                    for T in SEQ_LENGTHS:
                        for S in strides_for(T):
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
                                                    "ff_hidden": ff["hidden_dim"],
                                                    "ff_hidden2": ff["hidden_dim2"],
                                                    "dropout": dr,
                                                    "rnn_type": rnn_type,
                                                    "rnn_hidden": rnn["hidden"],
                                                    "rnn_layers": rnn["layers"],
                                                    "bidirectional": False,
                                                    "seq_len": T,
                                                    "stride": S,
                                                    "optimizer": opt,
                                                    "lr": lr,
                                                    "weight_decay": wd,
                                                    "batch_size": bs,
                                                    "norm_mode": "none" if not DO_STANDARDIZE else "zscore",
                                                    "keep_au_c_raw": KEEP_AU_C_RAW,
                                                }

def _make_model_and_preproc(cfg_run: dict,
                            master_feature_cols: List[str],
                            for_ids_stats: List[str],
                            feature_cols_from: List[str]|None = None):
    """
    Helper to build model + preprocessor (computes scaler if needed).
    feature_cols_from: if provided, avoid rebuilding dataset just to know selected columns.
    """
    dummy_ds = None
    if feature_cols_from is None:
        dummy_ds = StreamingSequenceDataset(for_ids_stats, cfg_run, master_feature_cols)
        feature_cols_from = dummy_ds.feature_cols
        input_dim = dummy_ds.input_dim
        col_idx = dummy_ds.col_idx
    else:
        # derive col_idx from master_feature_cols + feature_cols_from
        idx = [master_feature_cols.index(n) for n in feature_cols_from]
        col_idx = np.asarray(idx, dtype=np.int64)
        input_dim = len(feature_cols_from)

    cfg_norm = cfg_run["norm_mode"].lower()
    if cfg_norm in ("zscore", "zscore+l2"):
        mean, std = compute_mean_std_from_cache(
            video_ids=for_ids_stats,
            master_feature_cols=master_feature_cols,
            col_idx=col_idx,
            device=DEVICE,
            batch_size=max(131072, int(cfg_run["batch_size"]) * int(cfg_run["seq_len"]))
        )
        adj_mean = mean.clone(); adj_std = std.clone()
        if cfg_run.get("keep_au_c_raw", True):
            auc_idx = [i for i, n in enumerate(feature_cols_from) if n.endswith("_c")]
            if auc_idx:
                adj_mean[auc_idx] = 0.0
                adj_std[auc_idx]  = 1.0
        preproc = build_preprocessor(cfg_norm, feature_cols_from, DEVICE, adj_mean, adj_std, keep_au_c_raw=False)
    else:
        preproc = build_preprocessor(cfg_norm, feature_cols_from, DEVICE, None, None,
                                     keep_au_c_raw=cfg_run.get("keep_au_c_raw", True))

    model = TemporalFFRNN(
        input_dim=input_dim,
        ff_hidden=int(cfg_run["ff_hidden"]),
        ff_hidden2=None if cfg_run["ff_hidden2"] is None else int(cfg_run["ff_hidden2"]),
        dropout=float(cfg_run["dropout"]),
        rnn_type=str(cfg_run["rnn_type"]),
        rnn_hidden=int(cfg_run["rnn_hidden"]),
        rnn_layers=int(cfg_run["rnn_layers"]),
        bidirectional=bool(cfg_run["bidirectional"]),
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)

    return model, preproc, feature_cols_from, col_idx

def train_one_config(cfg_run: dict, master_feature_cols: List[str]) -> Dict[str, float]:
    cfg_dir = _cfg_dir_for(cfg_run)
    last_ckpt = os.path.join(cfg_dir, "last_ckpt.pt")
    best_path = os.path.join(cfg_dir, "best_state.pt")
    metrics_path = os.path.join(cfg_dir, "metrics.json")

    # Skip if done
    if SKIP_DONE_CONFIGS and os.path.isfile(metrics_path):
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            if m.get("done", False):
                print(f"[skip] {os.path.basename(cfg_dir)} already done.")
                return {
                    "val_seq_acc": float(m.get("val_seq_acc", "nan")),
                    "val_vid_acc": float(m.get("val_vid_acc", "nan")),
                    "val_frame_acc": float(m.get("val_frame_acc", "nan")),
                    "val_loss": float(m.get("val_loss", "nan")),
                    "test_seq_acc": float(m.get("test_seq_acc", "nan")),
                    "test_vid_acc": float(m.get("test_vid_acc", "nan")),
                    "test_frame_acc": float(m.get("test_frame_acc", "nan")),
                }
        except Exception:
            pass

    # Splits
    with _time("Read split lists"):
        train_ids = _read_list(_require_file(TRAIN_LIST, "TRAIN_LIST"))
        val_ids   = _read_list(_require_file(VAL_LIST,   "VAL_LIST"))
        test_ids  = _read_list(_require_file(TEST_LIST,  "TEST_LIST"))
        if INCLUDE_LIST or EXCLUDE_LIST:
            train_ids = _apply_include_exclude(train_ids, INCLUDE_LIST, EXCLUDE_LIST)
            val_ids   = _apply_include_exclude(val_ids,   INCLUDE_LIST, EXCLUDE_LIST)
            test_ids  = _apply_include_exclude(test_ids,  INCLUDE_LIST, EXCLUDE_LIST)

    # Datasets
    with _time("Build StreamingSequenceDataset (train)"):
        train_ds = StreamingSequenceDataset(train_ids, cfg_run, master_feature_cols)
    with _time("Build StreamingSequenceDataset (val)"):
        val_ds   = StreamingSequenceDataset(val_ids,   cfg_run, master_feature_cols)

    # Loaders
    with _time(f"Dataloader (train) bs={cfg_run['batch_size']} N={len(train_ds)}"):
        train_loader = _make_loader(train_ds, cfg_run["batch_size"], shuffle=True)
    with _time(f"Dataloader (val)   bs={cfg_run['batch_size']} N={len(val_ds)}"):
        val_loader   = _make_loader(val_ds,   cfg_run["batch_size"], shuffle=False)

    # ---------- PREPROCESSOR ----------
    cfg_norm = cfg_run["norm_mode"].lower()   # 'none' | 'l2' | 'zscore' | 'zscore+l2'
    if cfg_norm in ("zscore", "zscore+l2"):
        train_ids_stats = train_ids
        mean, std = compute_mean_std_from_cache(
            video_ids=train_ids_stats,
            master_feature_cols=master_feature_cols,
            col_idx=train_ds.col_idx,
            device=DEVICE,
            batch_size=max(131072, int(cfg_run["batch_size"]) * int(cfg_run["seq_len"]))
       )
        adj_mean = mean.clone(); adj_std = std.clone()
        if cfg_run.get("keep_au_c_raw", True):
            auc_idx = [i for i, n in enumerate(train_ds.feature_cols) if n.endswith("_c")]
            if auc_idx:
                adj_mean[auc_idx] = 0.0
                adj_std[auc_idx]  = 1.0
        scaler_path = os.path.join(cfg_dir, "scaler.pt")
        torch.save({"mean": adj_mean.cpu(), "std": adj_std.cpu()}, scaler_path)
        print(f"[norm] saved scaler -> {scaler_path}")
        preproc = build_preprocessor(
            norm_mode=cfg_norm, feature_cols=train_ds.feature_cols,
            device=DEVICE, mean=adj_mean, std=adj_std, keep_au_c_raw=False
        )
    else:
        D = len(train_ds.feature_cols)
        scaler_path = os.path.join(cfg_dir, "scaler.pt")
        torch.save({"mean": torch.zeros(D), "std": torch.ones(D)}, scaler_path)
        preproc = build_preprocessor(
            norm_mode=cfg_norm, feature_cols=train_ds.feature_cols,
            device=DEVICE, mean=None, std=None, keep_au_c_raw=cfg_run.get("keep_au_c_raw", True)
        )

    # Persist per-config feature cols (for test-time)
    featcols_local = os.path.join(cfg_dir, "feature_cols.json")
    with open(featcols_local, "w") as f:
        json.dump(train_ds.feature_cols, f)

    # Model + opt + sched
    model = TemporalFFRNN(
        input_dim=train_ds.input_dim,
        ff_hidden=int(cfg_run["ff_hidden"]),
        ff_hidden2=None if cfg_run["ff_hidden2"] is None else int(cfg_run["ff_hidden2"]),
        dropout=float(cfg_run["dropout"]),
        rnn_type=str(cfg_run["rnn_type"]),
        rnn_hidden=int(cfg_run["rnn_hidden"]),
        rnn_layers=int(cfg_run["rnn_layers"]),
        bidirectional=bool(cfg_run["bidirectional"]),
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)

    optim = _build_optimizer(cfg_run["optimizer"], model.parameters(), lr=float(cfg_run["lr"]), wd=float(cfg_run["weight_decay"]))
    ce = nn.CrossEntropyLoss()
    plateau_mode = "max" if ES_MONITOR == "val_acc" else "min"
    scheduler = ReduceLROnPlateau(optim, mode=plateau_mode, factor=PLATEAU_FACTOR,
                                  patience=PLATEAU_PATIENCE, min_lr=MIN_LR)

    # Resume
    start_epoch = 1
    best_metric = -math.inf if ES_MONITOR == "val_acc" else math.inf
    no_improve  = 0
    last_ckpt = os.path.join(cfg_dir, "last_ckpt.pt")
    if RESUME_CONFIGS and os.path.isfile(last_ckpt):
        print(f"[resume] {last_ckpt}")
        ckpt = _load_ckpt(last_ckpt, DEVICE)
        try: model.load_state_dict(ckpt["model"])
        except RuntimeError as e: print(f"[warn] state_dict mismatch; starting fresh: {e}")
        try: optim.load_state_dict(ckpt["optim"])
        except Exception: pass
        if ckpt.get("sched") is not None:
            try: scheduler.load_state_dict(ckpt["sched"])
            except Exception: pass
        best_metric = ckpt.get("best_metric", best_metric)
        no_improve  = int(ckpt.get("no_improve", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    # Train (grid phase)
    print(f"[TRAINING] Starting training for {EPOCHS - start_epoch + 1} epochs…")
    for epoch in range(start_epoch, EPOCHS + 1):
        ep_t0 = time.time()
        model.train()
        run_loss, data_wait, compute = 0.0, 0.0, 0.0
        correct, total = 0, 0 

        for xb, yb in train_loader:
            t0 = time.time()
            xb = xb.to(DEVICE, non_blocking=True).float()
            yb = yb.to(DEVICE, non_blocking=True)
            t1 = time.time()

            B, L, D = xb.shape
            xb = xb.view(B*L, D); xb = preproc(xb); xb = xb.view(B, L, D)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            if CLIP_NORM and CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optim.step()
            t2 = time.time()
            
            run_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.numel()

            data_wait += (t1 - t0)
            compute   += (t2 - t1)

        tr_loss = run_loss / max(1, len(train_loader))
        tr_seq_acc = correct / max(1, total) 

        va_loss, va_seq_acc, _, _ = _eval_epoch_seq(model, val_loader, DEVICE, preproc)

        sched_value = va_seq_acc if plateau_mode == "max" else va_loss
        improved = (sched_value > best_metric) if ES_MONITOR == "val_acc" else (sched_value < best_metric)
        if improved:
            best_metric = sched_value; no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        scheduler.step(sched_value)

        ep_time = time.time() - ep_t0
        print(f"[{os.path.basename(cfg_dir)}] ep {epoch:03d} | tr_loss {tr_loss:.4f} | val_loss {va_loss:.4f} "
              f"| val_seq_acc {va_seq_acc:.4f} | lr {optim.param_groups[0]['lr']:.2e} "
              f"| no_improve {no_improve}/{ES_PATIENCE} | ep_time {ep_time:.2f}s "
              f"(data {data_wait:.2f}s, compute {compute:.2f}s)")

        _save_ckpt(last_ckpt, epoch, model, optim, scheduler, best_metric, no_improve, best_state_path=best_path)
        if no_improve >= ES_PATIENCE:
            print("[early-stop] patience reached."); break

    # Final val with best
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    # ===== Validation metrics (sequence) =====
    val_loss, val_seq_acc, val_frame_acc, _ = _eval_epoch_seq(model, val_loader, DEVICE, preproc)

    # ===== Per-config TEST evaluation (REQUIRED) =====
    # Build TEST dataset/loader with same cfg
    test_ids = _read_list(_require_file(TEST_LIST, "TEST_LIST"))
    if INCLUDE_LIST or EXCLUDE_LIST:
        test_ids = _apply_include_exclude(test_ids, INCLUDE_LIST, EXCLUDE_LIST)
    with _time("Build StreamingSequenceDataset (TEST)"):
        test_ds = StreamingSequenceDataset(test_ids, cfg_run, master_feature_cols)
    with _time(f"Dataloader (TEST) bs={cfg_run['batch_size']} N={len(test_ds)}"):
        test_loader = _make_loader(test_ds, cfg_run["batch_size"], shuffle=False)

    # Sequence-level TEST
    test_loss, test_seq_acc, _, _ = _eval_epoch_seq(model, test_loader, DEVICE, preproc)

    # Frame-level TEST (prob aggregation)
    y_true_fr, y_pred_fr = _collect_frame_preds_stream(model, test_ids, cfg_run, master_feature_cols, preproc)
    test_frame_acc = float((y_true_fr == y_pred_fr).mean()) if y_true_fr.size else float("nan")

    # Video-level TEST (avg frame prob -> argmax)
    y_true_vid, y_pred_vid = _collect_video_preds_from_frames(model, test_ids, cfg_run, master_feature_cols, preproc)
    test_vid_acc = float((y_true_vid == y_pred_vid).mean()) if y_true_vid.size else float("nan")

    # metrics.json (VAL + TEST)
    out = {
        "done": True,
        "val_seq_acc": float(val_seq_acc),
        "val_vid_acc": float(_eval_video_level_from_frames(model, val_ids, cfg_run, master_feature_cols, preproc)),
        "val_frame_acc": float(val_frame_acc),
        "val_loss": float(val_loss),
        "test_seq_acc": float(test_seq_acc),
        "test_vid_acc": float(test_vid_acc),
        "test_frame_acc": float(test_frame_acc),
        "use_vgg": bool(cfg_run["use_vgg"]),
        "use_resnet": bool(cfg_run["use_resnet"]),
        "use_au_c": bool(cfg_run["use_au_c"]),
        "use_au_r": bool(cfg_run["use_au_r"]),
        "ff_hidden": int(cfg_run["ff_hidden"]),
        "ff_hidden2": None if cfg_run["ff_hidden2"] is None else int(cfg_run["ff_hidden2"]),
        "dropout": float(cfg_run["dropout"]),
        "rnn_type": str(cfg_run["rnn_type"]),
        "rnn_hidden": int(cfg_run["rnn_hidden"]),
        "rnn_layers": int(cfg_run["rnn_layers"]),
        "bidirectional": bool(cfg_run["bidirectional"]),
        "seq_len": int(cfg_run["seq_len"]),
        "stride": int(cfg_run["stride"]),
        "optimizer": str(cfg_run["optimizer"]),
        "lr": float(cfg_run["lr"]),
        "weight_decay": float(cfg_run["weight_decay"]),
        "batch_size": int(cfg_run["batch_size"]),
        "norm_mode": str(cfg_run["norm_mode"]),
        "keep_au_c_raw": bool(cfg_run["keep_au_c_raw"]),
        "tag": _config_tag(cfg_run),
    }
    tmp = os.path.join(_cfg_dir_for(cfg_run), "metrics.json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, os.path.join(_cfg_dir_for(cfg_run), "metrics.json"))

    print(f"[TEST] seq_acc={test_seq_acc:.4f} | frame_acc={test_frame_acc:.4f} | vid_acc={test_vid_acc:.4f}")

    return out

# ======================
#   FINAL RETRAIN (TRAIN+VAL) WITH EARLY STOP ON TRAIN ACC
# ======================
def _save_confmat_and_report(root_dir: str, prefix: str,
                             y_true: np.ndarray, y_pred: np.ndarray):
    os.makedirs(root_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_path_npy = os.path.join(root_dir, f"{prefix}_confusion_matrix.npy")
    cm_path_csv = os.path.join(root_dir, f"{prefix}_confusion_matrix.csv")
    np.save(cm_path_npy, cm)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(cm_path_csv)
    print(f"[final] saved confusion matrix -> {cm_path_npy} / {cm_path_csv}")

    report_dict = classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    rep_json = os.path.join(root_dir, f"{prefix}_classification_report.json")
    rep_txt  = os.path.join(root_dir, f"{prefix}_classification_report.txt")
    with open(rep_json, "w") as f:
        json.dump(report_dict, f, indent=2)
    with open(rep_txt, "w") as f:
        f.write(classification_report(
            y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES, zero_division=0
        ))
    print(f"[final] saved classification report -> {rep_json} / {rep_txt}")

def _train_final_on_trainval(cfg_run: dict, master_feature_cols: List[str], tag: str):
    """
    Retrain on TRAIN+VAL combined with early stopping on TRAIN accuracy.
    Evaluate on TEST and save confusion matrices + reports for both
    sequence-level and video-level predictions.
    """
    # --- build combined ids ---
    train_ids = _read_list(_require_file(TRAIN_LIST, "TRAIN_LIST"))
    val_ids   = _read_list(_require_file(VAL_LIST,   "VAL_LIST"))
    test_ids  = _read_list(_require_file(TEST_LIST,  "TEST_LIST"))
    if INCLUDE_LIST or EXCLUDE_LIST:
        train_ids = _apply_include_exclude(train_ids, INCLUDE_LIST, EXCLUDE_LIST)
        val_ids   = _apply_include_exclude(val_ids,   INCLUDE_LIST, EXCLUDE_LIST)
        test_ids  = _apply_include_exclude(test_ids,  INCLUDE_LIST, EXCLUDE_LIST)
    trainval_ids = sorted(set(train_ids) | set(val_ids))

    # --- dataset + loaders ---
    trainval_ds = StreamingSequenceDataset(trainval_ids, cfg_run, master_feature_cols)
    trainval_loader = _make_loader(trainval_ds, cfg_run["batch_size"], shuffle=True)

    # --- preproc + model ---
    model, preproc, feature_cols, _ = _make_model_and_preproc(
        cfg_run, master_feature_cols, for_ids_stats=trainval_ids
    )

    optim = _build_optimizer(cfg_run["optimizer"], model.parameters(), lr=float(cfg_run["lr"]), wd=float(cfg_run["weight_decay"]))
    ce = nn.CrossEntropyLoss()

    # --- early stop on TRAIN accuracy ---
    best_acc = -1.0
    no_imp = 0
    final_dir = os.path.join(FINAL_DIR, tag)
    os.makedirs(final_dir, exist_ok=True)
    best_path = os.path.join(final_dir, "final_best_state.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for xb, yb in trainval_loader:
            xb = xb.to(DEVICE, non_blocking=True).float()
            yb = yb.to(DEVICE, non_blocking=True)
            B, L, D = xb.shape
            xb = xb.view(B*L, D); xb = preproc(xb); xb = xb.view(B, L, D)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            if CLIP_NORM and CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optim.step()
            run_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.numel()
        tr_loss = run_loss / max(1, len(trainval_loader))
        tr_acc  = correct / max(1, total)
        improved = tr_acc > best_acc
        if improved:
            best_acc = tr_acc
            no_imp = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_imp += 1
        print(f"[final:{tag}] ep {epoch:03d} | tr_loss {tr_loss:.4f} | tr_seq_acc {tr_acc:.4f} | best {best_acc:.4f} | no_imp {no_imp}/{FINAL_PATIENCE_ESTOP}")
        if no_imp >= FINAL_PATIENCE_ESTOP:
            print(f"[final:{tag}] early-stop (train-acc) reached.")
            break

    # load best
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    # ---- TEST evaluation ----
    test_ds = StreamingSequenceDataset(test_ids, cfg_run, master_feature_cols)
    test_loader = _make_loader(test_ds, cfg_run["batch_size"], shuffle=False)

    # sequence-level
    test_loss, test_seq_acc, _, _ = _eval_epoch_seq(model, test_loader, DEVICE, preproc)
    y_true_seq, y_pred_seq = _collect_seq_preds(model, test_loader, DEVICE, preproc)
    _save_confmat_and_report(final_dir, "seq", y_true_seq, y_pred_seq)

    # video-level (via frames)
    y_true_vid, y_pred_vid = _collect_video_preds_from_frames(model, test_ids, cfg_run, master_feature_cols, preproc)
    test_vid_acc = float((y_true_vid == y_pred_vid).mean()) if y_true_vid.size else float("nan")
    _save_confmat_and_report(final_dir, "video", y_true_vid, y_pred_vid)

    # also frame-level (optional; not required for confusion matrix)
    y_true_fr, y_pred_fr = _collect_frame_preds_stream(model, test_ids, cfg_run, master_feature_cols, preproc)
    test_frame_acc = float((y_true_fr == y_pred_fr).mean()) if y_true_fr.size else float("nan")

    metrics = {
        "final_on": "train+val",
        "monitor": "train_seq_acc",
        "best_train_seq_acc": float(best_acc),
        "test_seq_acc": float(test_seq_acc),
        "test_vid_acc": float(test_vid_acc),
        "test_frame_acc": float(test_frame_acc),
        "classes": CLASS_NAMES,
        "best_state_path": best_path,
    }
    with open(os.path.join(final_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[final:{tag}] TEST: seq_acc={test_seq_acc:.4f} | vid_acc={test_vid_acc:.4f} | frame_acc={test_frame_acc:.4f}")
    return metrics

# ======================
#           MAIN
# ======================
def main():
    # seeds
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    # Build/load master feature list
    master_feature_cols = get_master_feature_cols()

    # Grid
    grid = list(_iter_grid())
    print(f"[grid] total configurations: {len(grid)}")

    results = []
    for i, cfg_run in enumerate(grid, 1):
        tag = _config_tag(cfg_run)
        print("\n" + "="*100)
        print(f"[{i}/{len(grid)}] {tag}")
        print("="*100)

        cfg_t0 = time.time()
        info = train_one_config(cfg_run, master_feature_cols)
        cfg_dur = time.time() - cfg_t0

        row = {**cfg_run, **info, "tag": tag, "config_time_sec": round(cfg_dur, 2)}
        results.append(row)

        print(f"[VAL]  seq_acc={info['val_seq_acc']:.4f} | vid_acc={info['val_vid_acc']:.4f} | frame_acc≈{info['val_frame_acc']:.4f}")
        print(f"[TEST] seq_acc={info['test_seq_acc']:.4f} | vid_acc={info['test_vid_acc']:.4f} | frame_acc={info['test_frame_acc']:.4f}")
        print(f"[TIMING] Configuration total time: {cfg_dur:.2f}s")

    # summary CSV
    df = pd.DataFrame(results).sort_values("val_seq_acc", ascending=False)
    out_csv = os.path.join(ART_DIR_SUB, "rnn_grid_results_RAVDESS.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[done] wrote grid results -> {out_csv}")
    show_cols = ["tag","val_seq_acc","val_vid_acc","val_frame_acc","test_seq_acc","test_vid_acc","test_frame_acc"]
    try:
        print(df[show_cols].to_string(index=False))
    except Exception:
        print(df.to_string(index=False))

    # ======================
    #   FINAL RETRAINING
    # ======================
    if len(df) == 0:
        print("[final] No configurations to retrain.")
        return

    # Pick best by SEQUENCE validation accuracy
    #best_seq_row = df.iloc[0]
    df_sorted_seq=df.sort_values("test_seq_acc", ascending=False)
    best_seq_row = df_sorted_seq.iloc[0]
    best_seq_tag = best_seq_row["tag"]
    print(f"\n[final] Best-by-SEQ model: {best_seq_tag} (test_seq_acc={best_seq_row['test_seq_acc']:.4f})")

    # Pick best by VIDEO validation accuracy
    df_sorted_vid = df.sort_values("test_vid_acc", ascending=False)
    best_vid_row = df_sorted_vid.iloc[0]
    best_vid_tag = best_vid_row["tag"]
    print(f"[final] Best-by-VIDEO model: {best_vid_tag} (test_vid_acc={best_vid_row['test_vid_acc']:.4f})")

    # Rebuild cfg_run dicts from the tag rows with dtype coercion
    def row_to_cfg(row: pd.Series) -> dict:
        ff2 = _none_if_nan(row["ff_hidden2"])
        return {
            "use_vgg":       _py_bool(row["use_vgg"]),
            "use_resnet":    _py_bool(row["use_resnet"]),
            "use_au_c":      _py_bool(row["use_au_c"]),
            "use_au_r":      _py_bool(row["use_au_r"]),
            "ff_hidden":     _py_int(row["ff_hidden"]),
            "ff_hidden2":    None if ff2 is None else _py_int(ff2),
            "dropout":       _py_float(row["dropout"]),
            "rnn_type":      str(row["rnn_type"]),
            "rnn_hidden":    _py_int(row["rnn_hidden"]),
            "rnn_layers":    _py_int(row["rnn_layers"]),
            "bidirectional": _py_bool(row["bidirectional"]),
            "seq_len":       _py_int(row["seq_len"]),
            "stride":        _py_int(row["stride"]),
            "optimizer":     str(row["optimizer"]),
            "lr":            _py_float(row["lr"]),
            "weight_decay":  _py_float(row["weight_decay"]),
            "batch_size":    _py_int(row["batch_size"]),
            "norm_mode":     str(row["norm_mode"]),
            "keep_au_c_raw": _py_bool(row["keep_au_c_raw"]),
        }

    cfg_best_seq = row_to_cfg(best_seq_row)
    cfg_best_vid = row_to_cfg(best_vid_row)

    # Final retrain (train+val, early stop on TRAIN accuracy), evaluate on TEST
    print("\n[final] Retraining best-by-SEQ on TRAIN+VAL and evaluating on TEST …")
    _ = _train_final_on_trainval(cfg_best_seq, master_feature_cols, tag=f"best_by_seq__{best_seq_tag}")

    print("\n[final] Retraining best-by-VIDEO on TRAIN+VAL and evaluating on TEST …")
    _ = _train_final_on_trainval(cfg_best_vid, master_feature_cols, tag=f"best_by_video__{best_vid_tag}")

if __name__ == "__main__":
    main()

