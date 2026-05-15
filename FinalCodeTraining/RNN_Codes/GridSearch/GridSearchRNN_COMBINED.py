#!/usr/bin/env python3
# grid_search_rnn_streaming_combined.py
# Grid search for CNN+RNN with streaming windows on COMBINED (CREMA-D + RAVDESS; 6-class canonical).
# Per-config: train on TRAIN with early stop on VAL, then EVALUATE ON TEST (seq/frame/video).
# After grid: do three final retrains (TRAIN+VAL) with early stop on TRAIN ACC:
#   - best-by-frame test acc
#   - best-by-sequence test acc
#   - best-by-video test acc
# Each final retrain writes confusion matrices and full classification reports.

import os, sys, json, math, time, random, gc
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

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

# ---- project imports (yours) ----
#import config as CFG
from utils.features import save_feature_cols, Standardize
from models.CNN_RNNmodel import TemporalFFRNN

# ======================
#      CONSTANTS
# ======================
MAX_EPOCHS      = 300
ES_PATIENCE_VAL = 15              # for VAL-based early stop in grid
ES_PATIENCE_TR  = 15              # for TRAIN-ACC-based early stop in final retrain
ES_MONITOR      = "val_acc"       # "val_acc" or "val_loss" during grid
SKIP_FIRST_N    = 0
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME          = True
SEED            = 42

CPU_COUNT       = os.cpu_count() or 4
NUM_WORKERS     = min(26, max(0, CPU_COUNT - 2))
PIN_MEMORY      = torch.cuda.is_available()
PREFETCH_FACTOR = 8

def _loader_kws():
    base = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    if NUM_WORKERS > 0:
        base.update(dict(prefetch_factor=PREFETCH_FACTOR, persistent_workers=True))
    return base

def _safe_collate_seq(batch):
    # batch: list of (B=T window, D), label -> we receive already batched tensors (T,D)
    xs, ys = zip(*batch)
    X = torch.stack(xs, 0)  # (B, T, D)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y

def _safe_collate_frame(batch):
    xs, ys = zip(*batch)
    X = torch.stack(xs, 0)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y

def _current_lr(optim):
    return optim.param_groups[0]['lr'] if optim is not None and optim.param_groups else float('nan')

def _set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ======================
#   COMBINED DATA PATHS
# ======================
# Combined split TXT files you generated earlier (ids formatted as "crema::<vid>" or "ravdess::Actor_xx/<vid>")
COMB_SPLIT_PATH = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/data/"
TRAIN_LIST = os.path.join(COMB_SPLIT_PATH, "train_videos_COMBINED.txt")
VAL_LIST   = os.path.join(COMB_SPLIT_PATH, "val_videos_COMBINED.txt")
TEST_LIST  = os.path.join(COMB_SPLIT_PATH, "test_videos_COMBINED.txt")

# Dataset roots
CREMA_ROOT   = "/media/root918/OS/[REDACTED]Project/CREMA-D/copiedFiles/"
RAVDESS_ROOT = "/media/root918/OS/[REDACTED]Project/copiedFilesRAVDESS/"

# Where artifacts go
PROJECT_DIR  = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD"
ART_DIR_TAG  = "combined_GridSearch_unscaled_RNN"
ART_DIR_SUB  = os.path.join(PROJECT_DIR, "artifacts", ART_DIR_TAG)
GRID_OUT_DIR = os.path.join(ART_DIR_SUB, "grid_COMBINED_RNN")
CONFIGS_DIR  = os.path.join(ART_DIR_SUB, "configs_COMBINED_RNN")
os.makedirs(GRID_OUT_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)

# Existing caches: candidate directories under each video folder
CANDIDATE_CACHE_DIRS = ["cache"]

# ======================
#   LABEL SPACE (6-class canonical)
# ======================
CANONICAL = ["angry","disgust","fear","happy","neutral","sad"]
EMOTION_TO_IDX = {e:i for i,e in enumerate(CANONICAL)}
IDX_TO_EMO     = {v:k for k,v in EMOTION_TO_IDX.items()}

# Map variants to canonical; None => drop frame
ALIASES = {
    "angry":"angry","anger":"angry","Anger":"angry","ANGER":"angry",
    "disgust":"disgust","Disgust":"disgust","DISGUST":"disgust",
    "fear":"fear","fearful":"fear","Fear":"fear","FEAR":"fear","Fearful":"fear",
    "happy":"happy","Happy":"happy","HAPPY":"happy",
    "neutral":"neutral","Neutral":"neutral","NEUTRAL":"neutral",
    "sad":"sad","sadness":"sad","Sad":"sad","SAD":"sad",
    # dropped classes if they appear:
    "calm":None,"Calm":None,"CALM":None,
    "surprised":None,"Surprised":None,"SURPRISED":None,
    "surprise":None,"Surprise":None,"SURPRISE":None,
}

def _ravdess_label_to_idx(s: str) -> Optional[int]:
    if s is None: return None
    t = str(s).strip()
    t = ALIASES.get(t, ALIASES.get(t.lower(), None))
    if t is None: return None
    return EMOTION_TO_IDX.get(t, None)

# CREMA mapping you used for MLP:
# {'H':0,'S':1,'A':2,'N':3,'D':4,'F':5}
# -> canonical: [H,S,A,N,D,F] → [happy, sad, angry, neutral, disgust, fear] → indices [3,5,0,4,1,2]
CREMA_INT_TO_CANON = np.array([3, 5, 0, 4, 1, 2], dtype=np.int64)

# ======================
#   FEATURE SELECTION GRID (suffix-based)
# ======================
DO_STANDARDIZE = False
KEEP_AU_C_RAW  = True

FF_ARCH_GRID = [
    {"hidden_dim": 512,  "hidden_dim2": None},
    {"hidden_dim": 1024, "hidden_dim2": None},
    {"hidden_dim": 1024, "hidden_dim2": 512},
    {"hidden_dim": 512,  "hidden_dim2": 256},
]
OPTIMIZERS   = ["adam"]
LRS          = [1e-4]
WEIGHT_DECAY = [1e-5]
DROPOUTS     = [0.5]
BATCH_SIZES  = [512]

FEATURE_SETS = [
    {"name":"RESNET+AU",     "use_vgg":False, "use_resnet":True,  "use_au_c":True,  "use_au_r":True },
    {"name":"VGG+RESNET+AU", "use_vgg":True,  "use_resnet":True,  "use_au_c":True,  "use_au_r":True },
    {"name":"RESNET",        "use_vgg":False, "use_resnet":True,  "use_au_c":False, "use_au_r":False},
]

# Sequence lengths/strides
SEQ_LENGTHS = [10, 30]
def strides_for(T: int) -> List[int]:
    # same heuristic as before: T and T//2
    return sorted({T, max(1, T // 2)})

# Normalization
def _maybe_build_scaler(feature_names: List[str], loader: DataLoader, device) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    if not DO_STANDARDIZE:
        D = len(feature_names)
        return nn.Identity(), torch.zeros(D), torch.ones(D)
    # compute mean/std over frames from sequence windows (flatten T across batch)
    D = len(feature_names)
    s1 = torch.zeros(D, device=device, dtype=torch.float64)
    s2 = torch.zeros(D, device=device, dtype=torch.float64)
    n  = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True).float()  # (B,T,D)
            B, T, D_ = xb.shape
            x = xb.view(B*T, D_)
            n += x.shape[0]
            s1 += x.sum(dim=0).double()
            s2 += (x.double().pow(2)).sum(dim=0)
    mean = (s1 / max(1, n)).float()
    var  = (s2 / max(1, n)).float() - mean.pow(2)
    var  = torch.clamp(var, min=1e-12)
    std  = torch.sqrt(var)
    if KEEP_AU_C_RAW:
        auc_idx = [i for i, n in enumerate(feature_names) if n.endswith("_c")]
        if auc_idx:
            idx = torch.tensor(auc_idx, dtype=torch.long, device=device)
            mean.index_fill_(0, idx, 0.0)
            std.index_fill_(0, idx, 1.0)
    scaler = Standardize(mean.cpu(), std.cpu()).to(device)
    return scaler, mean.cpu(), std.cpu()

# ======================
#   SPLITS & SCHEMA ASSERTIONS
# ======================
def _require_file(path, desc):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")
    return path

def _read_ids(list_path: str) -> List[str]:
    with open(list_path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def _parse_id(combined_id: str) -> Tuple[str,str]:
    # "crema::<vid>" or "ravdess::Actor_01/<vid_dir>"
    if "::" not in combined_id:
        raise ValueError(f"Invalid combined id (expected 'dataset::video_id'): {combined_id}")
    ds, vid = combined_id.split("::", 1)
    ds = ds.strip().lower()
    if ds not in ("crema","ravdess"):
        raise ValueError(f"Unknown dataset tag '{ds}' in {combined_id}")
    return ds, vid

def _vid_root(dataset: str) -> str:
    return CREMA_ROOT if dataset == "crema" else RAVDESS_ROOT

def _find_cache_dir(cid: str) -> str:
    """
    Return the cache directory containing X.npy and either y.npy or y_str.npy.
    """
    ds, vid = _parse_id(cid)
    vroot = os.path.join(_vid_root(ds), vid)

    # Accept: X.npy + (y.npy or y_str.npy)
    for cdir in CANDIDATE_CACHE_DIRS:
        p = os.path.join(vroot, cdir)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "X.npy")):
            y_path = os.path.join(p, "y.npy")
            ystr_path = os.path.join(p, "y_str.npy")
            if os.path.isfile(y_path) or os.path.isfile(ystr_path):
                return p
    raise FileNotFoundError(f"No cache found for {cid}")

def _read_feature_cols(cache_dir: str):
    """
    Support both meta layouts:
      - CREMA:   {"feature_cols": [...]}
      - RAVDESS: {"feature_cols_master": [...]}
    """
    meta = os.path.join(cache_dir, "meta.json")
    if not os.path.isfile(meta):
        return None
    try:
        with open(meta) as f:
            m = json.load(f)
        cols = m.get("feature_cols", None)
        if cols is None:
            cols = m.get("feature_cols_master", None)
        return list(cols) if cols is not None else None
    except Exception:
        return None

def assert_consistent_feature_schema(ids: List[str]) -> List[str]:
    """
    Ensures every cached video has the exact same feature_cols (same set + order).
    Returns that common feature_cols if consistent; raises otherwise.
    """
    common = None
    for cid in ids:
        cdir = _find_cache_dir(cid)
        cols = _read_feature_cols(cdir)
        if cols is None:
            raise RuntimeError(f"{cid}: missing or invalid meta feature columns; cannot assert schema.")
        if common is None:
            common = cols
        else:
            if len(cols) != len(common) or any(a != b for a, b in zip(cols, common)):
                raise RuntimeError(
                    f"Inconsistent feature schema in {cid}.\n"
                    f"Expected first-video schema length {len(common)}, got {len(cols)}."
                )
    print(f"[features] Consistent feature schema confirmed: {len(common)} columns.")
    return common

def _selected_names_for_cfg(all_cols: List[str], use_vgg, use_resnet, use_au_c, use_au_r) -> List[str]:
    sel = []
    for c in all_cols:
        if c.endswith("_vgg")    and use_vgg:    sel.append(c)
        if c.endswith("_resnet") and use_resnet: sel.append(c)
        if c.endswith("_c")      and use_au_c:   sel.append(c)
        if c.endswith("_r")      and use_au_r:   sel.append(c)
    if not sel:
        raise ValueError("No columns selected by this feature combo (check suffixes and schema).")
    return sel

def _indices_from_names(all_cols: List[str], selected_names: List[str]) -> np.ndarray:
    name_to_pos = {n:i for i,n in enumerate(all_cols)}
    return np.asarray([name_to_pos[n] for n in selected_names], dtype=np.int64)

# ======================
#   CACHE LOADERS
# ======================
def _load_cached_arrays(cid: str):
    """
    Load X and y for a combined id.
    - Prefer y.npy if present.
    - Otherwise use y_str.npy and decode byte strings to str if needed.
    """
    cdir = _find_cache_dir(cid)
    x_path = os.path.join(cdir, "X.npy")
    y_int_path = os.path.join(cdir, "y.npy")
    y_str_path = os.path.join(cdir, "y_str.npy")

    if not os.path.isfile(x_path):
        raise FileNotFoundError(f"Missing X.npy at {cdir}")
    X = np.load(x_path, mmap_mode="r")

    y = None
    if os.path.isfile(y_int_path):
        y = np.load(y_int_path, allow_pickle=True)
    elif os.path.isfile(y_str_path):
        y = np.load(y_str_path, allow_pickle=True)
        if y.dtype.kind == 'S':  # bytes
            y = np.char.decode(y, 'utf-8')
    else:
        raise FileNotFoundError(f"Missing y.npy / y_str.npy at {cdir}")

    return X, y

# ======================
#   DATASETS
# ======================
class StreamingSequenceDatasetCombined(Dataset):
    """
    Builds overlapping sequences from per-video caches on the fly (combined CREMA/RAVDESS).
    Window label = mode of canonical 6-class labels over real frames only.
    """
    def __init__(self, ids: List[str], sel_idx: np.ndarray, sel_names: List[str],
                 seq_len: int, stride: int, skip_first_n: int):
        self.ids = list(ids)
        self.sel_idx = np.asarray(sel_idx, dtype=np.int64)
        self.sel_names = list(sel_names)
        self.T = int(seq_len); self.S = int(stride)
        self.skip_first_n = int(skip_first_n)

        self.index = []   # (vid_idx, start)
        self._arrays = [] # list of (X_memmap, y_mapped_np)
        self._lengths = []# per-video lengths after skipping
        self.input_dim = len(self.sel_idx)

        total_seqs = 0
        used = 0
        for cid in self.ids:
            ds, _ = _parse_id(cid)
            X, y_raw = _load_cached_arrays(cid)
            n = X.shape[0]
            start = min(self.skip_first_n, n) if self.skip_first_n > 0 else 0

            # map labels to 6-class canonical
            if ds == "crema":
                y_local = np.asarray(y_raw, dtype=np.int64)
                mask = (y_local >= 0) & (y_local < 6) & (np.arange(n) >= start)
                keep_idx = np.where(mask)[0]
                y_map = CREMA_INT_TO_CANON[y_local[keep_idx]]
            else:
                if isinstance(y_raw, np.ndarray) and y_raw.dtype.kind == 'S':
                    y_raw = np.char.decode(y_raw, 'utf-8')
                mapped = np.array([_ravdess_label_to_idx(v) for v in y_raw], dtype=object)
                keep_idx = np.where((pd.notna(mapped)) & (np.arange(n) >= start))[0]
                if keep_idx.size == 0:
                    continue
                y_map = np.array([int(mapped[i]) for i in keep_idx], dtype=np.int64)

            if keep_idx.size == 0:
                continue

            # store arrays (memmap for X)
            Xm = np.load(_find_cache_dir(cid) + "/X.npy", mmap_mode="r")  # reload to ensure memmap
            self._arrays.append((Xm, keep_idx, y_map))
            self._lengths.append(keep_idx.size)
            vid_idx = len(self._arrays) - 1
            N = keep_idx.size

            if N >= self.T:
                for s in range(0, N - self.T + 1, self.S):
                    self.index.append((vid_idx, s))
                tail_start = ((N - self.T + self.S - 1) // self.S) * self.S
                if tail_start < N and tail_start > (N - self.T):
                    self.index.append((vid_idx, tail_start))
            else:
                self.index.append((vid_idx, 0))
            used += 1

        total_seqs = len(self.index)
        print(f"[seq-dataset] videos used: {used} | sequences: {total_seqs:,} | dim: {self.input_dim}")

        self.feature_cols = self.sel_names

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        vid_idx, start = self.index[i]
        Xm, keep_idx, y_map = self._arrays[vid_idx]
        N = keep_idx.size
        T = self.T

        # slice selected columns
        def sel_rows(a, rows):
            return np.asarray(a[rows][:, self.sel_idx], dtype=np.float32, order="C")

        if start + T <= N:
            rows = keep_idx[start:start+T]
            win = sel_rows(Xm, rows)
            lab_win = y_map[start:start+T]
            n_real = T
        else:
            rows_all = keep_idx
            sel_all = sel_rows(Xm, rows_all)
            win = np.empty((T, sel_all.shape[1]), dtype=np.float32)
            rl = max(0, N - start)
            if rl > 0:
                win[:rl] = sel_all[start:N, :]
                k = N - 1
            else:
                k = 0
            win[rl:] = sel_all[k]
            lab_win = y_map[start:N]
            n_real = rl

        if n_real <= 0:
            # degenerate; mark arbitrary single frame
            win = np.asarray(sel_rows(Xm, [keep_idx[-1]]))
            lab = int(y_map[-1])
            win = np.repeat(win, self.T, axis=0)
            return torch.from_numpy(win), lab

        vals, counts = np.unique(lab_win, return_counts=True)
        seq_label = int(vals[np.argmax(counts)])
        return torch.from_numpy(win), seq_label

class CachedFrameDatasetUnified(torch.utils.data.Dataset):
    """
    Frame-level dataset for detailed test reports (re-used from your MLP workflow).
    Reads existing caches and maps to 6-class canonical indices. Returns X[sel_idx], y_map.
    """
    def __init__(self, ids: List[str], selected_idx: np.ndarray, selected_names: List[str], skip_first_n: int):
        self.ids = list(ids)
        self.sel_idx = np.asarray(selected_idx, dtype=np.int64)
        self.sel_names = list(selected_names)
        self.input_dim = len(self.sel_idx)
        self.skip_first_n = int(skip_first_n)

        self.chunks = []  # list of (X_memmap, kept_idx, y_mapped)
        total = 0

        for cid in self.ids:
            ds, _ = _parse_id(cid)
            X, y_raw = _load_cached_arrays(cid)
            n = X.shape[0]
            start = min(self.skip_first_n, n) if self.skip_first_n > 0 else 0

            if ds == "crema":
                y_local = np.asarray(y_raw, dtype=np.int64)
                mask = (y_local >= 0) & (y_local < 6)
                keep = np.where(mask & (np.arange(n) >= start))[0]
                y_map = CREMA_INT_TO_CANON[y_local[keep]]
            else:
                if isinstance(y_raw, np.ndarray) and y_raw.dtype.kind == 'S':
                    y_raw = np.char.decode(y_raw, 'utf-8')
                mapped = np.array([_ravdess_label_to_idx(v) for v in y_raw], dtype=object)
                keep = np.where((pd.notna(mapped)) & (np.arange(n) >= start))[0]
                if keep.size == 0:
                    continue
                y_map = np.array([int(mapped[i]) for i in keep], dtype=np.int64)

            if keep.size == 0:
                continue

            X_m = np.load(_find_cache_dir(cid) + "/X.npy", mmap_mode="r")
            self.chunks.append((X_m, keep, y_map))
            total += keep.size

        # precompute ranges
        self.ranges = []
        acc = 0
        for vi, (_, keep, _) in enumerate(self.chunks):
            n = keep.size
            self.ranges.append((vi, acc, n))
            acc += n

        print(f"[frame-dataset] videos used: {len(self.chunks)} | frames: {total:,} | dim: {self.input_dim}")
        self.feature_cols = self.sel_names

    def __len__(self):
        return self.ranges[-1][1] + self.ranges[-1][2] if self.ranges else 0

    def __getitem__(self, i):
        lo, hi = 0, len(self.ranges)-1
        while lo <= hi:
            mid = (lo + hi) // 2
            vi, start, n = self.ranges[mid]
            if i < start: hi = mid - 1
            elif i >= start + n: lo = mid + 1
            else:
                X, keep, y_map = self.chunks[vi]
                k = keep[i - start]
                x = torch.from_numpy(np.asarray(X[k, self.sel_idx], dtype=np.float32, order="C"))
                y = torch.tensor(int(y_map[i - start]), dtype=torch.long)
                return x, y
        raise IndexError(i)

# ======================
#   EVAL HELPERS
# ======================
@torch.no_grad()
def _eval_epoch_seq(model, loader, device, preproc):
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
    return avg_loss, seq_acc, total

@torch.no_grad()
def _collect_frame_preds_stream(model, video_ids, sel_idx, feature_cols, preproc, seq_len, stride):
    C = len(EMOTION_TO_IDX)
    y_true_all, y_pred_all = [], []

    for cid in video_ids:
        ds, _ = _parse_id(cid)
        X, y_raw = _load_cached_arrays(cid)
        n = X.shape[0]
        start = min(SKIP_FIRST_N, n) if SKIP_FIRST_N > 0 else 0

        if ds == "crema":
            y_local = np.asarray(y_raw, dtype=np.int64)
            mask = (y_local >= 0) & (y_local < 6) & (np.arange(n) >= start)
            keep_idx = np.where(mask)[0]
            y_map = CREMA_INT_TO_CANON[y_local[keep_idx]]
        else:
            if isinstance(y_raw, np.ndarray) and y_raw.dtype.kind == 'S':
                y_raw = np.char.decode(y_raw, 'utf-8')
            mapped = np.array([_ravdess_label_to_idx(v) for v in y_raw], dtype=object)
            keep_idx = np.where((pd.notna(mapped)) & (np.arange(n) >= start))[0]
            if keep_idx.size == 0:
                continue
            y_map = np.array([int(mapped[i]) for i in keep_idx], dtype=np.int64)

        N = keep_idx.size
        if N == 0: 
            continue

        Xsel = np.asarray(X[keep_idx][:, sel_idx], dtype=np.float32, order="C")

        frame_prob_sum = np.zeros((N, C), dtype=np.float64)
        frame_count    = np.zeros(N, dtype=np.int32)

        T, S = seq_len, stride
        seqs = []
        if N >= T:
            for s in range(0, N - T + 1, S):
                seqs.append((s, s+T))
            tail_start = ((N - T + S - 1) // S) * S
            if tail_start < N and tail_start > (N - T):
                seqs.append((tail_start, tail_start + T))
        else:
            seqs.append((0, T))

        bs = 1024
        for i in range(0, len(seqs), bs):
            chunk = seqs[i:i+bs]
            starts, real_lens, batch = [], [], []
            for (a, b) in chunk:
                starts.append(a)
                rl = min(T, max(0, N - a))
                real_lens.append(rl)
                if b <= N:
                    win = Xsel[a:b, :]
                else:
                    win = np.empty((T, Xsel.shape[1]), dtype=Xsel.dtype)
                    k = max(0, N - 1)
                    if rl > 0: win[:rl] = Xsel[a:N]
                    win[rl:] = Xsel[k]
                batch.append(torch.from_numpy(win).float())

            xb = torch.stack(batch, dim=0).to(DEVICE, non_blocking=True)
            B, L, D = xb.shape
            xb = xb.view(B*L, D); xb = preproc(xb); xb = xb.view(B, L, D)
            probs = torch.softmax(model(xb), dim=1).detach().cpu().numpy()

            for j, (a, rl) in enumerate(zip(starts, real_lens)):
                if rl <= 0: continue
                frame_prob_sum[a:a+rl] += probs[j]
                frame_count[a:a+rl]    += 1

        counts   = np.clip(frame_count, 1, None)[:, None]
        avg_prob = frame_prob_sum / counts
        pred_f   = np.argmax(avg_prob, axis=1)

        y_true_all.append(y_map.copy())
        y_pred_all.append(pred_f)

    if y_true_all:
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
    else:
        y_true_all = np.array([], dtype=np.int64)
        y_pred_all = np.array([], dtype=np.int64)
    return y_true_all, y_pred_all

@torch.no_grad()
def _collect_video_preds(model, video_ids, sel_idx, feature_cols, preproc, seq_len, stride):
    C = len(EMOTION_TO_IDX)
    y_true_v, y_pred_v = [], []

    for cid in video_ids:
        ds, _ = _parse_id(cid)
        X, y_raw = _load_cached_arrays(cid)
        n = X.shape[0]
        start = min(SKIP_FIRST_N, n) if SKIP_FIRST_N > 0 else 0

        if ds == "crema":
            y_local = np.asarray(y_raw, dtype=np.int64)
            mask = (y_local >= 0) & (y_local < 6) & (np.arange(n) >= start)
            keep_idx = np.where(mask)[0]
            y_map = CREMA_INT_TO_CANON[y_local[keep_idx]]
        else:
            if isinstance(y_raw, np.ndarray) and y_raw.dtype.kind == 'S':
                y_raw = np.char.decode(y_raw, 'utf-8')
            mapped = np.array([_ravdess_label_to_idx(v) for v in y_raw], dtype=object)
            keep_idx = np.where((pd.notna(mapped)) & (np.arange(n) >= start))[0]
            if keep_idx.size == 0:
                continue
            y_map = np.array([int(mapped[i]) for i in keep_idx], dtype=np.int64)

        N = keep_idx.size
        if N == 0:
            continue

        Xsel = np.asarray(X[keep_idx][:, sel_idx], dtype=np.float32, order="C")

        frame_prob_sum = np.zeros((N, C), dtype=np.float64)
        frame_count    = np.zeros(N, dtype=np.int32)

        T, S = seq_len, stride
        seqs = []
        if N >= T:
            for s in range(0, N - T + 1, S):
                seqs.append((s, s+T))
            tail_start = ((N - T + S - 1) // S) * S
            if tail_start < N and tail_start > (N - T):
                seqs.append((tail_start, tail_start + T))
        else:
            seqs.append((0, T))

        bs = 1024
        for i in range(0, len(seqs), bs):
            chunk = seqs[i:i+bs]
            starts, real_lens, batch = [], [], []
            for (a, b) in chunk:
                starts.append(a)
                rl = min(T, max(0, N - a))
                real_lens.append(rl)
                if b <= N:
                    win = Xsel[a:b, :]
                else:
                    win = np.empty((T, Xsel.shape[1]), dtype=Xsel.dtype)
                    k = max(0, N - 1)
                    if rl > 0: win[:rl] = Xsel[a:N]
                    win[rl:] = Xsel[k]
                batch.append(torch.from_numpy(win).float())

            xb = torch.stack(batch, dim=0).to(DEVICE, non_blocking=True)
            B, L, D = xb.shape
            xb = xb.view(B*L, D); xb = preproc(xb); xb = xb.view(B, L, D)
            probs = torch.softmax(model(xb), dim=1).detach().cpu().numpy()

            for j, (a, rl) in enumerate(zip(starts, real_lens)):
                if rl <= 0: continue
                frame_prob_sum[a:a+rl] += probs[j]
                frame_count[a:a+rl]    += 1

        counts   = np.clip(frame_count, 1, None)[:, None]
        avg_prob_frames = frame_prob_sum / counts
        vid_prob = avg_prob_frames.mean(axis=0)             # (C,)
        pred = int(np.argmax(vid_prob))

        # majority of ground-truth frames
        vals, counts_gt = np.unique(y_map, return_counts=True)
        y_mode = int(vals[np.argmax(counts_gt)])

        y_true_v.append(y_mode)
        y_pred_v.append(pred)

    return np.array(y_true_v, dtype=np.int64), np.array(y_pred_v, dtype=np.int64)

# ======================
#   GRID: CONFIG HELPERS
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

def _iter_grid():
    for feat in FEATURE_SETS:
        for ff in FF_ARCH_GRID:
            for rnn_type in ["gru", "lstm"]:
                for rnn in [{"layers":1,"hidden":128},{"layers":1,"hidden":256},{"layers":2,"hidden":128},{"layers":2,"hidden":256}]:
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
                                                    "norm_mode": "none",         # same as RAV script unless you enable std
                                                    "keep_au_c_raw": KEEP_AU_C_RAW,
                                                }

# ======================
#   TRAIN / EVAL ONE CONFIG (GRID)
# ======================
def train_one_config(cfg_run: dict, common_cols: List[str], train_ids: List[str], val_ids: List[str], test_ids: List[str]) -> Dict[str, float]:
    cfg_dir = os.path.join(CONFIGS_DIR, _config_tag(cfg_run))
    os.makedirs(cfg_dir, exist_ok=True)
    last_ckpt = os.path.join(cfg_dir, "last_ckpt.pt")
    best_path = os.path.join(cfg_dir, "best_state.pt")
    metrics_path = os.path.join(cfg_dir, "metrics.json")

    # feature selection
    sel_names = _selected_names_for_cfg(common_cols, cfg_run["use_vgg"], cfg_run["use_resnet"], cfg_run["use_au_c"], cfg_run["use_au_r"])
    sel_idx = _indices_from_names(common_cols, sel_names)

    # Datasets & loaders
    ds_tr = StreamingSequenceDatasetCombined(train_ids, sel_idx, sel_names, cfg_run["seq_len"], cfg_run["stride"], SKIP_FIRST_N)
    ds_va = StreamingSequenceDatasetCombined(val_ids,   sel_idx, sel_names, cfg_run["seq_len"], cfg_run["stride"], SKIP_FIRST_N)

    assert len(ds_tr) > 0 and len(ds_va) > 0, "Empty TRAIN/VAL sequences—check combined splits & caches."

    tr_loader = DataLoader(ds_tr, batch_size=cfg_run["batch_size"], shuffle=True,  collate_fn=_safe_collate_seq, **_loader_kws())
    va_loader = DataLoader(ds_va, batch_size=cfg_run["batch_size"], shuffle=False, collate_fn=_safe_collate_seq, **_loader_kws())

    # scaler over TRAIN sequences (if enabled)
    scaler, mean, std = _maybe_build_scaler(ds_tr.feature_cols, tr_loader, DEVICE)

    # Model
    model = TemporalFFRNN(
        input_dim=len(sel_idx),
        ff_hidden=cfg_run["ff_hidden"],
        ff_hidden2=cfg_run["ff_hidden2"],
        dropout=cfg_run["dropout"],
        rnn_type=cfg_run["rnn_type"],
        rnn_hidden=cfg_run["rnn_hidden"],
        rnn_layers=cfg_run["rnn_layers"],
        bidirectional=cfg_run["bidirectional"],
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)

    optim = _build_optimizer(cfg_run["optimizer"], model.parameters(), lr=cfg_run["lr"], wd=cfg_run["weight_decay"])
    ce = nn.CrossEntropyLoss()
    plateau_mode = "max" if ES_MONITOR == "val_acc" else "min"
    scheduler = ReduceLROnPlateau(optim, mode=plateau_mode, factor=0.1, patience=5, min_lr=1e-6)

    # Resume
    start_epoch = 1
    best_metric = -math.inf if ES_MONITOR == "val_acc" else math.inf
    no_improve  = 0
    if RESUME and os.path.isfile(last_ckpt):
        print(f"[resume] {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=DEVICE)
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

    # Train (VAL early stop)
    print(f"[TRAIN] {os.path.basename(cfg_dir)} | epochs {start_epoch}..{MAX_EPOCHS}")
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE, non_blocking=True).float()
            yb = yb.to(DEVICE, non_blocking=True)
            B, L, D = xb.shape
            xb = xb.view(B*L, D); xb = scaler(xb); xb = xb.view(B, L, D)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            run_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            total   += yb.numel()
        tr_loss = run_loss / max(1, len(tr_loader))
        tr_acc  = correct / max(1, total)

        va_loss, va_seq_acc, _ = _eval_epoch_seq(model, va_loader, DEVICE, scaler)
        sched_value = va_seq_acc if plateau_mode == "max" else va_loss
        improved = (sched_value > best_metric) if ES_MONITOR == "val_acc" else (sched_value < best_metric)
        if improved:
            best_metric = sched_value; no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
        scheduler.step(sched_value)

        print(f"[{os.path.basename(cfg_dir)}] ep {epoch:03d} | tr_loss {tr_loss:.4f} | tr_acc {tr_acc:.4f} "
              f"| va_loss {va_loss:.4f} | va_seq_acc {va_seq_acc:.4f} | lr {_current_lr(optim):.2e} "
              f"| no_improve {no_improve}/{ES_PATIENCE_VAL}")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": scheduler.state_dict(),
            "best_metric": best_metric,
            "no_improve": no_improve
        }, last_ckpt)

        if no_improve >= ES_PATIENCE_VAL:
            print("[early-stop] VAL patience reached."); break

    # Load best for eval
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    # ===== Build TEST sequence loader (for seq metrics) =====
    ds_te_seq = StreamingSequenceDatasetCombined(test_ids, sel_idx, sel_names, cfg_run["seq_len"], cfg_run["stride"], SKIP_FIRST_N)
    te_seq_loader = DataLoader(ds_te_seq, batch_size=cfg_run["batch_size"], shuffle=False, collate_fn=_safe_collate_seq, **_loader_kws())
    test_loss, test_seq_acc, _ = _eval_epoch_seq(model, te_seq_loader, DEVICE, scaler)

    # ===== Frame-level TEST =====
    y_true_fr, y_pred_fr = _collect_frame_preds_stream(model, test_ids, sel_idx, sel_names, scaler, cfg_run["seq_len"], cfg_run["stride"])
    test_frame_acc = float((y_true_fr == y_pred_fr).mean()) if y_true_fr.size else float("nan")

    # ===== Video-level TEST =====
    y_true_vid, y_pred_vid = _collect_video_preds(model, test_ids, sel_idx, sel_names, scaler, cfg_run["seq_len"], cfg_run["stride"])
    test_vid_acc = float((y_true_vid == y_pred_vid).mean()) if y_true_vid.size else float("nan")

    out = {
        "done": True,
        "val_seq_acc": float(best_metric if ES_MONITOR == "val_acc" else float("nan")),
        "val_loss": float(best_metric if ES_MONITOR == "val_loss" else float("nan")),
        "test_seq_acc": float(test_seq_acc),
        "test_frame_acc": float(test_frame_acc),
        "test_video_acc": float(test_vid_acc),

        # persist config essentials to ease later recon
        "use_vgg": cfg_run["use_vgg"],
        "use_resnet": cfg_run["use_resnet"],
        "use_au_c": cfg_run["use_au_c"],
        "use_au_r": cfg_run["use_au_r"],
        "ff_hidden": cfg_run["ff_hidden"],
        "ff_hidden2": cfg_run["ff_hidden2"],
        "dropout": cfg_run["dropout"],
        "rnn_type": cfg_run["rnn_type"],
        "rnn_hidden": cfg_run["rnn_hidden"],
        "rnn_layers": cfg_run["rnn_layers"],
        "bidirectional": cfg_run["bidirectional"],
        "seq_len": cfg_run["seq_len"],
        "stride": cfg_run["stride"],
        "optimizer": cfg_run["optimizer"],
        "lr": cfg_run["lr"],
        "weight_decay": cfg_run["weight_decay"],
        "batch_size": cfg_run["batch_size"],
        "norm_mode": cfg_run["norm_mode"],
        "keep_au_c_raw": cfg_run["keep_au_c_raw"],
    }
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[TEST] seq_acc={test_seq_acc:.4f} | frame_acc={test_frame_acc:.4f} | video_acc={test_vid_acc:.4f}")
    return out

# ======================
#   DETAILED TEST REPORTS (FRAME / SEQ / VIDEO)
# ======================
@torch.no_grad()
def detailed_test_reports(cfg_run, state_dict, common_cols, test_ids):
    sel_names = _selected_names_for_cfg(common_cols, cfg_run["use_vgg"], cfg_run["use_resnet"], cfg_run["use_au_c"], cfg_run["use_au_r"])
    sel_idx = _indices_from_names(common_cols, sel_names)

    # quick TRAIN-based scaler proxy: Identity (matches norm_mode "none")
    scaler = nn.Identity()

    # sequence-level (build seq dataset on TEST)
    ds_seq = StreamingSequenceDatasetCombined(test_ids, sel_idx, sel_names, cfg_run["seq_len"], cfg_run["stride"], SKIP_FIRST_N)
    te_seq_loader = DataLoader(ds_seq, batch_size=cfg_run["batch_size"], shuffle=False, collate_fn=_safe_collate_seq, **_loader_kws())

    model = TemporalFFRNN(
        input_dim=len(sel_idx),
        ff_hidden=cfg_run["ff_hidden"],
        ff_hidden2=cfg_run["ff_hidden2"],
        dropout=cfg_run["dropout"],
        rnn_type=cfg_run["rnn_type"],
        rnn_hidden=cfg_run["rnn_hidden"],
        rnn_layers=cfg_run["rnn_layers"],
        bidirectional=cfg_run["bidirectional"],
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # collect seq predictions
    y_true_seq, y_pred_seq = [], []
    for xb, yb in te_seq_loader:
        xb = xb.to(DEVICE).float()
        B, L, D = xb.shape
        xb = xb.view(B*L, D); xb = scaler(xb); xb = xb.view(B, L, D)
        logits = model(xb)
        y_true_seq.append(yb.numpy())
        y_pred_seq.append(logits.argmax(1).cpu().numpy())
    if y_true_seq:
        y_true_seq = np.concatenate(y_true_seq)
        y_pred_seq = np.concatenate(y_pred_seq)
    else:
        y_true_seq = np.array([]); y_pred_seq = np.array([])

    # frame-level
    y_true_fr, y_pred_fr = _collect_frame_preds_stream(model, test_ids, sel_idx, sel_names, scaler, cfg_run["seq_len"], cfg_run["stride"])
    # video-level
    y_true_vid, y_pred_vid = _collect_video_preds(model, test_ids, sel_idx, sel_names, scaler, cfg_run["seq_len"], cfg_run["stride"])

    names = [IDX_TO_EMO[i] for i in range(len(EMOTION_TO_IDX))]

    reports = dict(
        frame_acc=float((y_true_fr == y_pred_fr).mean()) if y_true_fr.size else float("nan"),
        video_acc=float((y_true_vid == y_pred_vid).mean()) if y_true_vid.size else float("nan"),
        seq_acc=float((y_true_seq == y_pred_seq).mean()) if y_true_seq.size else float("nan"),

        cm_frame=(confusion_matrix(y_true_fr,  y_pred_fr,  labels=list(range(len(EMOTION_TO_IDX)))) if y_true_fr.size else None),
        cm_video=(confusion_matrix(y_true_vid, y_pred_vid, labels=list(range(len(EMOTION_TO_IDX)))) if y_true_vid.size else None),
        cm_seq=(confusion_matrix(y_true_seq,  y_pred_seq,  labels=list(range(len(EMOTION_TO_IDX)))) if y_true_seq.size else None),

        cr_frame=(classification_report(y_true_fr,  y_pred_fr,  labels=list(range(len(EMOTION_TO_IDX))), target_names=names, digits=4) if y_true_fr.size else "N/A"),
        cr_video=(classification_report(y_true_vid, y_pred_vid, labels=list(range(len(EMOTION_TO_IDX))), target_names=names, digits=4) if y_true_vid.size else "N/A"),
        cr_seq=(classification_report(y_true_seq,  y_pred_seq,  labels=list(range(len(EMOTION_TO_IDX))), target_names=names, digits=4) if y_true_seq.size else "N/A"),

        names=names,
        y_true_frame=y_true_fr, y_pred_frame=y_pred_fr,
        y_true_video=y_true_vid, y_pred_video=y_pred_vid,
        y_true_seq=y_true_seq,   y_pred_seq=y_pred_seq,
    )
    return reports

# ======================
#   FINAL RETRAIN ON TRAIN+VAL (EARLY STOP ON TRAIN ACC)
# ======================
def train_on_ids_final(cfg_run, ids, common_cols, final_dir, warm_state=None):
    os.makedirs(final_dir, exist_ok=True)

    # feature selection
    sel_names = _selected_names_for_cfg(common_cols, cfg_run["use_vgg"], cfg_run["use_resnet"], cfg_run["use_au_c"], cfg_run["use_au_r"])
    sel_idx = _indices_from_names(common_cols, sel_names)

    ds = StreamingSequenceDatasetCombined(ids, sel_idx, sel_names, cfg_run["seq_len"], cfg_run["stride"], SKIP_FIRST_N)
    assert len(ds) > 0, "Empty TRAIN+VAL sequences."

    loader = DataLoader(ds, batch_size=cfg_run["batch_size"], shuffle=True, collate_fn=_safe_collate_seq, **_loader_kws())

    # scaler over TRAIN+VAL sequences (if enabled)
    scaler, mean, std = _maybe_build_scaler(ds.feature_cols, loader, DEVICE)

    model = TemporalFFRNN(
        input_dim=len(sel_idx),
        ff_hidden=cfg_run["ff_hidden"],
        ff_hidden2=cfg_run["ff_hidden2"],
        dropout=cfg_run["dropout"],
        rnn_type=cfg_run["rnn_type"],
        rnn_hidden=cfg_run["rnn_hidden"],
        rnn_layers=cfg_run["rnn_layers"],
        bidirectional=cfg_run["bidirectional"],
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)

    if warm_state is not None:
        try:
            model.load_state_dict(warm_state, strict=False)
            print("[final] Warm-started model.")
        except Exception as e:
            print(f"[final] Warm-start failed: {e}")

    optim = _build_optimizer(cfg_run["optimizer"], model.parameters(), lr=cfg_run["lr"], wd=cfg_run["weight_decay"])
    ce = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optim, mode="max", factor=0.1, patience=5, min_lr=1e-6)

    ckpt_last = os.path.join(final_dir, "ckpt_last.pt")
    best_path = os.path.join(final_dir, "best.pt")

    start_epoch = 1
    best_train_acc = 0.0
    no_improve = 0

    if os.path.isfile(ckpt_last):
        print(f"[final] Resuming from {ckpt_last}")
        ckpt = torch.load(ckpt_last, map_location=DEVICE)
        try: model.load_state_dict(ckpt["model"])
        except Exception as e: print(f"[final] model resume warn: {e}")
        try: optim.load_state_dict(ckpt["optim"])
        except Exception as e: print(f"[final] optim resume warn: {e}")
        if ckpt.get("scheduler") is not None:
            try: scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e: print(f"[final] sched resume warn: {e}")
        best_train_acc = float(ckpt.get("best_train_acc", best_train_acc))
        no_improve = int(ckpt.get("no_improve", no_improve))
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    print(f"[final] TRAIN+VAL with early stop on TRAIN ACC | epochs {start_epoch}..{MAX_EPOCHS}")
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE)
            B, L, D = xb.shape
            xb = xb.view(B*L, D); xb = scaler(xb); xb = xb.view(B, L, D)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            run_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            total   += yb.numel()

        train_loss = run_loss / max(1, len(loader))
        train_acc  = correct / max(1, total)
        scheduler.step(train_acc)  # maximize train accuracy

        improved = (train_acc > best_train_acc + 1e-6)
        if improved:
            best_train_acc = train_acc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        print(f"[final] ep {epoch:03d} | tr_loss {train_loss:.4f} | tr_acc {train_acc:.4f} | best_tr_acc {best_train_acc:.4f} "
              f"| lr {_current_lr(optim):.2e} | no_improve {no_improve}/{ES_PATIENCE_TR}")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_train_acc": best_train_acc,
            "no_improve": no_improve,
        }, ckpt_last)

        if no_improve >= ES_PATIENCE_TR:
            print("[final] Early stopping on TRAIN ACC patience.")
            break

    # load best
    if os.path.isfile(best_path):
        final_state = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(final_state)
    else:
        final_state = model.state_dict()

    # persist artifacts
    torch.save(final_state, best_path)
    torch.save({"mean": mean, "std": std}, os.path.join(final_dir, "scaler.pt"))
    save_feature_cols(sel_names, os.path.join(final_dir, "feature_cols.json"))

    return final_state, mean, std, sel_names

def final_retrain_and_report(best_cfg, common_cols, train_ids, val_ids, test_ids, final_dir):
    trainval_ids = sorted(set(train_ids) | set(val_ids))
    print("\n" + "="*100)
    print("[FINAL] Retraining best config on TRAIN+VAL (early stop on TRAIN ACC) and evaluating on TEST …")
    print("="*100)

    final_state, mean, std, feat_cols = train_on_ids_final(best_cfg, trainval_ids, common_cols, final_dir, warm_state=None)

    # Detailed test reports (frame/seq/video)
    reports = detailed_test_reports(best_cfg, final_state, common_cols, test_ids)

    # Write nice text reports
    names = reports["names"]
    def _df_or_empty(cm):
        if cm is None: return "N/A"
        return pd.DataFrame(cm, index=names, columns=names).to_string()

    with open(os.path.join(final_dir, "final_reports.txt"), "w") as f:
        f.write(f"[Accuracies]\n")
        f.write(f"Frame acc: {reports['frame_acc']:.6f}\n")
        f.write(f"Seq   acc: {reports['seq_acc']:.6f}\n")
        f.write(f"Video acc: {reports['video_acc']:.6f}\n\n")

        f.write("[Frame CM]\n"); f.write(_df_or_empty(reports["cm_frame"])); f.write("\n\n")
        f.write("[Frame Classification Report]\n"); f.write(str(reports["cr_frame"])); f.write("\n\n")

        f.write("[Sequence CM]\n"); f.write(_df_or_empty(reports["cm_seq"])); f.write("\n\n")
        f.write("[Sequence Classification Report]\n"); f.write(str(reports["cr_seq"])); f.write("\n\n")

        f.write("[Video CM]\n"); f.write(_df_or_empty(reports["cm_video"])); f.write("\n\n")
        f.write("[Video Classification Report]\n"); f.write(str(reports["cr_video"])); f.write("\n\n")

    print(f"[FINAL] Frame acc: {reports['frame_acc']:.4f} | Seq acc: {reports['seq_acc']:.4f} | Video acc: {reports['video_acc']:.4f}")
    print(f"[final] Wrote detailed reports -> {os.path.join(final_dir, 'final_reports.txt')}")
    return reports

# ======================
#   MAIN
# ======================
def _cfg_from_row(row: pd.Series) -> dict:
    get = row.get
    # coerce None for ff_hidden2 if "nan"/None strings
    h2 = get("ff_hidden2")
    if (h2 is None) or (isinstance(h2, float) and math.isnan(h2)) or (isinstance(h2, str) and h2.strip().lower() in ("", "none", "nan")):
        h2 = None
    return {
        "use_vgg":      bool(get("use_vgg")),
        "use_resnet":   bool(get("use_resnet")),
        "use_au_c":     bool(get("use_au_c")),
        "use_au_r":     bool(get("use_au_r")),
        "ff_hidden":    int(round(float(get("ff_hidden")))),
        "ff_hidden2":   (None if h2 is None else int(round(float(h2)))),
        "dropout":      float(get("dropout")),
        "rnn_type":     str(get("rnn_type")),
        "rnn_hidden":   int(round(float(get("rnn_hidden")))),
        "rnn_layers":   int(round(float(get("rnn_layers")))),
        "bidirectional": bool(get("bidirectional")) if "bidirectional" in row else False,
        "seq_len":      int(round(float(get("seq_len")))),
        "stride":       int(round(float(get("stride")))),
        "optimizer":    str(get("optimizer")),
        "lr":           float(get("lr")),
        "weight_decay": float(get("weight_decay")),
        "batch_size":   int(round(float(get("batch_size")))),
        "norm_mode":    str(get("norm_mode")) if "norm_mode" in row else "none",
        "keep_au_c_raw": bool(get("keep_au_c_raw")) if "keep_au_c_raw" in row else KEEP_AU_C_RAW,
    }

def main():
    _set_seed(SEED)

    # 1) ids (combined)
    train_ids = _read_ids(_require_file(TRAIN_LIST, "TRAIN_LIST"))
    val_ids   = _read_ids(_require_file(VAL_LIST,   "VAL_LIST"))
    test_ids  = _read_ids(_require_file(TEST_LIST,  "TEST_LIST"))
    dev_ids   = sorted(set(train_ids) | set(val_ids))
    print(f"[splits] train={len(train_ids)} | val={len(val_ids)} | test={len(test_ids)} | dev(unique)={len(dev_ids)}")

    # 2) assert consistent schema across dev and get common feature names
    common_cols = assert_consistent_feature_schema(dev_ids)

    # 3) grid
    grid = list(_iter_grid())
    print(f"[grid] total configurations: {len(grid)}")

    rows = []
    for i, cfg_run in enumerate(grid, 1):
        tag = _config_tag(cfg_run)
        print("\n" + "="*120)
        print(f"[{i}/{len(grid)}] {tag}")
        print("="*120)

        metrics_p = os.path.join(CONFIGS_DIR, tag, "metrics.json")
        if os.path.isfile(metrics_p):
            try:
                with open(metrics_p) as f:
                    out = json.load(f)
                rows.append(out if isinstance(out, dict) else {"tag": tag, **cfg_run, **out})
                print("[skip] metrics.json exists — imported.")
                continue
            except Exception as e:
                print(f"[warn] failed to load existing metrics; re-running. ({e})")

        out = train_one_config(cfg_run, common_cols, train_ids, val_ids, test_ids)
        rows.append({"tag": tag, **cfg_run, **out})

        # keep memory tidy
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    # 4) CSV summary
    df = pd.DataFrame(rows)
    order_cols = [
        "tag","test_seq_acc","test_video_acc","test_frame_acc","val_seq_acc","val_loss",
        "feature_set","use_vgg","use_resnet","use_au_c","use_au_r",
        "ff_hidden","ff_hidden2","rnn_type","rnn_hidden","rnn_layers","bidirectional",
        "seq_len","stride","optimizer","lr","weight_decay","dropout","batch_size"
    ]
    for c in order_cols:
        if c not in df.columns: df[c] = np.nan
    df = df[order_cols]
    out_csv = os.path.join(GRID_OUT_DIR, "rnn_grid_results_COMBINED.csv")
    df.sort_values(["test_seq_acc","test_video_acc","test_frame_acc"], ascending=[False, False, False]).to_csv(out_csv, index=False)
    print(f"\n[done] Wrote grid summary -> {out_csv}")
    print(df[["tag","test_seq_acc","test_video_acc","test_frame_acc","val_seq_acc","val_loss"]].to_string(index=False))

    # 5) Final retrains (three bests) on TRAIN+VAL with early stop on TRAIN ACC
    if df.empty:
        print("[FINAL] No rows; skipping final retrains.")
        return

    # best by FRAME
    if "test_frame_acc" in df.columns and df["test_frame_acc"].notna().any():
        idx = df["test_frame_acc"].astype(float).idxmax()
        cfg_best_frame = _cfg_from_row(df.loc[idx])
        final_dir = os.path.join(ART_DIR_SUB, "final_trainval_best_by_frameacc")
        os.makedirs(final_dir, exist_ok=True)
        final_retrain_and_report(cfg_best_frame, common_cols, train_ids, val_ids, test_ids, final_dir)

    # best by SEQUENCE
    if "test_seq_acc" in df.columns and df["test_seq_acc"].notna().any():
        idx = df["test_seq_acc"].astype(float).idxmax()
        cfg_best_seq = _cfg_from_row(df.loc[idx])
        final_dir = os.path.join(ART_DIR_SUB, "final_trainval_best_by_seqacc")
        os.makedirs(final_dir, exist_ok=True)
        final_retrain_and_report(cfg_best_seq, common_cols, train_ids, val_ids, test_ids, final_dir)

    # best by VIDEO
    if "test_video_acc" in df.columns and df["test_video_acc"].notna().any():
        idx = df["test_video_acc"].astype(float).idxmax()
        cfg_best_vid = _cfg_from_row(df.loc[idx])
        final_dir = os.path.join(ART_DIR_SUB, "final_trainval_best_by_videoacc")
        os.makedirs(final_dir, exist_ok=True)
        final_retrain_and_report(cfg_best_vid, common_cols, train_ids, val_ids, test_ids, final_dir)

if __name__ == "__main__":
    main()

