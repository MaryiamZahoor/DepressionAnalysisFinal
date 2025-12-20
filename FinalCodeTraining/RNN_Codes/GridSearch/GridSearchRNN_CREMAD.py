# grid_search_rnn_streaming.py
# Grid search for CNN+RNN with streaming, per-video cache, and detailed timings.
# Video-level eval = argmax over average per-frame probabilities (no sequence aggregation).

import os, sys, json, math, time
import numpy as np
import pandas as pd
from typing import List, Dict

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
#   USER CONSTANTS
# ======================
SPLIT_PATH  = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/"
TRAIN_LIST  = os.path.join(SPLIT_PATH, "train_videos_full.txt")
VAL_LIST    = os.path.join(SPLIT_PATH, "val_videos_full.txt")
TEST_LIST   = os.path.join(SPLIT_PATH, "test_videos_full.txt")

# Where to cache the master feature list (put it in the data dir)
MASTER_FEATURES_JSON = os.path.join(SPLIT_PATH, "master_feature_cols.json")
MASTER_SCAN_LIMIT = 10   # only scan first N videos we find

INCLUDE_LIST = None
EXCLUDE_LIST = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/exclude_videos.txt"

LABEL_COL          = getattr(CFG, "SPLIT_LABEL_COL", "GT_Emotion")
SKIP_FIRST_N       = getattr(CFG, "SKIP_FRAME", 0)
COMBINED_CSV_NAME  = getattr(CFG, "COMBINED_CSV_NAME", "combined.csv")
OUTPUT_DIR         = getattr(CFG, "OUTPUT_DIR")
EMOTION_TO_IDX     = getattr(CFG, "emotion_to_idx")
IDX_TO_EMO         = {v:k for k,v in EMOTION_TO_IDX.items()}

ART_DIR_TAG   = "cremad_GridSearch_unscaled_RNN"
PROJECT_DIR   = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/"
ART_DIR_SUB   = os.path.join(PROJECT_DIR, "artifacts", ART_DIR_TAG)
os.makedirs(ART_DIR_SUB, exist_ok=True)
BEST_WEIGHTS  = os.path.join(ART_DIR_SUB, f"best_{ART_DIR_TAG}.pt")
GRID_OUT_DIR  = os.path.join(os.path.dirname(BEST_WEIGHTS), "grid_config")
os.makedirs(GRID_OUT_DIR, exist_ok=True)

EPOCHS = 300
SEED   = getattr(CFG, "SEED", 42)

DO_STANDARDIZE = True
KEEP_AU_C_RAW  = True

FF_ARCH_GRID = [
    {"hidden_dim": 256,  "hidden_dim2": None},
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
CLIP_NORM    = 1.0
ES_MONITOR   = "val_acc"      # "val_acc" or "val_loss"
ES_PATIENCE  = 15
PLATEAU_PATIENCE = 5
PLATEAU_FACTOR  = 0.1
MIN_LR          = 1e-6

FEATURE_SETS = [
    {"name":"VGG",           "use_vgg":True,  "use_resnet":False, "use_au_c":False, "use_au_r":False},
    {"name":"RESNET",        "use_vgg":False, "use_resnet":True,  "use_au_c":False, "use_au_r":False},
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
    return os.path.join(OUTPUT_DIR, vid, "cache")

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
      - include all present VGG/RESNET features,
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
def build_video_cache(vid: str, master_feature_cols: List[str]) -> bool:
    """
    Return True if (re)built; False if cache already exists and is valid.
    """
    xnp, ynp, meta = _cache_paths(vid)
    os.makedirs(os.path.dirname(xnp), exist_ok=True)
    if os.path.isfile(xnp) and os.path.isfile(ynp) and os.path.isfile(meta):
        return False

    csvp = _vid_csv_path(vid)
    if not os.path.isfile(csvp):
        raise FileNotFoundError(f"CSV missing for {vid}: {csvp}")

    df = pd.read_csv(csvp)
    df = harmonize_vgg_cols(df)
    if SKIP_FIRST_N > 0: df = df.iloc[SKIP_FIRST_N:].reset_index(drop=True)

    # labels
    y = (df[LABEL_COL].astype(str).str.upper()
         .map(EMOTION_TO_IDX).dropna().astype(np.int64))
    idx = y.index

    # features in the canonical master order
    X = (df.loc[idx, master_feature_cols]
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0).astype("float32").to_numpy(copy=False))
    y = y.to_numpy(copy=False)

    np.save(xnp, X)
    np.save(ynp, y)
    with open(meta, "w") as f:
        json.dump({"feature_cols": master_feature_cols}, f)
    return True

class StreamingSequenceDataset(Dataset):
    """
    Dataset that:
      • Ensures per-video cache exists
      • Memory-maps X/y (mmap_mode='r') to avoid large RAM
      • Builds sequence windows on the fly (T, stride)
      • Selects columns by name via master list → indices once
      • Labels window by the MODE over REAL (unpadded) frames only
    """
    def __init__(self, video_ids: List[str], cfg_run: dict, master_feature_cols: List[str]):
        self.vids = list(video_ids)
        self.T    = int(cfg_run["seq_len"])
        self.S    = int(cfg_run["stride"])

        with _time(f"Build/verify cache (N_videos={len(self.vids)})"):
            for vid in self.vids:
                try:
                    created = build_video_cache(vid, master_feature_cols)
                    if created:
                        print(f"[cache] built {vid}")
                except Exception as e:
                    print(f"[cache] skip {vid}: {e}")

        # choose columns
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
                # add a tail start if not aligned
                tail_start = ((N - self.T + self.S - 1) // self.S) * self.S  # ceil((N-T)/S)*S
                if tail_start < N and tail_start > (N - self.T):
                    self.index.append((vid, tail_start))
            else:
                self.index.append((vid, 0))

        # light LRU of memmaps
        self._arrays = {}

        print(f"[stream-ds] Selected {len(self.col_idx)} features | sequences={len(self.index)}")

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
            n_real = self.T
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

        # window label = mode over REAL frames only
        vals, counts = np.unique(lab_real, return_counts=True)
        seq_label = int(vals[np.argmax(counts)])
        return torch.from_numpy(win), torch.tensor(seq_label, dtype=torch.long)

# ======================
#     TRAIN/EVAL UTILS
# ======================
def _save_scaler(mean: torch.Tensor, std: torch.Tensor, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"mean": mean.detach().cpu(), "std": std.detach().cpu()}, save_path)

def _make_loader(ds, batch_size: int, shuffle: bool):
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
    frame_acc_approx = seq_acc
    return avg_loss, seq_acc, frame_acc_approx, total

# ---------- Frame & Video aggregators (probability-based) ----------
@torch.no_grad()
def _collect_frame_preds_stream(model, video_ids, cfg_run, master_feature_cols, preproc):
    """
    Aggregate overlapping window probabilities back onto frames.
    For each frame, average the softmax prob over all windows that contain it.
    Returns: y_true_frames, y_pred_frames (concatenated across videos).
    """
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

        bs = cfg_run["batch_size"]
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
    """
    Video label via per-frame probability averaging:
      1) Build overlapping windows; get window softmax probs.
      2) Average probs onto frames (as above).
      3) Average frame probs over the whole video; take argmax.
      GT = mode of frame GT labels.
    Returns: y_true_v, y_pred_v
    """
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

        bs = cfg_run["batch_size"]
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

def train_one_config(cfg_run: dict, master_feature_cols: List[str]) -> Dict[str, float]:
    cfg_dir = _cfg_dir_for(cfg_run)
    last_ckpt = os.path.join(cfg_dir, "last_ckpt.pt")
    best_path = os.path.join(cfg_dir, "best_state.pt")
    metrics_path = os.path.join(cfg_dir, "metrics.json")
    config_t0 = time.time()

    # Skip if done
    if SKIP_DONE_CONFIGS and os.path.isfile(metrics_path):
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            if m.get("done", False):
                print(f"[skip] {os.path.basename(cfg_dir)} already done.")
                return {
                    "val_seq_acc": float(m["val_seq_acc"]),
                    "val_vid_acc": float(m["val_vid_acc"]),
                    "val_frame_acc": float(m["val_frame_acc"]),
                    "val_loss": float(m["val_loss"]),
                }
        except Exception:
            pass

    # Splits
    print("creating or loading master feature columns")
    with _time("Read split lists"):
        train_ids = _read_list(_require_file(TRAIN_LIST, "TRAIN_LIST"))
        val_ids   = _read_list(_require_file(VAL_LIST,   "VAL_LIST"))
        if INCLUDE_LIST or EXCLUDE_LIST:
            train_ids = _apply_include_exclude(train_ids, INCLUDE_LIST, EXCLUDE_LIST)
            val_ids   = _apply_include_exclude(val_ids,   INCLUDE_LIST, EXCLUDE_LIST)

    # Datasets
    with _time("Build StreamingSequenceDataset (train)"):
        train_ds = StreamingSequenceDataset(train_ids, cfg_run, master_feature_cols)
    with _time("Build StreamingSequenceDataset (val)"):
        val_ds   = StreamingSequenceDataset(val_ids,   cfg_run, master_feature_cols)

    print(f"[features] N={len(train_ds.feature_cols)}")
    if len(train_ds.feature_cols):
        head = ", ".join(train_ds.feature_cols[:8])
        tail = ", ".join(train_ds.feature_cols[-8:])
        print(f"[features] head: [{head}] ... tail: [{tail}]")

    # Persist per-config feature cols
    featcols_local = os.path.join(cfg_dir, "feature_cols.json")
    with open(featcols_local, "w") as f:
        json.dump(train_ds.feature_cols, f)

    # Loaders
    with _time(f"Dataloader (train) bs={cfg_run['batch_size']} N={len(train_ds)}"):
        train_loader = _make_loader(train_ds, cfg_run["batch_size"], shuffle=True)
    with _time(f"Dataloader (val)   bs={cfg_run['batch_size']} N={len(val_ds)}"):
        val_loader   = _make_loader(val_ds,   cfg_run["batch_size"], shuffle=False)

    # ---------- PREPROCESSOR ----------
    mean = std = None
    cfg_norm = cfg_run["norm_mode"].lower()   # 'none' | 'l2' | 'zscore' | 'zscore+l2'
    if cfg_norm in ("zscore", "zscore+l2"):
        train_ids_stats = _read_list(_require_file(TRAIN_LIST, "TRAIN_LIST"))
        if INCLUDE_LIST or EXCLUDE_LIST:
            train_ids_stats = _apply_include_exclude(train_ids_stats, INCLUDE_LIST, EXCLUDE_LIST)
        mean, std = compute_mean_std_from_cache(
            video_ids=train_ids_stats,
            master_feature_cols=master_feature_cols,
            col_idx=train_ds.col_idx,
            device=DEVICE,
            batch_size=max(131072, cfg_run["batch_size"] * cfg_run["seq_len"])
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

    # Model + opt + sched
    with _time("Model init"):
        model = TemporalFFRNN(
            input_dim=train_ds.input_dim,
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
    scheduler = ReduceLROnPlateau(optim, mode=plateau_mode, factor=PLATEAU_FACTOR,
                                  patience=PLATEAU_PATIENCE, min_lr=MIN_LR)

    # Resume
    start_epoch = 1
    best_metric = -math.inf if ES_MONITOR == "val_acc" else math.inf
    no_improve  = 0
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

    # Train
    print(f"[TRAINING] Starting training for {EPOCHS - start_epoch + 1} epochs…")
    train_all_t0 = time.time()
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

        val_t0 = time.time()
        va_loss, va_seq_acc, va_frame_acc_approx, _ = _eval_epoch_seq(model, val_loader, DEVICE, preproc)
        val_ep_time = time.time() - val_t0

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
              f"(data {data_wait:.2f}s, compute {compute:.2f}s, val {val_ep_time:.2f}s)")

        _save_ckpt(last_ckpt, epoch, model, optim, scheduler, best_metric, no_improve, best_state_path=best_path)
        if no_improve >= ES_PATIENCE:
            print("[early-stop] patience reached."); break

    total_train_time = time.time() - train_all_t0
    print(f"[TIMING] Total training time (all epochs): {total_train_time:.2f}s")

    # Final val with best
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    with _time("Final validation (sequence)"):
        val_loss, val_seq_acc, val_frame_acc, _ = _eval_epoch_seq(model, val_loader, DEVICE, preproc)

    with _time("Video-level validation (frame-prob argmax)"):
        val_vid_acc = _eval_video_level_from_frames(model, val_ids, cfg_run, master_feature_cols, preproc)

    # metrics.json
    tmp = metrics_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "done": True,
            "val_seq_acc": float(val_seq_acc),
            "val_vid_acc": float(val_vid_acc),
            "val_frame_acc": float(val_frame_acc),
            "val_loss": float(val_loss)
        }, f, indent=2)
    os.replace(tmp, metrics_path)

    print(f"[TIMING] Total time for configuration: {time.time() - config_t0:.2f}s")

    return {
        "val_seq_acc": float(val_seq_acc),
        "val_vid_acc": float(val_vid_acc),
        "val_frame_acc": float(val_frame_acc),
        "val_loss": float(val_loss)
    }

# ======================
#  FULL TRAIN+VAL TRAIN
# ======================
def train_full_on_trainval(cfg_run: dict,
                           master_feature_cols: List[str],
                           save_dir: str,
                           save_name: str = "best_full_trainval.pt") -> str:
    """
    Train on TRAIN+VAL union (no external validation). Early-stop on training loss.
    Returns the saved weights path.
    """
    weights_path = os.path.join(save_dir, save_name)

    # load lists
    train_ids = _read_list(_require_file(TRAIN_LIST, "TRAIN_LIST"))
    val_ids   = _read_list(_require_file(VAL_LIST,   "VAL_LIST"))
    if INCLUDE_LIST or EXCLUDE_LIST:
        train_ids = _apply_include_exclude(train_ids, INCLUDE_LIST, EXCLUDE_LIST)
        val_ids   = _apply_include_exclude(val_ids,   INCLUDE_LIST, EXCLUDE_LIST)
    all_ids = sorted(set(train_ids) | set(val_ids))

    # dataset/loader over the union
    full_ds = StreamingSequenceDataset(all_ids, cfg_run, master_feature_cols)
    full_loader = _make_loader(full_ds, cfg_run["batch_size"], shuffle=True)

    # persist per-config feature cols (for test-time)
    featcols_local = os.path.join(save_dir, "feature_cols.json")
    with open(featcols_local, "w") as f:
        json.dump(full_ds.feature_cols, f)

    # ---------- PREPROCESSOR ----------
    cfg_norm = cfg_run["norm_mode"].lower()
    if cfg_norm in ("zscore", "zscore+l2"):
        mean, std = compute_mean_std_from_cache(
            video_ids=all_ids,
            master_feature_cols=master_feature_cols,
            col_idx=full_ds.col_idx,
            device=DEVICE,
            batch_size=max(131072, cfg_run["batch_size"] * cfg_run["seq_len"])
        )
        adj_mean, adj_std = mean.clone(), std.clone()
        if cfg_run.get("keep_au_c_raw", True):
            auc_idx = [i for i, n in enumerate(full_ds.feature_cols) if n.endswith("_c")]
            if auc_idx:
                adj_mean[auc_idx] = 0.0
                adj_std[auc_idx]  = 1.0
        scaler_path = os.path.join(save_dir, "scaler.pt")
        torch.save({"mean": adj_mean.cpu(), "std": adj_std.cpu()}, scaler_path)
        print(f"[norm][full] saved scaler -> {scaler_path}")
        preproc = build_preprocessor(cfg_norm, full_ds.feature_cols, DEVICE,
                                     mean=adj_mean, std=adj_std, keep_au_c_raw=False)
    else:
        scaler_path = os.path.join(save_dir, "scaler.pt")
        D = len(full_ds.feature_cols)
        torch.save({"mean": torch.zeros(D), "std": torch.ones(D)}, scaler_path)
        preproc = build_preprocessor(cfg_norm, full_ds.feature_cols, DEVICE,
                                     mean=None, std=None,
                                     keep_au_c_raw=cfg_run.get("keep_au_c_raw", True))

    # model + opt + sched (plateau on train loss)
    model = TemporalFFRNN(
        input_dim=full_ds.input_dim,
        ff_hidden=cfg_run["ff_hidden"],
        ff_hidden2=cfg_run["ff_hidden2"],
        dropout=cfg_run["dropout"],
        rnn_type=cfg_run["rnn_type"],
        rnn_hidden=cfg_run["rnn_hidden"],
        rnn_layers=cfg_run["rnn_layers"],
        bidirectional=cfg_run["bidirectional"],
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)

    optim = _build_optimizer(cfg_run["optimizer"], model.parameters(),
                             lr=cfg_run["lr"], wd=cfg_run["weight_decay"])
    ce = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optim, mode="min",
                                  factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE,
                                  min_lr=MIN_LR)

    best_tr_loss = math.inf
    no_improve   = 0
    print(f"[FULL FIT] TRAIN+VAL union | epochs={EPOCHS} | early-stop on train loss (patience={ES_PATIENCE})")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, data_wait, compute = 0.0, 0.0, 0.0
        correct, total = 0, 0  
        ep_t0 = time.time()

        for xb, yb in full_loader:
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

        tr_loss = run_loss / max(1, len(full_loader))
        tr_seq_acc = correct / max(1, total)
        scheduler.step(tr_loss)

        if tr_loss + 1e-8 < best_tr_loss:
            best_tr_loss = tr_loss
            no_improve = 0
            torch.save(model.state_dict(), weights_path)
        else:
            no_improve += 1

        ep_time = time.time() - ep_t0
        print(f"[FULL FIT] ep {epoch:03d} | train_loss {tr_loss:.4f} | train_seq_acc {tr_seq_acc:.4f} "
              f"| best_train_loss {best_tr_loss:.4f} | no_improve {no_improve}/{ES_PATIENCE} "
              f"| ep_time {ep_time:.2f}s (data {data_wait:.2f}s, compute {compute:.2f}s)")

        if no_improve >= ES_PATIENCE:
            print("[FULL FIT][early-stop] patience reached.")
            break

    if not os.path.isfile(weights_path):
        torch.save(model.state_dict(), weights_path)
    return weights_path

# ======================
#       TEST EVAL (frame + video-from-frames)
# ======================
@torch.no_grad()
def _collect_seq_preds(model, loader, device, preproc):
    """Return y_true, y_pred for sequence-level eval."""
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True)
        B, L, D = xb.shape
        xb = xb.view(B*L, D); xb = preproc(xb); xb = xb.view(B, L, D)
        logits = model(xb)
        y_pred.append(logits.argmax(1).detach().cpu().numpy())
        y_true.append(yb.detach().cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)

@torch.no_grad()
def eval_on_test(cfg_run: dict,
                 cfg_dir: str,
                 master_feature_cols: List[str],
                 weights_path: str | None = None):
    """
    Evaluate on TEST using the provided weights_path (or <cfg_dir>/best_state.pt).
    Reports: sequence-level acc + report, frame-level acc + report,
             video-level acc via frame-prob aggregation with argmax.
    """
    test_total_t0 = time.time()

    if weights_path is None:
        weights_path = os.path.join(cfg_dir, "best_state.pt")
    featcols_local = os.path.join(cfg_dir, "feature_cols.json")

    if not os.path.isfile(weights_path):
        print(f"[test-eval] No weights at {weights_path}; skipping."); return
    if not os.path.isfile(featcols_local):
        print(f"[test-eval] Missing feature_cols.json in {cfg_dir}; skipping."); return

    feature_cols = json.load(open(featcols_local))

    with _time("Read TEST list"):
        test_ids = _read_list(_require_file(TEST_LIST, "TEST_LIST"))
        if INCLUDE_LIST or EXCLUDE_LIST:
            test_ids = _apply_include_exclude(test_ids, INCLUDE_LIST, EXCLUDE_LIST)

    with _time("Build StreamingSequenceDataset (TEST)"):
        test_ds = StreamingSequenceDataset(test_ids, cfg_run, master_feature_cols)
    with _time(f"Dataloader (TEST) bs={cfg_run['batch_size']} N={len(test_ds)}"):
        test_loader = _make_loader(test_ds, cfg_run["batch_size"], shuffle=False)

    scaler_path = os.path.join(cfg_dir, "scaler.pt")
    if os.path.isfile(scaler_path) and cfg_run["norm_mode"] in ("zscore", "zscore+l2"):
        d = torch.load(scaler_path, map_location=DEVICE)
        mean, std = d["mean"].to(DEVICE), d["std"].to(DEVICE)
        preproc = build_preprocessor(
            norm_mode=cfg_run["norm_mode"],
            feature_cols=feature_cols,
            device=DEVICE,
            mean=mean, std=std,
            keep_au_c_raw=cfg_run["keep_au_c_raw"],
        )
        print(f"[norm][test] loaded scaler from {scaler_path}")
    else:
        preproc = build_preprocessor("none", feature_cols, DEVICE, None, None, keep_au_c_raw=True)
        if cfg_run["norm_mode"] in ("zscore", "zscore+l2"):
            print("[norm][test] WARNING: norm_mode requested but scaler.pt not found; using identity.")

    with _time("Rebuild model + load weights"):
        model = TemporalFFRNN(
            input_dim=test_ds.input_dim,
            ff_hidden=cfg_run["ff_hidden"],
            ff_hidden2=cfg_run["ff_hidden2"],
            dropout=cfg_run["dropout"],
            rnn_type=cfg_run["rnn_type"],
            rnn_hidden=cfg_run["rnn_hidden"],
            rnn_layers=cfg_run["rnn_layers"],
            bidirectional=cfg_run["bidirectional"],
            num_classes=len(EMOTION_TO_IDX),
        ).to(DEVICE)
        state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state)

    # === Sequence-level ===
    with _time("TEST sequence-level eval (loss/acc)"):
        test_loss, test_seq_acc, _, _ = _eval_epoch_seq(model, test_loader, DEVICE, preproc)
    with _time("Collect sequence-level predictions"):
        y_true_seq, y_pred_seq = _collect_seq_preds(model, test_loader, DEVICE, preproc)

    # === Frame-level ===
    with _time("Collect frame-level predictions"):
        y_true_fr, y_pred_fr = _collect_frame_preds_stream(model, test_ids, cfg_run, master_feature_cols, preproc)
    test_frame_acc = float((y_true_fr == y_pred_fr).mean()) if y_true_fr.size else float("nan")

    # === Video-level (from frames, argmax over avg probs) ===
    with _time("Video-level from FRAMES (avg probs -> argmax)"):
        y_true_vid, y_pred_vid = _collect_video_preds_from_frames(model, test_ids, cfg_run, master_feature_cols, preproc)
    test_vid_acc = float((y_true_vid == y_pred_vid).mean()) if y_true_vid.size else float("nan")

    classes = [IDX_TO_EMO[i] for i in range(len(EMOTION_TO_IDX))]

    print(f"[TEST] seq_acc={test_seq_acc:.4f} | frame_acc={test_frame_acc:.4f} | vid_acc_frameAgg={test_vid_acc:.4f}")

    # Sequence-level report
    if y_true_seq.size:
        print("\n[TEST][SEQUENCE] classification report:")
        print(classification_report(y_true_seq, y_pred_seq, target_names=classes, digits=3))
        print("[TEST][SEQUENCE] confusion matrix:")
        print(confusion_matrix(y_true_seq, y_pred_seq))
    else:
        print("\n[TEST][SEQUENCE] No sequence predictions collected.")

    # Frame-level report
    if y_true_fr.size:
        print("\n[TEST][FRAME] classification report:")
        print(classification_report(y_true_fr, y_pred_fr, target_names=classes, digits=3))
        print("[TEST][FRAME] confusion matrix:")
        print(confusion_matrix(y_true_fr, y_pred_fr))
    else:
        print("\n[TEST][FRAME] No frame predictions collected.")

    # Video-level report
    if y_true_vid.size:
        print("\n[TEST][VIDEO] classification report:")
        print(classification_report(y_true_vid, y_pred_vid, target_names=classes, digits=3))
        print("[TEST][VIDEO] confusion matrix:")
        print(confusion_matrix(y_true_vid, y_pred_vid))
    else:
        print("\n[TEST][VIDEO] No video predictions collected.")

    print(f"\n[TIMING] Total TEST evaluation time: {time.time() - test_total_t0:.2f}s")

# ======================
#           MAIN
# ======================
def main():
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)

    master_feature_cols = get_master_feature_cols()

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

        print(f"[VAL] seq_acc={info['val_seq_acc']:.4f} | vid_acc(frame-agg)={info['val_vid_acc']:.4f} | frame_acc≈seq_acc={info['val_frame_acc']:.4f}")
        print(f"[TIMING] Configuration total time: {cfg_dur:.2f}s")

    # summary CSV
    df = pd.DataFrame(results).sort_values("val_seq_acc", ascending=False)
    out_csv = os.path.join(ART_DIR_SUB, "rnn_grid_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[done] wrote grid results -> {out_csv}")
    show_cols = ["tag","val_seq_acc","val_vid_acc","val_frame_acc","val_loss"]
    print(df[show_cols].to_string(index=False))

    # pick best-by-seq and best-by-video (frame-agg)
    if len(results):
        best_by_seq = max(results, key=lambda d: d["val_seq_acc"])
        best_by_vid = max(results, key=lambda d: d["val_vid_acc"])

        # --- BEST BY SEQ: retrain on TRAIN+VAL, then test ---
        best_tag_seq = _config_tag(best_by_seq)
        best_dir_seq = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/cremad_GridSearch_unscaled_RNN/bestModels/bestModels_std_300epoch_corrected/best_full_trainval_seq/"
        os.makedirs(best_dir_seq, exist_ok=True)
        print(best_tag_seq)
        print("\n=== FULL TRAIN+VAL: BEST by SEQUENCE ACC ===")
        full_seq_weights = train_full_on_trainval(best_by_seq, master_feature_cols,
                                                  save_dir=best_dir_seq,
                                                  save_name="best_full_trainval_seq.pt")
        print("\n=== TEST evaluation (weights from full TRAIN+VAL, best-by-seq) ===")
        print(best_tag_seq)
        eval_on_test(best_by_seq, best_dir_seq, master_feature_cols, weights_path=full_seq_weights)

        # --- BEST BY VIDEO (frame-agg): retrain on TRAIN+VAL, then test ---
        if best_by_vid["tag"] != best_by_seq["tag"]:
            best_tag_vid = _config_tag(best_by_vid)
            best_dir_vid = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/cremad_GridSearch_unscaled_RNN/bestModels/bestModels_std_300epoch_corrected/best_full_trainval_video/"
            os.makedirs(best_dir_vid, exist_ok=True)
            print("\n=== FULL TRAIN+VAL: BEST by VIDEO ACC (frame-agg) ===")
            full_vid_weights = train_full_on_trainval(best_by_vid, master_feature_cols,
                                                      save_dir=best_dir_vid,
                                                      save_name="best_full_trainval_video.pt")
            print("\n=== TEST evaluation (weights from full TRAIN+VAL, best-by-video) ===")
            print(best_tag_vid)
            eval_on_test(best_by_vid, best_dir_vid, master_feature_cols, weights_path=full_vid_weights)

if __name__ == "__main__":
    main()

