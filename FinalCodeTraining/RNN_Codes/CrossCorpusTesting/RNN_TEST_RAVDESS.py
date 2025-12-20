# eval_ravdess_cached_seq_vid_frame.py
# Evaluate a TemporalFFRNN (trained on CREMA-D) on RAVDESS.
# - Cache per-video arrays: cache/X.npy (ALL features in MASTER order), cache/y_str.npy (original labels)
# - On load, select ONLY the columns present in the trained config's feature_cols.json
# - Never rewrites CSVs; label strings are mapped to CREMA-D classes at evaluation time
# - Prints sequence-, video-, and frame-level metrics (+ confusion matrices)

import os, sys, json, time, math
import numpy as np
import pandas as pd
from typing import List, Tuple

# --- clean env for torch import (avoids LD_LIBRARY_PATH conflicts) ---
if os.environ.get("LD_LIBRARY_PATH") and not os.environ.get("_LDLIBPATH_CLEANED"):
    env = dict(os.environ); env.pop("LD_LIBRARY_PATH", None); env["_LDLIBPATH_CLEANED"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)
os.environ.pop("LD_LIBRARY_PATH", None)

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ========== CONFIG: EDIT ME ==========
RAVDESS_ROOT = "/media/root918/OS/MaryiamProject/copiedFilesRAVDESS/"   # contains Actor_XX/<video_dir>/<csv>
WEIGHTS_PATH = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/cremad_GridSearch_unscaled_RNN/bestModels/bestModels_std_300epoch_corrected/best_full_trainval_seq/best_full_trainval_seq.pt"

# master list = ALL feature names in the unified order used to cache X.npy
MASTER_FEATURE_COLS_JSON = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/cremad_GridSearch_unscaled_RNN/master_feature_cols.json"

# trained-config list = ONLY the features used by the best model
FEATCOLS_JSON = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/cremad_GridSearch_unscaled_RNN/bestModels/bestModels_std_300epoch_corrected/best_full_trainval_seq/feature_cols.json"

COMBINED_CSV_NAME = "au_resnet_vgg_with_gt.csv"
LABEL_COL_PREFERRED = "emotion"

# TemporalFFRNN architecture (should match the saved model) 

#best seq model config
MODEL_CFG = {
    "ff_hidden": 1024,
    "ff_hidden2": 512,         # None or int
    "dropout": 0.5,
    "rnn_type": "lstm",          # "gru" or "lstm"
    "rnn_hidden": 128,
    "rnn_layers": 2,
    "bidirectional": False,
    "num_classes": 6,           # CREMA-D 6-class
}

#best video model config
#MODEL_CFG = {
 #   "ff_hidden": 1024,
  #  "ff_hidden2": None,         # None or int
   # "dropout": 0.5,
    #"rnn_type": "gru",          # "gru" or "lstm"
    #"rnn_hidden": 256,
    #"rnn_layers": 1,
    #"bidirectional": False,
    #"num_classes": 6,           # CREMA-D 6-class
#}
# Sequence windowing
SEQ_LEN = 30
STRIDE  = 30

# Runtime opts
SKIP_FIRST_N = 0
BATCH_SIZE   = 512
PLOT_CM      = True
SEED         = 42

# --- Normalization options ---
SCALER_SOURCE  = "target"           # "train" -> use saved scaler.pt, "target" -> compute on RAVDESS
NORM_MODE     = "zscore"          # "none" | "l2" | "zscore" | "zscore+l2"
KEEP_AU_C_RAW = True            # if True, *_c columns are left unscaled under zscore
SCALER_PATH   = os.path.join(os.path.dirname(FEATCOLS_JSON), "scaler.pt")  # saved during training

# =====================================

# ---- project imports ----
from utils.features import load_feature_cols, Standardize  # Standardize kept for future use
from models.CNN_RNNmodel import TemporalFFRNN

# ======================
#   LABEL MAPPING (RAVDESS -> CREMA-D 6-class)
# ======================
_NAME_TO_LETTER = {
    "neutral": "N",
    "happy":   "H",
    "sad":     "S",
    "anger":   "A",
    "angry":   "A",
    "fearful": "F",
    "disgust": "D",
    # calm/surprised intentionally unmapped -> ignored
}
EMOTION_TO_IDX = {"H":0,"S":1,"A":2,"N":3,"D":4,"F":5}
IDX_TO_EMO     = {v:k for k,v in EMOTION_TO_IDX.items()}

def _label_to_idx(s: str) -> int:
    if s is None: return -1
    s = str(s).strip()
    if not s: return -1
    if len(s) == 1:
        return EMOTION_TO_IDX.get(s.upper(), -1)
    letter = _NAME_TO_LETTER.get(s.lower())
    return EMOTION_TO_IDX.get(letter, -1) if letter else -1

def _map_y_str_to_idx(y_str_arr: np.ndarray) -> np.ndarray:
    return np.fromiter((_label_to_idx(s) for s in y_str_arr), dtype=np.int64)

# ======================
#   IO / PATH HELPERS
# ======================
def _require_file(path, desc):
    if not os.path.isfile(path): raise FileNotFoundError(f"Missing {desc}: {path}")
    return path

def list_ravdess_video_ids(root_dir: str) -> list[str]:
    ids = []
    actors = sorted(d for d in os.listdir(root_dir) if d.startswith("Actor_"))
    for actor in actors:
        ap = os.path.join(root_dir, actor)
        if not os.path.isdir(ap): continue
        for vid in sorted(os.listdir(ap)):
            vp = os.path.join(ap, vid)
            if os.path.isdir(vp):
                ids.append(f"{actor}/{vid}")
    if not ids:
        raise RuntimeError(f"No RAVDESS videos found under: {root_dir}")
    return ids

def _cache_paths(root: str, vid_id: str) -> Tuple[str, str, str]:
    vdir = os.path.join(root, vid_id)
    cdir = os.path.join(vdir, "cache")
    os.makedirs(cdir, exist_ok=True)
    Xp = os.path.join(cdir, "X.npy")
    Yp = os.path.join(cdir, "y_str.npy")   # original label strings
    Mp = os.path.join(cdir, "meta.json")   # stores the master feature list used to build X.npy
    return Xp, Yp, Mp

# ======================
#   CACHE BUILDING (write ALL features in MASTER order)
# ======================
def build_cache_for_video(root: str,
                          vid_id: str,
                          master_feature_cols: List[str],
                          combined_csv_name: str,
                          label_col_preferred: str,
                          skip_first_n: int) -> None:
    Xp, Yp, Mp = _cache_paths(root, vid_id)
    if os.path.isfile(Xp) and os.path.isfile(Yp) and os.path.isfile(Mp):
        return

    csvp = os.path.join(root, vid_id, combined_csv_name)
    if not os.path.isfile(csvp):
        print(f"[cache][skip] missing CSV for {vid_id}: {csvp}")
        return

    try:
        df = pd.read_csv(csvp)
    except Exception as e:
        print(f"[cache][skip] cannot read {csvp}: {e}")
        return

    missing = [c for c in master_feature_cols if c not in df.columns]
    if missing:
        print(f"[cache][skip] {vid_id}: missing {len(missing)} of {len(master_feature_cols)} master feature columns")
        return

    feats = (df[master_feature_cols]
             .replace([np.inf, -np.inf], np.nan)
             .fillna(0.0)
             .astype("float32"))

    if skip_first_n and skip_first_n > 0:
        if len(feats) <= skip_first_n:
            print(f"[cache][skip] {vid_id}: too few frames after SKIP_FIRST_N")
            return
        feats = feats.iloc[skip_first_n:].reset_index(drop=True)
        df    = df.iloc[skip_first_n:].reset_index(drop=True)

    # prefer given label col, else fall back to 'emotion'
    used_label_col = label_col_preferred if label_col_preferred in df.columns else ("emotion" if "emotion" in df.columns else None)
    if used_label_col is None:
        print(f"[cache][skip] {vid_id}: no label column ('{label_col_preferred}' or 'emotion')")
        return

    # Save features in MASTER order and labels as fixed-width unicode (memmappable)
    np.save(Xp, feats.to_numpy(copy=False))
    y_str = df[used_label_col].astype(str).to_numpy(dtype="U32", copy=False)
    np.save(Yp, y_str)
    with open(Mp, "w") as f:
        json.dump({"feature_cols_master": master_feature_cols,
                   "label_source_col": used_label_col}, f)

def _load_labels_memsafe(Yp: str) -> np.ndarray:
    try:
        y = np.load(Yp, mmap_mode="r")
        if y.dtype.kind in ("U", "S", "i", "f"):
            return y
    except ValueError:
        pass
    y = np.load(Yp, allow_pickle=True)
    if y.dtype == object:
        y = y.astype("U32")
        try:
            tmp = Yp + ".tmp"
            np.save(tmp, y)
            os.replace(tmp, Yp)
        except Exception:
            pass
    return y

def ensure_cache_for_all(root: str,
                         video_ids: List[str],
                         master_feature_cols: List[str],
                         combined_csv_name: str,
                         label_col_preferred: str,
                         skip_first_n: int) -> None:
    for (i,vid) in enumerate(video_ids):
        build_cache_for_video(root, vid, master_feature_cols, combined_csv_name, label_col_preferred, skip_first_n)
        if (i%50==0):
          print("done: ", i) 
        

# ======================
#   WINDOWING + FORWARD
# ======================
def _build_windows(X: np.ndarray, T: int, S: int, pad_short=True) -> Tuple[np.ndarray, List[int]]:
    N, D = X.shape
    if N <= 0:
        return np.zeros((0, T, D), dtype=np.float32), []
    if N < T:
        if not pad_short: return np.zeros((0, T, D), dtype=np.float32), []
        pad = np.repeat(X[[-1], :], T - N, axis=0)
        return np.stack([np.concatenate([X, pad], axis=0)], axis=0).astype("float32"), [0]
    starts, wins = [], []
    for s in range(0, N - T + 1, S):
        wins.append(X[s:s+T]); starts.append(s)
    if starts and starts[-1] != (N - T):
        wins.append(X[N - T:N]); starts.append(N - T)
    return np.stack(wins, axis=0).astype("float32"), starts

@torch.no_grad()
def _predict_seq_logits(model: nn.Module, Xseq: np.ndarray, device, scaler: nn.Module, bs=128) -> np.ndarray | None:
    if Xseq is None or len(Xseq) == 0:
        return None
    outs = []
    model.eval()
    for i in range(0, len(Xseq), bs):
        # contiguous copy -> avoids non-writable memmap warning
        b = torch.from_numpy(np.ascontiguousarray(Xseq[i:i+bs])).float().to(device)   # (B, T, D)
        B, T, D = b.shape
        b2 = b.reshape(B*T, D)
        b2 = scaler(b2)     # currently Identity, kept for future
        b  = b2.reshape(B, T, D)
        logits = model(b)   # (B, C)
        outs.append(logits.detach().cpu().numpy())
    return np.vstack(outs)

def _frame_probs_from_windows(X: np.ndarray, T: int, S: int, device, model, scaler, bs: int):
    Xseq, starts = _build_windows(X, T, S, pad_short=True)
    if Xseq.shape[0] == 0: return None, None
    logits = _predict_seq_logits(model, Xseq, device, scaler, bs=max(1, bs//2))
    if logits is None: return None, None
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()  # (W, C)
    N = X.shape[0]; C = probs.shape[1]
    sum_probs = np.zeros((N, C), dtype=np.float64)
    counts    = np.zeros((N,),    dtype=np.int32)
    for w, s in enumerate(starts):
        e = min(s + T, N)
        sum_probs[s:e] += probs[w]
        counts[s:e]    += 1
    mask = counts > 0
    if not mask.any(): return None, None
    frame_probs = np.zeros((N, C), dtype=np.float32)
    frame_probs[mask] = (sum_probs[mask] / counts[mask, None])
    return frame_probs, mask

# ======================
#   DATASET (SEQUENCE FROM CACHE, with column selection)
# ======================
class StreamingSequenceDatasetFromCache(Dataset):
    """
    Builds sequence windows from cache/X.npy and cache/y_str.npy (ALL features in MASTER order).
    On __init__, compute column indices that map MASTER -> TRAINED feature set.
    Sequence label = majority (mode) among valid mapped frames in the window.
    """
    def __init__(self, root: str, video_ids: List[str], T: int, S: int,
                 trained_feature_cols: List[str], master_feature_cols: List[str]):
        self.root = root
        self.vids = list(video_ids)
        self.T, self.S = int(T), int(S)
        self._arrays = {}
        self._lengths = {}
        self.index = []

        # map trained feature names to indices within master
        name_to_idx = {n: i for i, n in enumerate(master_feature_cols)}
        col_idx = []
        missing = []
        for n in trained_feature_cols:
            i = name_to_idx.get(n, None)
            if i is None:
                missing.append(n)
            else:
                col_idx.append(i)
        if missing:
            print(f"[warn] {len(missing)} trained features not found in master; they will be ignored.")
        if not col_idx:
            raise ValueError("No overlap between trained feature list and master feature list.")
        self.col_idx = np.array(sorted(set(col_idx)), dtype=np.int64)

        for vid in self.vids:
            Xp, Yp, _ = _cache_paths(root, vid)
            if not (os.path.isfile(Xp) and os.path.isfile(Yp)): 
                continue
            Xm = np.load(Xp, mmap_mode="r")
            N = len(Xm)
            self._lengths[vid] = N
            if N >= self.T:
                for s in range(0, N - self.T + 1, self.S):
                    self.index.append((vid, s))
                tail = N - self.T
                if len(self.index) == 0 or self.index[-1] != (vid, tail):
                    self.index.append((vid, tail))
            elif N > 0:
                self.index.append((vid, 0))

        # input_dim is len of selected columns
        self._input_dim = int(len(self.col_idx))
        print(f"[seq-ds] videos={len(self.vids)} | sequences={len(self.index)} | dim={self._input_dim}")

    def __len__(self): return len(self.index)
    @property
    def input_dim(self): return self._input_dim

    def _get_arrays(self, vid: str):
        arrs = self._arrays.get(vid)
        if arrs is None:
            Xp, Yp, _ = _cache_paths(self.root, vid)
            X_m = np.load(Xp, mmap_mode="r")         # ALL features in MASTER order
            y_m = _load_labels_memsafe(Yp)           # original strings
            arrs = (X_m, y_m)
            if len(self._arrays) > 16: self._arrays.clear()
            self._arrays[vid] = arrs
        return arrs

    def __getitem__(self, idx: int):
        vid, start = self.index[idx]
        X_m, y_m_str = self._get_arrays(vid)
        N = self._lengths[vid]; T = self.T

        if start + T <= N:
            win_all = X_m[start:start+T]
            lab_str = y_m_str[start:start+T]
        else:
            k = max(0, N-1)
            win_all = np.empty((T, X_m.shape[1]), dtype=X_m.dtype)
            end_temp = max(0, N - start)
            if end_temp > 0:
                win_all[:end_temp] = X_m[start:N]
                win_all[end_temp:] = X_m[k]
                lab_str = np.empty((T,), dtype=object)
                lab_str[:end_temp] = y_m_str[start:N]
                lab_str[end_temp:] = y_m_str[k]
            else:
                win_all[:] = X_m[k]
                lab_str = np.array([y_m_str[k]] * T, dtype=object)

        # select only trained columns
        win = np.ascontiguousarray(win_all[:, self.col_idx])

        lab_idx = np.fromiter((_label_to_idx(s) for s in lab_str), dtype=np.int64)
        valid = lab_idx >= 0
        if not np.any(valid):
            seq_label = -1
        else:
            vals, counts = np.unique(lab_idx[valid], return_counts=True)
            seq_label = int(vals[np.argmax(counts)])

        return torch.from_numpy(win), torch.tensor(seq_label, dtype=torch.long)
class L2Normalize(nn.Module):
    def __init__(self, eps: float = 1e-6): 
        super().__init__(); self.eps = eps
    def forward(self, x): 
        return x / (x.norm(dim=1, keepdim=True).clamp_min(self.eps))

def build_preprocessor(norm_mode: str,
                       feature_cols: list[str],
                       device: torch.device,
                       mean: torch.Tensor | None = None,
                       std:  torch.Tensor | None = None,
                       keep_au_c_raw: bool = True) -> nn.Module:
    mode = str(norm_mode).lower()
    if mode == "none": return nn.Identity().to(device)
    if mode == "l2":   return L2Normalize().to(device)

    # zscore or zscore+l2
    assert mean is not None and std is not None, "mean/std required for zscore* modes"
    if mean.dim() != 1: mean = mean.view(-1)
    if std.dim()  != 1: std  = std.view(-1)
    D = len(feature_cols)
    if mean.numel() != D or std.numel() != D:
        raise ValueError(f"mean/std length ({mean.numel()}/{std.numel()}) != feature dim ({D})")
    mean = mean.clone(); std = std.clone()

    if keep_au_c_raw and feature_cols:
        auc_idx = [i for i, n in enumerate(feature_cols) if n.endswith("_c")]
        if auc_idx:
            mean[auc_idx] = 0.0
            std[auc_idx]  = 1.0

    z = Standardize(mean, std).to(device)
    return z if mode == "zscore" else nn.Sequential(z, L2Normalize()).to(device)

@torch.no_grad()
def compute_target_mean_std_from_cache(root: str,
                                       video_ids: list[str],
                                       col_idx: np.ndarray,
                                       device: torch.device,
                                       batch_size: int = 262144) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute z-score stats on the *target* dataset (RAVDESS), using the same
    selected columns (col_idx) as the trained model.
    """
    S = None; SS = None; N = 0
    for vid in video_ids:
        Xp, Yp, _ = _cache_paths(root, vid)
        if not os.path.isfile(Xp): continue
        Xm = np.load(Xp, mmap_mode="r")  # (N, D_all)
        if Xm.shape[0] == 0: continue
        for i in range(0, Xm.shape[0], batch_size):
            xb = Xm[i:i+batch_size, :][:, col_idx]    # (B, D_sel)
            xb = torch.from_numpy(np.ascontiguousarray(xb)).float().to(device)
            if S is None:
                S  = xb.sum(0)
                SS = (xb * xb).sum(0)
                N  = xb.shape[0]
            else:
                S  = S  + xb.sum(0)
                SS = SS + (xb * xb).sum(0)
                N  = N  + xb.shape[0]
    if S is None or N == 0:
        D = len(col_idx)
        return torch.zeros(D, device=device), torch.ones(D, device=device)
    mean = S / N
    var  = (SS / N) - (mean * mean)
    std  = torch.sqrt(torch.clamp(var, min=1e-8))
    return mean, std

# ======================
#   EVAL HELPERS
# ======================
def _make_loader(ds: Dataset, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False,
                      num_workers=min(8, max(2, (os.cpu_count() or 4) - 1)),
                      pin_memory=torch.cuda.is_available(), persistent_workers=True)

@torch.no_grad()
def eval_sequence_level(model: nn.Module, loader: DataLoader, device, scaler: nn.Module):
    ce = nn.CrossEntropyLoss()
    model.eval()
    y_true, y_pred = [], []
    total, correct, run_loss, n_batches = 0, 0, 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device)
        B, L, D = xb.shape
        xb2 = xb.view(B*L, D); xb2 = scaler(xb2); xb = xb2.view(B, L, D)
        logits = model(xb)
        pred = logits.argmax(1)

        mask = (yb >= 0)
        if mask.any():
            yv = yb[mask]
            pv = pred[mask]
            run_loss += ce(logits[mask], yv).item()
            correct += (pv == yv).sum().item()
            total   += yv.numel()
            y_true.append(yv.cpu().numpy())
            y_pred.append(pv.cpu().numpy())
            n_batches += 1

    y_true = np.concatenate(y_true) if y_true else np.array([], dtype=int)
    y_pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=int)
    avg_loss = (run_loss / max(1, n_batches))
    acc = (correct / max(1, total)) if total else float("nan")
    return avg_loss, acc, y_true, y_pred

@torch.no_grad()
def eval_video_level_from_cache(root, video_ids, T, S, device, model, scaler, bs,
                                trained_feature_cols, master_feature_cols):
    # precompute col indices once
    name_to_idx = {n: i for i, n in enumerate(master_feature_cols)}
    col_idx = [name_to_idx[n] for n in trained_feature_cols if n in name_to_idx]
    col_idx = np.array(col_idx, dtype=np.int64)

    y_true_v, y_pred_v = [], []
    for vid in video_ids:
        Xp, Yp, _ = _cache_paths(root, vid)
        if not (os.path.isfile(Xp) and os.path.isfile(Yp)): continue
        X_all = np.load(Xp, mmap_mode="r")
        if X_all.shape[0] == 0: continue
        # select only trained columns
        X = X_all[:, col_idx]

        y = _map_y_str_to_idx(_load_labels_memsafe(Yp))

        frame_probs, mask = _frame_probs_from_windows(X, T, S, device, model, scaler, bs)
        if frame_probs is None: continue
        valid = (y >= 0) & mask
        if not np.any(valid): continue

        pred_idx = int(frame_probs[valid].mean(axis=0).argmax())
        vals, counts = np.unique(y[valid], return_counts=True)
        gt = int(vals[np.argmax(counts)])

        y_true_v.append(gt); y_pred_v.append(pred_idx)

    return (np.array(y_true_v, dtype=int), np.array(y_pred_v, dtype=int))

@torch.no_grad()
def eval_frame_level_from_cache(root, video_ids, T, S, device, model, scaler, bs,
                                trained_feature_cols, master_feature_cols):
    name_to_idx = {n: i for i, n in enumerate(master_feature_cols)}
    col_idx = [name_to_idx[n] for n in trained_feature_cols if n in name_to_idx]
    col_idx = np.array(col_idx, dtype=np.int64)

    y_true_f, y_pred_f = [], []
    for vid in video_ids:
        Xp, Yp, _ = _cache_paths(root, vid)
        if not (os.path.isfile(Xp) and os.path.isfile(Yp)): continue
        X_all = np.load(Xp, mmap_mode="r")
        if X_all.shape[0] == 0: continue

        # select only trained columns
        X = X_all[:, col_idx]
        y = _map_y_str_to_idx(_load_labels_memsafe(Yp))

        frame_probs, mask = _frame_probs_from_windows(X, T, S, device, model, scaler, bs)
        if frame_probs is None: continue
        valid = (y >= 0) & mask
        if not np.any(valid): continue

        y_true_f.append(y[valid])
        y_pred_f.append(frame_probs[valid].argmax(axis=1))

    if y_true_f:
        y_true_f = np.concatenate(y_true_f)
        y_pred_f = np.concatenate(y_pred_f)
    else:
        y_true_f = np.array([], dtype=int); y_pred_f = np.array([], dtype=int)
    return y_true_f, y_pred_f

def print_report_and_cm(title: str, y_true: np.ndarray, y_pred: np.ndarray, classes_letters: List[str], plot=False):
    print(f"\n[{title}]")
    if y_true.size == 0:
        print("nothing evaluated (empty set).")
        return
    names = []
    letter_to_name = {'H':'happy','S':'sad','A':'anger','N':'neutral','D':'disgust','F':'fearful'}
    for l in classes_letters:
        names.append(f"{l}:{letter_to_name.get(l, l)}")
    print(classification_report(y_true, y_pred, target_names=names, digits=3))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes_letters))))
    print("Confusion Matrix (counts):\n", cm)
    if plot:
        cmn = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=names)
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, include_values=False)
        thresh = np.nanmax(cmn)/2.0 if cmn.size else 0
        for i in range(cmn.shape[0]):
            for j in range(cmn.shape[1]):
                pct = 100.0 * (cmn[i, j] if cm.sum(axis=1, keepdims=True)[i] != 0 else 0)
                ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                        ha="center", va="center",
                        color="white" if cmn[i, j] > thresh else "black", fontsize=9)
        plt.title(f'Normalized Confusion Matrix ({title})')
        plt.tight_layout(); plt.show()

# ======================
#           MAIN
# ======================
def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- artifacts ---
    _require_file(WEIGHTS_PATH, "WEIGHTS_PATH")
    _require_file(FEATCOLS_JSON, "feature_cols.json (trained config)")
    _require_file(MASTER_FEATURE_COLS_JSON, "master_feature_cols.json (cache order)")

    trained_feature_cols = load_feature_cols(FEATCOLS_JSON)            # LIST for the best model
    master_feature_cols  = load_feature_cols(MASTER_FEATURE_COLS_JSON) # LIST for caching full X.npy

    # --- discover videos & ensure cache (ALL features in MASTER order) ---
    video_ids = list_ravdess_video_ids(RAVDESS_ROOT)
    ensure_cache_for_all(RAVDESS_ROOT, video_ids, master_feature_cols,
                         COMBINED_CSV_NAME, LABEL_COL_PREFERRED, SKIP_FIRST_N)

    # --- Dataset / Loader (sequence-level), pass both lists so we can select cols ---
    seq_ds = StreamingSequenceDatasetFromCache(
        RAVDESS_ROOT, video_ids, SEQ_LEN, STRIDE,
        trained_feature_cols=trained_feature_cols,
        master_feature_cols=master_feature_cols
    )
    seq_loader = _make_loader(seq_ds, BATCH_SIZE, shuffle=False)

   
    def _master_to_trained_idx(master_feature_cols: list[str], feature_cols: list[str]) -> np.ndarray:
        name_to_idx = {n: i for i, n in enumerate(master_feature_cols)}
        return np.array([name_to_idx[n] for n in feature_cols if n in name_to_idx], dtype=np.int64)
    # --- Scaler selection: use training scaler or target (RAVDESS) scaler ---
    feature_cols = trained_feature_cols[:]   # order the model expects
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    if NORM_MODE in ("zscore", "zscore+l2"):
        mean = std = None
        sc_src = SCALER_SOURCE
        if sc_src == "train":
            if os.path.isfile(SCALER_PATH):
                d = torch.load(SCALER_PATH, map_location=device)
                mean, std = d["mean"].to(device), d["std"].to(device)
                print(f"[norm] using TRAIN scaler: {SCALER_PATH}")
                print(f"[norm] loaded stats shape: mean={mean.shape}, std={std.shape}, features={len(feature_cols)}")

            else:
                print(f"[norm][warn] SCALER_SOURCE=train but scaler not found at {SCALER_PATH}; "
                      f"falling back to TARGET stats.")
                sc_src = "target"

        if sc_src == "target":
            col_idx = _master_to_trained_idx(master_feature_cols, feature_cols)
            print("[norm] computing TARGET (RAVDESS) mean/std for selected columns…")
            mean, std = compute_target_mean_std_from_cache(
                root=RAVDESS_ROOT, video_ids=video_ids, col_idx=col_idx, device=device
            )

        scaler = build_preprocessor(
            norm_mode=NORM_MODE,
            feature_cols=feature_cols,
            device=device,
            mean=mean, std=std,
            keep_au_c_raw=KEEP_AU_C_RAW
        )
    elif NORM_MODE == "l2":
        scaler = build_preprocessor("l2", feature_cols, device)
    else:
        scaler = build_preprocessor("none", feature_cols, device)



    # --- Rebuild model + load weights ---
    model = TemporalFFRNN(
        input_dim=seq_ds.input_dim,
        ff_hidden=MODEL_CFG["ff_hidden"],
        ff_hidden2=(None if MODEL_CFG["ff_hidden2"] in [None, 0] else MODEL_CFG["ff_hidden2"]),
        dropout=float(MODEL_CFG["dropout"]),
        rnn_type=str(MODEL_CFG["rnn_type"]).lower(),
        rnn_hidden=int(MODEL_CFG["rnn_hidden"]),
        rnn_layers=int(MODEL_CFG["rnn_layers"]),
        bidirectional=bool(MODEL_CFG["bidirectional"]),
        num_classes=int(MODEL_CFG["num_classes"]),
    ).to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # --- SEQUENCE-LEVEL EVAL ---
    seq_loss, seq_acc, y_true_seq, y_pred_seq = eval_sequence_level(model, seq_loader, device, scaler)
    print(f"\n[TEST][SEQUENCE] loss={seq_loss:.4f} | acc={seq_acc:.4f}")
    print_report_and_cm("SEQUENCE (RAVDESS)", y_true_seq, y_pred_seq, [IDX_TO_EMO[i] for i in range(6)], plot=PLOT_CM)

    # --- VIDEO-LEVEL EVAL (select trained cols at runtime) ---
    y_true_v, y_pred_v = eval_video_level_from_cache(
        RAVDESS_ROOT, video_ids, SEQ_LEN, STRIDE, device, model, scaler, BATCH_SIZE,
        trained_feature_cols=trained_feature_cols,
        master_feature_cols=master_feature_cols
    )
    vid_acc = float((y_true_v == y_pred_v).mean()) if y_true_v.size else float("nan")
    print(f"\n[TEST][VIDEO] acc={vid_acc:.4f}")
    print_report_and_cm("VIDEO (RAVDESS)", y_true_v, y_pred_v, [IDX_TO_EMO[i] for i in range(6)], plot=PLOT_CM)

    # --- FRAME-LEVEL EVAL (select trained cols at runtime) ---
    y_true_f, y_pred_f = eval_frame_level_from_cache(
        RAVDESS_ROOT, video_ids, SEQ_LEN, STRIDE, device, model, scaler, BATCH_SIZE,
        trained_feature_cols=trained_feature_cols,
        master_feature_cols=master_feature_cols
    )
    frame_acc = float((y_true_f == y_pred_f).mean()) if y_true_f.size else float("nan")
    print(f"\n[TEST][FRAME] acc={frame_acc:.4f}")
    print_report_and_cm("FRAME (RAVDESS)", y_true_f, y_pred_f, [IDX_TO_EMO[i] for i in range(6)], plot=PLOT_CM)

if __name__ == "__main__":
    main()

