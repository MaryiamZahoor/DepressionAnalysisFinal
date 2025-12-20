#!/usr/bin/env python3
import os, json, math, gc, random
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# ---- project imports (yours) ----
import config as CFG
from models.TwoLayerMLP import FrameClassifier
from utils.features import save_feature_cols, Standardize
# (we don't need harmonize_vgg_cols / build_au_master because we use existing caches)

# ======================
#      CONSTANTS
# ======================
MAX_EPOCHS   = CFG.EPOCHS
ES_PATIENCE  = 15
ES_MONITOR   = "val_acc"   # "val_acc" or "val_loss"
SKIP_FIRST_N = CFG.SKIP_FRAME
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME       = True
WARM_START_FINAL = False

CPU_COUNT       = os.cpu_count() or 4
NUM_WORKERS     = min(8, max(0, CPU_COUNT - 2))
PIN_MEMORY      = torch.cuda.is_available()
PREFETCH_FACTOR = 4
def _loader_kws():
    base = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if NUM_WORKERS > 0:
        base.update(dict(prefetch_factor=PREFETCH_FACTOR, persistent_workers=True))
    return base

def _safe_collate(batch):
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

def _save_ckpt(path, model, optim, scheduler, epoch, best_metric, best_state):
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict() if optim is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_state": best_state,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
    }, path)

def _load_ckpt(path, model, optim, scheduler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optim is not None and ckpt.get("optim") is not None:
        optim.load_state_dict(ckpt["optim"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "rng" in ckpt and ckpt["rng"] is not None:
        torch.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        np.random.set_state(ckpt["rng"]["numpy"])
        random.setstate(ckpt["rng"]["random"])
    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", -math.inf if ES_MONITOR == "val_acc" else math.inf)
    best_state  = ckpt.get("best_state", None)
    return epoch, best_metric, best_state

# ======================
#   COMBINED DATA PATHS
# ======================
# Combined split TXT files you generated earlier (ids formatted as "crema::<vid>" or "ravdess::Actor_xx/<vid>")
COMB_SPLIT_PATH = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/"
TRAIN_LIST = os.path.join(COMB_SPLIT_PATH, "train_videos_COMBINED.txt")
VAL_LIST   = os.path.join(COMB_SPLIT_PATH, "val_videos_COMBINED.txt")
TEST_LIST  = os.path.join(COMB_SPLIT_PATH, "test_videos_COMBINED.txt")


# Dataset roots
CREMA_ROOT   = "/media/root918/OS/MaryiamProject/CREMA-D/copiedFiles/"
RAVDESS_ROOT = "/media/root918/OS/MaryiamProject/copiedFilesRAVDESS/"

# Where artifacts go
PROJECT_DIR  = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/"
ART_DIR_TAG  = "combined_GridSearch_unscaled_MLP"
ART_DIR_SUB  = os.path.join(PROJECT_DIR, "artifacts", ART_DIR_TAG)
GRID_OUT_DIR = os.path.join(ART_DIR_SUB, "grid_COMBINED")
CONFIGS_DIR  = os.path.join(ART_DIR_SUB, "configs_COMBINED")
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
    # should not appear (you removed these videos), but safe-guard:
    "calm":None,"Calm":None,"CALM":None,
    "surprised":None,"Surprised":None,"SURPRISED":None,
    "surprise":None,"Surprise":None,"SURPRISE":None,
}

def _filter_ids_by_dataset(ids, dataset: str):
    ds = dataset.lower().strip()
    tag = ds + "::"
    return [cid for cid in ids if cid.lower().startswith(tag)]

def _ravdess_label_to_idx(s: str) -> Optional[int]:
    if s is None: return None
    t = str(s).strip()
    t = ALIASES.get(t, ALIASES.get(t.lower(), None))
    if t is None: return None
    return EMOTION_TO_IDX.get(t, None)

# CREMA mapping you provided:
# {'H':0,'S':1,'A':2,'N':3,'D':4,'F':5}
# -> canonical: [H,S,A,N,D,F] → [happy, sad, angry, neutral, disgust, fear] → indices [3,5,0,4,1,2]
CREMA_INT_TO_CANON = np.array([3, 5, 0, 4, 1, 2], dtype=np.int64)

# ======================
#   FEATURE SELECTION GRID (suffix-based)
# ======================
DO_STANDARDIZE = False
KEEP_AU_C_RAW  = True

ARCH_GRID = [
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
#   DATASET (existing caches; no padding)
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
        # bytes -> unicode if needed
        if y.dtype.kind == 'S':
            y = np.char.decode(y, 'utf-8')
    else:
        raise FileNotFoundError(f"Missing y.npy / y_str.npy at {cdir}")

    return X, y

class CachedFrameDatasetUnified(torch.utils.data.Dataset):
    """
    Reads *existing* per-video caches (CREMA: int labels; RAVDESS: string labels),
    maps to 6-class canonical indices, and outputs only the selected feature columns.
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
                # y_raw is int-coded 0..5
                y_local = y_raw.astype(np.int64, copy=False)
                mask = (y_local >= 0) & (y_local < 6)
                keep = np.where(mask & (np.arange(n) >= start))[0]
                y_map = CREMA_INT_TO_CANON[y_local[keep]]
            else:
                # ravdess strings -> canonical ints (drop None)
                # y_raw might already be strings, or sometimes bytes/objects
                if y_raw.dtype.kind == 'S':  # bytes
                    y_raw = np.char.decode(y_raw, 'utf-8')
                mapped = np.array([_ravdess_label_to_idx(v) for v in y_raw], dtype=object)
                keep = np.where((pd.notna(mapped)) & (np.arange(n) >= start))[0]
                if keep.size == 0:
                    continue
                y_map = np.array([int(mapped[i]) for i in keep], dtype=np.int64)

            if keep.size == 0:
                continue

            self.chunks.append((X, keep, y_map))
            total += keep.size

        # precompute ranges
        self.ranges = []
        acc = 0
        for vi, (_, keep, _) in enumerate(self.chunks):
            n = keep.size
            self.ranges.append((vi, acc, n))
            acc += n

        #print(f"[dataset] videos used: {len(self.chunks)} | frames: {total:,} | dim: {self.input_dim}")
        self.feature_cols = self.sel_names  # for scaler naming

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
                # slice only selected feature columns here (no padding)
                x = torch.from_numpy(np.asarray(X[k, self.sel_idx], dtype=np.float32, order="C"))
                y = torch.tensor(int(y_map[i - start]), dtype=torch.long)
                return x, y
        raise IndexError(i)

# ======================
#   UTILITIES
# ======================
@torch.no_grad()
def _compute_mean_std_per_feature(loader: DataLoader, device, feature_names: List[str]):
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

def _config_tag(cfg_run: dict) -> str:
    parts, fb = [], []
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
#   DATA LOADING
# ======================
def get_ids_train_val_test() -> Tuple[List[str], List[str], List[str]]:
    tr_ids = _read_ids(_require_file(TRAIN_LIST, "TRAIN_LIST"))
    va_ids = _read_ids(_require_file(VAL_LIST,   "VAL_LIST"))
    te_ids = _read_ids(_require_file(TEST_LIST,  "TEST_LIST"))
    return tr_ids, va_ids, te_ids

# ======================
#   TEST EVALUATION HELPERS (dataset-based)
# ======================
@torch.no_grad()
def evaluate_on_test(cfg_run, state_dict, selected_idx, selected_names, scaler_mean, scaler_std):
    vids = _read_ids(_require_file(TEST_LIST, "TEST_LIST"))
    ds_te = CachedFrameDatasetUnified(vids, selected_idx, selected_names, SKIP_FIRST_N)
    te_loader = DataLoader(ds_te, batch_size=16384, shuffle=False,
                           drop_last=False, collate_fn=_safe_collate, **_loader_kws())
    scaler = Standardize(scaler_mean, scaler_std).to(DEVICE) if DO_STANDARDIZE else nn.Identity()

    model = FrameClassifier(
        input_dim=len(selected_idx),
        hidden_dim=cfg_run["hidden_dim"],
        hidden_dim2=cfg_run["hidden_dim2"],
        dropout=cfg_run["dropout"],
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    y_true, y_pred = [], []
    for xb, yb in te_loader:
        xb = scaler(xb.to(DEVICE, non_blocking=True).float())
        yb = yb.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            logits = model(xb)
            y_pred.append(logits.argmax(1).cpu().numpy())
            y_true.append(yb.cpu().numpy())
    acc_frame = float("nan")
    if y_true:
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        acc_frame = float((y_true == y_pred).mean())

    # video-level via mean-softmax per video
    y_true_v, y_pred_v = [], []
    for cid in vids:
        ds_one = CachedFrameDatasetUnified([cid], selected_idx, selected_names, SKIP_FIRST_N)
        if len(ds_one) == 0:
            continue
        loader_one = DataLoader(ds_one, batch_size=16384, shuffle=False,
                                drop_last=False, collate_fn=_safe_collate, **_loader_kws())
        probs_sum = torch.zeros(len(EMOTION_TO_IDX), dtype=torch.float32, device=DEVICE)
        y_major = []
        for xb, yb in loader_one:
            xb = scaler(xb.to(DEVICE).float())
            with torch.no_grad():
                logits = model(xb)
                probs_sum += torch.softmax(logits, dim=1).sum(dim=0)
            y_major.append(yb.numpy())
        y_pred_v.append(int(probs_sum.argmax().item()))
        y_major = np.concatenate(y_major)
        y_true_v.append(int(np.bincount(y_major).argmax()))
    acc_video = float("nan")
    if y_true_v:
        y_true_v = np.array(y_true_v); y_pred_v = np.array(y_pred_v)
        acc_video = float((y_true_v == y_pred_v).mean())

    return {"frame_acc": acc_frame, "video_acc": acc_video}

#@torch.no_grad()
#def evaluate_on_test_detailed(cfg_run, state_dict, selected_idx, selected_names, scaler_mean, scaler_std):
 #   vids = _read_ids(_require_file(TEST_LIST, "TEST_LIST"))
 
@torch.no_grad()
def evaluate_on_test_detailed(cfg_run, state_dict, selected_idx, selected_names, scaler_mean, scaler_std,
                              video_ids=None):
                              
    vids = video_ids if video_ids is not None else _read_ids(_require_file(TEST_LIST, "TEST_LIST"))
    
    scaler = Standardize(scaler_mean, scaler_std).to(DEVICE) if DO_STANDARDIZE else nn.Identity()

    model = FrameClassifier(
        input_dim=len(selected_idx),
        hidden_dim=cfg_run["hidden_dim"],
        hidden_dim2=cfg_run["hidden_dim2"],
        dropout=cfg_run["dropout"],
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # frame-level
    y_true_f, y_pred_f = [], []
    for cid in vids:
        ds_one = CachedFrameDatasetUnified([cid], selected_idx, selected_names, SKIP_FIRST_N)
        if len(ds_one) == 0:
            continue
        loader = DataLoader(ds_one, batch_size=16384, shuffle=False,
                            drop_last=False, collate_fn=_safe_collate, **_loader_kws())
        for xb, yb in loader:
            xb = scaler(xb.to(DEVICE).float())
            with torch.no_grad():
                logits = model(xb)
            y_pred_f.append(logits.argmax(1).cpu().numpy())
            y_true_f.append(yb.numpy())
    if y_true_f:
        y_true_f = np.concatenate(y_true_f); y_pred_f = np.concatenate(y_pred_f)
    else:
        y_true_f = np.array([]); y_pred_f = np.array([])

    # video-level (mean-softmax)
    y_true_v, y_pred_v = [], []
    for cid in vids:
        ds_one = CachedFrameDatasetUnified([cid], selected_idx, selected_names, SKIP_FIRST_N)
        if len(ds_one) == 0:
            continue
        loader = DataLoader(ds_one, batch_size=16384, shuffle=False,
                            drop_last=False, collate_fn=_safe_collate, **_loader_kws())
        probs_sum = torch.zeros(len(EMOTION_TO_IDX), dtype=torch.float32, device=DEVICE)
        maj = []
        for xb, yb in loader:
            xb = scaler(xb.to(DEVICE).float())
            with torch.no_grad():
                logits = model(xb)
            probs_sum += torch.softmax(logits, dim=1).sum(dim=0)
            maj.append(yb.numpy())
        y_pred_v.append(int(probs_sum.argmax().item()))
        maj = np.concatenate(maj)
        y_true_v.append(int(np.bincount(maj).argmax()))
    y_true_v = np.array(y_true_v); y_pred_v = np.array(y_pred_v)

    target_names = [IDX_TO_EMO[i] for i in range(len(EMOTION_TO_IDX))]

    acc_frame = float((y_true_f == y_pred_f).mean()) if y_true_f.size else float("nan")
    acc_video = float((y_true_v == y_pred_v).mean()) if y_true_v.size else float("nan")

    cm_frame = confusion_matrix(y_true_f, y_pred_f, labels=list(range(len(EMOTION_TO_IDX)))) if y_true_f.size else None
    cr_frame = classification_report(y_true_f, y_pred_f, labels=list(range(len(EMOTION_TO_IDX))), target_names=target_names, digits=4) if y_true_f.size else "N/A"

    cm_video = confusion_matrix(y_true_v, y_pred_v, labels=list(range(len(EMOTION_TO_IDX)))) if y_true_v.size else None
    cr_video = classification_report(y_true_v, y_pred_v, labels=list(range(len(EMOTION_TO_IDX))), target_names=target_names, digits=4) if y_true_v.size else "N/A"

    return {
        "frame_acc": acc_frame, "video_acc": acc_video,
        "y_true_frame": y_true_f, "y_pred_frame": y_pred_f,
        "y_true_video": y_true_v, "y_pred_video": y_pred_v,
        "cm_frame": cm_frame, "cm_video": cm_video,
        "cr_frame": cr_frame, "cr_video": cr_video,
        "target_names": target_names,
    }

# ======================
#   TRAIN (ES on VAL) THEN TEST — PER CONFIG
# ======================
def train_val_es_then_test(cfg_run, train_ids, val_ids, common_cols):
    # feature selection
    sel_names = _selected_names_for_cfg(common_cols,
                                        cfg_run["use_vgg"], cfg_run["use_resnet"],
                                        cfg_run["use_au_c"], cfg_run["use_au_r"])
    sel_idx = _indices_from_names(common_cols, sel_names)

    # datasets
    ds_tr = CachedFrameDatasetUnified(train_ids, sel_idx, sel_names, SKIP_FIRST_N)
    ds_va = CachedFrameDatasetUnified(val_ids,   sel_idx, sel_names, SKIP_FIRST_N)

    print(f"[sizes] train frames: {len(ds_tr):,} | val frames: {len(ds_va):,}")
    assert len(ds_tr) > 0 and len(ds_va) > 0, "Empty TRAIN/VAL—check combined splits & caches."

    tr_loader = DataLoader(ds_tr, batch_size=cfg_run["batch_size"], shuffle=True,
                           drop_last=False, collate_fn=_safe_collate, **_loader_kws())
    va_loader = DataLoader(ds_va, batch_size=cfg_run["batch_size"], shuffle=False,
                           drop_last=False, collate_fn=_safe_collate, **_loader_kws())

    # scaler on TRAIN only (for selected features)
    if DO_STANDARDIZE:
        tmp_loader = DataLoader(ds_tr, batch_size=4096, shuffle=False,
                                collate_fn=_safe_collate, **_loader_kws())
        mean, std = _compute_mean_std_per_feature(tmp_loader, DEVICE, ds_tr.feature_cols)
        if KEEP_AU_C_RAW:
            auc_idx = [i for i, n in enumerate(ds_tr.feature_cols) if n.endswith("_c")]
            if auc_idx:
                idx = torch.tensor(auc_idx, dtype=torch.long)
                std[idx] = 1.0; mean[idx] = 0.0
        scaler = Standardize(mean, std).to(DEVICE)
    else:
        mean = torch.zeros(ds_tr.input_dim)
        std  = torch.ones(ds_tr.input_dim)
        scaler = nn.Identity()

    model = FrameClassifier(
        input_dim=ds_tr.input_dim,
        hidden_dim=cfg_run["hidden_dim"],
        hidden_dim2=cfg_run["hidden_dim2"],
        dropout=cfg_run["dropout"],
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)
    optim = _build_optimizer(cfg_run["optimizer"], model.parameters(),
                             lr=cfg_run["lr"], weight_decay=cfg_run["weight_decay"])
    ce = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optim, mode=("max" if ES_MONITOR=="val_acc" else "min"),
                                  factor=0.1, patience=5, min_lr=1e-6)

    cfg_dir   = os.path.join(CONFIGS_DIR, _config_tag(cfg_run))
    os.makedirs(cfg_dir, exist_ok=True)
    ckpt_last = os.path.join(cfg_dir, "ckpt_last.pt")
    best_path = os.path.join(cfg_dir, "best.pt")
    done_flag = os.path.join(cfg_dir, "done.flag")

    if os.path.isfile(done_flag):
        print(f"[grid][{_config_tag(cfg_run)}] Found {done_flag}; skipping training.")
        m_json = os.path.join(cfg_dir, "metrics.json")
        if os.path.isfile(m_json):
            with open(m_json) as f:
                return json.load(f)

    start_epoch = 1
    best_metric = -math.inf if ES_MONITOR == "val_acc" else math.inf
    best_state  = None
    no_improve  = 0
    best_val_loss, best_val_acc = float("inf"), 0.0
    best_train_acc = 0.0

    if RESUME and os.path.isfile(ckpt_last):
        print(f"[grid][{_config_tag(cfg_run)}] Resuming from {ckpt_last}")
        start_epoch, best_metric, best_state = _load_ckpt(ckpt_last, model, optim, scheduler)
        if best_state is not None:
            torch.save(best_state, best_path)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train(); run_loss = 0.0; correct=0; total=0
        for xb, yb in tr_loader:
            xb = scaler(xb.to(DEVICE, non_blocking=True).float())
            yb = yb.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward(); optim.step()
            run_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            total   += yb.numel()
        train_loss = run_loss/max(1,len(tr_loader))
        train_acc  = correct/max(1,total)
        best_train_acc = max(best_train_acc, float(train_acc))

        val_loss, val_acc = _eval_loader(model, va_loader, DEVICE, scaler)
        sched_value = val_acc if ES_MONITOR=="val_acc" else val_loss
        scheduler.step(sched_value)

        monitor = val_acc if ES_MONITOR == "val_acc" else val_loss
        improved = (monitor > best_metric) if ES_MONITOR == "val_acc" else (monitor < best_metric)
        if improved:
            best_metric = monitor; no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val_loss, best_val_acc = float(val_loss), float(val_acc)
            torch.save(best_state, best_path)
        else:
            no_improve += 1

        _save_ckpt(ckpt_last, model, optim, scheduler, epoch, best_metric, best_state)
        print(f"[grid][{_config_tag(cfg_run)}] ep {epoch:03d} | tr_loss {train_loss:.4f} | tr_acc {train_acc:.4f} "
              f"| va_loss {val_loss:.4f} | va_acc {val_acc:.4f} | best_{ES_MONITOR} {best_metric:.4f} "
              f"| lr {_current_lr(optim):.2e}")
        if no_improve >= ES_PATIENCE:
            print(f"[grid][{_config_tag(cfg_run)}] Early stopping after {epoch} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, best_path)

    if DO_STANDARDIZE:
        torch.save({"mean": mean, "std": std}, os.path.join(cfg_dir, "scaler.pt"))
    save_feature_cols(ds_tr.feature_cols, os.path.join(cfg_dir, "feature_cols.json"))
    torch.save(model.state_dict(), os.path.join(cfg_dir, "best.pt"))

    test_metrics = evaluate_on_test(
        cfg_run, model.state_dict(),
        selected_idx=sel_idx, selected_names=sel_names,
        scaler_mean=mean, scaler_std=std
    )

    open(done_flag, "a").close()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    return {
        "best_train_acc": float(best_train_acc),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_frame_acc": float(test_metrics.get("frame_acc", float("nan"))),
        "test_video_acc": float(test_metrics.get("video_acc", float("nan")))
    }

# ======================
#   FINAL TRAIN ON TRAIN+VAL
# ======================
def _save_final_artifacts(final_dir, model_state, mean, std, feature_cols, ckpt_extra=None):
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model_state, os.path.join(final_dir, "best.pt"))
    torch.save({"mean": mean, "std": std}, os.path.join(final_dir, "scaler.pt"))
    save_feature_cols(feature_cols, os.path.join(final_dir, "feature_cols.json"))
    if ckpt_extra is not None:
        with open(os.path.join(final_dir, "meta.json"), "w") as f:
            json.dump(ckpt_extra, f, indent=2)

def train_on_ids(cfg_run, ids, common_cols, final_dir, warm_start_state=None):
    sel_names = _selected_names_for_cfg(common_cols,
                                        cfg_run["use_vgg"], cfg_run["use_resnet"],
                                        cfg_run["use_au_c"], cfg_run["use_au_r"])
    sel_idx = _indices_from_names(common_cols, sel_names)

    ds = CachedFrameDatasetUnified(ids, sel_idx, sel_names, SKIP_FIRST_N)
    loader = DataLoader(ds, batch_size=cfg_run["batch_size"], shuffle=True,
                        drop_last=False, collate_fn=_safe_collate, **_loader_kws())

    if DO_STANDARDIZE:
        tmp_loader = DataLoader(ds, batch_size=4096, shuffle=False,
                                collate_fn=_safe_collate, **_loader_kws())
        mean, std = _compute_mean_std_per_feature(tmp_loader, DEVICE, ds.feature_cols)
        if KEEP_AU_C_RAW:
            auc_idx = [i for i, n in enumerate(ds.feature_cols) if n.endswith("_c")]
            if auc_idx:
                idx = torch.tensor(auc_idx, dtype=torch.long)
                std[idx] = 1.0; mean[idx] = 0.0
        scaler = Standardize(mean, std).to(DEVICE)
    else:
        mean = torch.zeros(ds.input_dim); std  = torch.ones(ds.input_dim); scaler = nn.Identity()

    model = FrameClassifier(
        input_dim=ds.input_dim,
        hidden_dim=cfg_run["hidden_dim"],
        hidden_dim2=cfg_run["hidden_dim2"],
        dropout=cfg_run["dropout"],
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)

    optim = _build_optimizer(cfg_run["optimizer"], model.parameters(),
                             lr=cfg_run["lr"], weight_decay=cfg_run["weight_decay"])
    ce = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optim, mode="min", factor=0.1, patience=5, min_lr=1e-6)

    os.makedirs(final_dir, exist_ok=True)
    ckpt_last = os.path.join(final_dir, "ckpt_last.pt")
    best_path = os.path.join(final_dir, "best.pt")
    scaler_path = os.path.join(final_dir, "scaler.pt")

    if WARM_START_FINAL and warm_start_state is not None:
        try:
            model.load_state_dict(warm_start_state, strict=False)
            print("[train+val] Warm-started model from provided state.")
        except Exception as e:
            print(f"[train+val] Warm-start failed: {e}")

    start_epoch = 1
    best_loss = float("inf")
    no_improve = 0
    best_state = None

    if os.path.isfile(ckpt_last):
        print(f"[train+val] Resuming from {ckpt_last}")
        ckpt = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        try: model.load_state_dict(ckpt["model"])
        except Exception as e: print(f"[train+val] Model state load failed on resume: {e}")
        if ckpt.get("optim") is not None:
            try: optim.load_state_dict(ckpt["optim"])
            except Exception as e: print(f"[train+val] Optim resume warn: {e}")
        if ckpt.get("scheduler") is not None:
            try: scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e: print(f"[train+val] Sched resume warn: {e}")
        if ckpt.get("best_state") is not None:
            best_state = ckpt["best_state"]
        best_loss = ckpt.get("best_metric", best_loss)
        start_epoch = ckpt.get("epoch", 0) + 1
        if os.path.isfile(scaler_path):
            try:
                sc = torch.load(scaler_path, map_location="cpu")
                mean, std = sc.get("mean", mean), sc.get("std", std)
                scaler = Standardize(mean, std).to(DEVICE) if DO_STANDARDIZE else nn.Identity()
                print("[train+val] Restored scaler from scaler.pt")
            except Exception as e:
                print(f"[train+val] Scaler restore failed: {e}")

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train(); run_loss = 0.0
        for xb, yb in loader:
            xb = scaler(xb.to(DEVICE, non_blocking=True).float())
            yb = yb.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward(); optim.step()
            run_loss += loss.item()

        train_loss = run_loss / max(1, len(loader))
        scheduler.step(train_loss)
        print(f"[train+val] ep {epoch:03d} | loss {train_loss:.4f} | lr {_current_lr(optim):.2e}")

        improved = train_loss < best_loss - 1e-6
        if improved:
            best_loss = train_loss; no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_final_artifacts(final_dir, best_state, mean, std, ds.feature_cols,
                                  ckpt_extra={"epoch": epoch, "best_train_loss": best_loss})
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                print(f"[train+val] Early stopping after {epoch} epochs (no improvement).")
                break

        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_loss,
            "best_state": best_state,
        }, ckpt_last)

    if best_state is not None:
        model.load_state_dict(best_state)
    
    # use a safe epoch when the loop didn’t run (resume already finished, etc.)
    epoch_safe = locals().get("epoch", start_epoch - 1)
    _save_final_artifacts(final_dir, model.state_dict(), mean, std, ds.feature_cols,
                      ckpt_extra={"epoch": epoch_safe, "best_train_loss": best_loss})

    #_save_final_artifacts(final_dir, model.state_dict(), mean, std, ds.feature_cols,
     #                     ckpt_extra={"epoch": epoch, "best_train_loss": best_loss})
                          
    return model.state_dict(), mean, std, ds.feature_cols

def retrain_on_trainval_and_report(best_cfg_dict, common_cols, train_ids, val_ids, warm_start_state=None):
    trainval_ids = sorted(set(train_ids) | set(val_ids))
    cfg_use = dict(best_cfg_dict)

    print("\n" + "="*100)
    print("[FINAL] Retraining best config on TRAIN+VAL (combined) …")
    print("="*100)

    final_dir = os.path.join(ART_DIR_SUB, "final_trainval_best_by_frameacc")
    final_state, mean, std, feat_cols = train_on_ids(
        cfg_use, trainval_ids, common_cols, final_dir,
        warm_start_state=(warm_start_state if WARM_START_FINAL else None)
    )
    # Build RAVDESS-only slice from the existing TEST list
    all_test_ids = _read_ids(_require_file(TEST_LIST, "TEST_LIST"))
    #only_ids = _filter_ids_by_dataset(all_test_ids, "ravdess")

    sel_idx = _indices_from_names(common_cols, feat_cols)
    detailed = evaluate_on_test_detailed(cfg_use, final_state, sel_idx, feat_cols, mean, std, video_ids=all_test_ids)

    #sel_idx = _indices_from_names(common_cols, feat_cols)
    #detailed = evaluate_on_test_detailed(cfg_use, final_state, sel_idx, feat_cols, mean, std)

    print("\n[FINAL] Test Accuracy")
    print(f"  Frame-level acc: {detailed['frame_acc']:.4f}")
    print(f"  Video-level acc: {detailed['video_acc']:.4f}")

    names = detailed["target_names"]
    if detailed["cm_frame"] is not None:
        print("\n[FINAL] Frame-level Confusion Matrix (rows=true, cols=pred):")
        print(pd.DataFrame(detailed["cm_frame"], index=names, columns=names).to_string())
    if detailed["cm_video"] is not None:
        print("\n[FINAL] Video-level Confusion Matrix (rows=true, cols=pred):")
        print(pd.DataFrame(detailed["cm_video"], index=names, columns=names).to_string())

    with open(os.path.join(final_dir, "final_reports.txt"), "w") as f:
        f.write(f"Frame acc: {detailed['frame_acc']:.6f}\n")
        f.write(f"Video acc: {detailed['video_acc']:.6f}\n\n")
        f.write("[Frame CM]\n")
        if detailed["cm_frame"] is not None:
            f.write(pd.DataFrame(detailed["cm_frame"], index=names, columns=names).to_string())
        f.write("\n\n[Frame Report]\n")
        f.write(str(detailed["cr_frame"]))
        f.write("\n\n[Video CM]\n")
        if detailed["cm_video"] is not None:
            f.write(pd.DataFrame(detailed["cm_video"], index=names, columns=names).to_string())
        f.write("\n\n[Video Report]\n")
        f.write(str(detailed["cr_video"]))
    print(f"\n[final] Wrote detailed reports -> {os.path.join(final_dir, 'final_reports.txt')}")
    return detailed

# ======================
#   MAIN
# ======================
def main():
    _set_seed(1337)

    # 1) ids (combined)
    train_ids, val_ids, test_ids = get_ids_train_val_test()
    dev_ids = sorted(set(train_ids) | set(val_ids))
    print(f"[splits] train={len(train_ids)} | val={len(val_ids)} | test={len(test_ids)} | dev(unique)={len(dev_ids)}")

    # 2) assert consistent schema across dev and get common feature names
    common_cols = assert_consistent_feature_schema(dev_ids)

    # 3) grid
    grid = list(_config_iter())
    print(f"[grid] total configurations: {len(grid)}")

    rows = []
    for i, cfg_run in enumerate(grid, 1):
        tag = (f"{cfg_run['feature_set']} | H1={cfg_run['hidden_dim']} "
               f"H2={cfg_run['hidden_dim2']} | opt={cfg_run['optimizer']} "
               f"lr={cfg_run['lr']} wd={cfg_run['weight_decay']} "
               f"do={cfg_run['dropout']} bs={cfg_run['batch_size']}")
        print("\n" + "="*100)
        print(f"[run {i}/{len(grid)}] {tag}")
        print("="*100)

        cfg_dir   = os.path.join(CONFIGS_DIR, _config_tag(cfg_run))
        done_flag = os.path.join(cfg_dir, "done.flag")
        metrics_p = os.path.join(cfg_dir, "metrics.json")

        if os.path.isfile(done_flag) and os.path.isfile(metrics_p):
            print(f"[run {i}/{len(grid)}] Already finished; importing metrics.json")
            try:
                with open(metrics_p) as f:
                    out = json.load(f)
                rows.append(out if isinstance(out, dict) else {"config": tag, **cfg_run, **out})
                continue
            except Exception as e:
                print(f"[warn] Failed to read metrics.json, re-running test later. ({e})")

        m = train_val_es_then_test(cfg_run, train_ids, val_ids, common_cols)
        os.makedirs(cfg_dir, exist_ok=True)
        out_dict = {"config": tag, **cfg_run, **m}
        with open(metrics_p, "w") as f:
            json.dump(out_dict, f, indent=2)
        rows.append(out_dict)

    # 4) CSV summary
    df = pd.DataFrame(rows).sort_values(
        ["best_val_acc","test_video_acc","test_frame_acc"], ascending=[False, False, False]
    )
    out_csv = os.path.join(GRID_OUT_DIR, "grid_results_val_train_test.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[done] Wrote grid summary -> {out_csv}")
    print(df.to_string(index=False))

    # 5) pick best by frame acc then final retrain on train+val
    if df.empty or "test_frame_acc" not in df.columns:
        print("[FINAL] No valid best config found. Skipping final retrain.")
        return

    best_idx = df["test_frame_acc"].astype(float).idxmax()
    best_row = df.loc[best_idx]
    get = (lambda k: best_row[k])
    cfg_use = {
        "use_vgg":      bool(get("use_vgg")),
        "use_resnet":   bool(get("use_resnet")),
        "use_au_c":     bool(get("use_au_c")),
        "use_au_r":     bool(get("use_au_r")),
        "hidden_dim":   int(round(float(get("hidden_dim")))),
        "hidden_dim2":  (
            None if (
                get("hidden_dim2") is None
                or (isinstance(get("hidden_dim2"), float) and math.isnan(get("hidden_dim2")))
                or (isinstance(get("hidden_dim2"), (np.floating,)) and np.isnan(get("hidden_dim2")))
                or (isinstance(get("hidden_dim2"), str) and get("hidden_dim2").strip().lower() in ("", "none", "nan"))
            ) else int(round(float(get("hidden_dim2"))))
        ),
        "optimizer":    str(get("optimizer")),
        "lr":           float(get("lr")),
        "weight_decay": float(get("weight_decay")),
        "dropout":      float(get("dropout")),
        "batch_size":   int(round(float(get("batch_size")))),
    }

    warm_state = None  # (enable if you want to warm start)
    retrain_on_trainval_and_report(cfg_use, common_cols, train_ids, val_ids, warm_start_state=warm_state)

if __name__ == "__main__":
    main()

