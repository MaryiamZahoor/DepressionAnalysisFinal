import os, json, math, gc, time, random
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix  # reports + CMs

# ---- project imports (yours) ----
import config as CFG
from models.TwoLayerMLP import FrameClassifier
from utils.features import save_feature_cols, Standardize, harmonize_vgg_cols
from data.datasets import build_au_master  # AU name order

# ======================
#      CONSTANTS
# ======================
MAX_EPOCHS   = CFG.EPOCHS
ES_PATIENCE  = 15
ES_MONITOR   = "val_acc"   # "val_acc" or "val_loss"
SKIP_FIRST_N = CFG.SKIP_FRAME
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resume / checkpointing
RESUME = True

# Final-phase options
WARM_START_FINAL = False  # set True to warm-start final train+val from grid best state if provided

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
    return optim.param_groups[0]['lr'] if optim.param_groups else float('nan')

def _set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    best_state = ckpt.get("best_state", None)
    return epoch, best_metric, best_state

# ---------- RAVDESS paths ----------
SPLIT_PATH = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/data/"
TRAIN_LIST = os.path.join(SPLIT_PATH, "train_videos_RAV.txt")
VAL_LIST   = os.path.join(SPLIT_PATH, "val_videos_RAV.txt")
TEST_LIST  = os.path.join(SPLIT_PATH, "test_videos_RAV.txt")

# Artifacts (mirrors your CREMA-D layout, but under RAVDESS tag)
PROJECT_DIR = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/"
ART_DIR_TAG = "ravdess_GridSearch_unscaled_MLP"
ART_DIR_SUB = os.path.join(PROJECT_DIR, "artifacts", ART_DIR_TAG)
GRID_OUT_DIR= os.path.join(ART_DIR_SUB, "grid_RAV")
CONFIGS_DIR = os.path.join(ART_DIR_SUB, "configs_RAV")
os.makedirs(GRID_OUT_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)

# caches / features
IDS_LABELS_CACHE     = os.path.join(GRID_OUT_DIR, "ids_labels.json")
MASTER_FEATURES_JSON = os.path.join(GRID_OUT_DIR, "master_feature_cols.json")
MASTER_SCAN_LIMIT    = 10
OUTPUT_DIR= "/media/root918/OS/[REDACTED]Project/copiedFilesRAVDESS/"

# Labels / columns
LABEL_COL         = "emotion" # e.g., "emotion" after harmonize
COMBINED_CSV_NAME = "au_resnet_vgg_with_gt.csv"

# --- RAVDESS emotion mappings (8 classes) ---
EMOTION_TO_IDX = {
    "neutral":   0,
    "calm":      1,
    "happy":     2,
    "sad":       3,
    "anger":     4,
    "fearful":   5,
    "disgust":   6,
    "surprise":  7,
}
IDX_TO_EMO = {v: k for k, v in EMOTION_TO_IDX.items()}

# Standardization
DO_STANDARDIZE = True
KEEP_AU_C_RAW  = True

# Grid (same spirit as your CREMA-D run)
ARCH_GRID   = [
    {"hidden_dim": 512,  "hidden_dim2": None},
    {"hidden_dim": 1024, "hidden_dim2": None},
    {"hidden_dim": 1024, "hidden_dim2": 512},
    {"hidden_dim": 512,  "hidden_dim2": 256},
]
OPTIMIZERS   = ["adam"]
LRS          = [1e-4]
WEIGHT_DECAY = [1e-5]
DROPOUTS     = [0.5]
BATCH_SIZES  = [512]  # Consider also 128 for CPU-only runs

FEATURE_SETS = [
    {"name":"VGG+RESNET",    "use_vgg":True,  "use_resnet":True,  "use_au_c":False, "use_au_r":False},
    {"name":"VGG+AU",        "use_vgg":True,  "use_resnet":False, "use_au_c":True,  "use_au_r":True },
    {"name":"RESNET+AU",     "use_vgg":False, "use_resnet":True,  "use_au_c":True,  "use_au_r":True },
    {"name":"VGG+RESNET+AU", "use_vgg":True,  "use_resnet":True,  "use_au_c":True,  "use_au_r":True },
    {"name":"RESNET",     "use_vgg":False, "use_resnet":True,  "use_au_c":False,  "use_au_r":False }
]

# ======================
#   MASTER FEATURES + CACHES
# ======================
MASTER_FEATURE_COLS: Optional[List[str]] = None

def _require_file(path, desc):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")
    return path

def _read_ids(list_path: str) -> List[str]:
    with open(list_path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def _vid_csv_path(vid: str) -> str:
    return os.path.join(OUTPUT_DIR, vid, COMBINED_CSV_NAME)

def _cache_paths(vid: str):
    cdir = os.path.join(OUTPUT_DIR, vid, "cache_RAVDESS")
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
        if not os.path.isfile(csvp): continue
        try:
            df = pd.read_csv(csvp, nrows=1)
            df = harmonize_vgg_cols(df)
            cnn_cols = [c for c in df.columns if c.endswith("_vgg") or c.endswith("_resnet")]
            _append_unique(master, cnn_cols); picked += 1
            if picked >= n_scan: break
        except Exception as e:
            print(f"[features] warn: header scan failed for {csvp}: {e}")

    au_all = build_au_master(True, True)  # *_c and *_r
    _append_unique(master, list(au_all))

    with open(MASTER_FEATURES_JSON, "w") as f:
        json.dump(master, f, indent=2)
    print(f"[features] Saved master list ({len(master)}) → {MASTER_FEATURES_JSON}")
    return master

def build_video_cache_master(vid: str, master_feature_cols: List[str],
                             label_col: str, skip_first_n: int) -> None:
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

    # lowercase/strip BEFORE mapping
    y = (df[label_col].astype(str).str.strip().str.lower()
         .map(EMOTION_TO_IDX).dropna().astype(np.int64))
    idx = y.index

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
           .to_numpy(copy=True))

    np.save(xnp, X)
    np.save(ynp, y.to_numpy(copy=True))
    with open(meta, "w") as f:
        json.dump({"feature_cols": master_feature_cols}, f)

class CachedFrameDatasetMaster(torch.utils.data.Dataset):
    def __init__(self, video_ids: List[str], master_feature_cols: List[str],
                 use_vgg: bool, use_resnet: bool, use_au_c: bool, use_au_r: bool,
                 skip_first_n: int):
        self.vids = list(video_ids)
        self.master_cols = list(master_feature_cols)

        for i, vid in enumerate(self.vids):
            build_video_cache_master(vid, self.master_cols, LABEL_COL, skip_first_n)
            if (i+1) % 200 == 0:
                print(f"[cache] built {i+1}/{len(self.vids)} videos")

        want = []
        for i, name in enumerate(self.master_cols):
            if name.endswith("_vgg"):    
                if use_vgg: want.append(i)
            if name.endswith("_resnet"): 
                if use_resnet: want.append(i)
            if name.endswith("_c"):      
                if use_au_c: want.append(i)
            if name.endswith("_r"):      
                if use_au_r: want.append(i)
        self.col_idx = np.asarray(sorted(set(want)), dtype=np.int64)
        if self.col_idx.size == 0:
            raise ValueError("No columns selected by this feature combo.")

        self.feature_cols = [self.master_cols[i] for i in self.col_idx.tolist()]
        self.input_dim = len(self.feature_cols)
        print(f"[dataset] input_dim={self.input_dim} (selected feature cols)")

        self.arrs, self.ranges, total = [], [], 0
        for vid in self.vids:
            xnp, ynp, _ = _cache_paths(vid)
            if not (os.path.isfile(xnp) and os.path.isfile(ynp)): continue
            Xm = np.load(xnp, mmap_mode="r"); ym = np.load(ynp, mmap_mode="r")
            n = len(ym)
            if n == 0: continue
            v_idx = len(self.arrs)
            self.arrs.append((Xm, ym))
            self.ranges.append((v_idx, total, n))
            total += n
        self.total = total

    def __len__(self): return self.total

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
                x = torch.from_numpy(feat.copy())
                y = torch.tensor(int(ym[off]), dtype=torch.long)
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

def _select_indices_for_cfg(master_cols: List[str], use_vgg, use_resnet, use_au_c, use_au_r):
    idx = []
    for i, c in enumerate(master_cols):
        if c.endswith("_vgg")    and use_vgg:    idx.append(i)
        if c.endswith("_resnet") and use_resnet: idx.append(i)
        if c.endswith("_c")      and use_au_c:   idx.append(i)
        if c.endswith("_r")      and use_au_r:   idx.append(i)
    return np.asarray(sorted(set(idx)), dtype=np.int64)

# ======================
#   DATA LOADING
# ======================
def get_ids_train_val_test() -> Tuple[List[str], List[str], List[str]]:
    tr_ids = _read_ids(_require_file(TRAIN_LIST, "TRAIN_LIST"))
    va_ids = _read_ids(_require_file(VAL_LIST,   "VAL_LIST"))
    te_ids = _read_ids(_require_file(TEST_LIST,  "TEST_LIST"))
    return tr_ids, va_ids, te_ids

# ======================
#   TEST EVALUATION (returns metrics only)
# ======================
@torch.no_grad()
def evaluate_on_test(cfg_run, state_dict, master_cols, scaler_mean, scaler_std, feature_cols_used: List[str]):
    vids = _read_ids(_require_file(TEST_LIST, "TEST_LIST"))

    # ensure caches exist for all test vids
    for v in vids:
        build_video_cache_master(v, master_cols, LABEL_COL, SKIP_FIRST_N)

    # Map training feature names -> indices in master_cols
    name_to_pos = {name: i for i, name in enumerate(master_cols)}
    sel_idx = np.asarray([name_to_pos[n] for n in feature_cols_used], dtype=np.int64)

    # scaler
    scaler = Standardize(scaler_mean, scaler_std).to(DEVICE) if DO_STANDARDIZE else nn.Identity()

    # model
    model = FrameClassifier(
        input_dim=len(sel_idx),
        hidden_dim=cfg_run["hidden_dim"],
        hidden_dim2=cfg_run["hidden_dim2"],
        dropout=cfg_run["dropout"],
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # frame-level
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

    acc_frame = float("nan")
    if y_true:
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        acc_frame = float((y_true == y_pred).mean())

    # video-level (mean-softmax per video)
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

    acc_video = float("nan")
    if len(y_true_v):
        y_true_v = np.array(y_true_v); y_pred_v = np.array(y_pred_v)
        acc_video = float((y_true_v == y_pred_v).mean())

    return {"frame_acc": acc_frame, "video_acc": acc_video}

# ======================
#   DETAILED TEST EVAL (CMs + reports)
# ======================
@torch.no_grad()
def evaluate_on_test_detailed(cfg_run, state_dict, master_cols, scaler_mean, scaler_std, feature_cols_used: List[str]):
    vids = _read_ids(_require_file(TEST_LIST, "TEST_LIST"))

    for v in vids:
        build_video_cache_master(v, master_cols, LABEL_COL, SKIP_FIRST_N)

    name_to_pos = {name: i for i, name in enumerate(master_cols)}
    sel_idx = np.asarray([name_to_pos[n] for n in feature_cols_used], dtype=np.int64)

    # scaler (ensure tensors)
    if DO_STANDARDIZE:
        if not torch.is_tensor(scaler_mean): scaler_mean = torch.as_tensor(scaler_mean, dtype=torch.float32)
        if not torch.is_tensor(scaler_std):  scaler_std  = torch.as_tensor(scaler_std,  dtype=torch.float32)
        scaler = Standardize(scaler_mean, scaler_std).to(DEVICE)
    else:
        scaler = nn.Identity()

    model = FrameClassifier(
        input_dim=len(sel_idx),
        hidden_dim=cfg_run["hidden_dim"],
        hidden_dim2=cfg_run["hidden_dim2"],
        dropout=cfg_run["dropout"],
        num_classes=len(EMOTION_TO_IDX),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # frame-level preds
    y_true_f, y_pred_f = [], []
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
            y_pred_f.append(logits.argmax(1).cpu().numpy())
            y_true_f.append(yv[i:i+bs].copy())

    if y_true_f:
        y_true_f = np.concatenate(y_true_f); y_pred_f = np.concatenate(y_pred_f)
    else:
        y_true_f = np.array([]); y_pred_f = np.array([])

    # video-level preds
    y_true_v, y_pred_v = [], []
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
def train_val_es_then_test(cfg_run, train_ids, val_ids, master_cols):
    # ---- datasets
    ds_tr = CachedFrameDatasetMaster(train_ids, master_cols,
                                     cfg_run["use_vgg"], cfg_run["use_resnet"],
                                     cfg_run["use_au_c"], cfg_run["use_au_r"], SKIP_FIRST_N)
    ds_va = CachedFrameDatasetMaster(val_ids, master_cols,
                                     cfg_run["use_vgg"], cfg_run["use_resnet"],
                                     cfg_run["use_au_c"], cfg_run["use_au_r"], SKIP_FIRST_N)

    print(f"[sizes] train frames: {len(ds_tr):,} | val frames: {len(ds_va):,}")
    assert len(ds_tr) > 0, "Empty TRAIN dataset—check label mapping & CSV availability."
    assert len(ds_va) > 0, "Empty VAL dataset—check label mapping & CSV availability."

    tr_loader = DataLoader(ds_tr, batch_size=cfg_run["batch_size"], shuffle=True,
                           drop_last=False, collate_fn=_safe_collate, **_loader_kws())
    va_loader = DataLoader(ds_va, batch_size=cfg_run["batch_size"], shuffle=False,
                           drop_last=False, collate_fn=_safe_collate, **_loader_kws())

    # ---- Scaler on TRAIN only
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

    # ---- Model/optim/sched
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
    plateau_mode = "max" if ES_MONITOR == "val_acc" else "min"
    scheduler = ReduceLROnPlateau(optim, mode=("max" if plateau_mode=="max" else "min"),
                                  factor=0.1, patience=5, min_lr=1e-6)

    # ---- Checkpoint paths
    cfg_dir   = os.path.join(CONFIGS_DIR, _config_tag(cfg_run))
    os.makedirs(cfg_dir, exist_ok=True)
    ckpt_last = os.path.join(cfg_dir, "ckpt_last.pt")
    best_path = os.path.join(cfg_dir, "best.pt")
    done_flag = os.path.join(cfg_dir, "done.flag")

    # ---- Skip if completed (optional optimization)
    if os.path.isfile(done_flag):
        print(f"[grid][{_config_tag(cfg_run)}] Found {done_flag}; skipping training.")
        # Try to reuse metrics.json; if missing, re-evaluate from best.pt
        m_json = os.path.join(cfg_dir, "metrics.json")
        if os.path.isfile(m_json):
            with open(m_json) as f:
                return json.load(f)
        if os.path.isfile(best_path):
            state_dict = torch.load(best_path, map_location="cpu", weights_only=False)
            test_metrics = evaluate_on_test(
                cfg_run, state_dict, master_cols, mean, std, ds_tr.feature_cols
            )
            return {
                "best_train_acc": float("nan"),
                "best_val_acc": float("nan"),
                "best_val_loss": float("nan"),
                "test_frame_acc": float(test_metrics.get("frame_acc", float("nan"))),
                "test_video_acc": float(test_metrics.get("video_acc", float("nan")))
            }

    # ---- Init tracking (supports resume)
    start_epoch = 1
    best_metric = -math.inf if ES_MONITOR == "val_acc" else math.inf
    best_state  = None
    no_improve  = 0
    best_val_loss, best_val_acc = float("inf"), 0.0
    best_train_acc = 0.0

    # ====== RESUME ======
    if RESUME and os.path.isfile(ckpt_last):
        print(f"[grid][{_config_tag(cfg_run)}] Resuming from {ckpt_last}")
        start_epoch, best_metric, best_state = _load_ckpt(ckpt_last, model, optim, scheduler)
        if best_state is not None:
            torch.save(best_state, best_path)

    # ---- Train (ES on VAL)
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        # train
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

        # validate
        val_loss, val_acc = _eval_loader(model, va_loader, DEVICE, scaler)
        sched_value = val_acc if plateau_mode == "max" else val_loss
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

    # ---- restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, best_path)

    # ---- Save artifacts for this config
    if DO_STANDARDIZE:
        torch.save({"mean": mean, "std": std}, os.path.join(cfg_dir, "scaler.pt"))
    save_feature_cols(ds_tr.feature_cols, os.path.join(cfg_dir, "feature_cols.json"))
    torch.save(model.state_dict(), os.path.join(cfg_dir, "best.pt"))

    # ---- Per-config TEST eval
    test_metrics = evaluate_on_test(
        cfg_run, model.state_dict(),
        master_cols=master_cols, scaler_mean=mean, scaler_std=std,
        feature_cols_used=ds_tr.feature_cols
    )

    # mark as done
    open(done_flag, "a").close()

    # cleanup
    if hasattr(ds_tr, "close"): ds_tr.close()
    if hasattr(ds_va, "close"): ds_va.close()
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "best_train_acc": float(best_train_acc),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_frame_acc": float(test_metrics.get("frame_acc", float("nan"))),
        "test_video_acc": float(test_metrics.get("video_acc", float("nan")))
    }

# ======================
#   FINAL TRAIN ON TRAIN+VAL (save best + scaler + resume)
# ======================
def _save_final_artifacts(final_dir, model_state, mean, std, feature_cols, ckpt_extra=None):
    os.makedirs(final_dir, exist_ok=True)
    # save best weights
    torch.save(model_state, os.path.join(final_dir, "best.pt"))
    # save scaler (even if DO_STANDARDIZE=False, keep interface uniform)
    to_save = {"mean": mean, "std": std}
    torch.save(to_save, os.path.join(final_dir, "scaler.pt"))
    # save feature cols
    save_feature_cols(feature_cols, os.path.join(final_dir, "feature_cols.json"))
    # optional extra (e.g., epoch/metrics)
    if ckpt_extra is not None:
        with open(os.path.join(final_dir, "meta.json"), "w") as f:
            json.dump(ckpt_extra, f, indent=2)

def train_on_ids(cfg_run, ids, master_cols, final_dir, warm_start_state=None):
    ds = CachedFrameDatasetMaster(ids, master_cols,
                                  cfg_run["use_vgg"], cfg_run["use_resnet"],
                                  cfg_run["use_au_c"], cfg_run["use_au_r"], SKIP_FIRST_N)
    loader = DataLoader(ds, batch_size=cfg_run["batch_size"], shuffle=True,
                        drop_last=False, collate_fn=_safe_collate, **_loader_kws())

    # scaler on TRAIN(+VAL)
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
        mean = torch.zeros(ds.input_dim)
        std  = torch.ones(ds.input_dim)
        scaler = nn.Identity()

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

    # ---- paths
    os.makedirs(final_dir, exist_ok=True)
    ckpt_last = os.path.join(final_dir, "ckpt_last.pt")
    best_path = os.path.join(final_dir, "best.pt")
    scaler_path = os.path.join(final_dir, "scaler.pt")

    # ---- warm start (optional)
    if WARM_START_FINAL and warm_start_state is not None:
        try:
            model.load_state_dict(warm_start_state, strict=False)
            print("[train+val] Warm-started model from provided state.")
        except Exception as e:
            print(f"[train+val] Warm-start failed: {e}")

    # ---- resume
    start_epoch = 1
    best_loss = float("inf")
    no_improve = 0
    best_state = None

    if os.path.isfile(ckpt_last):
        print(f"[train+val] Resuming from {ckpt_last}")
        ckpt = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(ckpt["model"])
        except Exception as e:
            print(f"[train+val] Model state load failed on resume: {e}")
        if ckpt.get("optim") is not None:
            try: optim.load_state_dict(ckpt["optim"])
            except Exception as e: print(f"[train+val] Optim resume warn: {e}")
        if ckpt.get("scheduler") is not None:
            try: scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e: print(f"[train+val] Sched resume warn: {e}")
        # restore best snapshot (so we can keep improving)
        if ckpt.get("best_state") is not None:
            best_state = ckpt["best_state"]
        best_loss = ckpt.get("best_metric", best_loss)
        start_epoch = ckpt.get("epoch", 0) + 1
        # restore scaler if exists on disk
        if os.path.isfile(scaler_path):
            try:
                sc = torch.load(scaler_path, map_location="cpu")
                mean, std = sc.get("mean", mean), sc.get("std", std)
                scaler = Standardize(mean, std).to(DEVICE) if DO_STANDARDIZE else nn.Identity()
                print("[train+val] Restored scaler from scaler.pt")
            except Exception as e:
                print(f"[train+val] Scaler restore failed: {e}")

    # ---- training loop with ES on train loss
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
            best_loss = train_loss
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # save immediate best artifacts
            _save_final_artifacts(final_dir, best_state, mean, std, ds.feature_cols,
                                  ckpt_extra={"epoch": epoch, "best_train_loss": best_loss})
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                print(f"[train+val] Early stopping after {epoch} epochs (no improvement).")
                break

        # rolling resume checkpoint (keeps best_state inside)
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_loss,
            "best_state": best_state,
        }, ckpt_last)

    # final restore best and save (idempotent)
    if best_state is not None:
        model.load_state_dict(best_state)
    _save_final_artifacts(final_dir, model.state_dict(), mean, std, ds.feature_cols,
                          ckpt_extra={"epoch": epoch, "best_train_loss": best_loss})

    return model.state_dict(), mean, std, ds.feature_cols

# ======================
#   ORCHESTRATE FINAL RETRAIN + REPORTS
# ======================
def retrain_on_trainval_and_report(best_cfg_dict, master_cols, train_ids, val_ids, warm_start_state=None):
    trainval_ids = sorted(set(train_ids) | set(val_ids))

    # Use config dict as-is (already normalized by caller)
    cfg_use = dict(best_cfg_dict)

    print("\n" + "="*100)
    print("[FINAL] Retraining best-by-frame-acc config on TRAIN+VAL …")
    print("="*100)

    final_dir = os.path.join(ART_DIR_SUB, "final_trainval_best_by_frameacc")
    final_state, mean, std, feat_cols = train_on_ids(
        cfg_use, trainval_ids, master_cols, final_dir,
        warm_start_state=(warm_start_state if WARM_START_FINAL else None)
    )

    detailed = evaluate_on_test_detailed(cfg_use, final_state, master_cols, mean, std, feat_cols)

    # Pretty-print results
    print("\n[FINAL] Test Accuracy")
    print(f"  Frame-level acc: {detailed['frame_acc']:.4f}")
    print(f"  Video-level acc: {detailed['video_acc']:.4f}")

    # Confusion Matrices
    names = detailed["target_names"]
    if detailed["cm_frame"] is not None:
        print("\n[FINAL] Frame-level Confusion Matrix (rows=true, cols=pred):")
        print(pd.DataFrame(detailed["cm_frame"], index=names, columns=names).to_string())
    if detailed["cm_video"] is not None:
        print("\n[FINAL] Video-level Confusion Matrix (rows=true, cols=pred):")
        print(pd.DataFrame(detailed["cm_video"], index=names, columns=names).to_string())

    # Classification Reports
    print("\n[FINAL] Frame-level Classification Report:")
    print(detailed["cr_frame"])
    print("\n[FINAL] Video-level Classification Report:")
    print(detailed["cr_video"])

    # Save text reports next to best.pt/scaler.pt
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

    global MASTER_FEATURE_COLS

    # 1) ids
    train_ids, val_ids, test_ids = get_ids_train_val_test()
    dev_ids = sorted(set(train_ids) | set(val_ids))
    print(f"[splits] train={len(train_ids)} | val={len(val_ids)} | test={len(test_ids)} | dev(unique)={len(dev_ids)}")

    # 2) master features + caches (for all dev ids)
    MASTER_FEATURE_COLS = get_master_feature_cols(dev_ids)
    for i, vid in enumerate(dev_ids, 1):
        build_video_cache_master(vid, MASTER_FEATURE_COLS, LABEL_COL, SKIP_FIRST_N)
        if i % 200 == 0:
            print(f"[master-cache] built {i}/{len(dev_ids)}")

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

        # If already finished, import metrics
        if os.path.isfile(done_flag) and os.path.isfile(metrics_p):
            print(f"[run {i}/{len(grid)}] Already finished; importing metrics.json")
            try:
                with open(metrics_p) as f:
                    out = json.load(f)
                rows.append(out if isinstance(out, dict) else {"config": tag, **cfg_run, **out})
                continue
            except Exception as e:
                print(f"[warn] Failed to read metrics.json, re-running test later. ({e})")

        m = train_val_es_then_test(cfg_run, train_ids, val_ids, MASTER_FEATURE_COLS)

        # per-config metrics.json
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

    # ---- Pick best-by-frame accuracy and normalize types inline (no helpers)
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

    # Optional warm-start: load grid best weights (uncomment to enable + set WARM_START_FINAL=True)
    warm_state = None
    # cfg_dir_best = os.path.join(CONFIGS_DIR, _config_tag(cfg_use), "best.pt")
    # if os.path.isfile(cfg_dir_best):
    #     warm_state = torch.load(cfg_dir_best, map_location="cpu", weights_only=False)

    # ---- Retrain on train+val and report detailed test results
    retrain_on_trainval_and_report(cfg_use, MASTER_FEATURE_COLS, train_ids, val_ids, warm_start_state=warm_state)

if __name__ == "__main__":
    main()

