#!/usr/bin/env python3
# CrossVal_MLP_COMBINED.py
# 10-fold CV for MLP on the combined dataset (CREMA-D + RAVDESS) using existing frame caches.

import os, re, json, math, gc, random
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# ---- project imports (yours) ----
import config as CFG
from models.TwoLayerMLP import FrameClassifier
from utils.features import save_feature_cols, Standardize

# ======================
#      CONSTANTS
# ======================
K_FOLDS      = 10
MAX_EPOCHS   = CFG.EPOCHS
ES_PATIENCE  = 15
ES_MONITOR   = "val_acc"  # "val_acc" or "val_loss"
SKIP_FIRST_N = CFG.SKIP_FRAME
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resume / speed-ups
RESUME_FOLDS = True
SKIP_DONE    = True

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
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)

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
COMB_SPLIT_PATH = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/data/"
TRAIN_LIST = os.path.join(COMB_SPLIT_PATH, "train_videos_COMBINED.txt")
VAL_LIST   = os.path.join(COMB_SPLIT_PATH, "val_videos_COMBINED.txt")
TEST_LIST  = os.path.join(COMB_SPLIT_PATH, "test_videos_COMBINED.txt")

# Dataset roots (existing caches live under each video folder)
CREMA_ROOT   = "/media/root918/OS/[REDACTED]Project/CREMA-D/copiedFiles/"
RAVDESS_ROOT = "/media/root918/OS/[REDACTED]Project/copiedFilesRAVDESS/"
CANDIDATE_CACHE_DIRS = ["cache"]  # existing cache folders that contain X.npy and y.npy/y_str.npy

# Artifacts
PROJECT_DIR  = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/"
ART_DIR_TAG  = "combined_GridSearch_unscaled_MLP"
ART_DIR_SUB  = os.path.join(PROJECT_DIR, "artifacts", ART_DIR_TAG)
GRID_OUT_DIR = os.path.join(ART_DIR_SUB, "grid_COMBINED")
CV_DIR       = os.path.join(ART_DIR_SUB, "cv_best_from_csv")
os.makedirs(GRID_OUT_DIR, exist_ok=True)
os.makedirs(CV_DIR, exist_ok=True)

# CSV produced by your combined MLP grid run
COMBINED_RESULTS_CSV = os.path.join(GRID_OUT_DIR, "grid_results_val_train_test.csv")

# Cache for CV labels (video-level majority in canonical 6-class)
IDS_LABELS_CACHE = os.path.join(GRID_OUT_DIR, "ids_labels_combined_all.json")

# ======================
#   LABEL SPACE (6-class canonical)
# ======================
CANONICAL = ["angry","disgust","fear","happy","neutral","sad"]
EMOTION_TO_IDX = {e:i for i,e in enumerate(CANONICAL)}
IDX_TO_EMO     = {v:k for k,v in EMOTION_TO_IDX.items()}

# Map RAVDESS strings -> canonical; None => drop
ALIASES = {
    "angry":"angry","anger":"angry","Anger":"angry","ANGER":"angry",
    "disgust":"disgust","Disgust":"disgust","DISGUST":"disgust",
    "fear":"fear","fearful":"fear","Fear":"fear","FEAR":"fear","Fearful":"fear",
    "happy":"happy","Happy":"happy","HAPPY":"happy",
    "neutral":"neutral","Neutral":"neutral","NEUTRAL":"neutral",
    "sad":"sad","sadness":"sad","Sad":"sad","SAD":"sad",
    # explicitly exclude (should be filtered earlier, but safe-guard):
    "calm":None,"Calm":None,"CALM":None,
    "surprise":None,"Surprise":None,"SURPRISE":None,"surprised":None,"Surprised":None,"SURPRISED":None,
}
def _ravdess_label_to_idx(s: str) -> Optional[int]:
    if s is None: return None
    t = str(s).strip()
    t = ALIASES.get(t, ALIASES.get(t.lower(), None))
    if t is None: return None
    return EMOTION_TO_IDX.get(t, None)

# CREMA mapping you used earlier: {'H':0,'S':1,'A':2,'N':3,'D':4,'F':5}
# canonical mapping order -> [happy, sad, angry, neutral, disgust, fear] indices [3,5,0,4,1,2]
CREMA_INT_TO_CANON = np.array([3, 5, 0, 4, 1, 2], dtype=np.int64)

# ======================
#   FEATURE PICKS (suffix-based)
# ======================
DO_STANDARDIZE = True
KEEP_AU_C_RAW  = True

def _require_file(path, desc):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")
    return path

def _read_ids(list_path: str) -> List[str]:
    with open(list_path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def _parse_id(combined_id: str) -> Tuple[str,str]:
    # "crema::<vid>" or "ravdess::Actor_xx/<vid_dir>"
    if "::" not in combined_id:
        raise ValueError(f"Invalid combined id: {combined_id}")
    ds, vid = combined_id.split("::", 1)
    ds = ds.strip().lower()
    if ds not in ("crema","ravdess"):
        raise ValueError(f"Unknown dataset tag '{ds}' in {combined_id}")
    return ds, vid

def _vid_root(dataset: str) -> str:
    return CREMA_ROOT if dataset == "crema" else RAVDESS_ROOT

def _find_cache_dir(cid: str) -> str:
    ds, vid = _parse_id(cid)
    vroot = os.path.join(_vid_root(ds), vid)
    for cdir in CANDIDATE_CACHE_DIRS:
        p = os.path.join(vroot, cdir)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "X.npy")):
            if os.path.isfile(os.path.join(p, "y.npy")) or os.path.isfile(os.path.join(p, "y_str.npy")):
                return p
    raise FileNotFoundError(f"No cache found for {cid}")

def _read_feature_cols(cache_dir: str):
    """
    Support both layouts:
      - CREMA:   meta.json -> {"feature_cols": [...]}
      - RAVDESS: meta.json -> {"feature_cols_master": [...]}
    """
    meta = os.path.join(cache_dir, "meta.json")
    if not os.path.isfile(meta): return None
    try:
        m = json.load(open(meta))
        cols = m.get("feature_cols", None)
        if cols is None: cols = m.get("feature_cols_master", None)
        return list(cols) if cols is not None else None
    except Exception:
        return None

def assert_consistent_feature_schema(ids: List[str]) -> List[str]:
    common = None
    for cid in ids:
        cdir = _find_cache_dir(cid)
        cols = _read_feature_cols(cdir)
        if cols is None:
            raise RuntimeError(f"{cid}: missing/invalid feature meta.json")
        if common is None:
            common = cols
        else:
            if len(cols) != len(common) or any(a!=b for a,b in zip(cols, common)):
                raise RuntimeError(f"Inconsistent feature schema at {cid}.")
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
        raise ValueError("No columns selected (check suffixes/schema).")
    return sel

def _indices_from_names(all_cols: List[str], selected_names: List[str]) -> np.ndarray:
    pos = {n:i for i,n in enumerate(all_cols)}
    return np.asarray([pos[n] for n in selected_names], dtype=np.int64)

# ======================
#   DATASET (reads existing caches)
# ======================
def _load_cached_arrays(cid: str):
    cdir = _find_cache_dir(cid)
    x_path = os.path.join(cdir, "X.npy")
    y_int_path = os.path.join(cdir, "y.npy")
    y_str_path = os.path.join(cdir, "y_str.npy")
    if not os.path.isfile(x_path): raise FileNotFoundError(f"Missing X.npy at {cdir}")
    X = np.load(x_path, mmap_mode="r")
    if os.path.isfile(y_int_path):
        y = np.load(y_int_path, allow_pickle=True)
    elif os.path.isfile(y_str_path):
        y = np.load(y_str_path, allow_pickle=True)
        if y.dtype.kind == 'S': y = np.char.decode(y, 'utf-8')
    else:
        raise FileNotFoundError(f"Missing y.npy / y_str.npy at {cdir}")
    return X, y

class CachedFrameDatasetUnified(torch.utils.data.Dataset):
    """
    Uses existing per-video caches, maps to 6-class canonical indices, and
    returns only selected feature columns.
    """
    def __init__(self, ids: List[str], selected_idx: np.ndarray, selected_names: List[str], skip_first_n: int):
        self.ids = list(ids)
        self.sel_idx = np.asarray(selected_idx, dtype=np.int64)
        self.sel_names = list(selected_names)
        self.input_dim = len(self.sel_idx)
        self.skip_first_n = int(skip_first_n)

        self.chunks = []  # (X_memmap, keep_idx, y_mapped)
        total = 0
        for cid in self.ids:
            ds, _ = _parse_id(cid)
            X, y_raw = _load_cached_arrays(cid)
            n = X.shape[0]
            start = min(self.skip_first_n, n) if self.skip_first_n > 0 else 0

            if ds == "crema":
                y_local = y_raw.astype(np.int64, copy=False)
                mask = (y_local >= 0) & (y_local < 6)
                keep = np.where(mask & (np.arange(n) >= start))[0]
                y_map = CREMA_INT_TO_CANON[y_local[keep]]
            else:
                if y_raw.dtype.kind == 'S':  # bytes -> str safety
                    y_raw = np.char.decode(y_raw, 'utf-8')
                mapped = np.array([_ravdess_label_to_idx(v) for v in y_raw], dtype=object)
                keep = np.where((pd.notna(mapped)) & (np.arange(n) >= start))[0]
                if keep.size == 0: continue
                y_map = np.array([int(mapped[i]) for i in keep], dtype=np.int64)

            if keep.size == 0: continue
            self.chunks.append((X, keep, y_map))
            total += keep.size

        self.ranges = []
        acc = 0
        for vi, (_, keep, _) in enumerate(self.chunks):
            self.ranges.append((vi, acc, keep.size))
            acc += keep.size

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
        n_total += xb.shape[0]
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

def _best_cfg_from_csv(csv_path: str) -> Dict:
    df = pd.read_csv(csv_path)
    if "test_frame_acc" not in df.columns:
        raise RuntimeError("CSV missing 'test_frame_acc' column")
    df["test_frame_acc"] = pd.to_numeric(df["test_frame_acc"], errors="coerce")
    row = df.sort_values("test_frame_acc", ascending=False).iloc[0]
    cfg = dict(
        use_vgg      = bool(row["use_vgg"]),
        use_resnet   = bool(row["use_resnet"]),
        use_au_c     = bool(row["use_au_c"]),
        use_au_r     = bool(row["use_au_r"]),
        hidden_dim   = int(float(row["hidden_dim"])),
        hidden_dim2  = (None if (pd.isna(row.get("hidden_dim2")) or str(row.get("hidden_dim2")).strip().lower() in ("", "none","nan"))
                        else int(float(row["hidden_dim2"]))),
        optimizer    = str(row["optimizer"]),
        lr           = float(row["lr"]),
        weight_decay = float(row["weight_decay"]),
        dropout      = float(row["dropout"]),
        batch_size   = int(float(row["batch_size"])),
        tag          = str(row.get("config","best_from_combined_csv"))
    )
    print(f"[best-from-csv] {cfg['tag']} | test_frame_acc={row['test_frame_acc']:.6f}")
    return cfg

def _fold_dir_for(cfg: Dict, fold: int) -> str:
    tag = re.sub(r"[^A-Za-z0-9_\-+=.]", "_", str(cfg.get("tag","CFG")))[:200]
    d = os.path.join(CV_DIR, tag, f"fold_{fold}")
    os.makedirs(d, exist_ok=True)
    return d

# ======================
#   ID & LABEL UTILITIES (for CV stratification)
# ======================
def _video_majority_label(cid: str) -> Optional[int]:
    ds, _ = _parse_id(cid)
    X, y_raw = _load_cached_arrays(cid)
    if X.shape[0] == 0: return None
    if ds == "crema":
        y = y_raw.astype(np.int64, copy=False)
        mask = (y >= 0) & (y < 6)
        if not mask.any(): return None
        y = y[mask]
        if y.size == 0: return None
        y = CREMA_INT_TO_CANON[y]
        return int(np.bincount(y).argmax())
    else:
        if y_raw.dtype.kind == 'S':
            y_raw = np.char.decode(y_raw, 'utf-8')
        mapped = np.array([_ravdess_label_to_idx(v) for v in y_raw], dtype=object)
        mapped = mapped[pd.notna(mapped)]
        if mapped.size == 0: return None
        y = np.array(mapped, dtype=np.int64)
        return int(np.bincount(y).argmax())

def _scan_ids_labels(all_ids: List[str]) -> Tuple[List[str], np.ndarray]:
    ids_out, labs = [], []
    for cid in all_ids:
        try:
            lab = _video_majority_label(cid)
        except Exception:
            lab = None
        if lab is None: continue
        ids_out.append(cid); labs.append(lab)
    if not ids_out:
        raise RuntimeError("No labeled videos found for CV.")
    return ids_out, np.asarray(labs, dtype=np.int64)

def get_ids_and_labels_cached() -> Tuple[List[str], np.ndarray]:
    train = _read_ids(_require_file(TRAIN_LIST, "TRAIN_LIST"))
    val   = _read_ids(_require_file(VAL_LIST,   "VAL_LIST"))
    test  = _read_ids(_require_file(TEST_LIST,  "TEST_LIST"))
    all_ids = sorted(set(train) | set(val) | set(test))
    if os.path.isfile(IDS_LABELS_CACHE):
        try:
            d = json.load(open(IDS_LABELS_CACHE))
            ids = list(d.get("ids", []))
            y   = np.array(d.get("labels", []), dtype=np.int64)
            if ids and len(ids) == len(y):
                print(f"[ids] loaded {len(ids)} from cache → {IDS_LABELS_CACHE}")
                return ids, y
        except Exception as e:
            print(f"[ids] cache read failed: {e}; rebuilding…")
    ids, y = _scan_ids_labels(all_ids)
    json.dump({"ids": ids, "labels": y.astype(int).tolist()}, open(IDS_LABELS_CACHE, "w"), indent=2)
    print(f"[ids] saved {len(ids)} → {IDS_LABELS_CACHE}")
    return ids, y

# ======================
#   TRAIN / EVAL (single fold)
# ======================
def _train_fold(cfg, tr_ids, va_ids, common_cols, tag_for_cols):
    # select feature names -> indices
    sel_names = _selected_names_for_cfg(common_cols, cfg["use_vgg"], cfg["use_resnet"], cfg["use_au_c"], cfg["use_au_r"])
    sel_idx   = _indices_from_names(common_cols, sel_names)

    ds_tr = CachedFrameDatasetUnified(tr_ids, sel_idx, sel_names, SKIP_FIRST_N)
    ds_va = CachedFrameDatasetUnified(va_ids, sel_idx, sel_names, SKIP_FIRST_N)

    tr_loader = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True,
                           drop_last=False, collate_fn=_safe_collate, **_loader_kws())
    va_loader = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False,
                           drop_last=False, collate_fn=_safe_collate, **_loader_kws())

    # scaler on train
    if DO_STANDARDIZE:
        tmp = DataLoader(ds_tr, batch_size=4096, shuffle=False, collate_fn=_safe_collate, **_loader_kws())
        mean, std = _compute_mean_std_per_feature(tmp, DEVICE, ds_tr.feature_cols)
        if KEEP_AU_C_RAW:
            auc_idx = [i for i, n in enumerate(ds_tr.feature_cols) if n.endswith("_c")]
            if auc_idx:
                idx = torch.tensor(auc_idx, dtype=torch.long)
                std[idx] = 1.0; mean[idx] = 0.0
        scaler = Standardize(mean, std).to(DEVICE)
    else:
        mean = torch.zeros(ds_tr.input_dim); std = torch.ones(ds_tr.input_dim)
        scaler = nn.Identity()

    model = FrameClassifier(input_dim=ds_tr.input_dim,
                            hidden_dim=cfg["hidden_dim"], hidden_dim2=cfg["hidden_dim2"],
                            dropout=cfg["dropout"], num_classes=len(EMOTION_TO_IDX)).to(DEVICE)

    optim = _build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ce    = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode=("max" if ES_MONITOR=="val_acc" else "min"),
                                                       factor=0.1, patience=5, min_lr=1e-6)

    cfg_dir   = tag_for_cols["fold_dir"]
    last_ckpt = os.path.join(cfg_dir, "last_ckpt.pt")
    best_path = os.path.join(cfg_dir, "best.pt")                # <-- rename to best.pt
    scaler_pt = os.path.join(cfg_dir, "scaler.pt")
    metrics_p = os.path.join(cfg_dir, "metrics.json")

    # save fold ids for traceability
    with open(os.path.join(cfg_dir, "fold_ids.json"), "w") as f:
        json.dump({"train_ids": tr_ids, "val_ids": va_ids}, f, indent=2)

    # resume
    start_epoch = 1
    best_metric = -math.inf if ES_MONITOR == "val_acc" else math.inf
    no_improve  = 0

    if RESUME_FOLDS and os.path.isfile(last_ckpt):
        print(f"[resume] {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        try: model.load_state_dict(ckpt["model"])
        except Exception as e: print(f"[resume warn] model: {e}")
        if ckpt.get("optim") is not None:
            try: optim.load_state_dict(ckpt["optim"])
            except Exception as e: print(f"[resume warn] optim: {e}")
        if ckpt.get("sched") is not None:
            try: sched.load_state_dict(ckpt["sched"])
            except Exception as e: print(f"[resume warn] sched: {e}")
        best_metric = ckpt.get("best_metric", best_metric)
        start_epoch = ckpt.get("epoch", 0) + 1
        if os.path.isfile(scaler_pt):
            try:
                sc = torch.load(scaler_pt, map_location="cpu")
                mean, std = sc.get("mean", mean), sc.get("std", std)
                scaler = Standardize(mean, std).to(DEVICE) if DO_STANDARDIZE else nn.Identity()
                print("[resume] restored scaler")
            except Exception as e:
                print(f"[resume warn] scaler: {e}")

    # train with ES on val_acc
    for ep in range(start_epoch, MAX_EPOCHS + 1):
        model.train(); run = 0.0
        for xb, yb in tr_loader:
            xb = scaler(xb.to(DEVICE, non_blocking=True).float())
            yb = yb.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward(); optim.step()
            run += loss.item()

        va_loss, va_acc = _eval_loader(model, va_loader, DEVICE, scaler)
        sched.step(va_acc if ES_MONITOR=="val_acc" else va_loss)

        improved = (va_acc > best_metric) if ES_MONITOR=="val_acc" else (va_loss < best_metric)
        if improved:
            best_metric = va_acc if ES_MONITOR=="val_acc" else va_loss
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1

        torch.save({
            "epoch": ep, "model": model.state_dict(),
            "optim": optim.state_dict(), "sched": sched.state_dict(),
            "best_metric": best_metric
        }, last_ckpt)

        print(f"[ep {ep:03d}] tr_loss {run/max(1,len(tr_loader)):.4f} | va_loss {va_loss:.4f} | va_acc {va_acc:.4f} | best {best_metric:.4f} | lr {_current_lr(optim):.2e}")

        if no_improve >= ES_PATIENCE:
            print("[early stop]")
            break

    # restore best
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"))

    # save scaler + feature cols once
    torch.save({"mean": mean, "std": std}, scaler_pt)
    save_feature_cols(ds_tr.feature_cols, os.path.join(cfg_dir, "feature_cols.json"))

    # ---------------------------
    # fold validation artifacts
    # ---------------------------
    # frame-level predictions, cm, report
    yT, yP = [], []
    with torch.no_grad():
        for xb, yb in va_loader:
            xb = Standardize(mean, std).to(DEVICE)(xb.to(DEVICE, non_blocking=True).float()) if DO_STANDARDIZE else xb.to(DEVICE, non_blocking=True).float()
            logits = model(xb)
            yP.append(logits.argmax(1).cpu().numpy())
            yT.append(yb.numpy())
    if yT:
        yT = np.concatenate(yT); yP = np.concatenate(yP)
        frame_acc = float((yT == yP).mean())
        # save preds
        np.savez_compressed(os.path.join(cfg_dir, "val_preds_frame.npz"), y_true=yT, y_pred=yP)
        # cm + report
        labels = list(range(len(EMOTION_TO_IDX)))
        target_names = [IDX_TO_EMO[i] for i in labels]
        cm_f = confusion_matrix(yT, yP, labels=labels)
        pd.DataFrame(cm_f, index=target_names, columns=target_names).to_csv(os.path.join(cfg_dir, "val_cm_frame.csv"))
        with open(os.path.join(cfg_dir, "val_report_frame.txt"), "w") as f:
            f.write(classification_report(yT, yP, labels=labels, target_names=target_names, digits=4))
    else:
        frame_acc = float("nan")

    # video-level (mean-softmax per video)
    video_acc = float("nan")
    try:
        vids = va_ids
        y_true_v, y_pred_v = [], []
        with torch.no_grad():
            for cid in vids:
                ds_one = CachedFrameDatasetUnified([cid], sel_idx, sel_names, SKIP_FIRST_N)
                if len(ds_one) == 0: continue
                loader_one = DataLoader(ds_one, batch_size=16384, shuffle=False,
                                        drop_last=False, collate_fn=_safe_collate, **_loader_kws())
                probs_sum = torch.zeros(len(EMOTION_TO_IDX), dtype=torch.float32, device=DEVICE)
                y_major = []
                for xb, yb in loader_one:
                    xb = Standardize(mean, std).to(DEVICE)(xb.to(DEVICE).float()) if DO_STANDARDIZE else xb.to(DEVICE).float()
                    logits = model(xb)
                    probs_sum += torch.softmax(logits, dim=1).sum(dim=0)
                    y_major.append(yb.numpy())
                y_pred_v.append(int(probs_sum.argmax().item()))
                y_major = np.concatenate(y_major)
                y_true_v.append(int(np.bincount(y_major).argmax()))
        if y_true_v:
            y_true_v = np.array(y_true_v); y_pred_v = np.array(y_pred_v)
            video_acc = float((y_true_v == y_pred_v).mean())
            # save preds
            np.savez_compressed(os.path.join(cfg_dir, "val_preds_video.npz"), y_true=y_true_v, y_pred=y_pred_v)
            # cm + report
            labels = list(range(len(EMOTION_TO_IDX)))
            target_names = [IDX_TO_EMO[i] for i in labels]
            cm_v = confusion_matrix(y_true_v, y_pred_v, labels=labels)
            pd.DataFrame(cm_v, index=target_names, columns=target_names).to_csv(os.path.join(cfg_dir, "val_cm_video.csv"))
            with open(os.path.join(cfg_dir, "val_report_video.txt"), "w") as f:
                f.write(classification_report(y_true_v, y_pred_v, labels=labels, target_names=target_names, digits=4))
    except Exception:
        pass

    json.dump({"done": True, "val_acc": frame_acc, "video_val_acc": video_acc},
              open(metrics_p, "w"), indent=2)

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return frame_acc, video_acc

# ======================
#          MAIN
# ======================
def main():
    _set_seed(1337)

    # 1) pick best config by **test_frame_acc** from combined CSV
    cfg = _best_cfg_from_csv(_require_file(COMBINED_RESULTS_CSV, "combined grid CSV"))
    cfg_tag = re.sub(r"[^A-Za-z0-9_\-+=.]", "_", str(cfg.get("tag","CFG")))[:200]
    root_tag_dir = os.path.join(CV_DIR, cfg_tag)
    os.makedirs(root_tag_dir, exist_ok=True)
    # persist the exact config used
    with open(os.path.join(root_tag_dir, "config_used.json"), "w") as f:
        json.dump({k: (None if v is None else (int(v) if isinstance(v, bool)==False and isinstance(v, float) and v.is_integer() else v)) for k,v in cfg.items()}, f, indent=2)

    # 2) collect ids & labels (cached) and assert feature schema on all usable videos
    ids, y = get_ids_and_labels_cached()
    common_cols = assert_consistent_feature_schema(ids)

    # 3) set up folds
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=CFG.SEED)
    fold_rows = []
    frame_accs, video_accs = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(ids, y), start=1):
        tr_ids = [ids[i] for i in tr_idx]
        va_ids = [ids[i] for i in va_idx]
        fold_dir = _fold_dir_for(cfg, fold)
        metrics_p = os.path.join(fold_dir, "metrics.json")

        if SKIP_DONE and os.path.isfile(metrics_p):
            try:
                m = json.load(open(metrics_p))
                if m.get("done", False):
                    print(f"[fold {fold:02d}] done; skipping")
                    fold_rows.append({"fold": fold, "frame_acc": m.get("val_acc", np.nan),
                                      "video_acc": m.get("video_val_acc", np.nan)})
                    frame_accs.append(m.get("val_acc", np.nan))
                    video_accs.append(m.get("video_val_acc", np.nan))
                    continue
            except Exception:
                pass

        f_acc, v_acc = _train_fold(cfg, tr_ids, va_ids, common_cols, {"fold_dir": fold_dir})
        fold_rows.append({"fold": fold, "frame_acc": f_acc, "video_acc": v_acc})
        frame_accs.append(f_acc)
        video_accs.append(v_acc)

    # 4) write summaries
    pd.DataFrame(fold_rows).to_csv(os.path.join(root_tag_dir, "fold_metrics.csv"), index=False)
    overall = {
        "frame_acc_mean_over_folds": float(np.nanmean(frame_accs)) if frame_accs else float("nan"),
        "video_acc_mean_over_folds": float(np.nanmean(video_accs)) if video_accs else float("nan"),
        "k_folds": K_FOLDS,
    }
    json.dump(overall, open(os.path.join(root_tag_dir, "overall_summary.json"), "w"), indent=2)

    print("\n[CV] saved under:", root_tag_dir)
    print("  - per-fold: last_ckpt.pt, best.pt, scaler.pt, feature_cols.json, metrics.json, fold_ids.json")
    print("              val_preds_frame.npz, val_cm_frame.csv, val_report_frame.txt")
    print("              val_preds_video.npz, val_cm_video.csv, val_report_video.txt")
    print("  - summary : fold_metrics.csv, overall_summary.json, config_used.json")
    print(f"  - mean frame acc: {overall['frame_acc_mean_over_folds']:.4f}")
    print(f"  - mean video acc: {overall['video_acc_mean_over_folds']:.4f}")

if __name__ == "__main__":
    main()

