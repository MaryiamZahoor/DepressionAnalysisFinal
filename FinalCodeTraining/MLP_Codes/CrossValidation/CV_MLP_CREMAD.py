#!/usr/bin/env python3
# crossval_best_from_csv.py
# Select BEST config from a CSV (by 'avg'), then 10-fold CV on ALL data (TRAIN∪VAL∪TEST).
# Uses existing per-video caches in master feature order. Saves per-fold checkpoints (resume),
# per-fold metrics, and pooled overall reports. Minimal printing; no per-fold reports/CMs printed.

import os, re, json, math, gc
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# ---- project imports (your modules) ----
import config as CFG
from models.TwoLayerMLP import FrameClassifier
from utils.features import save_feature_cols, Standardize, harmonize_vgg_cols
from data.datasets import build_au_master

# ======================
#      CONSTANTS
# ======================
K_FOLDS         = 10
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ES_PATIENCE     = 15
MAX_EPOCHS      = CFG.EPOCHS
ES_MONITOR      = "val_acc"   # early stop on validation accuracy
SKIP_FIRST_N    = CFG.SKIP_FRAME
DO_STANDARDIZE  = True
KEEP_AU_C_RAW   = True

RESUME_FOLDS    = True
SKIP_DONE_FOLDS = True

# ---------- Dataloader perf ----------
CPU_COUNT = os.cpu_count() or 4
NUM_WORKERS = min(8, max(0, CPU_COUNT - 2))
PIN_MEMORY  = torch.cuda.is_available()
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
    return optim.param_groups[0]['lr'] if optim.param_groups else float('nan')

# ---------- Paths (from your existing code) ----------
SPLIT_PATH = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/data/"
TRAIN_LIST = os.path.join(SPLIT_PATH, "train_videos_full.txt")
VAL_LIST   = os.path.join(SPLIT_PATH, "val_videos_full.txt")
TEST_LIST  = os.path.join(SPLIT_PATH, "test_videos_full.txt")

EXCLUDE_LIST = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/exclude_videos.txt"
INCLUDE_LIST = None

ART_DIR_TAG = "cremad_GridSearch_unscaled_MLP"
PROJECT_DIR = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/"
ART_DIR_SUB = os.path.join(PROJECT_DIR, "artifacts", ART_DIR_TAG)
GRID_OUT_DIR = os.path.join(ART_DIR_SUB, "grid_cv")
CONFIGS_DIR  = os.path.join(ART_DIR_SUB, "configs")

# CSV with per-config CV results (must contain columns: 'config', 'avg')
CV_RESULTS_CSV = os.path.join(GRID_OUT_DIR, "cv_results_by_config.csv")  # adjust if your filename differs

# ---------- Labels / columns ----------
LABEL_COL          = getattr(CFG, "SPLIT_LABEL_COL", "Actual_Emotion")
COMBINED_CSV_NAME  = getattr(CFG, "COMBINED_CSV_NAME", "affwild_resnet_au_vgg_with_gt.csv")
EMOTION_TO_IDX     = getattr(CFG, "emotion_to_idx")
IDX_TO_EMO         = {v: k for k, v in EMOTION_TO_IDX.items()}

# ======================
#  MASTER/CACHE HELPERS
# ======================
def _vid_csv_path(vid: str) -> str:
    return os.path.join(CFG.OUTPUT_DIR, vid, COMBINED_CSV_NAME)

def _cache_paths(vid: str):
    cdir = os.path.join(CFG.OUTPUT_DIR, vid, "cache")
    return os.path.join(cdir, "X.npy"), os.path.join(cdir, "y.npy"), os.path.join(cdir, "meta.json")

MASTER_FEATURES_JSON = os.path.join(GRID_OUT_DIR, "master_feature_cols.json")
MASTER_SCAN_LIMIT    = 10

def _append_unique(dst_list, names_iterable):
    seen = set(dst_list)
    for n in names_iterable:
        if n not in seen:
            dst_list.append(n); seen.add(n)

def get_master_feature_cols(ids_all: List[str]) -> List[str]:
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
            _append_unique(master, cnn_cols)
            picked += 1
            if picked >= MASTER_SCAN_LIMIT: break
        except Exception as e:
            print(f"[features] warn: {csvp} -> {e}")
    au_all = build_au_master(True, True)
    _append_unique(master, list(au_all))
    os.makedirs(os.path.dirname(MASTER_FEATURES_JSON), exist_ok=True)
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
    y = (df[label_col].astype(str).str.upper()
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
    np.save(xnp, X); np.save(ynp, y.to_numpy(copy=True))
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
                print(f"[cache] built {i+1}/{len(self.vids)}")
        want = []
        for i, name in enumerate(self.master_cols):
            if name.endswith("_vgg")    and use_vgg:    want.append(i)
            if name.endswith("_resnet") and use_resnet: want.append(i)
            if name.endswith("_c")      and use_au_c:   want.append(i)
            if name.endswith("_r")      and use_au_r:   want.append(i)
        self.col_idx = np.asarray(sorted(set(want)), dtype=np.int64)
        if self.col_idx.size == 0:
            raise ValueError("No columns selected by this feature combo.")
        self.feature_cols = [self.master_cols[i] for i in self.col_idx.tolist()]
        self.input_dim = len(self.feature_cols)
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
#      UTILITIES
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
    ids_out, y_out = [], []
    for vid in video_ids:
        csvp = _vid_csv_path(vid)
        if not os.path.isfile(csvp): continue
        try:
            s = pd.read_csv(csvp, usecols=[LABEL_COL])[LABEL_COL]
        except Exception:
            continue
        s = s.dropna().astype(str).str.upper()
        if s.empty: continue
        lab = s.mode().iat[0]
        if lab not in EMOTION_TO_IDX: continue
        ids_out.append(vid)
        y_out.append(EMOTION_TO_IDX[lab])
    if not ids_out:
        raise RuntimeError("No labeled videos found.")
    return ids_out, np.array(y_out, dtype=int)

def _compute_mean_std_per_feature(loader: DataLoader, device, feature_names: List[str]):
    D = len(feature_names); n_total = 0
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
    return mean.cpu(), std.cpu()

def _select_indices_for_cfg(master_cols: List[str], use_vgg, use_resnet, use_au_c, use_au_r):
    idx = []
    for i, c in enumerate(master_cols):
        if c.endswith("_vgg")    and use_vgg:    idx.append(i)
        if c.endswith("_resnet") and use_resnet: idx.append(i)
        if c.endswith("_c")      and use_au_c:   idx.append(i)
        if c.endswith("_r")      and use_au_r:   idx.append(i)
    return np.asarray(sorted(set(idx)), dtype=np.int64)

# =========================================================
#  SELECT BEST CONFIG FROM CSV (highest 'avg' row)
#  Example row format (your snippet):
#  "VGG+RESNET+AU | H1=1024 H2=512 | opt=adam lr=0.0001 wd=1e-05 do=0.5 bs=512"
# =========================================================
def _parse_config_from_csv_string(s: str) -> Dict:
    # Parts split by '|'
    # [0]: "VGG+RESNET+AU"
    # [1]: " H1=1024 H2=512 "
    # [2]: " opt=adam lr=0.0001 wd=1e-05 do=0.5 bs=512"
    parts = [p.strip() for p in s.split("|")]
    feats = parts[0]
    use_vgg    = "VGG" in feats
    use_resnet = "RESNET" in feats
    use_au_c   = "AU" in feats  # you use AU_c and AU_r together in these combos
    use_au_r   = "AU" in feats

    mH1 = re.search(r"H1\s*=\s*(\d+)", parts[1])
    mH2 = re.search(r"H2\s*=\s*(None|\d+)", parts[1], re.IGNORECASE)
    hidden_dim  = int(mH1.group(1)) if mH1 else 1024
    hidden_dim2 = None if (not mH2 or mH2.group(1).lower()=="none") else int(mH2.group(1))

    opt = re.search(r"opt\s*=\s*([A-Za-z0-9_]+)", parts[2]).group(1)
    lr  = float(re.search(r"lr\s*=\s*([0-9eE\.\-]+)", parts[2]).group(1))
    wd  = float(re.search(r"wd\s*=\s*([0-9eE\.\-]+)", parts[2]).group(1))
    do  = float(re.search(r"do\s*=\s*([0-9\.]+)", parts[2]).group(1))
    bs  = int(re.search(r"bs\s*=\s*(\d+)", parts[2]).group(1))

    return dict(
        feature_set=feats,
        use_vgg=use_vgg, use_resnet=use_resnet, use_au_c=use_au_c, use_au_r=use_au_r,
        hidden_dim=hidden_dim, hidden_dim2=hidden_dim2,
        optimizer=opt, lr=lr, weight_decay=wd, dropout=do, batch_size=bs
    )

def _sanitize_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-+=.]", "_", s)[:200]

def _load_best_config_from_csv(csv_path: str) -> Tuple[Dict, str]:
    df = pd.read_csv(csv_path)
    if "config" not in df.columns or "avg" not in df.columns:
        raise RuntimeError(f"CSV must contain 'config' and 'avg' columns: {csv_path}")
    best_row = df.sort_values("avg", ascending=False).iloc[0]
    cfg = _parse_config_from_csv_string(str(best_row["config"]))
    tag = _sanitize_tag(str(best_row["config"]))
    print(f"[best-from-csv] {best_row['config']} (avg={best_row['avg']:.4f})")
    return cfg, tag

# ======================
#          MAIN
# ======================
def main():
    # ----- union all ids -----
    train_ids = _read_ids(_require_file(TRAIN_LIST, "TRAIN_LIST"))
    val_ids   = _read_ids(_require_file(VAL_LIST,   "VAL_LIST"))
    test_ids  = _read_ids(_require_file(TEST_LIST,  "TEST_LIST"))
    all_ids   = sorted(set(train_ids) | set(val_ids) | set(test_ids))
    all_ids   = _apply_include_exclude(all_ids)
    ids, y    = _scan_labels(all_ids)
    print(f"[ALL] labeled videos: {len(ids)}")

    # ----- master list & caches -----
    master_cols = get_master_feature_cols(ids)
    for i, vid in enumerate(ids, 1):
        build_video_cache_master(vid, master_cols, LABEL_COL, SKIP_FIRST_N)
        if i % 300 == 0: print(f"[cache] ensured {i}/{len(ids)}")

    # ----- best config from CSV -----
    best_cfg, csv_tag = _load_best_config_from_csv(_require_file(CV_RESULTS_CSV, "CV results CSV"))

    CV_ROOT = os.path.join(ART_DIR_SUB, "cv_best_from_csv", csv_tag)
    os.makedirs(CV_ROOT, exist_ok=True)
    save_feature_cols(master_cols, os.path.join(CV_ROOT, "feature_cols.json"))

    # ----- CV -----
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=CFG.SEED)
    fold_rows = []
    y_true_all_frame, y_pred_all_frame = [], []
    y_true_all_video, y_pred_all_video = [], []

    sel_idx = _select_indices_for_cfg(
        master_cols, best_cfg["use_vgg"], best_cfg["use_resnet"], best_cfg["use_au_c"], best_cfg["use_au_r"]
    )

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(ids, y), start=1):
        tr_ids = [ids[i] for i in tr_idx]
        va_ids = [ids[i] for i in va_idx]

        fold_dir = os.path.join(CV_ROOT, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        metrics_path = os.path.join(fold_dir, "metrics.json")
        last_ckpt    = os.path.join(fold_dir, "last_ckpt.pt")
        best_state   = os.path.join(fold_dir, "best_state.pt")
        scaler_path  = os.path.join(fold_dir, "scaler.pt")

        # fast skip if done
        if SKIP_DONE_FOLDS and os.path.isfile(metrics_path):
            try:
                m = json.load(open(metrics_path, "r"))
                if m.get("done", False):
                    print(f"[fold {fold_id:02d}] done; skipping")
                    fold_rows.append({"fold": fold_id, "frame_acc": m.get("val_acc", np.nan),
                                      "video_acc": m.get("video_val_acc", np.nan),
                                      "val_loss": m.get("val_loss", np.nan)})
                    continue
            except Exception:
                pass

        # datasets
        ds_tr = CachedFrameDatasetMaster(tr_ids, master_cols,
                                         best_cfg["use_vgg"], best_cfg["use_resnet"],
                                         best_cfg["use_au_c"], best_cfg["use_au_r"],
                                         SKIP_FIRST_N)
        ds_va = CachedFrameDatasetMaster(va_ids, master_cols,
                                         best_cfg["use_vgg"], best_cfg["use_resnet"],
                                         best_cfg["use_au_c"], best_cfg["use_au_r"],
                                         SKIP_FIRST_N)

        tr_loader = DataLoader(ds_tr, batch_size=best_cfg["batch_size"], shuffle=True,
                               drop_last=False, collate_fn=_safe_collate, **_loader_kws())
        va_loader = DataLoader(ds_va, batch_size=best_cfg["batch_size"], shuffle=False,
                               drop_last=False, collate_fn=_safe_collate, **_loader_kws())

        # scaler on train fold
        if DO_STANDARDIZE:
            tmp_loader = DataLoader(ds_tr, batch_size=4096, shuffle=False,
                                    collate_fn=_safe_collate, **_loader_kws())
            mean, std = _compute_mean_std_per_feature(tmp_loader, DEVICE, ds_tr.feature_cols)
            torch.save({"mean": mean, "std": std}, scaler_path)
            scaler = Standardize(mean, std).to(DEVICE)
        else:
            scaler = nn.Identity()

        # model/optim/sched
        model = FrameClassifier(
            input_dim=ds_tr.input_dim,
            hidden_dim=best_cfg["hidden_dim"],
            hidden_dim2=best_cfg["hidden_dim2"],
            dropout=best_cfg["dropout"],
            num_classes=len(EMOTION_TO_IDX),
        ).to(DEVICE)

        if best_cfg["optimizer"].lower() == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=best_cfg["lr"], weight_decay=best_cfg["weight_decay"])
        else:
            optim = torch.optim.SGD(model.parameters(), lr=best_cfg["lr"], momentum=0.9,
                                    nesterov=True, weight_decay=best_cfg["weight_decay"])
        ce = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="max", factor=0.1, patience=5, min_lr=1e-6
        )

        # resume?
        start_epoch = 1
        best_metric = -math.inf
        no_improve  = 0
        if RESUME_FOLDS and os.path.isfile(last_ckpt):
            print(f"[fold {fold_id:02d}] resuming from last_ckpt.pt")
            ckpt = torch.load(last_ckpt, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
            optim.load_state_dict(ckpt["optim"])
            if ckpt.get("sched") is not None:
                scheduler.load_state_dict(ckpt["sched"])
            best_metric = ckpt.get("best_metric", best_metric)
            no_improve  = int(ckpt.get("no_improve", 0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1

        # train with early stopping on val_acc
        for epoch in range(start_epoch, MAX_EPOCHS + 1):
            model.train(); run_loss = 0.0
            for xb, yb in tr_loader:
                xb = scaler(xb.to(DEVICE, non_blocking=True).float())
                yb = yb.to(DEVICE, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = ce(logits, yb)
                loss.backward(); optim.step()
                run_loss += loss.item()

            # val
            model.eval(); total, correct, vloss = 0, 0, 0.0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb = scaler(xb.to(DEVICE, non_blocking=True).float())
                    yb = yb.to(DEVICE, non_blocking=True)
                    logits = model(xb)
                    vloss += ce(logits, yb).item()
                    correct += (logits.argmax(1) == yb).sum().item()
                    total   += yb.numel()
            val_acc  = correct / max(1, total)
            val_loss = vloss / max(1, len(va_loader))
            scheduler.step(val_acc)

            # save best weights & resume ckpt
            improved = (val_acc > best_metric)
            if improved:
                best_metric = val_acc
                torch.save(model.state_dict(), best_state)
                no_improve = 0
            else:
                no_improve += 1

            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "sched": scheduler.state_dict(),
                "best_metric": best_metric,
                "no_improve": no_improve,
            }, last_ckpt)

            print(f"[fold {fold_id:02d}] ep {epoch:03d} | tr_loss {run_loss/max(1,len(tr_loader)):.4f} "
                  f"| va_loss {val_loss:.4f} | va_acc {val_acc:.4f} | best {best_metric:.4f} "
                  f"| lr {_current_lr(optim):.2e}")

            if no_improve >= ES_PATIENCE:
                print(f"[fold {fold_id:02d}] early stop.")
                break

        # load best for eval on held-out fold
        if os.path.isfile(best_state):
            model.load_state_dict(torch.load(best_state, map_location=DEVICE))

        # frame-level pooled preds for this fold
        y_true_f, y_pred_f = [], []
        with torch.no_grad():
            for vid in va_ids:
                xnp, ynp, _ = _cache_paths(vid)
                if not (os.path.isfile(xnp) and os.path.isfile(ynp)): continue
                Xm = np.load(xnp, mmap_mode="r"); yv = np.load(ynp, mmap_mode="r")
                if Xm.shape[0] == 0: continue
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
            cm_f = confusion_matrix(y_true_f, y_pred_f, labels=list(range(len(IDX_TO_EMO))))
            frame_acc = (cm_f.trace()/cm_f.sum()) if cm_f.sum() > 0 else float("nan")
        else:
            frame_acc = float("nan")

        # video-level for this fold
        y_true_v, y_pred_v = [], []
        with torch.no_grad():
            for vid in va_ids:
                xnp, ynp, _ = _cache_paths(vid)
                if not (os.path.isfile(xnp) and os.path.isfile(ynp)): continue
                Xm = np.load(xnp, mmap_mode="r"); yv = np.load(ynp, mmap_mode="r")
                if Xm.shape[0] == 0: continue
                Xs = np.array(Xm[:, sel_idx], copy=True)
                X  = torch.from_numpy(Xs).float().to(DEVICE, non_blocking=True)
                X  = scaler(X)
                probs = torch.softmax(model(X), dim=1).mean(dim=0)
                y_pred_v.append(int(probs.argmax().item()))
                y_true_v.append(int(np.bincount(yv).argmax()))
        if len(y_true_v):
            cm_v = confusion_matrix(np.array(y_true_v), np.array(y_pred_v), labels=list(range(len(IDX_TO_EMO))))
            video_acc = (cm_v.trace()/cm_v.sum()) if cm_v.sum() > 0 else float("nan")
        else:
            video_acc = float("nan")

        with open(metrics_path, "w") as f:
            json.dump({
                "done": True,
                "val_acc": float(frame_acc),
                "val_loss": float(val_loss),
                "video_val_acc": float(video_acc),
                "epochs": int(min(MAX_EPOCHS, 10**9)),  # placeholder
            }, f, indent=2)

        if y_true_f.size:  # accumulate for overall pooled report
            y_true_all_frame.append(y_true_f); y_pred_all_frame.append(y_pred_f)
        if len(y_true_v):
            y_true_all_video.append(np.array(y_true_v)); y_pred_all_video.append(np.array(y_pred_v))

        fold_rows.append({"fold": fold_id, "frame_acc": float(frame_acc),
                          "video_acc": float(video_acc), "val_loss": float(val_loss)})

        # cleanup
        if hasattr(ds_tr, "close"): ds_tr.close()
        if hasattr(ds_va, "close"): ds_va.close()
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # ----- Save fold_metrics.csv -----
    pd.DataFrame(fold_rows).to_csv(os.path.join(CV_ROOT, "fold_metrics.csv"), index=False)

    # ----- Overall pooled reports -----
    classes = [IDX_TO_EMO[i] for i in range(len(IDX_TO_EMO))]
    overall = {}

    if y_true_all_frame:
        y_true_all_frame = np.concatenate(y_true_all_frame)
        y_pred_all_frame = np.concatenate(y_pred_all_frame)
        repF = classification_report(y_true_all_frame, y_pred_all_frame,
                                     target_names=classes, digits=4, output_dict=True)
        cmF  = confusion_matrix(y_true_all_frame, y_pred_all_frame,
                                labels=list(range(len(IDX_TO_EMO))))
        with open(os.path.join(CV_ROOT, "frame_report.json"), "w") as f:
            json.dump(repF, f, indent=2)
        pd.DataFrame(cmF, index=classes, columns=classes).to_csv(os.path.join(CV_ROOT, "frame_cm.csv"))
        overall["frame_acc_mean_over_folds"] = float(np.nanmean([r["frame_acc"] for r in fold_rows]))
    else:
        overall["frame_acc_mean_over_folds"] = float("nan")

    if y_true_all_video:
        y_true_all_video = np.concatenate(y_true_all_video)
        y_pred_all_video = np.concatenate(y_pred_all_video)
        repV = classification_report(y_true_all_video, y_pred_all_video,
                                     target_names=classes, digits=4, output_dict=True)
        cmV  = confusion_matrix(y_true_all_video, y_pred_all_video,
                                labels=list(range(len(IDX_TO_EMO))))
        with open(os.path.join(CV_ROOT, "video_report.json"), "w") as f:
            json.dump(repV, f, indent=2)
        pd.DataFrame(cmV, index=classes, columns=classes).to_csv(os.path.join(CV_ROOT, "video_cm.csv"))
        overall["video_acc_mean_over_folds"] = float(np.nanmean([r["video_acc"] for r in fold_rows]))
    else:
        overall["video_acc_mean_over_folds"] = float("nan")

    with open(os.path.join(CV_ROOT, "overall_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print("\n[CV] saved:")
    print(" - per-fold: last_ckpt.pt, best_state.pt, scaler.pt, metrics.json")
    print(" - overall : fold_metrics.csv, overall_summary.json, frame_report.json/frame_cm.csv, video_report.json/video_cm.csv")
    print(f"Root: {CV_ROOT}")

if __name__ == "__main__":
    main()

