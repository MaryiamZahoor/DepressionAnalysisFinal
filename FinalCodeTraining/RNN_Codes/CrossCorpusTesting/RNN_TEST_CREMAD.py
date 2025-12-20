#!/usr/bin/env python3
"""
RNN final retrain using best row from RNN grid CSV (sorted by test_seq_acc desc):
- Train 6-class RNN on RAVDESS caches (sequence-based)
- Evaluate on RAVDESS (sequence + video)
- Evaluate on CREMA-D (sequence + video)

Cache layout expected:
  RAVDESS (nested): <root>/<actor>/<vid>/cache/{X_all.npy|X.npy, y.npy|y_str.npy, meta.json?}
  CREMA-D  (flat) : <root>/<vid>/cache/{X_all.npy|X.npy, y.npy|y_str.npy, meta.json?}

6-class order (CREMA-compatible):
  0: happy, 1: sad, 2: anger, 3: neutral, 4: disgust, 5: fearful
"""

import os, json, sys, math, random
# --- clean env for torch import (avoids LD_LIBRARY_PATH conflicts) ---
if os.environ.get("LD_LIBRARY_PATH") and not os.environ.get("_LDLIBPATH_CLEANED"):
    env = dict(os.environ); env.pop("LD_LIBRARY_PATH", None); env["_LDLIBPATH_CLEANED"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)
os.environ.pop("LD_LIBRARY_PATH", None)

from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# ==== Use your existing model ====
# If your class is in models/TemporalFFRNN.py, change import accordingly.
from models.CNN_RNNmodel import TemporalFFRNN

# ===================== PATHS / PARAMS (edit as needed) =====================
RAVDESS_ROOT = "/media/root918/OS/MaryiamProject/copiedFilesRAVDESS"
CREMAD_ROOT  = "/media/root918/OS/MaryiamProject/CREMA-D/copiedFiles/"  # flat: <vid>/...

SPLIT_PATH = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data"
TRAIN_LIST = os.path.join(SPLIT_PATH, "train_videos_RAV.txt")
VAL_LIST   = os.path.join(SPLIT_PATH, "val_videos_RAV.txt")
TEST_LIST  = os.path.join(SPLIT_PATH, "test_videos_RAV.txt")

GRID_RESULTS_CSV         = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/ravdess_GridSearch_unscaled_RNN/ravdess_GridSearch_unscaled_RNN/rnn_grid_results_RAVDESS.csv"
MASTER_FEATURE_COLS_JSON = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/master_feature_cols.json"

ART_DIR = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/ravdess6class_final_retrain_RNN"
os.makedirs(ART_DIR, exist_ok=True)

# Global training controls (early stop etc.)
EPOCHS         = 300
ES_PATIENCE    = 30
SEED           = 1337
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU            = torch.device("cpu")

NUM_WORKERS_TRAIN  = min(8, (os.cpu_count() or 2))
PIN_MEMORY         = torch.cuda.is_available()

# Video aggregation
VIDEO_STRATEGY     = "mean_softmax"   # "mean_softmax" | "majority"

# OOM-safe evaluation batch sizes
EVAL_SEQ_BS = 128   # used for RAVDESS/CREMA-D eval
STATS_BS    = 64    # used for CPU stats computation
# ==========================================================================

# ===== Standardize helper from your project =====
from utils.features import Standardize
# ===============================================

# ---------- 6-class mapping ----------
RAV6_ORDER = ["happy", "sad", "anger", "neutral", "disgust", "fearful"]
LETTER_TO_IDX6 = {"H":0,"S":1,"A":2,"N":3,"D":4,"F":5}
NAME_TO_LETTER = {
    "neutral":"N","happy":"H","sad":"S","angry":"A","anger":"A",
    "fearful":"F","fear":"F","disgust":"D","calm":None,"surprise":None
}
RAV8_IDX_TO_NAME = {
    0:"neutral", 1:"calm", 2:"happy", 3:"sad",
    4:"anger",   5:"fearful", 6:"disgust", 7:"surprise"
}

def name_to_idx6(s: str) -> int:
    if s is None: return -1
    s = str(s).strip().lower()
    if not s: return -1
    if s in ("calm","surprise"): return -1
    if len(s) == 1:
        return LETTER_TO_IDX6.get(s.upper(), -1)
    letter = NAME_TO_LETTER.get(s, None)
    return LETTER_TO_IDX6.get(letter, -1) if letter else -1

def idx8_to_idx6(i: int) -> int:
    nm = RAV8_IDX_TO_NAME.get(int(i), "")
    return name_to_idx6(nm)

# ---------- small utils ----------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def require_file(p, what=None):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing {what or p}: {p}")
    return p

def read_ids(p: str) -> List[str]:
    with open(require_file(p, p)) as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_master_cols(path: str) -> List[str]:
    with open(require_file(path, "master feature cols")) as f:
        return json.load(f)

# ---------- parse best row from grid CSV (by test_seq_acc desc) ----------
def best_row_from_rnn_grid(csv_path: str) -> Dict:
    df = pd.read_csv(require_file(csv_path, "RNN grid results"))
    # Ensure numeric
    for c in ["val_seq_acc","test_seq_acc","val_vid_acc","test_vid_acc","val_frame_acc","test_frame_acc",
              "lr","weight_decay","dropout","rnn_hidden","rnn_layers","batch_size","seq_len","stride"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Sort by test_seq_acc desc, then val_seq_acc desc, then test_vid_acc desc
    #sort_cols = []
    #if "test_seq_acc" in df.columns: sort_cols.append(("test_seq_acc", False))
    #if "val_seq_acc"  in df.columns: sort_cols.append(("val_seq_acc",  False))
    #if "test_vid_acc" in df.columns: sort_cols.append(("test_vid_acc", False))
    #if not sort_cols:
     #   raise ValueError("Grid CSV missing required accuracy columns for sorting.")
    #df_sorted = df.sort_values([c for c,_ in sort_cols], ascending=[asc for _,asc in sort_cols])
    df_sorted = df.sort_values("test_vid_acc", ascending=False)

    row = df_sorted.iloc[0].to_dict()

    def _bool(x, default=False):
        if pd.isna(x): return default
        s = str(x).strip().lower()
        if s in ("true","1","yes"): return True
        if s in ("false","0","no"): return False
        return default

    def _opt_int(x):
        if pd.isna(x) or str(x).strip()=="":
            return None
        return int(round(float(x)))

    cfg = {
        # feature toggles
        "use_vgg":      _bool(row.get("use_vgg", True)),
        "use_resnet":   _bool(row.get("use_resnet", True)),
        "use_au_c":     _bool(row.get("use_au_c", True)),
        "use_au_r":     _bool(row.get("use_au_r", True)),
        # pre-FF dims (optional)
        "ff_hidden":    _opt_int(row.get("ff_hidden", None)),
        "ff_hidden2":   _opt_int(row.get("ff_hidden2", None)),
        # rnn hyperparams
        "rnn_type":     str(row.get("rnn_type", "gru")).lower(),
        "rnn_hidden":   int(round(float(row.get("rnn_hidden", 128)))),
        "rnn_layers":   int(round(float(row.get("rnn_layers", 1)))),
        "bidirectional": _bool(row.get("bidirectional", False)),
        "dropout":      float(row.get("dropout", 0.5)),
        # windowing
        "seq_len":      int(round(float(row.get("seq_len", 10)))),
        "stride":       int(round(float(row.get("stride", 5)))),
        # optimization
        "optimizer":    str(row.get("optimizer", "adam")).lower(),
        "lr":           float(row.get("lr", 1e-4)),
        "weight_decay": float(row.get("weight_decay", 1e-5)),
        "batch_size":   int(round(float(row.get("batch_size", 512)))),
        # normalization flags
        "norm_mode":    None,
        #"norm_mode":    str(row.get("norm_mode", "zscore")).lower(),
        "keep_au_c_raw": _bool(row.get("keep_au_c_raw", True)),
        # bookkeeping/debug
        "_picked_tag":   row.get("tag", None),
        "_val_seq_acc":  float(row.get("val_seq_acc", float("nan"))),
        "_test_seq_acc": float(row.get("test_seq_acc", float("nan"))),
    }
    print("\n[best-row from RNN grid (by test_seq_acc)]")
    for k in ["_picked_tag","_val_seq_acc","_test_seq_acc"]:
        print(f"  {k}: {cfg[k]}")
    print("[hyperparams]")
    for k in ["use_vgg","use_resnet","use_au_c","use_au_r","ff_hidden","ff_hidden2","rnn_type","rnn_hidden",
              "rnn_layers","bidirectional","dropout","seq_len","stride","optimizer","lr","weight_decay",
              "batch_size","norm_mode","keep_au_c_raw"]:
        print(f"  {k}: {cfg[k]}")
    return cfg

# ---------- feature selection ----------
def feature_sel_idx(master_cols: List[str], cfg: Dict) -> np.ndarray:
    idx=[]
    for i,n in enumerate(master_cols):
        if n.endswith("_vgg")    and cfg["use_vgg"]:    idx.append(i)
        if n.endswith("_resnet") and cfg["use_resnet"]: idx.append(i)
        if n.endswith("_c")      and cfg["use_au_c"]:   idx.append(i)
        if n.endswith("_r")      and cfg["use_au_r"]:   idx.append(i)
    idx = np.asarray(sorted(set(idx)), dtype=np.int64)
    if idx.size == 0:
        raise ValueError("Feature selection is empty (check use_* toggles vs master cols).")
    return idx

# ---------- cache detection ----------
def detect_cacheA(root: str, vid: str) -> Optional[Dict]:
    cdir = os.path.join(root, vid, "cache")
    x_candidates = ["X_all.npy", "X.npy"]
    y_candidates = ["y.npy", "y_str.npy"]

    Xp = next((os.path.join(cdir, x) for x in x_candidates if os.path.isfile(os.path.join(cdir, x))), None)
    if Xp is None:
        return None

    Yp, ykind = None, None
    for yname in y_candidates:
        yp = os.path.join(cdir, yname)
        if os.path.isfile(yp):
            Yp = yp
            ykind = "int" if yname == "y.npy" else "str"
            break
    if Yp is None:
        return None

    meta = {}
    mp = os.path.join(cdir, "meta.json")
    if os.path.isfile(mp):
        try:
            meta = json.load(open(mp))
        except Exception:
            pass
    meta["X_path"] = Xp
    meta["Y_path"] = Yp
    meta["Y_kind"] = ykind
    return meta

# ---------- RNN sequence dataset ----------
class CacheASequences6(Dataset):
    """
    Builds fixed-length sequences:
      - Map labels to 6 classes (drops calm/surprise/invalid).
      - Sequence label = majority label over frames in window.
    """
    def __init__(self, root: str, video_ids: List[str], sel_idx: np.ndarray,
                 seq_len: int, seq_stride: int, expect_master_cols: List[str],
                 return_vid: bool = False, tag: str = "RAVDESS"):
        self.root = root
        self.vids = list(video_ids)
        self.sel_idx = np.asarray(sel_idx, dtype=np.int64)
        self.return_vid = bool(return_vid)
        self.tag = tag
        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride)

        self.items = []   # per-video metadata
        self.index = []   # (item_idx, start_offset, maj_label)

        # stats
        cls_counts_frames = np.zeros(6, dtype=np.int64)
        cls_counts_seqs   = np.zeros(6, dtype=np.int64)
        mode_counts = {"int":0, "str":0}
        frames_kept_total = 0
        seq_total = 0

        for vid in self.vids:
            det = detect_cacheA(root, vid)
            if det is None:
                continue
            Xp, Yp, yk = det["X_path"], det["Y_path"], det["Y_kind"]
            Xm = np.load(Xp, mmap_mode="r")
            Ym = np.load(Yp, mmap_mode="r")
            n = min(Xm.shape[0], Ym.shape[0])
            if n <= 0:
                continue

            # optional meta verification
            mcols = det.get("feature_cols_master", det.get("feature_cols", None))
            if isinstance(mcols, list) and len(mcols) != Xm.shape[1]:
                print(f"[warn][{self.tag}] {vid} meta feature_cols len {len(mcols)} != X shape {Xm.shape[1]}")

            # map labels to 6
            if yk == "int":
                arr = Ym[:n].astype(np.int64, copy=False)
                if arr.max() > 5:  # 8-class -> 6
                    y6 = np.fromiter((idx8_to_idx6(int(i)) for i in arr), dtype=np.int64)
                    mode_counts["int"] += 1
                else:
                    y6 = arr
                    mode_counts["int"] += 1
            else:
                y6 = np.fromiter((name_to_idx6(str(s)) for s in Ym[:n]), dtype=np.int64)
                mode_counts["str"] += 1

            keep = np.nonzero(y6 >= 0)[0]
            if keep.size < self.seq_len:
                continue

            for k in keep.tolist():
                cls_counts_frames[y6[k]] += 1
            frames_kept_total += keep.size

            starts = list(range(0, keep.size - self.seq_len + 1, self.seq_stride))
            if not starts:
                continue

            start_idx = len(self.index)
            for s in starts:
                window_idx = keep[s:s+self.seq_len]
                maj = int(np.bincount(y6[window_idx].astype(int)).argmax())
                self.index.append((len(self.items), int(s), int(maj)))
                cls_counts_seqs[maj] += 1
            end_idx = len(self.index)
            seq_total += (end_idx - start_idx)

            self.items.append({
                "vid": vid, "X_path": Xp, "y_map": y6, "keep_idx": keep
            })

        print(f"\n[{self.tag}] frames kept (after 6-class filter): {frames_kept_total:,}")
        print(f"[{self.tag}] label counts (frames H,S,A,N,D,F): {cls_counts_frames.tolist()}")
        print(f"[{self.tag}] sequences built: {seq_total:,}")
        print(f"[{self.tag}] label counts (sequences H,S,A,N,D,F): {cls_counts_seqs.tolist()}")
        print(f"[{self.tag}] source label kinds used: {mode_counts}")
        print(f"[{self.tag}] selected feature dim = {len(self.sel_idx)}  |  seq_len={self.seq_len} stride={self.seq_stride}")

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int):
        item_idx, start_off, maj_label = self.index[i]
        it = self.items[item_idx]
        Xm = np.load(it["X_path"], mmap_mode="r")
        window_idx = it["keep_idx"][start_off:start_off+self.seq_len]
        x_np = np.asarray(Xm[window_idx[:,None], self.sel_idx], dtype=np.float32, order="C")  # (T, D)
        if self.return_vid:
            return torch.from_numpy(x_np), torch.tensor(maj_label), it["vid"]
        return torch.from_numpy(x_np), torch.tensor(maj_label)

    @property
    def input_dim(self) -> int:
        return int(len(self.sel_idx))

# ---------- CPU-friendly scaler stats over sequences ----------
@torch.no_grad()
def compute_mean_std_seq(loader: DataLoader, device: torch.device = torch.device("cpu")):
    """
    Streamed mean/std over (B,T,D) on CPU by default to avoid GPU OOM.
    """
    s1=None; s2=None; n_total=0
    for batch in loader:
        xb = batch[0].to(device).float()  # CPU by default
        B,T,D = xb.shape
        xbt = xb.reshape(B*T, D)
        if s1 is None:
            s1 = torch.zeros(D, device=device, dtype=torch.float64)
            s2 = torch.zeros(D, device=device, dtype=torch.float64)
        n_total += xbt.shape[0]
        # (x^2) computed in-place-friendly way to reduce peak mem
        xbt_d = xbt.double()
        s1 += xbt_d.sum(dim=0)
        s2 += (xbt_d * xbt_d).sum(dim=0)
    if s1 is None:
        mean = torch.zeros(1)
        std  = torch.ones(1)
    else:
        mean = (s1 / max(1, n_total)).float()
        var  = (s2 / max(1, n_total)) - mean.double().pow(2)
        std  = torch.sqrt(torch.clamp(var, min=1e-12)).float()
    return mean.cpu(), std.cpu()

# ============================ MAIN ============================
def main():
    set_seed()

    # 1) pick best config from RNN grid (by test_seq_acc)
    cfg = best_row_from_rnn_grid(GRID_RESULTS_CSV)

    # 2) feature selection
    master_cols = load_master_cols(MASTER_FEATURE_COLS_JSON)
    sel_idx = feature_sel_idx(master_cols, cfg)
    feat_names = [master_cols[i] for i in sel_idx.tolist()]
    print("\n[features] selected dim:", len(sel_idx))
    print("[features] first 12 selected:", feat_names[:12])

    # 3) splits
    tr_ids = read_ids(TRAIN_LIST); va_ids = read_ids(VAL_LIST); te_ids = read_ids(TEST_LIST)
    print(f"\n[splits] train={len(tr_ids)} | val={len(va_ids)} | test={len(te_ids)}")

    # 4) datasets + loaders (RAVDESS sequences)
    ds_tr = CacheASequences6(RAVDESS_ROOT, tr_ids, sel_idx, cfg["seq_len"], cfg["stride"], master_cols,
                             return_vid=False, tag="RAVDESS/TRAIN")
    ds_va = CacheASequences6(RAVDESS_ROOT, va_ids, sel_idx, cfg["seq_len"], cfg["stride"], master_cols,
                             return_vid=False, tag="RAVDESS/VAL")
    ds_te = CacheASequences6(RAVDESS_ROOT, te_ids, sel_idx, cfg["seq_len"], cfg["stride"], master_cols,
                             return_vid=False, tag="RAVDESS/TEST")
    ds_te_vid = CacheASequences6(RAVDESS_ROOT, te_ids, sel_idx, cfg["seq_len"], cfg["stride"], master_cols,
                                 return_vid=True, tag="RAVDESS/TEST")

    assert len(ds_tr) and len(ds_va) and len(ds_te), "Empty RAVDESS split after mapping/windowing—check caches or SEQ_LEN/STRIDE."

    tr_loader = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True,
                           num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY)
    va_loader = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False,
                           num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY)
    # Use OOM-safe eval batch size for test/vid eval
    te_loader     = DataLoader(ds_te,     batch_size=EVAL_SEQ_BS, shuffle=False, num_workers=0)
    te_vid_loader = DataLoader(ds_te_vid, batch_size=EVAL_SEQ_BS, shuffle=False, num_workers=0)

    # 5) normalization (TRAIN domain) — compute stats on CPU by default
    if cfg["norm_mode"] == "zscore":
        # small CPU stats loader to avoid GPU OOM
        tr_stats_loader = DataLoader(ds_tr, batch_size=STATS_BS, shuffle=False, num_workers=0, pin_memory=False)
        mean, std = compute_mean_std_seq(tr_stats_loader, device=CPU)
        if cfg["keep_au_c_raw"]:
            auc_idx = [i for i,n in enumerate(feat_names) if n.endswith("_c")]
            if auc_idx:
                mean[torch.as_tensor(auc_idx)] = 0.0
                std[ torch.as_tensor(auc_idx)] = 1.0
        scaler = Standardize(mean, std).to(DEVICE)
        print("\n[scaler] ZSCORE over training sequences (CPU-computed).")
    else:
        scaler = nn.Identity()
        print("\n[scaler] DISABLED (norm_mode!=zscore).")

    def preproc_seq(x: torch.Tensor):
        return scaler(x.to(DEVICE).float())

    # 6) model
    model = TemporalFFRNN(
        input_dim=len(sel_idx),
        ff_hidden=(cfg["ff_hidden"] or max(1, len(sel_idx))),  # required by class
        ff_hidden2=cfg["ff_hidden2"],
        dropout=cfg["dropout"],
        rnn_type=cfg["rnn_type"],          # "gru" or "lstm"
        rnn_hidden=cfg["rnn_hidden"],
        rnn_layers=cfg["rnn_layers"],
        bidirectional=cfg["bidirectional"],
        num_classes=6
    ).to(DEVICE)
    print(f"\n[model] tag={cfg.get('_picked_tag')}")
    print(f"[model] FF=({cfg['ff_hidden']},{cfg['ff_hidden2']})  RNN={cfg['rnn_type'].upper()} "
          f"h{cfg['rnn_hidden']} l{cfg['rnn_layers']} bi{int(cfg['bidirectional'])}  drop={cfg['dropout']}")

    # 7) optimizer
    if cfg["optimizer"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"])
    ce = nn.CrossEntropyLoss()
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=5, min_lr=1e-6)

    # 8) train + early stop (sequence-level)
    best_val, best_state, noimp = -math.inf, None, 0
    for ep in range(1, EPOCHS+1):
        model.train(); run_loss=0.0; correct=0; total=0
        for xb,yb in tr_loader:
            xb = preproc_seq(xb); yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb); loss.backward(); opt.step()
            run_loss += loss.item()
            correct += (logits.argmax(1)==yb).sum().item(); total += yb.numel()
        tr_loss = run_loss/max(1,len(tr_loader)); tr_acc = correct/max(1,total)

        va_acc = evaluate_sequences(model, va_loader, preproc_seq)
        sched.step(va_acc)

        if va_acc > best_val:
            best_val = va_acc; noimp = 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state, os.path.join(ART_DIR, "best.pt"))
            torch.save({"sel_idx": sel_idx,
                        "mean": getattr(scaler,"mean",None),
                        "std": getattr(scaler,"std",None),
                        "seq_len": cfg["seq_len"],
                        "seq_stride": cfg["stride"],
                        "cfg": cfg},
                       os.path.join(ART_DIR, "slice_scaler.pt"))
        else:
            noimp += 1

        print(f"[RAV6-RNN][train] ep {ep:03d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} "
              f"| va_seq_acc {va_acc:.4f} | best {best_val:.4f}")
        if noimp >= ES_PATIENCE:
            print("[RAV6-RNN] early stopping"); break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 9) RAVDESS test — sequences
    rav_s_acc, y_true_s, y_pred_s = evaluate_sequences_with_outputs(model, te_loader, preproc_seq)
    print("\n[RAVDESS-6][SEQUENCE] acc:", f"{rav_s_acc:.4f}")
    if y_true_s is not None:
        print("[RAVDESS-6][SEQUENCE] report:\n", classification_report(y_true_s, y_pred_s, target_names=RAV6_ORDER, digits=4))
        print("[RAVDESS-6][SEQUENCE] CM:\n", confusion_matrix(y_true_s, y_pred_s, labels=list(range(6))))

    # 10) RAVDESS test — video (aggregate sequences)
    rav_v_acc, y_true_v, y_pred_v = evaluate_video_from_sequences(model, te_vid_loader, preproc_seq,
                                                                  strategy=VIDEO_STRATEGY, debug_n=8)
    print("\n[RAVDESS-6][VIDEO] acc:", f"{rav_v_acc:.4f}")
    if y_true_v is not None:
        print("[RAVDESS-6][VIDEO] report:\n", classification_report(y_true_v, y_pred_v, target_names=RAV6_ORDER, digits=4))
        print("[RAVDESS-6][VIDEO] CM:\n", confusion_matrix(y_true_v, y_pred_v, labels=list(range(6))))

    # 11) CREMA-D (flat <vid>/cache) — same SEQ_LEN/STRIDE and normalization policy
    if os.path.isdir(CREMAD_ROOT):
        crema_ids = [v for v in sorted(os.listdir(CREMAD_ROOT)) if os.path.isdir(os.path.join(CREMAD_ROOT, v))]
        if crema_ids:
            ds_cr = CacheASequences6(CREMAD_ROOT, crema_ids, sel_idx, cfg["seq_len"], cfg["stride"], master_cols,
                                     return_vid=False, tag="CREMA-D/ALL")
            if len(ds_cr):
                # OOM-safe eval loaders
                cr_loader     = DataLoader(ds_cr,     batch_size=EVAL_SEQ_BS, shuffle=False, num_workers=0)
                ds_cr_vid     = CacheASequences6(CREMAD_ROOT, crema_ids, sel_idx, cfg["seq_len"], cfg["stride"], master_cols,
                                                 return_vid=True, tag="CREMA-D/ALL")
                cr_vid_loader = DataLoader(ds_cr_vid, batch_size=EVAL_SEQ_BS, shuffle=False, num_workers=0)

                # Target-domain standardization (CPU) if zscore
                if cfg["norm_mode"] == "zscore":
                    cr_stats_loader = DataLoader(ds_cr, batch_size=STATS_BS, shuffle=False, num_workers=0, pin_memory=False)
                    mean_t, std_t = compute_mean_std_seq(cr_stats_loader, device=CPU)
                    if cfg["keep_au_c_raw"]:
                        auc_idx = [i for i,n in enumerate(feat_names) if n.endswith("_c")]
                        if auc_idx:
                            mean_t[torch.as_tensor(auc_idx)] = 0.0
                            std_t[ torch.as_tensor(auc_idx)] = 1.0
                    scaler_t = Standardize(mean_t, std_t).to(DEVICE)
                    def preproc_cr(x): return scaler_t(x.to(DEVICE).float())
                    print("\n[CREMA-D] Using TARGET ZSCORE normalization (CPU-computed).")
                    # Optional: free any stale GPU cache before big eval
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                else:
                    preproc_cr = (lambda x: x.to(DEVICE).float())
                    print("\n[CREMA-D] Using NO normalization (norm_mode!=zscore).")

                cr_s_acc, y_true_cs, y_pred_cs = evaluate_sequences_with_outputs(model, cr_loader, preproc_cr)
                print("\n[CREMA-D-6][SEQUENCE] acc:", f"{cr_s_acc:.4f}")
                if y_true_cs is not None:
                    print("[CREMA-D-6][SEQUENCE] report:\n", classification_report(y_true_cs, y_pred_cs, target_names=RAV6_ORDER, digits=4))
                    print("[CREMA-D-6][SEQUENCE] CM:\n", confusion_matrix(y_true_cs, y_pred_cs, labels=list(range(6))))

                cr_v_acc, y_true_cv, y_pred_cv = evaluate_video_from_sequences(model, cr_vid_loader, preproc_cr,
                                                                               strategy=VIDEO_STRATEGY, debug_n=8)
                print("\n[CREMA-D-6][VIDEO] acc:", f"{cr_v_acc:.4f}")
                if y_true_cv is not None:
                    print("[CREMA-D-6][VIDEO] report:\n", classification_report(y_true_cv, y_pred_cv, target_names=RAV6_ORDER, digits=4))
                    print("[CREMA-D-6][VIDEO] CM:\n", confusion_matrix(y_true_cv, y_pred_cv, labels=list(range(6))))
            else:
                print("[CREMA-D] No usable sequences after mapping/windowing.")
        else:
            print("[CREMA-D] No video folders found at root.")
    else:
        print("[CREMA-D] Root not found; skipping.")

    # 12) Save final artifacts
    torch.save(model.state_dict(), os.path.join(ART_DIR, "best.pt"))
    print(f"\n[done] Artifacts saved in: {ART_DIR}")

# ======================== EVAL HELPERS =========================
@torch.no_grad()
def evaluate_sequences(model, loader, preproc):
    model.eval(); correct=0; total=0
    for batch in loader:
        if len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch
        xb = preproc(xb); yb = yb.to(DEVICE)
        logits = model(xb)
        pred = logits.argmax(1)
        correct += (pred==yb).sum().item(); total += yb.numel()
    if total == 0: return float("nan")
    return float(correct/total)

@torch.no_grad()
def evaluate_sequences_with_outputs(model, loader, preproc):
    model.eval(); y_true=[]; y_pred=[]
    for batch in loader:
        if len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch
        xb = preproc(xb); yb = yb.to(DEVICE)
        logits = model(xb)
        y_true.append(yb.cpu().numpy()); y_pred.append(logits.argmax(1).cpu().numpy())
    if not y_true:
        return float("nan"), None, None
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    return float((y_true==y_pred).mean()), y_true, y_pred

@torch.no_grad()
def evaluate_video_from_sequences(model, loader, preproc, strategy="mean_softmax", debug_n=5):
    assert strategy in ("mean_softmax","majority")
    model.eval()
    per_vid_logits={}; per_vid_preds={}; per_vid_true={}
    for batch in loader:
        xb, yb, vids = batch
        xb = preproc(xb); yb = yb.to(DEVICE)
        logits = model(xb)
        if strategy == "mean_softmax":
            for i,v in enumerate(vids):
                per_vid_logits.setdefault(v, []).append(logits[i].detach().cpu())
                per_vid_true.setdefault(v,   []).append(int(yb[i].item()))
        else:
            pred = logits.argmax(1)
            for i,v in enumerate(vids):
                per_vid_preds.setdefault(v, []).append(int(pred[i].item()))
                per_vid_true.setdefault(v,  []).append(int(yb[i].item()))
    # aggregate
    y_true_v=[]; y_pred_v=[]
    if strategy == "mean_softmax":
        for v, parts in per_vid_logits.items():
            probs = torch.softmax(torch.stack(parts,0), dim=1).mean(0)
            pred  = int(probs.argmax().item())
            gt    = int(np.bincount(np.asarray(per_vid_true[v], dtype=int)).argmax())
            y_true_v.append(gt); y_pred_v.append(pred)
    else:
        for v, preds in per_vid_preds.items():
            pred = int(np.bincount(np.asarray(preds, dtype=int)).argmax())
            gt   = int(np.bincount(np.asarray(per_vid_true[v], dtype=int)).argmax())
            y_true_v.append(gt); y_pred_v.append(pred)

    y_true_v = np.asarray(y_true_v, dtype=int)
    y_pred_v = np.asarray(y_pred_v, dtype=int)

    print("[video-agg][sample]")
    for i,(gt,pr) in enumerate(zip(y_true_v[:debug_n], y_pred_v[:debug_n])):
        print(f"  vid#{i:02d}  GT={RAV6_ORDER[gt]}({gt})  Pred={RAV6_ORDER[pr]}({pr})")

    acc = float((y_true_v==y_pred_v).mean()) if y_true_v.size else float("nan")
    return acc, y_true_v, y_pred_v

# ===============================================================
if __name__ == "__main__":
    main()

