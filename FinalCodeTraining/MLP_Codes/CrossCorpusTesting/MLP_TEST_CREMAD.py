#!/usr/bin/env python3
"""
Cache-only pipeline:
- Train 6-class MLP on RAVDESS (from <vid>/cache/ ... no CSVs)
- Eval on RAVDESS test (frame + video)
- Eval on CREMA-D (frame + video)

Assumptions for each video folder:
  <root>/<actor>/<vid>/cache/        [RAVDESS - nested]
  <root>/<vid>/cache/                 [CREMA-D  - flat]

Cache files accepted:
  - X_all.npy OR X.npy   (float32, ALL features in 'feature_cols_master' order)
  - y.npy (int) OR y_str.npy (unicode strings)
  - meta.json (optional) {"feature_cols_master": [...]}

If RAVDESS y are 8-class ints, we map to 6-class.
If RAVDESS/CREMA-D y are strings, we map by name to 6-class.

6-class order & indices (CREMA-compatible):
  0: happy, 1: sad, 2: anger, 3: neutral, 4: disgust, 5: fearful
"""

import os, json, math, random
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# ===================== PATHS / PARAMS (edit) =====================
RAVDESS_ROOT = "/media/root918/OS/MaryiamProject/copiedFilesRAVDESS"
CREMAD_ROOT  = "/media/root918/OS/MaryiamProject/CREMA-D/copiedFiles/"  # flat: <vid>/cache/

SPLIT_PATH = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data"
TRAIN_LIST = os.path.join(SPLIT_PATH, "train_videos_RAV.txt")
VAL_LIST   = os.path.join(SPLIT_PATH, "val_videos_RAV.txt")
TEST_LIST  = os.path.join(SPLIT_PATH, "test_videos_RAV.txt")

GRID_RESULTS_CSV         = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/ravdess_GridSearch_unscaled_MLP/grid_RAV/grid_results_val_train_test.csv"
MASTER_FEATURE_COLS_JSON = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/master_feature_cols.json"

ART_DIR = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/ravdess6class_train2"
os.makedirs(ART_DIR, exist_ok=True)

# training/eval
EPOCHS      = 300
ES_PATIENCE = 30
BATCH_TR    = 512
BATCH_EVAL  = 16384
SEED        = 1337
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DO_STANDARDIZE     = False
KEEP_AU_C_RAW      = True   # leaves *_c features unscaled
CREMAD_NORMALIZE   = "none"   # "target" | "train" | "none"
VIDEO_STRATEGY     = "mean_softmax"   # "mean_softmax" | "majority"
NUM_WORKERS_TRAIN  = min(8, (os.cpu_count() or 2))
PIN_MEMORY         = torch.cuda.is_available()
# ================================================================

# ===== Reuse your helpers =====
from models.TwoLayerMLP import FrameClassifier
from utils.features import Standardize
# ==============================

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

def best_cfg_from_csv(csv_path: str) -> Dict:
    df = pd.read_csv(require_file(csv_path, "grid results"))
    for col in ["best_val_acc", "test_video_acc", "test_frame_acc"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    row = df.sort_values(["best_val_acc","test_video_acc","test_frame_acc"],
                         ascending=[False, False, False]).iloc[0].to_dict()
    def norm_h2(x):
        if pd.isna(x): return None
        s = str(x).strip().lower()
        return None if s in ("", "none", "nan") else int(round(float(x)))
    cfg = {
        "use_vgg":      bool(row["use_vgg"]),
        "use_resnet":   bool(row["use_resnet"]),
        "use_au_c":     bool(row["use_au_c"]),
        "use_au_r":     bool(row["use_au_r"]),
        "hidden_dim":   int(round(float(row["hidden_dim"]))),
        "hidden_dim2":  norm_h2(row.get("hidden_dim2", None)),
        "optimizer":    str(row["optimizer"]),
        "lr":           float(row["lr"]),
        "weight_decay": float(row["weight_decay"]),
        "dropout":      float(row["dropout"]),
        "batch_size":   int(round(float(row.get("batch_size", BATCH_TR)))),
    }
    print("\n[best-config from grid csv]")
    for k,v in cfg.items(): print(f"  {k}: {v}")
    return cfg

def feature_sel_idx(master_cols: List[str], cfg: Dict) -> np.ndarray:
    idx=[]
    for i,n in enumerate(master_cols):
        if n.endswith("_vgg")    and cfg["use_vgg"]:    idx.append(i)
        if n.endswith("_resnet") and cfg["use_resnet"]: idx.append(i)
        if n.endswith("_c")      and cfg["use_au_c"]:   idx.append(i)
        if n.endswith("_r")      and cfg["use_au_r"]:   idx.append(i)
    idx = np.asarray(sorted(set(idx)), dtype=np.int64)
    if idx.size == 0:
        raise ValueError("Feature selection is empty.")
    return idx

# ---------- RAVDESS (nested) ----------
def list_video_ids(root: str) -> List[str]:
    vids=[]
    for a in sorted(os.listdir(root)):
        ap = os.path.join(root, a)
        if not os.path.isdir(ap): continue
        for v in sorted(os.listdir(ap)):
            if os.path.isdir(os.path.join(ap, v)):
                vids.append(f"{a}/{v}")
    return vids

# ---------- CREMA-D (flat) ----------
def list_cremad_video_ids(root: str) -> List[str]:
    vids=[]
    for v in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, v)):
            vids.append(v)  # each immediate subdir is a video folder
    return vids

# ---------- dataset from cache/<X_all|X.npy,y_*|y_str.npy> ----------
def detect_cacheA(root: str, vid: str) -> Optional[Dict]:
    """
    Accepts:
      - X_all.npy OR X.npy
      - y.npy (ints) OR y_str.npy (strings)
      - meta.json optional
    """
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

class CacheAFrames6(Dataset):
    """Reads frames from <root>/<vid>/cache/*. Filters to 6-class; can return vid for video eval."""
    def __init__(self, root: str, video_ids: List[str], sel_idx: np.ndarray, expect_master_cols: List[str],
                 return_vid: bool = False, tag: str = "RAVDESS"):
        self.root = root
        self.vids = list(video_ids)
        self.sel_idx = np.asarray(sel_idx, dtype=np.int64)
        self.return_vid = bool(return_vid)
        self.tag = tag

        self.items = []   # per-video metadata
        self.index = []   # (item_idx, frame_offset)

        # counters for sanity
        cls_counts = np.zeros(6, dtype=np.int64)
        mode_counts = {"int":0, "str":0}
        frame_kept_total = 0

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

            # verify meta feature order if present
            mcols = det.get("feature_cols_master", det.get("feature_cols", None))
            if isinstance(mcols, list) and len(mcols) != Xm.shape[1]:
                print(f"[warn][{self.tag}] {vid} meta feature_cols len {len(mcols)} != X shape {Xm.shape[1]}")

            # label mapping
            if yk == "int":
                arr = Ym[:n].astype(np.int64, copy=False)
                if arr.max() > 5:   # e.g., RAVDESS 8-class -> map to 6
                    y6 = np.fromiter((idx8_to_idx6(int(i)) for i in arr), dtype=np.int64)
                    mode_counts["int"] += 1
                else:
                    y6 = arr
                    mode_counts["int"] += 1
            else:
                y6 = np.fromiter((name_to_idx6(str(s)) for s in Ym[:n]), dtype=np.int64)
                mode_counts["str"] += 1

            keep = np.nonzero(y6 >= 0)[0]
            if keep.size == 0:
                continue

            # add index
            start = len(self.index)
            for off in keep.tolist():
                self.index.append((len(self.items), off))
                cls_counts[y6[off]] += 1
            end = len(self.index)
            frame_kept_total += keep.size

            self.items.append({
                "vid": vid, "X_path": Xp, "Y_path": Yp, "y_map": y6,
            })

        print(f"\n[{self.tag}] frames kept (after 6-class filter): {frame_kept_total:,}")
        print(f"[{self.tag}] label counts (H,S,A,N,D,F): {cls_counts.tolist()}")
        print(f"[{self.tag}] source label kinds used: {mode_counts}")
        print(f"[{self.tag}] selected feature dim = {len(self.sel_idx)}")

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int):
        item_idx, off = self.index[i]
        it = self.items[item_idx]
        Xm = np.load(it["X_path"], mmap_mode="r")
        x_np = np.asarray(Xm[off, self.sel_idx], dtype=np.float32, order="C")
        y = int(it["y_map"][off])
        if self.return_vid:
            return torch.from_numpy(x_np), torch.tensor(y), self.items[item_idx]["vid"]
        return torch.from_numpy(x_np), torch.tensor(y)

    @property
    def input_dim(self) -> int:
        return int(len(self.sel_idx))

# scaler stats
@torch.no_grad()
def compute_mean_std(loader: DataLoader, device=DEVICE):
    s1=None; s2=None; n_total=0
    for batch in loader:
        xb = batch[0].to(device).float()
        if s1 is None:
            D = xb.shape[1]
            s1 = torch.zeros(D, device=device, dtype=torch.float64)
            s2 = torch.zeros(D, device=device, dtype=torch.float64)
        n_total += xb.shape[0]
        s1 += xb.sum(dim=0).double()
        s2 += (xb.double().pow(2)).sum(dim=0)
    if s1 is None:
        mean = torch.zeros(1)
        std  = torch.ones(1)
    else:
        mean = (s1 / max(1, n_total)).float()
        var  = (s2 / max(1, n_total)) - mean.double().pow(2)
        std  = torch.sqrt(torch.clamp(var, min=1e-12)).float()
    return mean.cpu(), std.cpu()

# eval helpers
def evaluate_frame(model, loader, preproc):
    model.eval(); y_true=[]; y_pred=[]
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                xb, yb, _ = batch
            else:
                xb, yb = batch
            xb = preproc(xb.to(DEVICE).float()); yb = yb.to(DEVICE)
            logits = model(xb)
            y_true.append(yb.cpu().numpy()); y_pred.append(logits.argmax(1).cpu().numpy())
    if not y_true:
        return float("nan"), None, None
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    return float((y_true==y_pred).mean()), y_true, y_pred

@torch.no_grad()
def evaluate_video(model, loader, preproc, strategy="mean_softmax", debug_n=5):
    assert strategy in ("mean_softmax","majority")
    model.eval()
    per_vid_logits={}; per_vid_preds={}; per_vid_true={}
    for xb, yb, vids in loader:
        xb = preproc(xb.to(DEVICE).float()); yb = yb.to(DEVICE)
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

    # debug: show a few videos’ GT vs Pred
    print("[video-agg][sample]")
    for i,(gt,pr) in enumerate(zip(y_true_v[:debug_n], y_pred_v[:debug_n])):
        print(f"  vid#{i:02d}  GT={RAV6_ORDER[gt]}({gt})  Pred={RAV6_ORDER[pr]}({pr})")

    acc = float((y_true_v==y_pred_v).mean()) if y_true_v.size else float("nan")
    return acc, y_true_v, y_pred_v

# ============================ MAIN ============================
def main():
    set_seed()

    # 1) best config + master cols -> feature indices
    cfg = best_cfg_from_csv(GRID_RESULTS_CSV)
    master_cols = load_master_cols(MASTER_FEATURE_COLS_JSON)
    sel_idx = feature_sel_idx(master_cols, cfg)
    feat_names = [master_cols[i] for i in sel_idx.tolist()]
    print("\n[features] selected dim:", len(sel_idx))
    print("[features] first 12 selected:", feat_names[:12])

    # 2) splits
    tr_ids = read_ids(TRAIN_LIST); va_ids = read_ids(VAL_LIST); te_ids = read_ids(TEST_LIST)
    print(f"\n[splits] train={len(tr_ids)} | val={len(va_ids)} | test={len(te_ids)}")

    # 3) datasets + loaders (RAVDESS)
    ds_tr = CacheAFrames6(RAVDESS_ROOT, tr_ids, sel_idx, master_cols, return_vid=False, tag="RAVDESS/TRAIN")
    ds_va = CacheAFrames6(RAVDESS_ROOT, va_ids, sel_idx, master_cols, return_vid=False, tag="RAVDESS/VAL")
    ds_te = CacheAFrames6(RAVDESS_ROOT, te_ids, sel_idx, master_cols, return_vid=False, tag="RAVDESS/TEST")
    ds_te_vid = CacheAFrames6(RAVDESS_ROOT, te_ids, sel_idx, master_cols, return_vid=True, tag="RAVDESS/TEST")

    assert len(ds_tr) and len(ds_va) and len(ds_te), "Empty RAVDESS split after mapping—check caches."

    tr_loader = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True,
                           num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY)
    va_loader = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False,
                           num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY)
    te_loader = DataLoader(ds_te, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)
    te_vid_loader = DataLoader(ds_te_vid, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)

    # 4) scaler (TRAIN)
    if DO_STANDARDIZE:
        mean, std = compute_mean_std(tr_loader, device=DEVICE)
        # Keep *_c raw if we know which columns end with _c inside selection
        if KEEP_AU_C_RAW:
            # build mask by names
            auc_idx = [i for i,n in enumerate(feat_names) if n.endswith("_c")]
            if auc_idx:
                mean[torch.as_tensor(auc_idx)] = 0.0
                std[ torch.as_tensor(auc_idx)] = 1.0
        scaler = Standardize(mean, std).to(DEVICE)
        print("\n[scaler] mean/std computed on TRAIN (post-slice).")
    else:
        scaler = nn.Identity()
        print("\n[scaler] DISABLED.")

    def preproc(x: torch.Tensor):
        return scaler(x.to(DEVICE).float())

    # 5) model
    model = FrameClassifier(
        input_dim=len(sel_idx),
        hidden_dim=int(cfg["hidden_dim"]),
        hidden_dim2=(None if cfg["hidden_dim2"] in (None, 0) else int(cfg["hidden_dim2"])),
        dropout=float(cfg["dropout"]),
        num_classes=6
    ).to(DEVICE)
    print(f"\n[model] input_dim={len(sel_idx)}  hidden_dim={cfg['hidden_dim']}  hidden_dim2={cfg['hidden_dim2']}  dropout={cfg['dropout']}  num_classes=6")

    opt = (torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
           if cfg["optimizer"].lower()=="adam"
           else torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"]))
    ce = nn.CrossEntropyLoss()
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=5, min_lr=1e-6)

    # 6) train + early stop
    best_val, best_state, noimp = -math.inf, None, 0
    for ep in range(1, EPOCHS+1):
        model.train(); run_loss=0.0; correct=0; total=0
        for xb,yb in tr_loader:
            xb = preproc(xb); yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb); loss.backward(); opt.step()
            run_loss += loss.item()
            correct += (logits.argmax(1)==yb).sum().item(); total += yb.numel()
        tr_loss = run_loss/max(1,len(tr_loader)); tr_acc = correct/max(1,total)

        va_acc, _, _ = evaluate_frame(model, va_loader, preproc)
        sched.step(va_acc)

        if va_acc > best_val:
            best_val = va_acc; noimp = 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state, os.path.join(ART_DIR, "best.pt"))
            torch.save({"sel_idx": sel_idx, "mean": getattr(scaler,"mean",None), "std": getattr(scaler,"std",None)},
                       os.path.join(ART_DIR, "slice_scaler.pt"))
        else:
            noimp += 1

        print(f"[RAV6][train] ep {ep:03d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | va_acc {va_acc:.4f} | best {best_val:.4f}")
        if noimp >= ES_PATIENCE:
            print("[RAV6] early stopping"); break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 7) RAVDESS test — frame
    rav_f_acc, y_true_f, y_pred_f = evaluate_frame(model, te_loader, preproc)
    print("\n[RAVDESS-6][FRAME] acc:", f"{rav_f_acc:.4f}")
    if y_true_f is not None:
        print("[RAVDESS-6][FRAME] report:\n", classification_report(y_true_f, y_pred_f, target_names=RAV6_ORDER, digits=4))
        print("[RAVDESS-6][FRAME] CM:\n", confusion_matrix(y_true_f, y_pred_f, labels=list(range(6))))
        for i in range(min(8, y_true_f.shape[0])):
            print(f"[frame][sample {i}]  true={RAV6_ORDER[y_true_f[i]]}({y_true_f[i]})  pred={RAV6_ORDER[y_pred_f[i]]}({y_pred_f[i]})")

    # 8) RAVDESS test — video
    rav_vid_loader = te_vid_loader  # already built with return_vid=True
    rav_v_acc, y_true_v, y_pred_v = evaluate_video(model, rav_vid_loader, preproc, strategy=VIDEO_STRATEGY, debug_n=8)
    print("\n[RAVDESS-6][VIDEO] acc:", f"{rav_v_acc:.4f}")
    if y_true_v is not None:
        print("[RAVDESS-6][VIDEO] report:\n", classification_report(y_true_v, y_pred_v, target_names=RAV6_ORDER, digits=4))
        print("[RAVDESS-6][VIDEO] CM:\n", confusion_matrix(y_true_v, y_pred_v, labels=list(range(6))))

    # 9) CREMA-D (flat <vid>/cache)
    if os.path.isdir(CREMAD_ROOT):
        crema_ids = list_cremad_video_ids(CREMAD_ROOT)
        if crema_ids:
            ds_cr = CacheAFrames6(CREMAD_ROOT, crema_ids, sel_idx, master_cols, return_vid=False, tag="CREMA-D/ALL")
            if len(ds_cr):
                cr_loader = DataLoader(ds_cr, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)
                # normalization policy
                if CREMAD_NORMALIZE == "target" and DO_STANDARDIZE:
                    mean_t, std_t = compute_mean_std(cr_loader, device=DEVICE)
                    if KEEP_AU_C_RAW:
                        auc_idx = [i for i,n in enumerate(feat_names) if n.endswith("_c")]
                        if auc_idx:
                            mean_t[torch.as_tensor(auc_idx)] = 0.0
                            std_t[ torch.as_tensor(auc_idx)] = 1.0
                    scaler_t = Standardize(mean_t, std_t).to(DEVICE)
                    def preproc_cr(x): return scaler_t(x.to(DEVICE).float())
                    print("\n[CREMA-D] Using TARGET normalization.")
                elif CREMAD_NORMALIZE == "train":
                    preproc_cr = preproc
                    print("\n[CREMA-D] Using TRAIN normalization.")
                else:
                    def preproc_cr(x): return x.to(DEVICE).float()
                    print("\n[CREMA-D] Using NO normalization.")

                cr_f_acc, y_true_cf, y_pred_cf = evaluate_frame(model, cr_loader, preproc_cr)
                print("\n[CREMA-D-6][FRAME] acc:", f"{cr_f_acc:.4f}")
                if y_true_cf is not None:
                    print("[CREMA-D-6][FRAME] report:\n", classification_report(y_true_cf, y_pred_cf, target_names=RAV6_ORDER, digits=4))
                    print("[CREMA-D-6][FRAME] CM:\n", confusion_matrix(y_true_cf, y_pred_cf, labels=list(range(6))))
                    for i in range(min(8, y_true_cf.shape[0])):
                        print(f"[CREMA-D frame][sample {i}]  true={RAV6_ORDER[y_true_cf[i]]}({y_true_cf[i]})  pred={RAV6_ORDER[y_pred_cf[i]]}({y_pred_cf[i]})")

                # video-level on CREMA-D
                ds_cr_vid = CacheAFrames6(CREMAD_ROOT, crema_ids, sel_idx, master_cols, return_vid=True, tag="CREMA-D/ALL")
                cr_vid_loader = DataLoader(ds_cr_vid, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)
                cr_v_acc, y_true_cv, y_pred_cv = evaluate_video(model, cr_vid_loader, preproc_cr, strategy=VIDEO_STRATEGY, debug_n=8)
                print("\n[CREMA-D-6][VIDEO] acc:", f"{cr_v_acc:.4f}")
                if y_true_cv is not None:
                    print("[CREMA-D-6][VIDEO] report:\n", classification_report(y_true_cv, y_pred_cv, target_names=RAV6_ORDER, digits=4))
                    print("[CREMA-D-6][VIDEO] CM:\n", confusion_matrix(y_true_cv, y_pred_cv, labels=list(range(6))))
            else:
                print("[CREMA-D] No usable frames after mapping.")
        else:
            print("[CREMA-D] No video folders found at root.")
    else:
        print("[CREMA-D] Root not found; skipping.")

    # 10) save final artifacts
    torch.save(model.state_dict(), os.path.join(ART_DIR, "best.pt"))
    print(f"\n[done] Artifacts saved in: {ART_DIR}")

if __name__ == "__main__":
    main()

