# RESNET_AU_VGG_SimpleModel_from_npy.py
# Evaluate a FrameClassifier (MLP) on RAVDESS using per-video .npy caches.
# - cache/X_all.npy : ALL features in MASTER order (from MASTER_FEATURE_COLS_JSON)
# - cache/y_str.npy : original label strings (fixed-width unicode, memmappable)
# - At runtime, we slice columns to ONLY those present in trained feature_cols.json

import os, json, numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# seaborn is optional (skip plots if not installed)
try:
    import seaborn as sns
except Exception:
    sns = None

from typing import List
from models.TwoLayerMLP import FrameClassifier

# ---------------- CONFIG: EDIT THESE ----------------
RAVDESS_ROOT = "/media/root918/OS/MaryiamProject/copiedFilesRAVDESS/"   # contains Actor_xx/<vid_dir>/<CSV>
ART_DIR      = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/cremad_GridSearch_unscaled_MLP/bestModel_std/100_epochs_video_best/"
WEIGHTS      = os.path.join(ART_DIR, "best.pt")
FEATCOLS     = os.path.join(ART_DIR, "feature_cols.json")       # trained feature list
SCALER_P     = os.path.join(ART_DIR, "scaler.pt")               # optional if NORMALIZE="train"

# MLP architecture USED DURING TRAINING (write these explicitly)
# frames_best = video best config
MLP_CFG = {
    "hidden_dim": 1024,
    "hidden_dim2": 512,        # 0 or None => 1 layer; >0 => 2 layers
    "dropout": 0.5,
    "num_classes": 6,          # CREMA-D 6 classes
}

# Master list = ALL features to cache (order used to write X_all.npy)
MASTER_FEATURE_COLS_JSON = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/master_feature_cols.json"

# RAVDESS CSV naming
COMBINED_CSV_NAME   = "au_resnet_vgg_with_gt.csv"
LABEL_COL_PREFERRED = "emotion"

# Normalization
NORMALIZE      = "target"   # "train" | "target" | "none"
KEEP_AU_C_RAW  = True       # when computing target scaler, keep *_c columns unscaled

# Batching / seed
BATCH = 4096
SEED  = 42
# ---------------------------------------------------

# ----- Label mapping: RAVDESS -> 6-class CREMA-D -----
_NAME_TO_LETTER = {
    "neutral": "N",
    "happy":   "H",
    "sad":     "S",
    "angry":   "A",
    "anger":   "A",
    "fearful": "F",
    "fear":    "F",
    "disgust": "D",
    # calm/surprised not mapped -> dropped
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

# ----------- utils: paths, cache, safe label loading -----------
def _require_file(path, desc):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")
    return path

def list_ravdess_video_ids(root_dir: str) -> List[str]:
    ids: List[str] = []
    actors = sorted(d for d in os.listdir(root_dir) if d.startswith("Actor_"))
    for actor in actors:
        ap = os.path.join(root_dir, actor)
        if not os.path.isdir(ap): continue
        for vid in sorted(os.listdir(ap)):
            vp = os.path.join(ap, vid)
            if os.path.isdir(vp):
                ids.append(f"{actor}/{vid}")
    if not ids:
        raise RuntimeError(f"No videos found under {root_dir}")
    return ids

def _cache_paths(root: str, vid_id: str):
    vdir = os.path.join(root, vid_id)
    cdir = os.path.join(vdir, "cache")
    os.makedirs(cdir, exist_ok=True)
    return (
        os.path.join(cdir, "X_all.npy"),   # ALL features in master order
        os.path.join(cdir, "y_str.npy"),   # original labels (U32)
        os.path.join(cdir, "meta.json"),   # stores "feature_cols_master"
    )

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
            tmp = Yp + ".tmp"; np.save(tmp, y); os.replace(tmp, Yp)
        except Exception:
            pass
    return y

def load_feature_cols(path) -> list:
    with open(path) as f:
        return json.load(f)

def build_cache_for_video(root, vid_id, master_feature_cols, csv_name, label_col_preferred, skip_first_n=0):
    Xp, Yp, Mp = _cache_paths(root, vid_id)
    if os.path.isfile(Xp) and os.path.isfile(Yp) and os.path.isfile(Mp):
        try:
            meta = json.load(open(Mp))
            if meta.get("feature_cols_master") == master_feature_cols:
                return  # cache matches current master; reuse
        except Exception:
            pass  # fall-through to rebuild

    csvp = os.path.join(root, vid_id, csv_name)
    if not os.path.isfile(csvp):
        print(f"[cache][skip] missing CSV for {vid_id}")
        return

    try:
        df = pd.read_csv(csvp)
    except Exception as e:
        print(f"[cache][skip] cannot read {csvp}: {e}")
        return

    missing = [c for c in master_feature_cols if c not in df.columns]
    if missing:
        print(f"[cache][skip] {vid_id}: missing {len(missing)} master features")
        return

    feats = (df[master_feature_cols]
             .replace([np.inf, -np.inf], np.nan)
             .fillna(0.0)
             .astype("float32"))

    if skip_first_n and len(feats) <= skip_first_n:
        print(f"[cache][skip] {vid_id}: too few frames after skip")
        return
    if skip_first_n:
        feats = feats.iloc[skip_first_n:].reset_index(drop=True)
        df    = df.iloc[skip_first_n:].reset_index(drop=True)

    use_lbl = label_col_preferred if label_col_preferred in df.columns else ("emotion" if "emotion" in df.columns else None)
    if use_lbl is None:
        print(f"[cache][skip] {vid_id}: no label col")
        return

    np.save(Xp, feats.to_numpy(copy=False))
    y_str = df[use_lbl].astype(str).to_numpy(dtype="U32", copy=False)
    np.save(Yp, y_str)
    with open(Mp, "w") as f:
        json.dump({"feature_cols_master": master_feature_cols, "label_source_col": use_lbl}, f)

def ensure_cache(root, video_ids, master_feature_cols, csv_name, label_col_preferred, skip_first_n=0):
    for i, vid in enumerate(video_ids):
        if i % 50 == 0:
            print(f"[cache] processed {i}/{len(video_ids)}")
        build_cache_for_video(root, vid, master_feature_cols, csv_name, label_col_preferred, skip_first_n)

# --------- Dataset from cache (frame-level for MLP) ----------
import bisect

class FrameDatasetFromCache(Dataset):
    def __init__(self, root, video_ids, col_idx):
        self.root = root
        self.vids = list(video_ids)
        self.col_idx = np.asarray(col_idx, dtype=np.int64)
        if self.col_idx.size == 0:
            raise ValueError("col_idx is empty — no overlap between trained features and master list.")

        # items: (vid, N_frames)
        self.items = []
        cum = [0]  # cumulative start indices
        total = 0

        for vid in self.vids:
            Xp, Yp, _ = _cache_paths(self.root, vid)
            if not (os.path.isfile(Xp) and os.path.isfile(Yp)):
                continue
            try:
                N = np.load(Xp, mmap_mode="r").shape[0]
            except Exception:
                continue
            if N <= 0:
                continue
            self.items.append((vid, N))
            total += N
            cum.append(total)

        self.cum = np.array(cum, dtype=np.int64)   # len = #videos + 1
        self.total_frames = int(total)
        self.input_dim = int(self.col_idx.size)

        print(f"[frame-ds] videos={len(self.items)} | frames={self.total_frames} | dim={self.input_dim}")

    def __len__(self):
        return self.total_frames

    def _locate(self, index: int):
        # find video k such that cum[k] <= index < cum[k+1]
        k = bisect.bisect_right(self.cum, index) - 1
        if k < 0 or k >= len(self.items):
            raise IndexError
        vid, N = self.items[k]
        offset = index - self.cum[k]
        if offset < 0 or offset >= N:
            raise IndexError
        return vid, offset

    def __getitem__(self, index: int):
        vid, offset = self._locate(index)

        Xp, Yp, _ = _cache_paths(self.root, vid)
        Xm = np.load(Xp, mmap_mode="r")           # (N, D_master)
        y  = _load_labels_memsafe(Yp)             # (N,) U32 or memmap

        # select trained features; copy to make tensor writable/contiguous
        x = Xm[offset, self.col_idx].astype(np.float32, copy=True)
        y_idx = _label_to_idx(y[offset])          # may be -1 for unmapped

        return torch.from_numpy(x), torch.tensor(y_idx, dtype=torch.long)

# --------- Scaler + target stats ----------
class Standardize(nn.Module):
    def __init__(self, mean, std, eps=1e-6):
        super().__init__()
        self.register_buffer("mean", mean.clone().detach())
        self.register_buffer("std",  std.clone().detach())
        self.eps = eps
    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)

def load_scaler_from_training(scaler_path: str, device: torch.device):
    d = torch.load(scaler_path, map_location="cpu")
    return Standardize(d["mean"], d["std"]).to(device)

@torch.inference_mode()
def compute_target_mean_std(ds: FrameDatasetFromCache, device, batch_size=16384):
    """
    Compute mean/std on CPU for stability; move the resulting tensors to `device`.
    Only considers frames with mapped labels (yb >= 0) to match evaluation filtering.
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    cnt = 0
    mean = None
    m2 = None
    for xb, yb in loader:
        mask = (yb >= 0)
        if not mask.any():
            continue
        x = xb[mask].float()  # stay on CPU for stats
        cnt_b = x.shape[0]
        b_mean = x.mean(dim=0)
        b_var  = x.var(dim=0, unbiased=False)

        if mean is None:
            mean = b_mean
            m2   = b_var * cnt_b
            cnt  = cnt_b
        else:
            delta = b_mean - mean
            total = cnt + cnt_b
            mean = mean + delta * (cnt_b / max(1, total))
            m2   = m2 + b_var * cnt_b + (delta**2) * (cnt * cnt_b / max(1, total))
            cnt  = total

    if mean is None:
        d = ds.input_dim
        mean = torch.zeros(d)
        std  = torch.ones(d)
    else:
        var = m2 / max(1, cnt)
        std = torch.sqrt(torch.clamp(var, min=1e-8))
    return mean.to(device), std.to(device)

# --------- Eval helpers ----------
@torch.inference_mode()
def eval_frame_level(model, scaler, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device)
        mask = (yb >= 0)
        if not mask.any():
            continue
        xb = xb[mask]; yb = yb[mask]
        xb = scaler(xb)
        logits = model(xb)
        y_true.append(yb.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    if y_true:
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
    else:
        y_true = np.array([], dtype=int); y_pred = np.array([], dtype=int)
    return y_true, y_pred

@torch.inference_mode()
def eval_video_level(model, scaler, root, video_ids, trained_feature_cols, master_feature_cols, device, batch=16384):
    name_to_idx = {n: i for i, n in enumerate(master_feature_cols)}
    col_idx = np.array([name_to_idx[n] for n in trained_feature_cols if n in name_to_idx], dtype=np.int64)

    y_true_v, y_pred_v = [], []
    for vid in video_ids:
        Xp, Yp, _ = _cache_paths(root, vid)
        if not (os.path.isfile(Xp) and os.path.isfile(Yp)):
            continue
        X_all = np.load(Xp, mmap_mode="r")
        y_str = _load_labels_memsafe(Yp)
        N = min(len(X_all), len(y_str))
        if N == 0:
            continue

        X = X_all[:N, :][:, col_idx]
        y = np.fromiter((_label_to_idx(s) for s in y_str[:N]), dtype=np.int64)
        valid = (y >= 0)
        if not np.any(valid):
            continue

        preds = []
        for i in range(0, N, batch):
            xb_np = np.ascontiguousarray(X[i:i+batch])
            xb = torch.from_numpy(xb_np).float().to(device)
            msk = torch.from_numpy(valid[i:i+batch]).to(xb.device)
            if msk.any():
                xb = xb[msk]
                xb = scaler(xb)
                logits = model(xb)
                preds.append(logits.argmax(1).cpu().numpy())
        if not preds:
            continue
        pred_all = np.concatenate(preds)

        pred_idx = int(np.bincount(pred_all).argmax())
        vals, counts = np.unique(y[valid], return_counts=True)
        gt_idx = int(vals[np.argmax(counts)])

        y_true_v.append(gt_idx); y_pred_v.append(pred_idx)

    return np.array(y_true_v, dtype=int), np.array(y_pred_v, dtype=int)

def print_reports(y_true, y_pred, title="FRAME"):
    classes = [IDX_TO_EMO[i] for i in range(len(IDX_TO_EMO))]
    letter_to_name = {'H':'happy','S':'sad','A':'anger','N':'neutral','D':'disgust','F':'fearful'}
    disp_labels = [f"{c}:{letter_to_name[c]}" for c in classes]

    if y_true.size == 0:
        print(f"\n[{title}] No valid samples. Skipping report.")
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        return cm, disp_labels

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    acc = (y_true == y_pred).mean() if y_true.size else float("nan")
    print(f"\n[{title}] Accuracy: {acc:.4f}")
    print(f"[{title}] Confusion Matrix (counts):\n{cm}")
    print(f"[{title}] Classification report:")
    print(classification_report(y_true, y_pred, target_names=disp_labels, digits=3))
    return cm, disp_labels

def plot_confusion_matrix(cm, labels, title="Confusion Matrix", save_path=None, normalize="true"):
    if sns is None:
        print(f"[plot] seaborn not available — skipping plot: {title}")
        return

    cm = cm.astype(np.float64)
    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
        data = (cm / np.clip(denom, 1, None)) * 100.0
        cbar_label = "Row %"
    elif normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
        data = (cm / np.clip(denom, 1, None)) * 100.0
        cbar_label = "Col %"
    elif normalize == "all":
        total = cm.sum()
        data = (cm / max(total, 1)) * 100.0
        cbar_label = "Overall %"
    else:
        data = cm; cbar_label = "Count"

    annot = np.empty_like(cm, dtype=object)
    show_pct = normalize in ("true", "pred", "all")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cnt = int(cm[i, j])
            annot[i, j] = f"{cnt}\n({data[i, j]:.1f}%)" if show_pct else f"{cnt}"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data, annot=annot, fmt="", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": cbar_label}, linewidths=0.5, linecolor="white", square=True,
    )
    plt.title(title, fontsize=16, pad=16)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()

# ------------------------------- MAIN --------------------------------
if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _require_file(WEIGHTS,  "best weights")
    _require_file(FEATCOLS, "feature_cols.json (trained)")
    _require_file(MASTER_FEATURE_COLS_JSON, "master_feature_cols.json")

    trained_feature_cols = load_feature_cols(FEATCOLS)
    master_feature_cols  = load_feature_cols(MASTER_FEATURE_COLS_JSON)

    # map trained features -> indices in the master X.npy
    master_index = {name: i for i, name in enumerate(master_feature_cols)}
    col_idx = np.array([master_index[n] for n in trained_feature_cols if n in master_index], dtype=np.int64)
    missing = [n for n in trained_feature_cols if n not in master_index]
    if missing:
        print(f"[warn] {len(missing)} trained features not found in master list. e.g. {missing[:5]}")
    if col_idx.size == 0:
        raise RuntimeError("No overlap between trained features and master feature list.")

    # Discover videos & build/verify caches (ALL features in MASTER order)
    video_ids = list_ravdess_video_ids(RAVDESS_ROOT)
    ensure_cache(RAVDESS_ROOT, video_ids, master_feature_cols, COMBINED_CSV_NAME, LABEL_COL_PREFERRED, skip_first_n=0)

    # Build frame-level dataset from cache (select trained columns)
    frame_ds = FrameDatasetFromCache(RAVDESS_ROOT, video_ids, col_idx)

    n_workers = min(8, max(2, (os.cpu_count() or 4) - 1))
    frame_loader = DataLoader(
        frame_ds, batch_size=BATCH, shuffle=False, drop_last=False,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(n_workers > 0)
    )

    # Build model with your explicit MLP config
    hidden2 = None if not MLP_CFG["hidden_dim2"] else int(MLP_CFG["hidden_dim2"])
    model = FrameClassifier(
        input_dim=len(col_idx),
        hidden_dim=int(MLP_CFG["hidden_dim"]),
        hidden_dim2=hidden2,
        dropout=float(MLP_CFG["dropout"]),
        num_classes=int(MLP_CFG["num_classes"]),
    ).to(device)

    # Load weights (strict)
    state = torch.load(WEIGHTS, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    load_ok = model.load_state_dict(state, strict=True)
    if getattr(load_ok, "missing_keys", []) or getattr(load_ok, "unexpected_keys", []):
        print("[warn] load_state_dict issues:",
              "missing=", getattr(load_ok, "missing_keys", []),
              "unexpected=", getattr(load_ok, "unexpected_keys", []))
    model.eval()

    # Scaler selection
    if NORMALIZE == "train":
        if os.path.isfile(SCALER_P):
            print("[scaler] Using TRAINING scaler (from artifacts).")
            scaler = load_scaler_from_training(SCALER_P, device)
        else:
            print("[scaler] scaler.pt not found -> using Identity.")
            scaler = nn.Identity()
    elif NORMALIZE == "target":
        print("[scaler] Computing TARGET mean/std on RAVDESS (selected columns)…")
        mean_rav, std_rav = compute_target_mean_std(frame_ds, device, batch_size=16384)
        scaler = Standardize(mean_rav, std_rav).to(device)
        # Optional: keep *_c raw (valid as long as no trained feature is missing from master)
        if KEEP_AU_C_RAW:
            au_c_idx = [i for i, name in enumerate(trained_feature_cols) if name.endswith("_c")]
            if au_c_idx:
                scaler.mean[au_c_idx] = 0.0
                scaler.std[au_c_idx]  = 1.0
    elif NORMALIZE == "none":
        print("[scaler] DISABLED (raw features).")
        scaler = nn.Identity()
    else:
        raise ValueError(f"Unknown NORMALIZE={NORMALIZE}")

    # sanity-check scaler shape (when Standardize is used)
    if isinstance(scaler, Standardize):
        assert scaler.mean.numel() == len(col_idx) and scaler.std.numel() == len(col_idx), \
            f"Scaler dim {scaler.mean.numel()} != input dim {len(col_idx)}"

    # ------------ Frame-level ------------
    print("\n" + "="*60)
    print(f"FRAME-LEVEL EVALUATION — NORMALIZE={NORMALIZE}")
    print("="*60)
    y_true_f, y_pred_f = eval_frame_level(model, scaler, frame_loader, device)
    cm_f, labels_f = print_reports(y_true_f, y_pred_f, title=f"FRAME ({NORMALIZE})")
    plot_confusion_matrix(
        cm_f, labels_f,
        title=f"Frame-Level CM — {NORMALIZE} (row %)",
        save_path=os.path.join(ART_DIR, f"ravdess_frame_cm_{NORMALIZE}.png"),
        normalize="true"
    )

    # ------------ Video-level ------------
    print("\n" + "="*60)
    print(f"VIDEO-LEVEL EVALUATION — NORMALIZE={NORMALIZE}")
    print("="*60)
    y_true_v, y_pred_v = eval_video_level(
        model, scaler, RAVDESS_ROOT, video_ids,
        trained_feature_cols, master_feature_cols, device, batch=16384
    )
    cm_v, labels_v = print_reports(y_true_v, y_pred_v, title=f"VIDEO ({NORMALIZE})")
    plot_confusion_matrix(
        cm_v, labels_v,
        title=f"Video-Level CM — {NORMALIZE} (row %)",
        save_path=os.path.join(ART_DIR, f"ravdess_video_cm_{NORMALIZE}.png"),
        normalize="true"
    )

    # ------------ Summary ------------
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    acc_f = (y_true_f == y_pred_f).mean() if y_true_f.size else float("nan")
    acc_v = (y_true_v == y_pred_v).mean() if y_true_v.size else float("nan")
    print(f"Frames evaluated: {len(y_true_f)} | Frame Acc: {acc_f:.4f}")
    print(f"Videos evaluated: {len(y_true_v)} | Video Acc: {acc_v:.4f}")

