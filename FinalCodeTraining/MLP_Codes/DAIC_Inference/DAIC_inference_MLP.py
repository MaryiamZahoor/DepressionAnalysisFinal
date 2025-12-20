# daic_infer_streamed.py
# Streamed DAIC-WOZ inference with target-domain standardization on GPU (no giant combined CSVs).
# - Computes global mean/std over ALL DAIC frames in the trained feature order (no concatenation).
# - Optionally keeps *_c AUs raw (mean=0, std=1) to mirror training.
# - Runs inference per subject in chunks; writes per-frame probability CSV + per-video summary JSON.

import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
# ---- Import your FrameClassifier ----
from models.TwoLayerMLP import FrameClassifier

# ---------- EDIT THESE ----------
# Artifacts from the SAME training run:
MODEL_PATH     = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/combined_GridSearch_unscaled_MLP/final_trainval_best_by_frameacc/best.pt"
FEATCOLS_JSON  = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/combined_GridSearch_unscaled_MLP/final_trainval_best_by_frameacc/feature_cols.json"    # list[str]: e.g., 'feat_0_resnet', ..., 'AU01_c', ...
# (No scaler: we compute target-domain stats here)

# DAIC paths
DAIC_ROOT      = "/media/root918/DATA/Projects/Maryiam_Projects/DepressionAnalysis/DAIC_WOZ/Data/"
LABELS_CSV     = "/media/root918/DATA/Projects/Maryiam_Projects/DepressionAnalysis/DAIC_WOZ/Labels/detailed_lables.csv"
OUT_DIR        = "/media/root918/OS/MaryiamProject/DAIC_RESULTS_MLP_UNSTD_COMBINED/"         # per-subject CSV + summary JSON
os.makedirs(OUT_DIR, exist_ok=True)

# MLP architecture USED DURING TRAINING (must match!)
MLP_CFG = {
    "hidden_dim": 1024,
    "hidden_dim2": 512,     # 0/None => 1 hidden layer; >0 => 2 layers
    "dropout": 0.5,
    "num_classes": 6,       # CREMA-D 6 classes (H,S,A,N,D,F)
}

# Normalization: target-domain (global across all DAIC frames) or disabled
DO_STANDARDIZE_TARGET = False
KEEP_AU_C_RAW         = True   # set mean=0, std=1 for *_c features after computing target stats

# Feature dims (change only if your .mat dims differ)
RESNET_DIM = 2048
VGG_DIM    = 4096

# Chunking (tune for your VRAM)
CHUNK_SIZE = 32768

# Label mapping (must match training))
#EMOTION_TO_IDX = {"H":0,"S":1,"A":2,"N":3,"D":4,"F":5}
#IDX_TO_EMO     = {v:k for k,v in EMOTION_TO_IDX.items()}

#do when using combined dataset
CANONICAL = ["angry","disgust","fear","happy","neutral","sad"]
EMOTION_TO_IDX = {e:i for i,e in enumerate(CANONICAL)}
IDX_TO_EMO     = {v:k for k,v in EMOTION_TO_IDX.items()}
# -------------------------------

# ---- Minimal Standardize (same API as your utils.features.Standardize) ----
class Standardize(nn.Module):
    def __init__(self, mean, std, eps=1e-6):
        super().__init__()
        self.register_buffer("mean", mean.clone().detach())
        self.register_buffer("std",  std.clone().detach())
        self.eps = eps
    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)

  # make sure this import path is correct

# ---- Load DAIC depression labels (optional) ----
def load_depression_labels(csv_path):
    if not os.path.isfile(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return dict(zip(df["Participant"].astype(str), df["Depression_label"]))

# ---- Subject discovery ----
def list_subjects(daic_root):
    subjects = []
    for dir_name in sorted(os.listdir(daic_root)):
        subj_dir = os.path.join(daic_root, dir_name)
        if not os.path.isdir(subj_dir):
            continue
        sid = dir_name.split("_")[0]  # e.g., "301_P" -> "301"
        subjects.append((sid, subj_dir))
    return subjects

# ---- Source loaders ----
def _load_resnet(subj_dir, sid):
    p = os.path.join(subj_dir, "features", f"{sid}_CNN_ResNet.mat")
    if not os.path.isfile(p): return None
    m = scipy.io.loadmat(p)
    X = m.get("feature", None)
    return None if X is None else X  # (N, 2048)

def _load_vgg(subj_dir, sid):
    p = os.path.join(subj_dir, "features", f"{sid}_CNN_VGG.mat")
    if not os.path.isfile(p): return None
    m = scipy.io.loadmat(p)
    X = m.get("feature", None)
    return None if X is None else X  # (N, 4096)

def _load_aus(subj_dir, sid):
    p = os.path.join(subj_dir, "features", f"{sid}_OpenFace2.1.0_Pose_gaze_AUs.csv")
    if not os.path.isfile(p): return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return None

# ---- Build trained feature order & mapping ----
def load_feature_cols(path):
    with open(path) as f:
        return json.load(f)

def build_feature_mapping(feature_cols, resnet_dim=2048, vgg_dim=4096):
    """
    Returns:
      name_to_pos: dict feature_name -> column index in trained order
      res_take, res_put, vgg_take, vgg_put: numpy arrays for slicing/placing
      au_used_names: list of AU names present in feature_cols (order: as in feature_cols)
      au_put: numpy array positions in final vector for those AUs
    """
    name_to_pos = {n: j for j, n in enumerate(feature_cols)}
    # resnet
    res_take = np.array([i for i in range(resnet_dim)
                         if f"feat_{i}_resnet" in name_to_pos], dtype=np.int64)
    res_put  = np.array([name_to_pos[f"feat_{i}_resnet"]
                         for i in res_take], dtype=np.int64)
    # vgg
    vgg_take = np.array([i for i in range(vgg_dim)
                         if f"feat_{i}_vgg" in name_to_pos], dtype=np.int64)
    vgg_put  = np.array([name_to_pos[f"feat_{i}_vgg"]
                         for i in vgg_take], dtype=np.int64)
    # AUs: keep *exactly* the order they appear in feature_cols
    au_used_names = [n for n in feature_cols if n.endswith("_c") or n.endswith("_r")]
    au_put  = np.array([name_to_pos[n] for n in au_used_names], dtype=np.int64)
    return name_to_pos, res_take, res_put, vgg_take, vgg_put, au_used_names, au_put

# ---- Target-domain stats on GPU (streamed) ----
@torch.no_grad()
def compute_target_stats_gpu(subjects,
                             feature_cols,
                             res_take, res_put, vgg_take, vgg_put, au_used_names, au_put,
                             device,
                             chunk_size=32768,
                             keep_au_c_raw=False):
    D = len(feature_cols)
    res_put_t = torch.from_numpy(res_put).to(device) if res_put.size else None
    vgg_put_t = torch.from_numpy(vgg_put).to(device) if vgg_put.size else None
    au_put_t  = torch.from_numpy(au_put).to(device)  if au_put.size  else None

    s1 = torch.zeros(D, dtype=torch.float64, device=device)
    s2 = torch.zeros(D, dtype=torch.float64, device=device)
    total = 0

    for sid, subj_dir in subjects:
        R = _load_resnet(subj_dir, sid)
        V = _load_vgg(subj_dir, sid)
        DF = _load_aus(subj_dir, sid)
        if R is None or V is None or DF is None:
            continue
        n = min(R.shape[0], V.shape[0], len(DF))
        if n <= 0:
            continue

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)

            # RESNET
            if res_put_t is not None and res_take.size:
                r_np = R[start:end, :][:, res_take]
                r = torch.from_numpy(np.ascontiguousarray(r_np)).to(device).float()
                s1[res_put_t] += r.sum(dim=0, dtype=torch.float64)
                s2[res_put_t] += (r.double().pow(2)).sum(dim=0)
                del r, r_np

            # VGG
            if vgg_put_t is not None and vgg_take.size:
                v_np = V[start:end, :][:, vgg_take]
                v = torch.from_numpy(np.ascontiguousarray(v_np)).to(device).float()
                s1[vgg_put_t] += v.sum(dim=0, dtype=torch.float64)
                s2[vgg_put_t] += (v.double().pow(2)).sum(dim=0)
                del v, v_np

            # AUs
            if au_put_t is not None and len(au_used_names):
                try:
                    au_np = np.column_stack([DF[nm].to_numpy()[start:end] for nm in au_used_names]).astype(np.float32, copy=False)
                except KeyError:
                    au_np = None
                if au_np is not None:
                    a = torch.from_numpy(np.ascontiguousarray(au_np)).to(device).float()
                    s1[au_put_t] += a.sum(dim=0, dtype=torch.float64)
                    s2[au_put_t] += (a.double().pow(2)).sum(dim=0)
                    del a, au_np

        total += n
        del R, V, DF

    if total == 0:
        mean = torch.zeros(D, dtype=torch.float32)
        std  = torch.ones(D,  dtype=torch.float32)
        return mean.cpu(), std.cpu()

    mean = (s1 / total)
    var  = (s2 / total) - mean.pow(2)
    var  = torch.clamp(var, min=1e-12)
    std  = torch.sqrt(var)

    mean = mean.float().cpu()
    std  = std.float().cpu()

    if keep_au_c_raw:
        auc_idx = [i for i, n in enumerate(feature_cols) if n.endswith("_c")]
        if auc_idx:
            idx = torch.tensor(auc_idx, dtype=torch.long)
            mean.index_fill_(0, idx, 0.0)
            std.index_fill_(0, idx, 1.0)

    return mean, std

# ---- Per-subject inference (streamed) ----
@torch.no_grad()
def infer_subject(subject_id, subj_dir,
                  model, scaler,
                  feature_cols,
                  res_take, res_put, vgg_take, vgg_put, au_used_names, au_put,
                  device,
                  out_dir,
                  chunk_size=32768):
    # load sources
    R = _load_resnet(subj_dir, subject_id)
    V = _load_vgg(subj_dir, subject_id)
    DF = _load_aus(subj_dir, subject_id)
    if R is None or V is None or DF is None:
        print(f"[skip] {subject_id}: missing one or more feature files")
        return

    n = min(R.shape[0], V.shape[0], len(DF))
    if n <= 0:
        print(f"[skip] {subject_id}: no frames")
        return

    D = len(feature_cols)
    res_put_t = torch.from_numpy(res_put).to(device) if res_put.size else None
    vgg_put_t = torch.from_numpy(vgg_put).to(device) if vgg_put.size else None
    au_put_t  = torch.from_numpy(au_put).to(device)  if au_put.size  else None

    # prepare per-frame CSV stream
    cols = [IDX_TO_EMO[i] for i in range(len(IDX_TO_EMO))]  # H,S,A,N,D,F (by index)
    csv_path = os.path.join(out_dir, f"{subject_id}_emotion_prediction.csv")
    wrote_header = False

    # accumulate mean-softmax
    sum_probs = torch.zeros(len(IDX_TO_EMO), dtype=torch.float64, device=device)

    # (optional) depression label
    dep = DEPRESSION_LABELS.get(str(subject_id), "Unknown")

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        B = end - start

        xb = torch.zeros(B, D, dtype=torch.float32, device=device)

        # RESNET
        if res_put_t is not None and res_take.size:
            r_np = R[start:end, :][:, res_take]
            r = torch.from_numpy(np.ascontiguousarray(r_np)).to(device).float()
            xb.index_copy_(1, res_put_t, r)
            del r, r_np

        # VGG
        if vgg_put_t is not None and vgg_take.size:
            v_np = V[start:end, :][:, vgg_take]
            v = torch.from_numpy(np.ascontiguousarray(v_np)).to(device).float()
            xb.index_copy_(1, vgg_put_t, v)
            del v, v_np

        # AUs
        if au_put_t is not None and len(au_used_names):
            try:
                au_np = np.column_stack([DF[nm].to_numpy()[start:end] for nm in au_used_names]).astype(np.float32, copy=False)
            except KeyError:
                au_np = None
            if au_np is not None:
                a = torch.from_numpy(np.ascontiguousarray(au_np)).to(device).float()
                xb.index_copy_(1, au_put_t, a)
                del a, au_np

        # scale + predict
        xbs = scaler(xb)
        logits = model(xbs)
        probs = torch.softmax(logits, dim=1)  # (B, C)

        # update mean-softmax
        sum_probs += probs.double().sum(dim=0)

        # write this chunk to CSV (no huge memory)
        probs_cpu = probs.cpu().numpy()
        frame_idx = np.arange(start, end, dtype=np.int64)
        df = pd.DataFrame(probs_cpu, columns=cols)
        df.insert(0, "frame_idx", frame_idx)
        df.insert(1, "Subject_ID", subject_id)
        df.insert(2, "Depression_Label", dep)
        df["Predicted_Emotion"] = [IDX_TO_EMO[i] for i in probs_cpu.argmax(axis=1)]
        df.to_csv(csv_path, index=False, mode=("w" if not wrote_header else "a"), header=(not wrote_header))
        wrote_header = True

        del xb, xbs, logits, probs

    # summary JSON (mean softmax)
    mean_probs = (sum_probs / max(1, n)).float().cpu().numpy()
    vid_pred = IDX_TO_EMO[int(mean_probs.argmax())]
    #with open(os.path.join(out_dir, f"{subject_id}_summary.json"), "w") as f:
     #   json.dump({
      #      "Subject_ID": subject_id,
       #     "Depression_Label": dep,
        #    "mean_softmax": {cols[i]: float(mean_probs[i]) for i in range(len(cols))},
         #   "video_pred": vid_pred,
          #  "frames": int(n),
        #}, f, indent=2)

    print(f"[ok] {subject_id}: frames={n}, video_pred={vid_pred}, csv={os.path.basename(csv_path)}")

# ------------------------- MAIN -------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # speed on constant shapes
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # load artifacts
    feature_cols = load_feature_cols(FEATCOLS_JSON)
    D_in = len(feature_cols)
    print(f"Trained feature dim: {D_in}")

    # build mapping from trained feature list to per-source columns
    (name_to_pos,
     res_take, res_put,
     vgg_take, vgg_put,
     au_used_names, au_put) = build_feature_mapping(feature_cols, RESNET_DIM, VGG_DIM)

    # model (must match training!)
    hidden2 = None if (not MLP_CFG["hidden_dim2"]) else int(MLP_CFG["hidden_dim2"])
    model = FrameClassifier(
        input_dim=D_in,
        hidden_dim=int(MLP_CFG["hidden_dim"]),
        hidden_dim2=hidden2,
        dropout=float(MLP_CFG["dropout"]),
        num_classes=int(MLP_CFG["num_classes"])
    ).to(DEVICE)

    # load weights (handle plain or {'state_dict': ...})
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    load_ok = model.load_state_dict(state, strict=True)
    if getattr(load_ok, "missing_keys", []) or getattr(load_ok, "unexpected_keys", []):
        print("[warn] load_state_dict:", 
              "missing=", getattr(load_ok, "missing_keys", []),
              "unexpected=", getattr(load_ok, "unexpected_keys", []))
    model.eval()

    # subjects + labels
    subjects = list_subjects(DAIC_ROOT)
    print(f"Found subjects: {len(subjects)}")
    DEPRESSION_LABELS = load_depression_labels(LABELS_CSV)

    # scaler
    if DO_STANDARDIZE_TARGET:
        print("[scaler] Computing TARGET mean/std over all DAIC frames (GPU, streamed)…")
        mean_t, std_t = compute_target_stats_gpu(
            subjects, feature_cols,
            res_take, res_put, vgg_take, vgg_put, au_used_names, au_put,
            device=DEVICE,
            chunk_size=CHUNK_SIZE,
            keep_au_c_raw=KEEP_AU_C_RAW
        )
        scaler = Standardize(mean_t.to(DEVICE), std_t.to(DEVICE)).to(DEVICE)
        print("[scaler] Done.")
    else:
        print("[scaler] Disabled (Identity).")
        scaler = nn.Identity()

    # inference per subject (streamed)
    for sid, subj_dir in subjects:
        infer_subject(
            sid, subj_dir,
            model, scaler,
            feature_cols,
            res_take, res_put, vgg_take, vgg_put, au_used_names, au_put,
            device=DEVICE,
            out_dir=OUT_DIR,
            chunk_size=CHUNK_SIZE
        )

    print("\nAll done.")

