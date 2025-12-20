#!/usr/bin/env python3
# daic_infer_rnn_streamed.py
# Streamed DAIC-WOZ inference with a TemporalFFRNN (trained on CREMA-D).
# - Builds features per subject in the *trained* feature order
# - Computes target-domain mean/std across ALL DAIC frames (GPU, streamed)
# - Runs overlapping windows (SEQ_LEN, STRIDE) to get per-window logits
# - Optionally averages overlapping windows -> per-frame probs
# - Writes per-window CSV, optional per-frame CSV, and per-video summary JSON

import os, sys, json, time
import numpy as np
import pandas as pd
import scipy.io
from typing import List, Tuple

# --- clean env for torch import (avoids LD_LIBRARY_PATH conflicts) ---
if os.environ.get("LD_LIBRARY_PATH") and not os.environ.get("_LDLIBPATH_CLEANED"):
    env = dict(os.environ); env.pop("LD_LIBRARY_PATH", None); env["_LDLIBPATH_CLEANED"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)
os.environ.pop("LD_LIBRARY_PATH", None)

import torch
import torch.nn as nn

# ======== EDIT THESE ========
# Artifacts from the SAME training run (RNN):
WEIGHTS_PATH = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/artifacts/combined_GridSearch_unscaled_RNN/std_results/final_trainval_best_by_seqacc/best.pt"  
FEATCOLS_JSON = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/artifacts/artifacts/combined_GridSearch_unscaled_RNN/std_results/final_trainval_best_by_seqacc/feature_cols.json"  # trained feature list (order the model expects)

# DAIC paths
DAIC_ROOT  = "/media/root918/DATA/Projects/Maryiam_Projects/DepressionAnalysis/DAIC_WOZ/Data/"
LABELS_CSV = "/media/root918/DATA/Projects/Maryiam_Projects/DepressionAnalysis/DAIC_WOZ/Labels/detailed_lables.csv"

OUT_DIR    = "/media/root918/OS/MaryiamProject/DAIC_RESULTS_RNN_UNSTD_COMBINED/"
os.makedirs(OUT_DIR, exist_ok=True)

# Model arch (must match training):
MODEL_CFG = {
    "ff_hidden": 1024,
    "ff_hidden2": 512,       # None or int
    "dropout": 0.5,
    "rnn_type": "gru",      # "gru" or "lstm"
    "rnn_hidden": 256,
    "rnn_layers": 1,
    "bidirectional": False,
    "num_classes": 6,        # 6-class canonical
}

# Windowing (change as needed)
SEQ_LEN = 30
STRIDE  = 15  # overlap (e.g., half-window)

# Output toggles
WRITE_PER_WINDOW = True
WRITE_PER_FRAME  = True
WRITE_VIDEO_JSON = False

# Normalization
NORM_MODE       = "none"  # "none"|"zscore"
KEEP_AU_C_RAW   = True      # leave *_c as mean=0, std=1 after zscore
CHUNK_SIZE      = 32768     # frames per chunk when streaming stats & features

# Feature dims
RESNET_DIM = 2048
VGG_DIM    = 4096

# ===== FIXED: use the same canonical order as training =====
CANONICAL = ["angry","disgust","fear","happy","neutral","sad"]
EMOTION_TO_IDX = {e:i for i,e in enumerate(CANONICAL)}
IDX_TO_EMO     = {v:k for k,v in EMOTION_TO_IDX.items()}

DEBUG_PRINT_FEATURES = True  # prints verification lines for feature order/shape
# ===========================


# ---------- Small utilities ----------
def load_feature_cols(path: str) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)

class Standardize(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", mean.view(-1).clone().detach())
        self.register_buffer("std",  std.view(-1).clone().detach())
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

def load_depression_labels(csv_path: str) -> dict:
    if not os.path.isfile(csv_path): return {}
    df = pd.read_csv(csv_path)
    return dict(zip(df["Participant"].astype(str), df["Depression_label"]))

def list_subjects(daic_root: str) -> list[tuple[str, str]]:
    out = []
    for d in sorted(os.listdir(daic_root)):
        p = os.path.join(daic_root, d)
        if not os.path.isdir(p): continue
        sid = d.split("_")[0]  # "301_P" -> "301"
        out.append((sid, p))
    return out

def _load_resnet(subj_dir: str, sid: str):
    p = os.path.join(subj_dir, "features", f"{sid}_CNN_ResNet.mat")
    if not os.path.isfile(p): return None
    m = scipy.io.loadmat(p); X = m.get("feature")
    return None if X is None else X  # (N, 2048)

def _load_vgg(subj_dir: str, sid: str):
    p = os.path.join(subj_dir, "features", f"{sid}_CNN_VGG.mat")
    if not os.path.isfile(p): return None
    m = scipy.io.loadmat(p); X = m.get("feature")
    return None if X is None else X  # (N, 4096)

def _load_aus(subj_dir: str, sid: str):
    p = os.path.join(subj_dir, "features", f"{sid}_OpenFace2.1.0_Pose_gaze_AUs.csv")
    if not os.path.isfile(p): return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def build_feature_mapping(feature_cols: List[str],
                          resnet_dim: int = 2048,
                          vgg_dim: int = 4096):
    name_to_pos = {n: j for j, n in enumerate(feature_cols)}
    res_take = np.array([i for i in range(resnet_dim) if f"feat_{i}_resnet" in name_to_pos], dtype=np.int64)
    res_put  = np.array([name_to_pos[f"feat_{i}_resnet"] for i in res_take], dtype=np.int64)
    vgg_take = np.array([i for i in range(vgg_dim)    if f"feat_{i}_vgg"    in name_to_pos], dtype=np.int64)
    vgg_put  = np.array([name_to_pos[f"feat_{i}_vgg"]  for i in vgg_take], dtype=np.int64)
    au_used_names = [n for n in feature_cols if n.endswith("_c") or n.endswith("_r")]
    au_put  = np.array([name_to_pos[n] for n in au_used_names], dtype=np.int64)
    return name_to_pos, res_take, res_put, vgg_take, vgg_put, au_used_names, au_put

# ---------- Target-domain stats over DAIC (selected columns only) ----------
@torch.no_grad()
def compute_target_stats_gpu(subjects,
                             feature_cols,
                             res_take, res_put, vgg_take, vgg_put, au_used_names, au_put,
                             device: torch.device,
                             chunk_size: int = 32768,
                             keep_au_c_raw: bool = False):
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
        if R is None or V is None or DF is None: continue
        n = min(R.shape[0], V.shape[0], len(DF))
        if n <= 0: continue

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            B = end - start

            # Assemble xb in trained order (same as inference)
            xb = torch.zeros(B, D, dtype=torch.float32, device=device)

            if res_put_t is not None and res_take.size:
                r_np = R[start:end, :][:, res_take]
                r = torch.from_numpy(np.ascontiguousarray(r_np)).to(device).float()
                xb.index_copy_(1, res_put_t, r); del r, r_np

            if vgg_put_t is not None and vgg_take.size:
                v_np = V[start:end, :][:, vgg_take]
                v = torch.from_numpy(np.ascontiguousarray(v_np)).to(device).float()
                xb.index_copy_(1, vgg_put_t, v); del v, v_np

            if au_put_t is not None and len(au_used_names):
                try:
                    au_np = np.column_stack([DF[nm].to_numpy()[start:end] for nm in au_used_names]).astype(np.float32, copy=False)
                except KeyError:
                    au_np = None
                if au_np is not None:
                    a = torch.from_numpy(np.ascontiguousarray(au_np)).to(device).float()
                    xb.index_copy_(1, au_put_t, a); del a, au_np

            s1 += xb.sum(0, dtype=torch.float64)
            s2 += (xb.double().pow(2)).sum(0)
            total += B
            del xb

        del R, V, DF

    if total == 0:
        mean = torch.zeros(D, dtype=torch.float32, device=device)
        std  = torch.ones(D,  dtype=torch.float32, device=device)
    else:
        mean = (s1 / total)
        var  = (s2 / total) - mean.pow(2)
        var.clamp_(min=1e-12)
        std  = torch.sqrt(var)
        mean = mean.float(); std = std.float()

    if keep_au_c_raw:
        auc_idx = [i for i, n in enumerate(feature_cols) if n.endswith("_c")]
        if auc_idx:
            idx = torch.tensor(auc_idx, dtype=torch.long, device=device)
            mean.index_fill_(0, idx, 0.0)
            std.index_fill_(0, idx, 1.0)

    return mean, std

# ---------- Window helpers ----------
def build_windows_np(X: np.ndarray, T: int, S: int, pad_short=True) -> tuple[np.ndarray, list[int]]:
    N, D = X.shape
    if N <= 0: return np.zeros((0, T, D), np.float32), []
    if N < T:
        if not pad_short: return np.zeros((0, T, D), np.float32), []
        pad = np.repeat(X[[-1], :], T - N, axis=0)
        return np.stack([np.concatenate([X, pad], axis=0)], axis=0).astype("float32"), [0]
    starts, wins = [], []
    for s in range(0, N - T + 1, S):
        wins.append(X[s:s+T]); starts.append(s)
    if starts and starts[-1] != (N - T):
        wins.append(X[N-T:N]); starts.append(N - T)
    return np.stack(wins, axis=0).astype("float32"), starts

@torch.no_grad()
def predict_window_logits(model: nn.Module, Xseq: np.ndarray, device, scaler: nn.Module, bs: int = 256) -> np.ndarray | None:
    if Xseq is None or len(Xseq) == 0: return None
    outs = []
    model.eval()
    for i in range(0, len(Xseq), bs):
        b = torch.from_numpy(np.ascontiguousarray(Xseq[i:i+bs])).float().to(device)  # (B, T, D)
        B, T, D = b.shape
        b2 = b.reshape(B*T, D)
        b2 = scaler(b2)
        b  = b2.reshape(B, T, D)
        logits = model(b)  # (B, C)
        outs.append(logits.detach().cpu().numpy())
    return np.vstack(outs)

def frame_probs_from_windows(X: np.ndarray, T: int, S: int, device, model, scaler, bs: int):
    Xseq, starts = build_windows_np(X, T, S, pad_short=True)
    if Xseq.shape[0] == 0: return None, None, None
    logits = predict_window_logits(model, Xseq, device, scaler, bs=max(1, bs//2))
    if logits is None: return None, None, None
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()  # (W, C)

    N = X.shape[0]; C = probs.shape[1]
    sum_probs = np.zeros((N, C), dtype=np.float64)
    counts    = np.zeros((N,),    dtype=np.int32)

    per_window_rows = []  # (start, end, center, probs..., pred)

    for w, s in enumerate(starts):
        e = min(s + T, N)
        sum_probs[s:e] += probs[w]
        counts[s:e]    += 1
        center = s + (e - s)//2
        pred_w = int(probs[w].argmax())
        per_window_rows.append((s, e, center, probs[w], pred_w))

    mask = counts > 0
    frame_probs = np.zeros((N, C), dtype=np.float32)
    frame_probs[mask] = (sum_probs[mask] / counts[mask, None])

    return frame_probs, per_window_rows, mask

# ---------- Model ----------
# Import your TemporalFFRNN the same way as in your RAVDESS script
from models.CNN_RNNmodel import TemporalFFRNN  # make sure path is correct

# ---------- Per-subject inference ----------
@torch.no_grad()
def infer_subject_rnn(subject_id: str, subj_dir: str,
                      model: nn.Module, scaler: nn.Module,
                      feature_cols: List[str],
                      res_take, res_put, vgg_take, vgg_put, au_used_names, au_put,
                      device: torch.device,
                      out_dir: str,
                      seq_len: int, stride: int,
                      chunk_size: int = 32768,
                      write_per_window: bool = True,
                      write_per_frame: bool = True,
                      write_video_json: bool = True):
    R = _load_resnet(subj_dir, subject_id)
    V = _load_vgg(subj_dir, subject_id)
    DF = _load_aus(subj_dir, subject_id)
    if R is None or V is None or DF is None:
        print(f"[skip] {subject_id}: missing one or more feature files")
        return

    n = min(R.shape[0], V.shape[0], len(DF))
    if n <= 0:
        print(f"[skip] {subject_id}: no frames"); return

    D = len(feature_cols)
    res_put_t = torch.from_numpy(res_put).to(device) if res_put.size else None
    vgg_put_t = torch.from_numpy(vgg_put).to(device) if vgg_put.size else None
    au_put_t  = torch.from_numpy(au_put).to(device)  if au_put.size  else None

    # assemble full feature matrix X (N, D) in trained order — stream into CPU array
    X = np.zeros((n, D), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        B = end - start
        xb = torch.zeros(B, D, dtype=torch.float32, device=device)

        if res_put_t is not None and res_take.size:
            r_np = R[start:end, :][:, res_take]
            r = torch.from_numpy(np.ascontiguousarray(r_np)).to(device).float()
            xb.index_copy_(1, res_put_t, r); del r, r_np

        if vgg_put_t is not None and vgg_take.size:
            v_np = V[start:end, :][:, vgg_take]
            v = torch.from_numpy(np.ascontiguousarray(v_np)).to(device).float()
            xb.index_copy_(1, vgg_put_t, v); del v, v_np

        if au_put_t is not None and len(au_used_names):
            try:
                au_np = np.column_stack([DF[nm].to_numpy()[start:end] for nm in au_used_names]).astype(np.float32, copy=False)
            except KeyError:
                au_np = None
            if au_np is not None:
                a = torch.from_numpy(np.ascontiguousarray(au_np)).to(device).float()
                xb.index_copy_(1, au_put_t, a); del a, au_np

        # copy to host
        X[start:end] = xb.detach().cpu().numpy()
        del xb

    if DEBUG_PRINT_FEATURES:
        print(f"[{subject_id}] X shape: {X.shape} | first 5 cols: {feature_cols[:5]}")

    # ----- windowing + inference -----
    frame_probs, per_window_rows, mask = frame_probs_from_windows(
        X, seq_len, stride, device, model, scaler, bs=512
    )
    if per_window_rows is None:
        print(f"[skip] {subject_id}: no windows"); return

    cols = [IDX_TO_EMO[i] for i in range(len(IDX_TO_EMO))]
    # per-window CSV
    if write_per_window:
        wp = os.path.join(out_dir, f"{subject_id}_rnn_seq_probs.csv")
        data = []
        for (s, e, c, pvec, pred_idx) in per_window_rows:
            row = {
                "Subject_ID": subject_id,
                "start": int(s),
                "end": int(e),
                "center": int(c),
                "Predicted_Emotion": IDX_TO_EMO[pred_idx],
            }
            for i, name in enumerate(cols):
                row[name] = float(pvec[i])
            data.append(row)
        pd.DataFrame(data).to_csv(wp, index=False)

    # per-frame CSV (overlap-avg of window probs)
    if write_per_frame and frame_probs is not None:
        fp = os.path.join(out_dir, f"{subject_id}_rnn_frame_probs.csv")
        df = pd.DataFrame(frame_probs, columns=cols)
        df.insert(0, "frame_idx", np.arange(len(df), dtype=np.int64))
        df["Subject_ID"] = subject_id
        df["Predicted_Emotion"] = [IDX_TO_EMO[int(i)] for i in frame_probs.argmax(axis=1)]
        df.to_csv(fp, index=False)

    # video summary (mean of frame probs over valid frames)
    if write_video_json and frame_probs is not None:
        valid = (mask if mask is not None else np.ones((len(frame_probs),), dtype=bool))
        if valid.any():
            mean_probs = frame_probs[valid].mean(axis=0)
            vid_pred = IDX_TO_EMO[int(mean_probs.argmax())]
            with open(os.path.join(out_dir, f"{subject_id}_rnn_summary.json"), "w") as f:
                json.dump({
                    "Subject_ID": subject_id,
                    "mean_softmax": {cols[i]: float(mean_probs[i]) for i in range(len(cols))},
                    "video_pred": vid_pred,
                    "frames": int(len(frame_probs)),
                    "seq_len": int(seq_len),
                    "stride": int(stride),
                }, f, indent=2)

    print(f"[ok][{subject_id}] frames={n} | windows={len(per_window_rows)} | out={os.path.basename(out_dir)}")


# ========================= MAIN =========================
def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- artifacts ---
    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing WEIGHTS_PATH: {WEIGHTS_PATH}")
    if not os.path.isfile(FEATCOLS_JSON):
        raise FileNotFoundError(f"Missing FEATCOLS_JSON: {FEATCOLS_JSON}")

    feature_cols = load_feature_cols(FEATCOLS_JSON)   # order the model expects
    D_in = len(feature_cols)
    print("Trained feature dim:", D_in)
    #print(feature_cols)

    # --- DAIC subjects + labels ---
    subjects = list_subjects(DAIC_ROOT)
    print("Found subjects:", len(subjects))
    dep_labels = load_depression_labels(LABELS_CSV)  # not used for metrics here, but kept if needed later

    # --- feature mapping ---
    (_, res_take, res_put, vgg_take, vgg_put, au_used_names, au_put) = build_feature_mapping(
        feature_cols, RESNET_DIM, VGG_DIM
    )
    if DEBUG_PRINT_FEATURES:
        print(f"Selected AU columns (count={len(au_used_names)}): {au_used_names[:8]}...")

    # --- Build model + load weights ---
    model = TemporalFFRNN(
        input_dim=D_in,
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
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[warn] load_state_dict missing:", missing, "unexpected:", unexpected)
    model.eval()

    # --- Scaler (target-domain over DAIC) ---
    if NORM_MODE == "zscore":
        print("[scaler] Computing TARGET mean/std on DAIC (selected columns; GPU, streamed)…")
        mean_t, std_t = compute_target_stats_gpu(
            subjects, feature_cols,
            res_take, res_put, vgg_take, vgg_put, au_used_names, au_put,
            device=device, chunk_size=CHUNK_SIZE,
            keep_au_c_raw=KEEP_AU_C_RAW
        )
        scaler = Standardize(mean_t.to(device), std_t.to(device)).to(device)
        print("[scaler] Done.")
    else:
        scaler = nn.Identity().to(device)
        print("[scaler] Disabled.")

    # --- Inference per subject ---
    os.makedirs(OUT_DIR, exist_ok=True)
    for sid, subj_dir in subjects:
        infer_subject_rnn(
            subject_id=sid, subj_dir=subj_dir,
            model=model, scaler=scaler,
            feature_cols=feature_cols,
            res_take=res_take, res_put=res_put, vgg_take=vgg_take, vgg_put=vgg_put,
            au_used_names=au_used_names, au_put=au_put,
            device=device, out_dir=OUT_DIR,
            seq_len=SEQ_LEN, stride=STRIDE,
            chunk_size=CHUNK_SIZE,
            write_per_window=WRITE_PER_WINDOW,
            write_per_frame=WRITE_PER_FRAME,
            write_video_json=WRITE_VIDEO_JSON,
        )

    print("\nAll done.")

if __name__ == "__main__":
    main()

