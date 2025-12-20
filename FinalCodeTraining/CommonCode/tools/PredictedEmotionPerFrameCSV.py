# tools/dump_frame_predictions.py
import os, sys, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
# --- add project root to sys.path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
import config as CFG
from models.TwoLayerMLP import FrameClassifier      # must match training model
from utils.features import harmonize_vgg_cols, load_feature_cols, Standardize


def _require_file(path, desc):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")
    return path


def _load_scaler(path, device):
    d = torch.load(path, map_location="cpu")
    return Standardize(d["mean"], d["std"]).to(device)


def _maybe_read_model_meta(weights_path, args):
    """If model_meta.json exists next to weights, use values unless CLI overrides."""
    meta_path = os.path.join(os.path.dirname(weights_path), "model_meta.json")
    if not os.path.isfile(meta_path):
        return args
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if args.hidden_dim is None:
            args.hidden_dim = meta.get("hidden_dim", args.hidden_dim)
        if args.hidden_dim2 is None:
            args.hidden_dim2 = meta.get("hidden_dim2", args.hidden_dim2)
        if args.dropout is None:
            args.dropout = meta.get("dropout", args.dropout)
    except Exception:
        pass
    return args


def _pick_frame_id_column(df):
    """Best-effort frame identifier column."""
    for c in ["frame", "frame_resnet", "frame_vgg"]:
        if c in df.columns:
            return c
    return None


@torch.no_grad()
def _predict_frames_and_video(model, X, device, scaler, batch_size, num_classes):
    """
    Returns:
      pred_idx_frame  : (N,) int64          per-frame top-1 class
      pred_prob_frame : (N,) float32        per-frame top-1 prob
      video_pred_idx  : int                 video-level top-1 (mean softmax)
      video_pred_prob : float               video-level top-1 prob
    """
    n = X.shape[0]
    if n == 0:
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                None, None)

    sum_probs = np.zeros((num_classes,), dtype=np.float64)
    idx_list, pmx_list = [], []

    for i in range(0, n, batch_size):
        b = torch.from_numpy(X[i:i+batch_size]).float().to(device)
        b = scaler(b)
        probs = F.softmax(model(b), dim=1)              # [B, C]
        pmx, idx = probs.max(dim=1)

        idx_list.append(idx.detach().cpu().numpy())
        pmx_list.append(pmx.detach().cpu().numpy())
        sum_probs += probs.detach().cpu().numpy().sum(axis=0)

    pred_idx_frame  = np.concatenate(idx_list)
    pred_prob_frame = np.concatenate(pmx_list)

    mean_probs = (sum_probs / float(n)).astype(np.float64)
    video_pred_idx  = int(mean_probs.argmax())
    video_pred_prob = float(mean_probs.max())
    return pred_idx_frame, pred_prob_frame, video_pred_idx, video_pred_prob


def main():
    p = argparse.ArgumentParser("Dump per-frame predictions for TEST videos (+ video-level prediction)")
    # Artifacts
    p.add_argument("--test_list",  type=str, default=getattr(CFG, "TEST_LIST", None),
                   help="Path to test_videos.txt (defaults to CFG.TEST_LIST)")
    p.add_argument("--weights",    type=str, default=CFG.BEST_WEIGHTS,
                   help="Path to .pt weights saved during training")
    p.add_argument("--featcols",   type=str, default=CFG.FEATCOLS_JSON,
                   help="feature_cols.json saved during training")
    p.add_argument("--scaler",     type=str, default=CFG.SCALER_PATH,
                   help="scaler.pt saved during training")
    # Model arch (should match training). If left None, we'll try model_meta.json
    p.add_argument("--hidden_dim",  type=lambda x: None if x.lower()=="none" else int(x),
                   default=getattr(CFG, "HIDDEN_UNITS", None))
    p.add_argument("--hidden_dim2", type=lambda x: None if x.lower()=="none" else int(x),
                   default=getattr(CFG, "HIDDEN_UNITS2", None),
                   help="Second hidden layer size; use None if trained with one layer")
    p.add_argument("--dropout",     type=float, default=getattr(CFG, "DROPOUT", None))
    p.add_argument("--batch_size",  type=int, default=getattr(CFG, "BATCH_SIZE", 512))
    # Output
    p.add_argument("--out_csv",     type=str, default=None,
                   help="Where to write the CSV. Default: <weights_dir>/frame_predictions_test.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Required artifacts
    test_list  = _require_file(args.test_list,  "TEST_LIST")
    featcols_p = _require_file(args.featcols,   "FEATCOLS_JSON")
    scaler_p   = _require_file(args.scaler,     "SCALER_PATH")
    weights_p  = _require_file(args.weights,    "BEST_WEIGHTS")

    # Output CSV path
    if args.out_csv is None:
        args.out_csv = os.path.join(os.path.dirname(weights_p), "frame_predictions_test.csv")

    # Load artifacts
    feature_cols = load_feature_cols(featcols_p)
    scaler       = _load_scaler(scaler_p, device)

    # Resolve model architecture (CLI > model_meta.json > config)
    args = _maybe_read_model_meta(weights_p, args)
    if args.hidden_dim is None:
        args.hidden_dim = 512
    h2 = None if (args.hidden_dim2 in [None, 0]) else int(args.hidden_dim2)
    if args.dropout is None:
        args.dropout = 0.0

    # Build model matching training
    num_classes = len(CFG.emotion_to_idx)
    model = FrameClassifier(
        input_dim=len(feature_cols),
        hidden_dim=int(args.hidden_dim),
        hidden_dim2=h2,                # None => one hidden layer
        dropout=float(args.dropout),
        num_classes=num_classes,
    ).to(device)

    # Load weights
    state = torch.load(weights_p, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Mappings
    idx_to_emotion = CFG.idx_to_emotion

    # Prepare CSV
    wrote_header = False
    total_rows_written = 0
    n_skipped_missing_cols = 0
    n_skipped_no_csv       = 0
    n_empty_frames         = 0

    with open(test_list) as f:
        vids = [ln.strip() for ln in f if ln.strip()]

    for vid in vids:
        csvp = os.path.join(CFG.OUTPUT_DIR, vid, CFG.COMBINED_CSV_NAME)
        if not os.path.isfile(csvp):
            n_skipped_no_csv += 1
            continue

        try:
            df = pd.read_csv(csvp)
        except Exception:
            n_skipped_no_csv += 1
            continue

        df = harmonize_vgg_cols(df)

        # Ensure expected feature columns
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            n_skipped_missing_cols += 1
            continue

        # Build features matrix
        X = (df[feature_cols]
             .replace([np.inf, -np.inf], np.nan)
             .fillna(0.0)
             .astype("float32")
             .values)

        if X.shape[0] == 0:
            n_empty_frames += 1
            continue

        # Per-frame + video-level predictions
        pred_idx, pred_prob, v_idx, v_prob = _predict_frames_and_video(
            model, X, device, scaler, args.batch_size, num_classes
        )
        pred_label = [idx_to_emotion[int(i)] for i in pred_idx]
        video_pred_idx   = v_idx
        video_pred_label = idx_to_emotion[v_idx] if v_idx is not None else ""
        video_pred_prob  = v_prob if v_prob is not None else np.nan

        # Frame ids
        frame_col = _pick_frame_id_column(df)
        if frame_col is not None:
            frame_ids = df[frame_col].astype(str).tolist()
        else:
            frame_ids = [str(i) for i in range(len(df))]

        # Ground-truth columns if present
        gt = df["GT_Emotion"].astype(str).str.upper().tolist() if "GT_Emotion" in df.columns else [""] * len(df)
        ac = df["Actual_Emotion"].astype(str).str.upper().tolist() if "Actual_Emotion" in df.columns else [""] * len(df)

        # Build output rows (video-level prediction repeated for each frame)
        out_df = pd.DataFrame({
            "video_id":         vid,
            "frame_id":         frame_ids,
            "pred_idx":         pred_idx.astype(np.int64),
            "pred_label":       pred_label,
            "pred_prob":        pred_prob.astype(np.float32),
            "video_pred_idx":   video_pred_idx,
            "video_pred_label": video_pred_label,
            "video_pred_prob":  video_pred_prob,
            "GT_Emotion":       gt,
            "Actual_Emotion":   ac,
        })

        out_df.to_csv(args.out_csv, index=False, mode="a", header=(not wrote_header))
        wrote_header = True
        total_rows_written += len(out_df)

    print(f"[done] wrote {total_rows_written} frame rows to: {args.out_csv}")
    print("[summary] skipped:",
          f"no_csv={n_skipped_no_csv},",
          f"missing_cols={n_skipped_missing_cols},",
          f"empty_videos={n_empty_frames}")


if __name__ == "__main__":
    main()

