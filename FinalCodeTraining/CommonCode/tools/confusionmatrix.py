#!/usr/bin/env python3
import sys, os, argparse, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
# --- add project root to sys.path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
import config as CFG  # for label set if you want (emotion_to_idx / idx_to_emotion)

EMOTIONS = list(CFG.emotion_to_idx.keys())  # e.g. ['H','S','A','N','D','F']


def _read_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    # Normalize label strings to uppercase (if present)
    for col in ["GT_Emotion", "Actual_Emotion", "pred_label", "video_pred_label"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()
    return df


def _agg_per_video(df, use_label="Actual_Emotion"):
    """
    Aggregate to one row per video:
      - true_label: mode of use_label per video (or GT_Emotion if use_label missing)
      - pred_label: from video_pred_label if present (else mode of pred_label)
    """
    if use_label not in df.columns:
        print(f"[warn] {use_label} not in CSV; falling back to GT_Emotion if present.")
        use_label = "GT_Emotion" if "GT_Emotion" in df.columns else None
    if use_label is None:
        raise ValueError("No valid ground-truth label column found in CSV.")

    have_video_pred = "video_pred_label" in df.columns

    rows = []
    for vid, g in df.groupby("video_id", as_index=False):
        # true label (mode over frames)
        lab_series = g[use_label].dropna().astype(str).str.upper()
        if lab_series.empty:
            continue
        true_lab = lab_series.mode().iat[0]

        # predicted label: prefer video_pred_label, else per-frame mode
        if have_video_pred:
            pred_series = g["video_pred_label"].dropna().astype(str).str.upper()
            if pred_series.empty:
                # rare, but fallback
                pf = g["pred_label"].dropna().astype(str).str.upper() if "pred_label" in g.columns else pd.Series([])
                if pf.empty:
                    continue
                pred_lab = pf.mode().iat[0]
            else:
                # they should all be the same; grab first
                pred_lab = pred_series.iloc[0]
        else:
            if "pred_label" not in g.columns:
                continue
            pf = g["pred_label"].dropna().astype(str).str.upper()
            if pf.empty:
                continue
            pred_lab = pf.mode().iat[0]

        rows.append((vid, true_lab, pred_lab))

    agg = pd.DataFrame(rows, columns=["video_id", "true_label", "pred_label"])
    # Filter to only recognized emotions, just in case
    agg = agg[agg["true_label"].isin(EMOTIONS) & agg["pred_label"].isin(EMOTIONS)].reset_index(drop=True)
    return agg


def _plot_cm(cm, labels, title, normalize=False, out_png=None):
    if normalize:
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label',
           title=title)
    ax.set_xlabel('Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = np.nanmax(cm) / 2.0 if cm.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    color="white" if val > thresh else "black")
    fig.tight_layout()

    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _one_vs_rest_confusion(true_labels, pred_labels, pos_label):
    """
    Build 2x2 confusion (TP, FN / FP, TN) for pos_label vs rest.
    Returns cm (2x2) and (precision, recall, f1, support).
    """
    y_true = np.array([1 if t == pos_label else 0 for t in true_labels], dtype=np.int64)
    y_pred = np.array([1 if p == pos_label else 0 for p in pred_labels], dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])  # [[TP, FN],[FP, TN]]
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    return cm, (prec, rec, f1, sup)


def main():
    ap = argparse.ArgumentParser("Build confusion matrices per emotion from per-frame CSV")
    ap.add_argument("--csv", required=True, help="Path to frame_predictions_test.csv (from dump_frame_predictions.py)")
    ap.add_argument("--use_label", default="Actual_Emotion",
                    help='Ground-truth column to use: "Actual_Emotion" (default) or "GT_Emotion"')
    ap.add_argument("--normalize", action="store_true", help="Normalize rows for plots")
    ap.add_argument("--out_dir", default=None, help="Directory to save confusion matrix PNGs (optional)")
    args = ap.parse_args()

    df = _read_csv(args.csv)
    agg = _agg_per_video(df, use_label=args.use_label)

    if agg.empty:
        print("[error] No usable per-video records found in CSV.")
        return

    # Overall multi-class confusion
    y_true = agg["true_label"].tolist()
    y_pred = agg["pred_label"].tolist()

    # Make sure label order is consistent & complete
    labels = [e for e in EMOTIONS if (e in set(y_true) or e in set(y_pred))]
    if not labels:
        print("[error] No labels present after filtering.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n[OVERALL] confusion matrix (counts):")
    print(pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]))
    print("\n[OVERALL] classification report:")
    print(pd.DataFrame.from_dict(
        {lab: Counter([p for t,p in zip(y_true,y_pred) if t==lab]) for lab in labels},
        orient="index"
    ).fillna(0).astype(int))
    # Also scikit-learn report:
    try:
        from sklearn.metrics import classification_report as cr
        print(cr(y_true, y_pred, labels=labels, target_names=labels, digits=3))
    except Exception:
        pass

    # Plot overall
    if args.out_dir or True:
        out_overall = os.path.join(args.out_dir, "cm_overall.png") if args.out_dir else None
        _plot_cm(cm, labels, title=f"Confusion Matrix — {args.use_label} (per-video)", normalize=args.normalize, out_png=out_overall)
        if out_overall:
            print(f"[save] overall CM -> {out_overall}")

    # One-vs-rest per emotion
    print("\n[ONE-VS-REST] per emotion (TP FN / FP TN) with Precision/Recall/F1:")
    for lab in labels:
        cm2, (prec, rec, f1, sup) = _one_vs_rest_confusion(y_true, y_pred, pos_label=lab)
        cm2_df = pd.DataFrame(cm2, index=[f"true_{lab}", f"true_not_{lab}"], columns=[f"pred_{lab}", f"pred_not_{lab}"])
        print(f"\nClass: {lab}")
        print(cm2_df)
        print(f"precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  support={int(sup)}")

        # Optional plot of the 2x2
        if args.out_dir:
            outp = os.path.join(args.out_dir, f"cm_one_vs_rest_{lab}.png")
            _plot_cm(cm2, [lab, f"not_{lab}"], title=f"1-vs-rest: {lab}", normalize=args.normalize, out_png=outp)
            print(f"[save] {lab} 1-vs-rest CM -> {outp}")


if __name__ == "__main__":
    main()

