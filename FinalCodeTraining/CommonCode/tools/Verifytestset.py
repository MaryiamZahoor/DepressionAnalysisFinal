# tools/verify_labels_equal.py
import os, sys, argparse, pandas as pd
import numpy as np

# --- add project root to sys.path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
import config as CFG

COL_GT  = "GT_Emotion"
COL_ACT = "Actual_Emotion"

def _norm(s):
    """Normalize label series for comparison."""
    return s.astype(str).str.strip().str.upper()

def _require(p, name):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing {name}: {p}")
    return p

def main():
    ap = argparse.ArgumentParser("Verify (and optionally fix) that GT_Emotion == Actual_Emotion per frame")
    ap.add_argument("--list", default=getattr(CFG, "TEST_LIST", None),
                    help="Path to a split list (e.g., TEST_LIST). Defaults to CFG.TEST_LIST.")
    ap.add_argument("--report_csv", default=None,
                    help="Optional: write a CSV with per-video mismatch counts and examples.")
    ap.add_argument("--fix_to", choices=["gt","actual"], default=None,
                    help="Optional: overwrite one column with the other when they differ "
                         "(gt = write GT_Emotion from Actual_Emotion; actual = write Actual_Emotion from GT_Emotion).")
    ap.add_argument("--limit_examples", type=int, default=5,
                    help="How many example mismatches to show per video in report.")
    args = ap.parse_args()

    list_path = _require(args.list, "list of videos")

    with open(list_path) as f:
        vids = [ln.strip() for ln in f if ln.strip()]

    total_frames = compared_frames = mismatch_frames = 0
    missing_cols = 0
    no_csv = 0
    empties = 0

    rows_report = []

    for vid in vids:
        csvp = os.path.join(CFG.OUTPUT_DIR, vid, CFG.COMBINED_CSV_NAME)
        if not os.path.isfile(csvp):
            no_csv += 1
            continue

        try:
            df = pd.read_csv(csvp, usecols=[COL_GT, COL_ACT])
        except Exception:
            # Try to read and check columns existence
            try:
                df = pd.read_csv(csvp)
            except Exception:
                no_csv += 1
                continue
            if COL_GT not in df.columns or COL_ACT not in df.columns:
                missing_cols += 1
                continue
            df = df[[COL_GT, COL_ACT]]

        if df.empty:
            empties += 1
            continue

        total_frames += len(df)
        mask = df[COL_GT].notna() & df[COL_ACT].notna()
        if not mask.any():
            empties += 1
            continue

        g = _norm(df.loc[mask, COL_GT])
        a = _norm(df.loc[mask, COL_ACT])

        cmp = (g.values == a.values)
        mism = (~cmp)
        num_cmp = mask.sum()
        num_mism = int(mism.sum())

        compared_frames += num_cmp
        mismatch_frames += num_mism

        if num_mism > 0:
            # Collect a few examples
            ex_idx = np.where(mism)[0][:args.limit_examples]
            examples = [{"index": int(df.index[mask].to_numpy()[i]),
                         "gt": g.to_numpy()[i],
                         "actual": a.to_numpy()[i]} for i in ex_idx]
            rows_report.append({
                "video_id": vid,
                "frames": int(len(df)),
                "compared_frames": int(num_cmp),
                "mismatch_frames": int(num_mism),
                "mismatch_rate": float(num_mism / num_cmp),
                "examples": examples
            })

            # Optional: fix in place
            if args.fix_to is not None:
                fix_src, fix_dst = (COL_ACT, COL_GT) if args.fix_to == "gt" else (COL_GT, COL_ACT)
                # Only modify rows where both present; copy normalized (but keep original case? choose normalized for safety)
                df.loc[mask & (df[fix_dst].astype(str).str.strip().str.upper()
                               != df[fix_src].astype(str).str.strip().str.upper()),
                       fix_dst] = df.loc[mask, fix_src]
                df.to_csv(csvp, index=False)

    # Summary
    print(f"[videos] total listed: {len(vids)}")
    print(f"[frames] total in CSVs (all rows): {total_frames}")
    print(f"[frames] compared (both labels present): {compared_frames}")
    print(f"[frames] mismatches: {mismatch_frames} "
          f"({0.0 if compared_frames==0 else 100.0*mismatch_frames/compared_frames:.2f}%)")
    if no_csv or missing_cols or empties:
        print(f"[skips] no_csv={no_csv}, missing_cols={missing_cols}, empty_or_no_overlap={empties}")

    if rows_report:
        print(f"[info] videos with mismatches: {len(rows_report)}")
        for r in rows_report[:10]:
            print(f"  {r['video_id']}: {r['mismatch_frames']}/{r['compared_frames']} "
                  f"({100.0*r['mismatch_rate']:.2f}%)")
            for ex in r["examples"]:
                print(f"     idx={ex['index']} GT={ex['gt']} ACTUAL={ex['actual']}")

    # Optional: write detailed report
    if args.report_csv:
        # Flatten examples minimally (first example only to keep it tidy)
        out = []
        for r in rows_report:
            first = r["examples"][0] if r["examples"] else {"index": None, "gt": None, "actual": None}
            out.append({
                "video_id": r["video_id"],
                "frames": r["frames"],
                "compared_frames": r["compared_frames"],
                "mismatch_frames": r["mismatch_frames"],
                "mismatch_rate": r["mismatch_rate"],
                "example_index": first["index"],
                "example_gt": first["gt"],
                "example_actual": first["actual"],
            })
        pd.DataFrame(out).to_csv(args.report_csv, index=False)
        print(f"[write] detailed report -> {args.report_csv}")

if __name__ == "__main__":
    main()

