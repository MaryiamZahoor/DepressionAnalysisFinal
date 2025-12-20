# tools/audit_splits.py
import os, sys, argparse, pandas as pd
from collections import Counter, defaultdict

# --- add project root to sys.path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import OUTPUT_DIR, COMBINED_CSV_NAME, TRAIN_LIST, VAL_LIST, TEST_LIST, emotion_to_idx,Project_DIR, SPLIT_LABEL_COL


def _read_ids(path):
    if not os.path.isfile(path): 
        return set()
    with open(path) as f: 
        return {ln.strip() for ln in f if ln.strip()}


def scan_videos(root):
    """
    Return dict: vid -> (status, info)
      status in {"eligible","no_csv","csv_read_error","no_labels","invalid_label"}
      info: majority label (if eligible/invalid_label) or exception text (if csv_read_error)
    """
    report = {}
    for vid in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        csvp = os.path.join(root, vid, COMBINED_CSV_NAME)
        if not os.path.isfile(csvp):
            report[vid] = ("no_csv", None); continue
        try:
            s = pd.read_csv(csvp, usecols=[SPLIT_LABEL_COL])[SPLIT_LABEL_COL]
        except Exception as e:
            report[vid] = ("csv_read_error", str(e)); continue
        s = s.dropna().astype(str).str.upper()
        if s.empty:
            report[vid] = ("no_labels", None); continue
        lab = s.mode().iat[0]
        if lab not in emotion_to_idx:
            report[vid] = ("invalid_label", lab); continue
        report[vid] = ("eligible", lab)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=30, help="Max items to print per category (0 = all)")
    ap.add_argument("--save_missing", action="store_true", help="Save missing-CSV list to a text file")
    args = ap.parse_args()
    show_n = None if args.show == 0 else args.show

    # 1) Scan the directory tree
    report = scan_videos(OUTPUT_DIR)
    all_dirs = set(report.keys())
    eligible = {vid for vid, (st, _) in report.items() if st == "eligible"}

    # 2) Read split lists
    train_ids = _read_ids(TRAIN_LIST)
    val_ids   = _read_ids(VAL_LIST)
    test_ids  = _read_ids(TEST_LIST)
    listed    = train_ids | val_ids | test_ids

    # 3) Basic counts
    print(f"[found on disk] total dirs: {len(all_dirs)} | eligible (have usable label): {len(eligible)}")
    print(f"[lists] train:{len(train_ids)}  val:{len(val_ids)}  test:{len(test_ids)}  union:{len(listed)}")

    # 4) Problems in lists
    overlap_tv  = train_ids & val_ids
    overlap_tt  = train_ids & test_ids
    overlap_vt  = val_ids   & test_ids
    if overlap_tv or overlap_tt or overlap_vt:
        print("\n[ERROR] Overlaps between splits detected:")
        if overlap_tv: print(f"  train ∩ val  = {len(overlap_tv)}")
        if overlap_tt: print(f"  train ∩ test = {len(overlap_tt)}")
        if overlap_vt: print(f"  val   ∩ test = {len(overlap_vt)}")

    # 5) Missing from lists
    missing_from_lists_eligible = sorted(eligible - listed)
    print(f"\n[missing] eligible videos not present in any split: {len(missing_from_lists_eligible)}")
    if show_n:
        for vid in missing_from_lists_eligible[:show_n]:
            print("  ", vid, "label=", report[vid][1])

    # 6) Extra items in lists
    in_lists_not_on_disk = sorted(listed - all_dirs)
    print(f"\n[extra] listed but directory missing: {len(in_lists_not_on_disk)}")
    if show_n:
        for vid in in_lists_not_on_disk[:show_n]:
            print("  ", vid)

    listed_not_eligible = sorted({v for v in listed if v in all_dirs and report[v][0] != "eligible"})
    print(f"\n[extra] listed but not eligible (no/invalid label or CSV issue): {len(listed_not_eligible)}")
    if show_n:
        for vid in listed_not_eligible[:show_n]:
            st, info = report[vid]
            print(f"  {vid}  status={st}  info={info}")

    # 7) Reasons for non-eligible
    reasons = Counter(st for _, (st, _) in report.items() if st != "eligible")
    if reasons:
        print("\n[non-eligible reasons] (among all directories):")
        for k, v in reasons.items():
            print(f"  {k:16s} : {v}")

    # --- NEW SECTION: list missing CSVs explicitly ---
    missing_csv = [vid for vid, (st, _) in report.items() if st == "no_csv"]
    print(f"\n[missing CSV] videos with no {COMBINED_CSV_NAME}: {len(missing_csv)}")
    if show_n:
        for vid in missing_csv[:show_n]:
            print("  ", vid, "→", os.path.join(OUTPUT_DIR, vid, COMBINED_CSV_NAME))

    # Optional: save missing CSVs to file
    if args.save_missing and missing_csv:
        out_txt = os.path.join(Project_DIR, "missing_csv_videos.txt")
        with open(out_txt, "w") as f:
            for vid in sorted(missing_csv):
                f.write(f"{vid}\t{os.path.join(OUTPUT_DIR, vid, COMBINED_CSV_NAME)}\n")
        print(f"[write] Saved list to: {out_txt}")

    # 8) Missing-by-label breakdown
    by_lab = defaultdict(list)
    for vid in missing_from_lists_eligible:
        by_lab[report[vid][1]].append(vid)
    if by_lab:
        print("\n[missing by label] (eligible but not in any split)")
        for lab in sorted(by_lab.keys()):
            vids = by_lab[lab]
            print(f"  {lab}: {len(vids)}")
            if show_n:
                for vid in vids[:show_n]: print("     ", vid)


if __name__ == "__main__":
    main()

