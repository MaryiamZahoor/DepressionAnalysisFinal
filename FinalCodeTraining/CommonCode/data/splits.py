import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

# --- add project root to sys.path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    OUTPUT_DIR, COMBINED_CSV_NAME, SEED,
    emotion_to_idx,
)
#SPLIT LISTS
SPLIT_PATH= "/media/root918/DATA/Projects/Projects/DepressionAnalysis/CREMAD_EXP/Project/data/2SplitTrain_Val/"

TRAIN_LIST   = os.path.join(SPLIT_PATH, "train_videos_full.txt")
VAL_LIST     = os.path.join(SPLIT_PATH, "val_videos_full.txt")
TEST_LIST    = os.path.join(SPLIT_PATH, "test_videos_full.txt")

#INCLUDE_LIST = "/media/root918/DATA/Projects/Projects/DepressionAnalysis/CREMAD_EXP/matching_videos.txt"   # one video_id per line, or None
INCLUDE_LIST = None
EXCLUDE_LIST = "/media/root918/DATA/Projects/Projects/DepressionAnalysis/CREMAD_EXP/Project/exclude_videos.txt"   # one video_id per line, or None

SPLIT_LABEL_COL = "Actual_Emotion"  # or "GT_Emotion"

# -------- helpers --------

def _read_list(path):
    """Read newline-separated IDs; return None if path is falsy or file missing."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        s = {ln.strip() for ln in f if ln.strip()}
    return s or None

def _write(path, series):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for vid in series.tolist():
            f.write(vid + "\n")


def _scan_videos(include_set=None, exclude_set=None, verbose=True):
    """
    Walk OUTPUT_DIR, keep only directories in include_set (if provided),
    always drop anything in exclude_set. Exclude wins on overlap.
    Return DataFrame [video_id, label].
    """
    present_dirs = {d for d in os.listdir(OUTPUT_DIR)
                    if os.path.isdir(os.path.join(OUTPUT_DIR, d))}

    # Start from allowed candidates
    if include_set is not None:
        allowed = present_dirs & include_set
        missing_from_include = include_set - present_dirs
        if verbose and missing_from_include:
            print(f"[warn] {len(missing_from_include)} IDs in include list "
                  f"not found on disk (showing up to 10):",
                  sorted(list(missing_from_include))[:10])
    else:
        allowed = set(present_dirs)

    # Exclude wins on overlap
    if exclude_set:
        allowed_before = len(allowed)
        allowed -= exclude_set
        if verbose:
            overlap = (exclude_set & present_dirs)
            if overlap:
                print(f"[info] excluded {len(exclude_set)} IDs "
                      f"(on-disk overlap removed: {len(exclude_set & allowed)})")
        if verbose and allowed_before != len(allowed):
            print(f"[info] candidates after exclude: {len(allowed)}")

    rows = []
    dropped_no_csv = dropped_bad_csv = dropped_no_labels = dropped_invalid_lab = 0

    for vid in sorted(allowed):
        vdir = os.path.join(OUTPUT_DIR, vid)
        csvp = os.path.join(vdir, COMBINED_CSV_NAME)
        if not os.path.isfile(csvp):
            dropped_no_csv += 1
            continue
        try:
            s = pd.read_csv(csvp, usecols=[SPLIT_LABEL_COL])[SPLIT_LABEL_COL]
        except Exception:
            dropped_bad_csv += 1
            continue
        s = s.dropna().astype(str).str.upper()
        if s.empty:
            dropped_no_labels += 1
            continue
        lab = s.mode().iat[0]
        if lab not in emotion_to_idx:
            dropped_invalid_lab += 1
            continue
        rows.append((vid, lab))

    if verbose:
        print(f"[scan] candidates on disk after include/exclude: {len(allowed)}")
        print(f"[scan] usable (eligible) videos: {len(rows)}")
        if dropped_no_csv or dropped_bad_csv or dropped_no_labels or dropped_invalid_lab:
            print("[scan] dropped:",
                  f"no_csv={dropped_no_csv},",
                  f"csv_read_error={dropped_bad_csv},",
                  f"no_labels={dropped_no_labels},",
                  f"invalid_label={dropped_invalid_lab}")

    if not rows:
        raise RuntimeError("No labeled videos found after include/exclude filtering.")

    return pd.DataFrame(rows, columns=["video_id", "label"])

def _check_stratify_ok(df, val_ratio, test_ratio):
    """
    Validate that a stratified split is feasible.
    - If test_ratio > 0 (two stratified splits), require >=3 samples per class.
    - If test_ratio == 0 (single stratified split), require >=2 samples per class.
    """
    counts = df["label"].value_counts().sort_index()
    if test_ratio > 0:
        min_needed = 3
        if (counts < min_needed).any():
            raise ValueError(
                "Not enough samples in some classes for stratified train/val/test splitting.\n"
                f"Counts per label:\n{counts.to_string()}\n"
                f"Each class needs at least {min_needed} samples."
            )
    else:
        min_needed = 2
        if (counts < min_needed).any():
            raise ValueError(
                "Not enough samples in some classes for a stratified train/val split.\n"
                f"Counts per label:\n{counts.to_string()}\n"
                f"Each class needs at least {min_needed} samples."
            )


# -------- main API --------

def make_or_load_splits(train_ratio=0.9, val_ratio=0.1, test_ratio=0.0,
                        force=False, include_list_path=None, exclude_list_path=None):
    """
    Creates (or reuses) video lists.
    When test_ratio == 0.0, uses the entire dataset for a single stratified split:
      - train: 90%
      - val  : 10%
    No TEST_LIST is created in that case.
    """
    include_list_path = include_list_path or (globals().get("INCLUDE_LIST", None))
    exclude_list_path = exclude_list_path or (globals().get("EXCLUDE_LIST", None))

    # Allow reuse only if the expected lists exist for this mode
    if not force:
        if test_ratio > 0:
            if os.path.isfile(TRAIN_LIST) and os.path.isfile(VAL_LIST) and os.path.isfile(TEST_LIST):
                print("[split] Using existing train/val/test lists.")
                return TRAIN_LIST, VAL_LIST, TEST_LIST
        else:
            if os.path.isfile(TRAIN_LIST) and os.path.isfile(VAL_LIST):
                print("[split] Using existing train/val lists (no test).")
                return TRAIN_LIST, VAL_LIST, TEST_LIST  # TEST_LIST may not exist (that’s fine)

    include_set = _read_list(include_list_path)
    exclude_set = _read_list(exclude_list_path)

    if include_set is not None:
        print(f"[cfg] include list provided: {len(include_set)} IDs")
    else:
        print("[cfg] no include list (use all videos on disk)")
    if exclude_set is not None:
        print(f"[cfg] exclude list provided: {len(exclude_set)} IDs")
    else:
        print("[cfg] no exclude list")

    df_vid = _scan_videos(include_set=include_set, exclude_set=exclude_set, verbose=True)

    # sanity for stratification
    _check_stratify_ok(df_vid, val_ratio=val_ratio, test_ratio=test_ratio)

    if test_ratio > 0:
        # ---- original 3-way path (unchanged) ----
        vid_trainval, vid_test = train_test_split(
            df_vid, test_size=test_ratio, stratify=df_vid["label"],
            random_state=SEED, shuffle=True
        )
        val_share = val_ratio / (train_ratio + val_ratio)
        vid_train, vid_val = train_test_split(
            vid_trainval, test_size=val_share, stratify=vid_trainval["label"],
            random_state=SEED, shuffle=True
        )

        # write
        os.makedirs(os.path.dirname(TRAIN_LIST), exist_ok=True)
        _write(TRAIN_LIST, vid_train["video_id"])
        _write(VAL_LIST,   vid_val["video_id"])
        _write(TEST_LIST,  vid_test["video_id"])

        print(f"[split] Created lists — train:{len(vid_train)}  val:{len(vid_val)}  test:{len(vid_test)}")
        print("[split] label counts (train):"); print(vid_train["label"].value_counts().sort_index())
        print("[split] label counts (val):");   print(vid_val["label"].value_counts().sort_index())
        print("[split] label counts (test):");  print(vid_test["label"].value_counts().sort_index())

    else:
        # ---- NEW: 2-way path (train/val only) ----
        # one stratified split over the whole dataset
        vid_train, vid_val = train_test_split(
            df_vid, test_size=val_ratio, stratify=df_vid["label"],
            random_state=SEED, shuffle=True
        )

        # write train/val only
        os.makedirs(os.path.dirname(TRAIN_LIST), exist_ok=True)
        _write(TRAIN_LIST, vid_train["video_id"])
        _write(VAL_LIST,   vid_val["video_id"])

        # optionally remove stale TEST_LIST if it exists
        if os.path.isfile(TEST_LIST):
            try:
                os.remove(TEST_LIST)
                print(f"[split] Removed old TEST_LIST: {TEST_LIST}")
            except OSError:
                pass

        print(f"[split] Created lists — train:{len(vid_train)}  val:{len(vid_val)}  (no test)")
        print("[split] label counts (train):"); print(vid_train["label"].value_counts().sort_index())
        print("[split] label counts (val):");   print(vid_val["label"].value_counts().sort_index())

    return TRAIN_LIST, VAL_LIST, TEST_LIST


# If you want to run this file directly:
if __name__ == "__main__":
    make_or_load_splits(force=True)

