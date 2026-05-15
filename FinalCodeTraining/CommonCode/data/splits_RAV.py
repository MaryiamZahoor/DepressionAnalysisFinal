import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

# --- add project root to sys.path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTPUT_DIR= "/media/root918/OS/Project/copiedFilesRAVDESS/"          # RAVDESS processed root: Actor_xx/<vid_dir>/<CSV>
SEED       = 42

# =========================
# RAVDESS-specific config
# =========================
COMBINED_CSV_NAME = "au_resnet_vgg_with_gt.csv"
PREFERRED_LABEL_COLS = "emotion"

# -------- SPLIT LIST PATHS (edit to your preferred location) --------
SPLIT_PATH = "/media/root918/OS/Project/CNN_RNN_RAVDESS/data/"

TRAIN_LIST = os.path.join(SPLIT_PATH, "train_videos_RAV.txt")
VAL_LIST   = os.path.join(SPLIT_PATH, "val_videos_RAV.txt")
TEST_LIST  = os.path.join(SPLIT_PATH, "test_videos_RAV.txt")

INCLUDE_LIST = None
EXCLUDE_LIST = None

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

def _pick_label_column(df: pd.DataFrame) -> str | None:
        if PREFERRED_LABEL_COLS in df.columns:
            return PREFERRED_LABEL_COLS
        return None

def _norm_label(s) -> str | None:
    """
    Normalize label for consistent stratification.
    Keeps the original class vocabulary (no mapping/dropping).
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # normalize case; RAVDESS commonly has lower/Title case strings
    return s.lower()

def _scan_videos(include_set=None, exclude_set=None, verbose=True):
    """
    Walk OUTPUT_DIR (RAVDESS-style: Actor_xx/<vid_dir>), keep only directories in include_set (if provided),
    always drop anything in exclude_set. Exclude wins on overlap.
    Return DataFrame [video_id, label] where label is normalized string (e.g., 'neutral','calm','surprised',...).
    """
    present_dirs = set()
    for actor in sorted(d for d in os.listdir(OUTPUT_DIR) if d.startswith("Actor_")):
        ap = os.path.join(OUTPUT_DIR, actor)
        if not os.path.isdir(ap):
            continue
        for vid in sorted(os.listdir(ap)):
            vp = os.path.join(ap, vid)
            if os.path.isdir(vp):
                present_dirs.add(f"{actor}/{vid}")

    # Start from allowed candidates
    if include_set is not None:
        allowed = present_dirs & include_set
        missing_from_include = include_set - present_dirs
        if verbose and missing_from_include:
            print(f"[warn] {len(missing_from_include)} IDs in include list not found on disk (up to 10 shown):",
                  sorted(list(missing_from_include))[:10])
    else:
        allowed = set(present_dirs)

    # Exclude wins on overlap
    if exclude_set:
        allowed_before = len(allowed)
        allowed -= exclude_set
        if verbose and allowed_before != len(allowed):
            print(f"[info] candidates after exclude: {len(allowed)}")

    rows = []
    dropped_no_csv = dropped_bad_csv = dropped_no_labels = 0

    for vid in sorted(allowed):
        vdir = os.path.join(OUTPUT_DIR, vid)
        csvp = os.path.join(vdir, COMBINED_CSV_NAME)
        
        if not os.path.isfile(csvp):
            dropped_no_csv += 1
            continue
        try:
            df = pd.read_csv(csvp)
        except Exception:
            dropped_bad_csv += 1
            continue

        lbl_col = _pick_label_column(df)
        if lbl_col is None:
            dropped_no_labels += 1
            continue

        s = df[lbl_col].dropna().astype(str)
        if s.empty:
            dropped_no_labels += 1
            continue

        # Normalize and compute per-video majority label over all classes
        mapped = s.map(_norm_label).dropna()
        if mapped.empty:
            dropped_no_labels += 1
            continue

        lab = mapped.mode().iat[0]  # majority label string (e.g., 'calm', 'surprised')
        rows.append((vid, lab))

    if verbose:
        print(f"[scan] RAVDESS candidates on disk after include/exclude: {len(allowed)}")
        print(f"[scan] usable (eligible) videos: {len(rows)}")
        if dropped_no_csv or dropped_bad_csv or dropped_no_labels:
            print("[scan] dropped:",
                  f"no_csv={dropped_no_csv},",
                  f"csv_read_error={dropped_bad_csv},",
                  f"no_labels_or_empty={dropped_no_labels}")

    if not rows:
        raise RuntimeError("No labeled videos found after include/exclude filtering.")

    return pd.DataFrame(rows, columns=["video_id", "label"])

def _check_stratify_ok(df, val_ratio, test_ratio):
    """
    Validate that a stratified split is feasible across *all* classes.
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

def make_or_load_splits(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                        force=False, include_list_path=None, exclude_list_path=None):
    """
    Creates (or reuses) video lists for RAVDESS with *all classes kept*.
    Default: 80/10/10 with stratification by the exact label strings found in CSVs.
    When test_ratio == 0.0, creates only TRAIN/VAL (no TEST_LIST).
    """
    include_list_path = include_list_path or (globals().get("INCLUDE_LIST", None))
    exclude_list_path = exclude_list_path or (globals().get("EXCLUDE_LIST", None))

    # Reuse existing lists unless force=True
    if not force:
        if test_ratio > 0:
            if os.path.isfile(TRAIN_LIST) and os.path.isfile(VAL_LIST) and os.path.isfile(TEST_LIST):
                print("[split] Using existing train/val/test lists.")
                return TRAIN_LIST, VAL_LIST, TEST_LIST
        else:
            if os.path.isfile(TRAIN_LIST) and os.path.isfile(VAL_LIST):
                print("[split] Using existing train/val lists (no test).")
                return TRAIN_LIST, VAL_LIST, TEST_LIST

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

    # sanity for stratification across *all* classes
    _check_stratify_ok(df_vid, val_ratio=val_ratio, test_ratio=test_ratio)

    if test_ratio > 0:
        # 3-way split: (train+val) vs test, then train vs val
        vid_trainval, vid_test = train_test_split(
            df_vid, test_size=test_ratio, stratify=df_vid["label"],
            random_state=SEED, shuffle=True
        )
        val_share = val_ratio / (train_ratio + val_ratio)
        vid_train, vid_val = train_test_split(
            vid_trainval, test_size=val_share, stratify=vid_trainval["label"],
            random_state=SEED, shuffle=True
        )

        os.makedirs(os.path.dirname(TRAIN_LIST), exist_ok=True)
        _write(TRAIN_LIST, vid_train["video_id"])
        _write(VAL_LIST,   vid_val["video_id"])
        _write(TEST_LIST,  vid_test["video_id"])

        print(f"[split] Created lists — train:{len(vid_train)}  val:{len(vid_val)}  test:{len(vid_test)}")
        print("[split] label counts (train):"); print(vid_train["label"].value_counts().sort_index())
        print("[split] label counts (val):");   print(vid_val["label"].value_counts().sort_index())
        print("[split] label counts (test):");  print(vid_test["label"].value_counts().sort_index())

    else:
        # 2-way split: train/val only
        vid_train, vid_val = train_test_split(
            df_vid, test_size=val_ratio, stratify=df_vid["label"],
            random_state=SEED, shuffle=True
        )

        os.makedirs(os.path.dirname(TRAIN_LIST), exist_ok=True)
        _write(TRAIN_LIST, vid_train["video_id"])
        _write(VAL_LIST,   vid_val["video_id"])

        # clean up old test list if any
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
    # Example: force a fresh 80/10/10 split across all classes
    make_or_load_splits(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, force=True)

