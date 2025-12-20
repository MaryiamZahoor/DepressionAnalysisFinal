#!/usr/bin/env python3
import os
import pandas as pd
from typing import Optional, Iterable

# ------------------ EDIT THESE PATHS ------------------

# CREMA-D processed root: <video_id>/<CSV>
CREMA_OUTPUT_DIR     = "/media/root918/OS/MaryiamProject/CREMA-D/copiedFiles/"
CREMA_COMBINED_CSV   = "affwild_resnet_au_vgg_with_gt.csv"
CREMA_LABEL_COL      = "Actual_Emotion"   # or "GT_Emotion"

# RAVDESS split text files and label CSV
RAV_TRAIN_TXT = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/train_videos_RAV.txt"
RAV_VAL_TXT   = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/val_videos_RAV.txt"
RAV_TEST_TXT  = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/test_videos_RAV.txt"

# Example label CSV path with columns: actor_id, emotion, is_song, filename
# filename looks like: "02-01-05-01-01-02-01.mp4"
RAV_LABEL_CSV = "/media/root918/DATA/Projects/Maryiam_Projects/DepressionAnalysis/CREMAD_EXP/RAVDESS_testing/ravdess_groundtruth_video_only.xlsx"

# CREMA-D split files (already created)
CREMA_TRAIN_TXT = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/train_videos_full.txt"
CREMA_VAL_TXT   = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/val_videos_full.txt"
CREMA_TEST_TXT  = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/test_videos_full.txt"  # may not exist

# Output merged split files
OUT_DIR = "/media/root918/OS/MaryiamProject/CNN_RNN_CREMAD/data/"
os.makedirs(OUT_DIR, exist_ok=True)
COMB_TRAIN_TXT = os.path.join(OUT_DIR, "train_videos_COMBINED.txt")
COMB_VAL_TXT   = os.path.join(OUT_DIR, "val_videos_COMBINED.txt")
COMB_TEST_TXT  = os.path.join(OUT_DIR, "test_videos_COMBINED.txt")

# Also write rich CSVs (handy for loaders / debugging)
COMB_TRAIN_CSV = os.path.join(OUT_DIR, "train_videos_COMBINED.csv")
COMB_VAL_CSV   = os.path.join(OUT_DIR, "val_videos_COMBINED.csv")
COMB_TEST_CSV  = os.path.join(OUT_DIR, "test_videos_COMBINED.csv")

# ------------------ LABEL MAPPING ------------------

CANONICAL = {"angry", "disgust", "fear", "happy", "neutral", "sad"}
ALIASES = {
    # canonical + variants
    "angry":"angry","anger":"angry","ANGER":"angry","Anger":"angry","A":"angry",
    "disgust":"disgust","DISGUST":"disgust","Disgust":"disgust","D":"disgust",
    "fear":"fear","fearful":"fear","FEAR":"fear","Fear":"fear","Fearful":"fear","F":"fear",
    "happy":"happy","HAPPY":"happy","Happy":"happy","H":"happy",
    "neutral":"neutral","NEUTRAL":"neutral","Neutral":"neutral","N":"neutral",
    "sad":"sad","sadness":"sad","SAD":"sad","Sad":"sad","S":"sad",
    # drop (RAVDESS-only) for canonical build
    "calm":None,"Calm":None,"CALM":None,
    "surprised":None,"Surprised":None,"SURPRISED":None,
}

def to_canonical(lbl: Optional[str]) -> Optional[str]:
    if lbl is None: return None
    s = str(lbl).strip()
    if s in ALIASES: return ALIASES[s]
    s_low = s.lower()
    return ALIASES.get(s_low, None)

# ------------------ HELPERS ------------------

def read_list(path: str) -> list[str]:
    if not path or not os.path.isfile(path): return []
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def write_txt(path: str, items: Iterable[str]) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(it + "\n")

def majority_label_from_crema_csv(csv_path: str, label_col: str) -> Optional[str]:
    try:
        df = pd.read_csv(csv_path, usecols=[label_col])
        s = df[label_col].dropna().astype(str)
        return None if s.empty else s.mode().iat[0]
    except Exception:
        return None

# RAW (no mapping) label for a CREMA video_id
def crema_raw_label_for_video(vid_id: str) -> Optional[str]:
    csvp = os.path.join(CREMA_OUTPUT_DIR, vid_id, CREMA_COMBINED_CSV)
    if not os.path.isfile(csvp): return None
    return majority_label_from_crema_csv(csvp, CREMA_LABEL_COL)

# Canonical (6-class) label for a CREMA video_id
def crema_canonical_label_for_video(vid_id: str) -> Optional[str]:
    raw = crema_raw_label_for_video(vid_id)
    return to_canonical(raw)

# ------------------ RAVDESS LABEL LOOKUPS ------------------

def _ravdess_df(label_csv_path: str) -> pd.DataFrame:
    df = pd.read_excel(label_csv_path)
    df = df.dropna(subset=["actor_id","emotion","filename"]).copy()
    df["actor_id"] = df["actor_id"].astype(int)
    df["Actor"] = df["actor_id"].apply(lambda x: f"Actor_{x:02d}")
    df["stem"] = df["filename"].astype(str).str.replace(".mp4","",regex=False)
    df["key"] = df["Actor"] + "/" + df["stem"]
    return df

# RAW map (keeps calm, surprised, etc.)
def build_ravdess_raw_label_map(label_csv_path: str) -> dict[str,str]:
    df = _ravdess_df(label_csv_path)
    return dict(zip(df["key"], df["emotion"].astype(str)))

# Canonical map (drops calm/surprised)
def build_ravdess_canonical_label_map(label_csv_path: str) -> dict[str,str]:
    df = _ravdess_df(label_csv_path)
    df["label"] = df["emotion"].map(to_canonical)
    df = df[df["label"].isin(CANONICAL)]
    return dict(zip(df["key"], df["label"]))

# ------------------ BUILD TABLES ------------------

def build_crema_table(ids: list[str], raw: bool=False) -> pd.DataFrame:
    rows = []
    for vid in ids:
        lbl = crema_raw_label_for_video(vid) if raw else crema_canonical_label_for_video(vid)
        if raw:
            if lbl is not None:
                rows.append(("crema", vid, lbl))
        else:
            if lbl in CANONICAL:
                rows.append(("crema", vid, lbl))
    return pd.DataFrame(rows, columns=["dataset","video_id","label"])

def build_ravdess_table(ids: list[str], label_map: dict[str,str], raw: bool=False) -> pd.DataFrame:
    rows = []
    for vid in ids:
        lbl = label_map.get(vid, None)
        if raw:
            if lbl is not None:
                rows.append(("ravdess", vid, lbl))
        else:
            if lbl in CANONICAL:
                rows.append(("ravdess", vid, lbl))
    return pd.DataFrame(rows, columns=["dataset","video_id","label"])

def print_counts(title: str, df: pd.DataFrame) -> None:
    print(title)
    if df.empty:
        print("(no items)\n")
        return
    vc = df["label"].value_counts().sort_index()
    print(vc.to_string())
    print()

def merge_and_write(crema_txt: str, rav_txt: str,
                    out_txt: str, out_csv: str,
                    rav_map_canon: dict[str,str],
                    rav_map_raw: dict[str,str]):
    crema_ids = read_list(crema_txt)
    rav_ids   = read_list(rav_txt)

    # --- canonical (6-class) combined build (unchanged behavior) ---
    df_crema_canon = build_crema_table(crema_ids, raw=False)
    df_rav_canon   = build_ravdess_table(rav_ids, rav_map_canon, raw=False)
    df_all = pd.concat([df_crema_canon, df_rav_canon], ignore_index=True)

    # Stable ID format in TXT
    stable_ids = (df_all["dataset"] + "::" + df_all["video_id"]).tolist()
    write_txt(out_txt, stable_ids)

    # Helpful CSV
    df_all.to_csv(out_csv, index=False)

    # Log summary (combined canonical)
    print(f"[write] {out_txt} (n={len(df_all)})")
    print(df_all["label"].value_counts().sort_index())
    print()

    # --- RAW distributions per dataset (KEEP calm & surprised) ---
    df_crema_raw = build_crema_table(crema_ids, raw=True)
    df_rav_raw   = build_ravdess_table(rav_ids, rav_map_raw, raw=True)

    print_counts("[CREMA-D raw label distribution]", df_crema_raw)
    print_counts("[RAVDESS raw label distribution]", df_rav_raw)

# ------------------ MAIN ------------------

if __name__ == "__main__":
    # build both maps for RAVDESS
    rav_map_canon = build_ravdess_canonical_label_map(RAV_LABEL_CSV)
    rav_map_raw   = build_ravdess_raw_label_map(RAV_LABEL_CSV)

    # train
    merge_and_write(CREMA_TRAIN_TXT, RAV_TRAIN_TXT, COMB_TRAIN_TXT, COMB_TRAIN_CSV, rav_map_canon, rav_map_raw)
    # val
    merge_and_write(CREMA_VAL_TXT,   RAV_VAL_TXT,   COMB_VAL_TXT,   COMB_VAL_CSV,   rav_map_canon, rav_map_raw)
    # test
    merge_and_write(CREMA_TEST_TXT,  RAV_TEST_TXT,  COMB_TEST_TXT,  COMB_TEST_CSV,  rav_map_canon, rav_map_raw)

