import os, pandas as pd
from collections import Counter
from config import OUTPUT_DIR, COMBINED_CSV_NAME, TRAIN_LIST, VAL_LIST, TEST_LIST, emotion_to_idx


def _maj(vid):
    s = pd.read_csv(os.path.join(OUTPUT_DIR, vid, COMBINED_CSV_NAME), usecols=["GT_Emotion"])["GT_Emotion"]
    s = s.dropna().astype(str).str.upper()
    return None if s.empty else (s.mode().iat[0] if s.mode().iat[0] in emotion_to_idx else None)

def _summarize(path, name):
    vids = [ln.strip() for ln in open(path) if ln.strip()]
    cnt = Counter(_maj(v) for v in vids); cnt.pop(None, None)
    total = sum(cnt.values())
    print(f"\n[{name}] total videos: {total}")
    for lab, idx in sorted(emotion_to_idx.items(), key=lambda kv: kv[1]):
        c = cnt.get(lab, 0); pct = 100.0*c/total if total>0 else 0
        print(f"  {idx}:{lab}  {c:4d}  ({pct:5.2f}%)")

def main():
    _summarize(TRAIN_LIST, "TRAIN")
    _summarize(VAL_LIST,   "VAL")
    _summarize(TEST_LIST,  "TEST")

if __name__ == "__main__":
    main()

