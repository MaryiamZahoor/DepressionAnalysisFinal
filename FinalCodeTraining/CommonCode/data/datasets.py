import os,sys, numpy as np, pandas as pd, torch

# --- add project root to sys.path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
    
from torch.utils.data import Dataset
from utils.features import harmonize_vgg_cols, pick_ordered_feature_cols, pick_present_aus
from config import USE_RESNET, USE_VGG_FC6, COMBINED_CSV_NAME, emotion_to_idx, SPLIT_LABEL_COL,SKIP_FRAME


def build_au_master(use_binary, use_regression):
    AU_binary = ["AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c"]
    AU_reg    = ["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r"]
    cols=[]
    if use_binary: cols += AU_binary
    if use_regression: cols += AU_reg
    return cols

def _read_ids(path):
    with open(path) as f: return [ln.strip() for ln in f if ln.strip()]

class FrameLevelEmotionDataset(Dataset):
    def __init__(self, root_folder, list_path, au_cols_master,
                 use_resnet=USE_RESNET, use_vgg=USE_VGG_FC6, feature_cols_lock=None,label_col=SPLIT_LABEL_COL,skip_first_n = SKIP_FRAME, combined_csv_name=COMBINED_CSV_NAME):
        self.samples=[]
        self.feature_cols = feature_cols_lock
        vids = _read_ids(list_path)
        self.label_col = label_col
        self.skip_first_n = skip_first_n
        self.combined_csv_name = combined_csv_name

        for vid in vids:
            csvp = os.path.join(root_folder, vid, combined_csv_name)
            if not os.path.isfile(csvp): continue
            try:
                df = pd.read_csv(csvp)
                df = harmonize_vgg_cols(df)
                
                # --- NEW: drop the first N rows for this video ---
                if skip_first_n > 0:
                    df = df.iloc[skip_first_n:].reset_index(drop=True)
                # --------------------------------------------------
                
                if self.feature_cols is None:
                    feats = pick_ordered_feature_cols(df, use_resnet, use_vgg)
                    aus   = pick_present_aus(df, au_cols_master)
                    self.feature_cols = list(feats) + list(aus)

                cols = self.feature_cols
                X = (df[cols].replace([np.inf,-np.inf], np.nan).fillna(0.0).astype("float32"))
                y = df[self.label_col].astype(str).str.upper().map(emotion_to_idx)
                m = y.notna()
                X = X.loc[m]; y=y.loc[m].astype(int)
                for i in range(len(y)):
                    self.samples.append((X.iloc[i].values, int(y.iloc[i])))
            except Exception:
                continue

        if not self.samples: raise RuntimeError("No samples loaded.")
        self.input_dim = len(self.feature_cols)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x,y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

