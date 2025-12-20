# data/seq_dataset.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.features import harmonize_vgg_cols, pick_ordered_feature_cols


def build_au_master(use_binary, use_regression):
    AU_binary = ["AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c",
                 "AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c",
                 "AU20_c","AU23_c","AU25_c","AU26_c"]
    AU_reg    = ["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r",
                 "AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r",
                 "AU20_r","AU23_r","AU25_r","AU26_r"]
    cols = []
    if use_binary:    cols += AU_binary
    if use_regression: cols += AU_reg
    return cols


class VideoSequenceDataset(Dataset):
    """
    Builds sliding windows (length T, stride) from per-video CSVs.
    - Keeps only frames with valid labels (present in emotion_to_idx).
    - Window label = mode over labels within the window.
    - Short videos (N < T) are **not skipped**: we create ONE window padded
      by repeating the last frame until length T.
    - If feature_cols_lock is None, it is discovered from the first valid CSV.
    """
    def __init__(self,
                 root_folder: str,
                 list_path: str,
                 emotion_to_idx: dict,
                 feature_cols_lock: list[str] | None,
                 label_col: str,
                 T: int = 30,
                 stride: int = 30,
                 skip_first_n: int = 0,
                 use_resnet: bool = False,
                 use_vgg: bool = True,
                 au_cols_master: list[str] | None = None,
                 combined_csv_name: str = "combined.csv"):
        super().__init__()
        self.root = root_folder
        self.combined_csv_name = combined_csv_name

        self.label_col     = label_col
        self.T             = int(T)
        self.stride        = int(stride)
        self.skip_first_n  = max(0, int(skip_first_n))

        self.emotion_to_idx = emotion_to_idx
        self.use_resnet     = use_resnet
        self.use_vgg        = use_vgg
        self.au_cols_master = au_cols_master or []

        with open(list_path) as f:
            self.vids = [ln.strip() for ln in f if ln.strip()]

        self.feature_cols = None if feature_cols_lock is None else list(feature_cols_lock)
        self.input_dim    = None if self.feature_cols is None else len(self.feature_cols)

        # samples: (vid, start_idx, window_np[T,D], label_int)
        self.samples: list[tuple[str, int, np.ndarray, int]] = []
        self._build()

    def _discover_cols(self, df: pd.DataFrame):
        df = harmonize_vgg_cols(df)
        feat_cols = pick_ordered_feature_cols(df, self.use_resnet, self.use_vgg)
        au_cols   = [c for c in self.au_cols_master if c in df.columns]
        cols = feat_cols + au_cols
        return df, cols

    @staticmethod
    def _pad_to_len(arr_2d: np.ndarray, T: int) -> np.ndarray:
        """
        arr_2d: (N, D), N <= T. Pad by repeating the last row until length T.
        Returns (T, D) float32.
        """
        N, D = arr_2d.shape
        if N == 0:
            # no frames; return zeros (won't happen if we check earlier)
            return np.zeros((T, D), dtype=np.float32)
        if N >= T:
            return arr_2d[:T].astype(np.float32, copy=False)
        pad = np.repeat(arr_2d[[-1], :], T - N, axis=0)
        out = np.concatenate([arr_2d, pad], axis=0)
        return out.astype(np.float32, copy=False)

    def _build(self):
        for vid in self.vids:
            csvp = os.path.join(self.root, vid, self.combined_csv_name)
            if not os.path.isfile(csvp):
                continue
            try:
                df = pd.read_csv(csvp)
            except Exception:
                continue

            df, cols_candidate = self._discover_cols(df)

            # lock feature columns if not provided yet and candidate exists
            if self.feature_cols is None:
                self.feature_cols = cols_candidate
                self.input_dim = len(self.feature_cols)

            # ensure all required feature columns exist
            if (not self.feature_cols) or any(c not in df.columns for c in self.feature_cols):
                continue

            # features and labels
            feats = (df[self.feature_cols]
                     .replace([np.inf, -np.inf], np.nan)
                     .fillna(0.0)
                     .astype("float32"))

            if self.label_col not in df.columns:
                # if missing label col, skip this video
                continue

            lbl_series = df[self.label_col].astype(str).str.upper()
            lbl_map    = lbl_series.map(self.emotion_to_idx)
            mask_ok    = lbl_map.notna()

            feats = feats.loc[mask_ok].reset_index(drop=True)
            lbl_map = lbl_map.loc[mask_ok].astype(int).reset_index(drop=True)

            # optional initial skip
            if self.skip_first_n > 0:
                if len(feats) <= self.skip_first_n:
                    # no usable frames remain, but still record a **padded** sample of whatever is there?
                    # Here: if nothing remains, fall back to "no frames" -> skip this video entirely.
                    continue
                feats = feats.iloc[self.skip_first_n:].reset_index(drop=True)
                lbl_map = lbl_map.iloc[self.skip_first_n:].reset_index(drop=True)

            N = len(feats)
            if N == 0:
                continue

            X = feats.values  # (N, D)
            y = lbl_map.values  # (N,)

            # --- Case 1: short video (N < T) → create ONE padded window starting at 0 ---
            if N < self.T:
                w = self._pad_to_len(X, self.T)  # (T, D), last-frame repeated
                lab = int(np.bincount(y).argmax())  # mode over available frames
                self.samples.append((vid, 0, w, lab))
                continue  # done with this video

            # --- Case 2: N >= T → standard sliding windows ---
            # windows: s = 0, stride, 2*stride, ... while s+T <= N
            for s in range(0, N - self.T + 1, self.stride):
                w = X[s:s + self.T]     # (T, D)
                wy = y[s:s + self.T]
                lab = int(np.bincount(wy).argmax())  # mode within window
                self.samples.append((vid, s, w.astype(np.float32, copy=False), lab))

            # ensure tail coverage: if final start wasn't exactly N-T, add last aligned window
            if (N - self.T) >= 0 and (len(self.samples) == 0 or self.samples[-1][1] != (N - self.T)):
                s = N - self.T
                w = X[s:s + self.T]
                wy = y[s:s + self.T]
                lab = int(np.bincount(wy).argmax())
                self.samples.append((vid, s, w.astype(np.float32, copy=False), lab))

        if self.feature_cols is None:
            raise RuntimeError("Could not discover feature columns from any video.")

        if self.input_dim is None:
            self.input_dim = len(self.feature_cols)

        if len(self.samples) == 0:
            raise RuntimeError("No sequence windows built. Check paths/labels/lengths.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, _, w, lab = self.samples[idx]
        return torch.tensor(w, dtype=torch.float32), torch.tensor(lab, dtype=torch.long)

