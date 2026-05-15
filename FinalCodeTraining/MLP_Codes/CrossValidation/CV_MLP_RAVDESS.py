#!/usr/bin/env python3
# crossval_rav_best_from_csv.py
# Pick best-by test_frame_acc from RAV CSV, then 10-fold CV over ALL labeled videos.

import os, re, json, math, gc, random
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# ---- project imports (yours) ----
import config as CFG
from models.TwoLayerMLP import FrameClassifier
from utils.features import save_feature_cols, Standardize, harmonize_vgg_cols
from data.datasets import build_au_master

# ======================
#      CONSTANTS
# ======================
K_FOLDS     = 10
MAX_EPOCHS  = CFG.EPOCHS
ES_PATIENCE = 15
ES_MONITOR  = "val_acc"
SKIP_FIRST_N= CFG.SKIP_FRAME
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME_FOLDS= True
SKIP_DONE   = True

CPU_COUNT       = os.cpu_count() or 4
NUM_WORKERS     = min(8, max(0, CPU_COUNT - 2))
PIN_MEMORY      = torch.cuda.is_available()
PREFETCH_FACTOR = 4
def _loader_kws():
    base = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if NUM_WORKERS > 0: base.update(dict(prefetch_factor=PREFETCH_FACTOR, persistent_workers=True))
    return base

def _safe_collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)

def _cur_lr(opt): return opt.param_groups[0]['lr'] if opt.param_groups else float('nan')
def _set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

# ---------- RAVDESS paths ----------
SPLIT_PATH = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/data/"
TRAIN_LIST = os.path.join(SPLIT_PATH, "train_videos_RAV.txt")
VAL_LIST   = os.path.join(SPLIT_PATH, "val_videos_RAV.txt")
TEST_LIST  = os.path.join(SPLIT_PATH, "test_videos_RAV.txt")

PROJECT_DIR = "/media/root918/OS/[REDACTED]Project/CNN_RNN_CREMAD/"
ART_DIR_TAG = "ravdess_GridSearch_unscaled_MLP"
ART_DIR_SUB = os.path.join(PROJECT_DIR, "artifacts", ART_DIR_TAG)
GRID_OUT_DIR= os.path.join(ART_DIR_SUB, "grid_RAV")
CONFIGS_DIR = os.path.join(ART_DIR_SUB, "cv_best_from_csv")
os.makedirs(GRID_OUT_DIR, exist_ok=True); os.makedirs(CONFIGS_DIR, exist_ok=True)

# CSV with columns: use_vgg,use_resnet,use_au_c,use_au_r,hidden_dim,hidden_dim2,optimizer,lr,weight_decay,dropout,batch_size,test_frame_acc
RAV_RESULTS_CSV = os.path.join(GRID_OUT_DIR, "grid_results_val_train_test.csv")

# caches / features
OUTPUT_DIR = "/media/root918/OS/[REDACTED]Project/copiedFilesRAVDESS/"
MASTER_FEATURES_JSON = os.path.join(GRID_OUT_DIR, "master_feature_cols.json")
IDS_LABELS_CACHE     = os.path.join(GRID_OUT_DIR, "ids_labels_all.json")
MASTER_SCAN_LIMIT    = 10

# Labels / columns (RAVDESS 8 classes)
LABEL_COL         = "emotion"
EMOTION_TO_IDX = {
    "neutral":0,"calm":1,"happy":2,"sad":3,"anger":4,"fearful":5,"disgust":6,"surprise":7
}
IDX_TO_EMO = {v:k for k,v in EMOTION_TO_IDX.items()}

DO_STANDARDIZE = True
KEEP_AU_C_RAW  = True

# ======================
#   MASTER/CACHE HELPERS
# ======================
def _require_file(p, desc):
    if not os.path.isfile(p): raise FileNotFoundError(f"Missing {desc}: {p}")
    return p

def _read_ids(p: str) -> List[str]:
    with open(p) as f: return [ln.strip() for ln in f if ln.strip()]

def _vid_csv_path(vid: str) -> str:
    return os.path.join(OUTPUT_DIR, vid, "au_resnet_vgg_with_gt.csv")

def _cache_paths(vid: str):
    cdir = os.path.join(OUTPUT_DIR, vid, "cache_RAVDESS")
    return os.path.join(cdir,"X.npy"), os.path.join(cdir,"y.npy"), os.path.join(cdir,"meta.json")

def get_master_feature_cols(ids_all: List[str]) -> List[str]:
    if os.path.isfile(MASTER_FEATURES_JSON):
        cols = json.load(open(MASTER_FEATURES_JSON))
        print(f"[features] Loaded master list ({len(cols)}) → {MASTER_FEATURES_JSON}")
        return cols
    master, picked = [], 0
    for vid in ids_all:
        csvp = _vid_csv_path(vid)
        if not os.path.isfile(csvp): continue
        try:
            df = pd.read_csv(csvp, nrows=1)
            df = harmonize_vgg_cols(df)
            cnn_cols = [c for c in df.columns if c.endswith("_vgg") or c.endswith("_resnet")]
            for c in cnn_cols:
                if c not in master: master.append(c)
            picked += 1
            if picked >= MASTER_SCAN_LIMIT: break
        except Exception as e:
            print(f"[features] warn: {csvp} -> {e}")
    au_all = build_au_master(True, True)
    for a in au_all:
        if a not in master: master.append(a)
    json.dump(master, open(MASTER_FEATURES_JSON,"w"), indent=2)
    print(f"[features] Saved master list ({len(master)}) → {MASTER_FEATURES_JSON}")
    return master

def build_video_cache_master(vid: str, master_cols: List[str]) -> None:
    xnp, ynp, meta = _cache_paths(vid)
    os.makedirs(os.path.dirname(xnp), exist_ok=True)
    if os.path.isfile(xnp) and os.path.isfile(ynp) and os.path.isfile(meta): return
    csvp = _vid_csv_path(vid)
    if not os.path.isfile(csvp):
        np.save(xnp, np.empty((0, len(master_cols)), np.float32))
        np.save(ynp, np.empty((0,), np.int64)); json.dump({"feature_cols": master_cols}, open(meta,"w"))
        return
    df = pd.read_csv(csvp); df = harmonize_vgg_cols(df)
    if SKIP_FIRST_N>0: df = df.iloc[SKIP_FIRST_N:].reset_index(drop=True)
    y = (df[LABEL_COL].astype(str).str.strip().str.lower()
         .map(EMOTION_TO_IDX).dropna().astype(np.int64))
    idx = y.index
    cols = []
    for c in master_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]): cols.append(df.loc[idx, c])
        else: cols.append(pd.Series(0.0, index=idx))
    X = (pd.concat(cols,axis=1).replace([np.inf,-np.inf],np.nan).fillna(0.0).astype("float32").to_numpy(copy=True))
    np.save(xnp, X); np.save(ynp, y.to_numpy(copy=True))
    json.dump({"feature_cols": master_cols}, open(meta,"w"))

class CachedFrameDatasetMaster(torch.utils.data.Dataset):
    def __init__(self, vids: List[str], master_cols: List[str], use_vgg, use_resnet, use_au_c, use_au_r):
        self.vids = list(vids); self.master_cols = list(master_cols)
        for i,v in enumerate(self.vids,1):
            build_video_cache_master(v, self.master_cols)
            if i%200==0: print(f"[cache] ensured {i}/{len(self.vids)}")
        want=[]
        for i,c in enumerate(self.master_cols):
            if c.endswith("_vgg") and use_vgg: want.append(i)
            if c.endswith("_resnet") and use_resnet: want.append(i)
            if c.endswith("_c") and use_au_c: want.append(i)
            if c.endswith("_r") and use_au_r: want.append(i)
        self.sel = np.asarray(sorted(set(want)), np.int64)
        if self.sel.size==0: raise ValueError("No columns selected.")
        self.feature_cols=[self.master_cols[i] for i in self.sel.tolist()]
        self.input_dim=len(self.feature_cols)
        self.arrs, self.ranges, total=[], [], 0
        for v in self.vids:
            xnp,ynp,_=_cache_paths(v)
            if not (os.path.isfile(xnp) and os.path.isfile(ynp)): continue
            Xm=np.load(xnp, mmap_mode="r"); ym=np.load(ynp, mmap_mode="r")
            n=len(ym);
            if n==0: continue
            vidx=len(self.arrs); self.arrs.append((Xm,ym)); self.ranges.append((vidx,total,n)); total+=n
        self.total=total
    def __len__(self): return self.total
    def __getitem__(self, i):
        lo,hi=0,len(self.ranges)-1
        while lo<=hi:
            mid=(lo+hi)//2; vidx,start,n=self.ranges[mid]
            if i<start: hi=mid-1
            elif i>=start+n: lo=mid+1
            else:
                Xm,ym=self.arrs[vidx]; off=i-start
                x=torch.from_numpy(np.asarray(Xm[off,self.sel],np.float32).copy())
                y=torch.tensor(int(ym[off]),dtype=torch.long); return x,y
        raise IndexError(i)

# ======================
#   UTILITIES
# ======================
def _compute_mean_std(loader: DataLoader, feature_names: List[str]):
    D=len(feature_names); n=0
    s1=torch.zeros(D, device=DEVICE, dtype=torch.float64)
    s2=torch.zeros(D, device=DEVICE, dtype=torch.float64)
    for xb,_ in loader:
        xb=xb.to(DEVICE, non_blocking=True).float()
        n+=xb.shape[0]; s1+=xb.sum(0).double(); s2+=(xb.double().pow(2)).sum(0)
    mean=(s1/max(1,n)).float()
    var =(s2/max(1,n))-mean.double().pow(2)
    std =torch.sqrt(torch.clamp(var, min=1e-12)).float()
    if DO_STANDARDIZE and KEEP_AU_C_RAW:
        idx=[i for i,nm in enumerate(feature_names) if nm.endswith("_c")]
        if idx:
            idx=torch.tensor(idx,device=DEVICE); mean.index_fill_(0,idx,0.0); std.index_fill_(0,idx,1.0)
    return mean.cpu(), std.cpu()

def _build_opt(name, params, lr, wd):
    if str(name).lower()=="adam": return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)

def _eval(model, loader, scaler):
    model.eval(); ce=nn.CrossEntropyLoss(); tot=0; cor=0; loss=0.0
    with torch.no_grad():
        for xb,yb in loader:
            xb=scaler(xb.to(DEVICE,non_blocking=True).float()); yb=yb.to(DEVICE,non_blocking=True)
            logits=model(xb); loss+=ce(logits,yb).item()
            cor+=(logits.argmax(1)==yb).sum().item(); tot+=yb.numel()
    return (loss/max(1,len(loader))), (cor/max(1,tot))

def _read_all_ids() -> Tuple[List[str], np.ndarray]:
    ids = sorted(set(_read_ids(_require_file(TRAIN_LIST,"TRAIN")) |
                     set(_read_ids(_require_file(VAL_LIST,"VAL"))) |
                     set(_read_ids(_require_file(TEST_LIST,"TEST")))) )

def _scan_labels(ids_all: List[str]) -> Tuple[List[str], np.ndarray]:
    ids_out, y_out = [], []
    for vid in ids_all:
        p=_vid_csv_path(vid)
        if not os.path.isfile(p): continue
        try:
            s=pd.read_csv(p, usecols=[LABEL_COL])[LABEL_COL]
        except Exception: continue
        s=s.dropna().astype(str).str.strip().str.lower()
        if s.empty: continue
        lab=s.mode().iat[0]
        if lab not in EMOTION_TO_IDX: continue
        ids_out.append(vid); y_out.append(EMOTION_TO_IDX[lab])
    if not ids_out: raise RuntimeError("No labeled videos found.")
    return ids_out, np.array(y_out, dtype=np.int64)

def get_ids_and_labels_cached() -> Tuple[List[str], np.ndarray]:
    # union all ids
    train=_read_ids(_require_file(TRAIN_LIST,"TRAIN"))
    val  =_read_ids(_require_file(VAL_LIST,"VAL"))
    test =_read_ids(_require_file(TEST_LIST,"TEST"))
    all_ids=sorted(set(train)|set(val)|set(test))
    if os.path.isfile(IDS_LABELS_CACHE):
        try:
            d=json.load(open(IDS_LABELS_CACHE)); ids=list(d.get("ids",[])); y=np.array(d.get("labels",[]),dtype=np.int64)
            if len(ids)==len(y) and ids: print(f"[ids] loaded {len(ids)} from cache → {IDS_LABELS_CACHE}"); return ids,y
        except Exception as e: print(f"[ids] cache read failed: {e}; rebuilding…")
    ids,y=_scan_labels(all_ids)
    os.makedirs(os.path.dirname(IDS_LABELS_CACHE), exist_ok=True)
    json.dump({"ids":ids,"labels":y.astype(int).tolist()}, open(IDS_LABELS_CACHE,"w"), indent=2)
    print(f"[ids] saved {len(ids)} → {IDS_LABELS_CACHE}")
    return ids,y

def _best_cfg_from_csv(csv_path: str) -> Dict:
    df=pd.read_csv(csv_path)
    if "test_frame_acc" not in df.columns: raise RuntimeError("CSV missing 'test_frame_acc'")
    df["test_frame_acc"]=pd.to_numeric(df["test_frame_acc"], errors="coerce")
    row=df.sort_values("test_frame_acc", ascending=False).iloc[0]
    cfg=dict(
        use_vgg=bool(row["use_vgg"]),
        use_resnet=bool(row["use_resnet"]),
        use_au_c=bool(row["use_au_c"]),
        use_au_r=bool(row["use_au_r"]),
        hidden_dim=int(float(row["hidden_dim"])),
        hidden_dim2= None if (pd.isna(row.get("hidden_dim2")) or str(row.get("hidden_dim2")).strip().lower() in ("", "none","nan")) else int(float(row["hidden_dim2"])),
        optimizer=str(row["optimizer"]),
        lr=float(row["lr"]),
        weight_decay=float(row["weight_decay"]),
        dropout=float(row["dropout"]),
        batch_size=int(float(row["batch_size"])),
        tag=str(row.get("config","best_from_rav_csv"))
    )
    print(f"[best-from-csv] {cfg['tag']} | test_frame_acc={row['test_frame_acc']:.6f}")
    return cfg

def _fold_dir_for(cfg: Dict, fold: int) -> str:
    tag=re.sub(r"[^A-Za-z0-9_\-+=.]", "_", str(cfg.get("tag","CFG")))[:200]
    d=os.path.join(CONFIGS_DIR, tag, f"fold_{fold}")
    os.makedirs(d, exist_ok=True); return d

def _select_indices(master_cols: List[str], cfg: Dict) -> np.ndarray:
    idx=[]
    for i,c in enumerate(master_cols):
        if c.endswith("_vgg") and cfg["use_vgg"]: idx.append(i)
        if c.endswith("_resnet") and cfg["use_resnet"]: idx.append(i)
        if c.endswith("_c") and cfg["use_au_c"]: idx.append(i)
        if c.endswith("_r") and cfg["use_au_r"]: idx.append(i)
    return np.asarray(sorted(set(idx)), dtype=np.int64)

# ======================
#          MAIN
# ======================
def main():
    _set_seed(1337)

    # 1) pick best RAV config by sequence (video) test accuracy
    best_cfg=_best_cfg_from_csv(_require_file(RAV_RESULTS_CSV,"RAV results CSV"))

    # 2) ids + labels (cached), master features & ensure caches
    ids,y=get_ids_and_labels_cached()
    master_cols=get_master_feature_cols(ids)
    for i,v in enumerate(ids,1):
        build_video_cache_master(v, master_cols)
        if i%300==0: print(f"[cache] ensured {i}/{len(ids)}")

    # 3) CV
    skf=StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=CFG.SEED)
    sel_idx=_select_indices(master_cols, best_cfg)
    fold_rows=[]; yF_all=[]; yFhat_all=[]; yV_all=[]; yVhat_all=[]

    for fold,(tr_idx,va_idx) in enumerate(skf.split(ids,y), start=1):
        tr_ids=[ids[i] for i in tr_idx]; va_ids=[ids[i] for i in va_idx]
        fold_dir=_fold_dir_for(best_cfg, fold)
        metrics_path=os.path.join(fold_dir,"metrics.json")
        last_ckpt=os.path.join(fold_dir,"last_ckpt.pt")
        best_path =os.path.join(fold_dir,"best_state.pt")
        scaler_pt =os.path.join(fold_dir,"scaler.pt")

        if SKIP_DONE and os.path.isfile(metrics_path):
            try:
                m=json.load(open(metrics_path)); 
                if m.get("done",False):
                    print(f"[fold {fold:02d}] done; skipping")
                    fold_rows.append({"fold":fold,"frame_acc":m.get("val_acc",np.nan),
                                      "video_acc":m.get("video_val_acc",np.nan),
                                      "val_loss":m.get("val_loss",np.nan)})
                    continue
            except Exception: pass

        # datasets
        ds_tr=CachedFrameDatasetMaster(tr_ids, master_cols, best_cfg["use_vgg"],best_cfg["use_resnet"],best_cfg["use_au_c"],best_cfg["use_au_r"])
        ds_va=CachedFrameDatasetMaster(va_ids, master_cols, best_cfg["use_vgg"],best_cfg["use_resnet"],best_cfg["use_au_c"],best_cfg["use_au_r"])
        tr_loader=DataLoader(ds_tr, batch_size=best_cfg["batch_size"], shuffle=True,  drop_last=False, collate_fn=_safe_collate, **_loader_kws())
        va_loader=DataLoader(ds_va, batch_size=best_cfg["batch_size"], shuffle=False, drop_last=False, collate_fn=_safe_collate, **_loader_kws())

        # scaler on train fold
        if DO_STANDARDIZE:
            tmp=DataLoader(ds_tr, batch_size=4096, shuffle=False, collate_fn=_safe_collate, **_loader_kws())
            mean,std=_compute_mean_std(tmp, ds_tr.feature_cols)
            torch.save({"mean":mean,"std":std}, scaler_pt)
            scaler=Standardize(mean, std).to(DEVICE)
        else:
            scaler=nn.Identity()

        # model/optim/sched
        model=FrameClassifier(input_dim=ds_tr.input_dim, hidden_dim=best_cfg["hidden_dim"],
                              hidden_dim2=best_cfg["hidden_dim2"], dropout=best_cfg["dropout"],
                              num_classes=len(EMOTION_TO_IDX)).to(DEVICE)
        opt=_build_opt(best_cfg["optimizer"], model.parameters(), best_cfg["lr"], best_cfg["weight_decay"])
        ce=nn.CrossEntropyLoss()
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=5, min_lr=1e-6)

        # resume?
        start_ep=1; best_metric=-math.inf; no_imp=0
        if RESUME_FOLDS and os.path.isfile(last_ckpt):
            print(f"[fold {fold:02d}] resuming")
            ckpt=torch.load(last_ckpt, map_location=DEVICE)
            try: model.load_state_dict(ckpt["model"])
            except Exception as e: print(f"[resume warn] model: {e}")
            if ckpt.get("optim") is not None:
                try: opt.load_state_dict(ckpt["optim"])
                except Exception as e: print(f"[resume warn] optim: {e}")
            if ckpt.get("sched") is not None:
                try: sched.load_state_dict(ckpt["sched"])
                except Exception as e: print(f"[resume warn] sched: {e}")
            best_metric=float(ckpt.get("best_metric", best_metric))
            start_ep=int(ckpt.get("epoch",0))+1

        # train loop (ES on val_acc)
        for ep in range(start_ep, MAX_EPOCHS+1):
            model.train(); run=0.0
            for xb,yb in tr_loader:
                xb=scaler(xb.to(DEVICE,non_blocking=True).float()); yb=yb.to(DEVICE,non_blocking=True)
                opt.zero_grad(set_to_none=True); logits=model(xb); loss=ce(logits,yb)
                loss.backward(); opt.step(); run+=loss.item()
            va_loss,va_acc=_eval(model, va_loader, scaler)
            sched.step(va_acc)
            improved=va_acc>best_metric
            if improved:
                best_metric=va_acc; torch.save(model.state_dict(), best_path); no_imp=0
            else:
                no_imp+=1
            torch.save({"epoch":ep, "model":model.state_dict(), "optim":opt.state_dict(),
                        "sched":sched.state_dict(), "best_metric":best_metric}, last_ckpt)
            print(f"[fold {fold:02d}] ep {ep:03d} | tr_loss {run/max(1,len(tr_loader)):.4f} | va_loss {va_loss:.4f} | va_acc {va_acc:.4f} | best {best_metric:.4f} | lr {_cur_lr(opt):.2e}")
            if no_imp>=ES_PATIENCE: print(f"[fold {fold:02d}] early stop"); break

        # load best & evaluate fold (frame/video acc only; no per-fold reports)
        if os.path.isfile(best_path): model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        # frame
        yf,yfhat=[],[]
        with torch.no_grad():
            for vid in va_ids:
                xnp,ynp,_=_cache_paths(vid)
                if not (os.path.isfile(xnp) and os.path.isfile(ynp)): continue
                Xm=np.load(xnp,mmap_mode="r"); yv=np.load(ynp,mmap_mode="r")
                if Xm.shape[0]==0: continue
                bs=16384
                for i in range(0, Xm.shape[0], bs):
                    chunk=np.array(Xm[i:i+bs][:, ds_tr.sel], copy=True)
                    xb=torch.from_numpy(chunk).float().to(DEVICE,non_blocking=True)
                    xb=scaler(xb); logits=model(xb)
                    yfhat.append(logits.argmax(1).cpu().numpy()); yf.append(yv[i:i+bs].copy())
        if yf: yf=np.concatenate(yf); yfhat=np.concatenate(yfhat); frame_acc=float((yf==yfhat).mean())
        else:  frame_acc=float("nan")
        # video
        yv_true,yv_pred=[],[]
        with torch.no_grad():
            for vid in va_ids:
                xnp,ynp,_=_cache_paths(vid)
                if not (os.path.isfile(xnp) and os.path.isfile(ynp)): continue
                Xm=np.load(xnp,mmap_mode="r"); yv=np.load(ynp,mmap_mode="r")
                if Xm.shape[0]==0: continue
                Xs=np.array(Xm[:, ds_tr.sel], copy=True)
                X=torch.from_numpy(Xs).float().to(DEVICE,non_blocking=True); X=scaler(X)
                probs=torch.softmax(model(X),dim=1).mean(0)
                yv_pred.append(int(probs.argmax().item())); yv_true.append(int(np.bincount(yv).argmax()))
        if yv_true:
            yv_true=np.array(yv_true); yv_pred=np.array(yv_pred); video_acc=float((yv_true==yv_pred).mean())
        else:
            video_acc=float("nan")

        json.dump({"done":True,"val_acc":frame_acc,"video_val_acc":video_acc,"val_loss":float("nan")},
                  open(metrics_path,"w"), indent=2)

        if isinstance(yf, np.ndarray) and yf.size: yF_all.append(yf); yFhat_all.append(yfhat)
        if isinstance(yv_true, np.ndarray) and yv_true.size: yV_all.append(yv_true); yVhat_all.append(yv_pred)
        fold_rows.append({"fold":fold,"frame_acc":frame_acc,"video_acc":video_acc,"val_loss":float("nan")})

        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # 4) Save fold table + pooled reports
    root=os.path.join(CONFIGS_DIR, re.sub(r"[^A-Za-z0-9_\-+=.]", "_", str(best_cfg["tag"]))[:200])
    os.makedirs(root, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(os.path.join(root,"fold_metrics.csv"), index=False)

    classes=[IDX_TO_EMO[i] for i in range(len(IDX_TO_EMO))]
    overall={}
    if yF_all:
        yt=np.concatenate(yF_all); yp=np.concatenate(yFhat_all)
        rep=classification_report(yt, yp, target_names=classes, digits=4, output_dict=True)
        cm =confusion_matrix(yt, yp, labels=list(range(len(IDX_TO_EMO))))
        json.dump(rep, open(os.path.join(root,"frame_report.json"),"w"), indent=2)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(root,"frame_cm.csv"))
        overall["frame_acc_mean_over_folds"]=float(np.nanmean([r["frame_acc"] for r in fold_rows]))
    else:
        overall["frame_acc_mean_over_folds"]=float("nan")

    if yV_all:
        yt=np.concatenate(yV_all); yp=np.concatenate(yVhat_all)
        rep=classification_report(yt, yp, target_names=classes, digits=4, output_dict=True)
        cm =confusion_matrix(yt, yp, labels=list(range(len(IDX_TO_EMO))))
        json.dump(rep, open(os.path.join(root,"video_report.json"),"w"), indent=2)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(root,"video_cm.csv"))
        overall["video_acc_mean_over_folds"]=float(np.nanmean([r["video_acc"] for r in fold_rows]))
    else:
        overall["video_acc_mean_over_folds"]=float("nan")

    json.dump(overall, open(os.path.join(root,"overall_summary.json"),"w"), indent=2)

    print("\n[CV] saved under:", root)
    print("  - per-fold: last_ckpt.pt, best_state.pt, scaler.pt, metrics.json")
    print("  - tables  : fold_metrics.csv, overall_summary.json, frame_report.json/frame_cm.csv, video_report.json/video_cm.csv")

if __name__ == "__main__":
    main()

