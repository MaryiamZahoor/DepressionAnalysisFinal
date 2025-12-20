# utils/run_summary.py
import os, sys, json, torch

def _is_literal(v):
    return isinstance(v, (str, int, float, bool)) or v is None

def config_to_dict(cfg):
    d = {}
    for k in dir(cfg):
        if k.startswith("_"): continue
        v = getattr(cfg, k)
        if callable(v): continue
        if _is_literal(v) or isinstance(v, (list, tuple, dict)):
            d[k] = v
    return d

def describe_device():
    """
    Return device info WITHOUT hard-failing if cuDNN/driver init breaks.
    """
    info = {"device": "cpu"}
    if torch.cuda.is_available():
        try:
            i = torch.cuda.current_device()
            cap = torch.cuda.get_device_capability(i)
            info.update({
                "device": "cuda",
                "cuda_index": i,
                "cuda_name": torch.cuda.get_device_name(i),
                "cuda_capability": f"{cap[0]}.{cap[1]}",
                "cuda_version": torch.version.cuda,
            })
            # cuDNN (safe)
            try:
                # This may raise if runtime cuDNN is incompatible — catch and stringify
                info["cudnn_version"] = torch.backends.cudnn.version()
            except Exception as e:
                info["cudnn_version"] = f"unavailable ({e.__class__.__name__})"
        except Exception as e:
            info["device"] = "cuda (init failed)"
            info["cuda_error"] = f"{e.__class__.__name__}: {e}"
    return info

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)

def print_run_summary(cfg, args, model, train_ds=None, val_ds=None,
                      extra=None, save_json_path=None):
    """Pretty-print a complete run header and optionally save to JSON."""
    header = {}

    # Config + CLI
    header["config"] = config_to_dict(cfg)
    if args is not None:
        header["args"] = {k: getattr(args, k) for k in vars(args)}

    # Env (build then update with device to avoid **unpacking exceptions)
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    env.update(describe_device())
    header["env"] = env

    # Data
    if train_ds is not None:
        data_info = {
            "train_frames": len(train_ds),
            "val_frames": len(val_ds) if val_ds is not None else None,
            "input_dim": getattr(train_ds, "input_dim", None),
        }
        feat_cols = getattr(train_ds, "feature_cols", None)
        if feat_cols is not None:
            data_info["feature_cols_count"] = len(feat_cols)
            data_info["feature_cols_head"] = feat_cols[:10]  # preview only
        header["data"] = data_info

    # Model (repr + params)
    total, trainable = count_params(model)
    header["model"] = {
        "class": model.__class__.__name__,
        "repr": str(model),
        "total_params": total,
        "trainable_params": trainable,
    }

    if extra:
        header["extra"] = extra

    print("\n========== RUN SUMMARY ==========")
    print(json.dumps(header, indent=2, default=str))
    print("=================================\n")

    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, "w") as f:
            json.dump(header, f, indent=2, default=str)

    return header

