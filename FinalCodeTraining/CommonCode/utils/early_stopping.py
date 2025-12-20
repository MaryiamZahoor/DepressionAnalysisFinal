# utils/early_stopping.py
import torch

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0,
                 monitor="val_loss", mode="min", save_path=None, verbose=True):
        """
        monitor: "val_loss" or "val_acc" (or any scalar you pass in)
        mode   : "min" for loss, "max" for accuracy
        """
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best = None
        self.early_stop = False
        self.save_path = save_path
        self.verbose = verbose

    def _is_better(self, curr, best):
        if self.mode == "min":
            return curr < best - self.min_delta
        else:  # "max"
            return curr > best + self.min_delta

    def __call__(self, current_value, model=None):
        if self.best is None:
            self.best = current_value
            if self.save_path and model is not None:
                torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"[ES] init best {self.monitor}={self.best:.6f}")
            return

        if self._is_better(current_value, self.best):
            self.best = current_value
            self.counter = 0
            if self.save_path and model is not None:
                torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"[ES] improved {self.monitor} -> {self.best:.6f} (counter reset)")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[ES] no improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("[ES] early stopping triggered")

