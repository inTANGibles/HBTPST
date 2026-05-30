"""Validation-based early stopping (lower metric is better)."""
from __future__ import annotations


class EarlyStopper:
    def __init__(self, patience: int, min_epochs: int = 0):
        self.patience = max(0, int(patience))
        self.min_epochs = max(0, int(min_epochs))
        self.best: float | None = None
        self.bad_epochs = 0
        self.stopped_epoch: int | None = None

    def step(self, metric: float, epoch: int) -> bool:
        """Return True if training should stop."""
        if self.best is None or metric < self.best:
            self.best = float(metric)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if epoch + 1 >= self.min_epochs and self.patience >= 0 and self.bad_epochs >= self.patience:
            self.stopped_epoch = epoch + 1
            return True
        return False
