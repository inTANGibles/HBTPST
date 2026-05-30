"""Unified trajectory train/val/test split for all benchmark methods."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from benchmark.common.dataset import TrajSplit, build_traj_split


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "SplitConfig":
        sp = cfg.get("split", {})
        return cls(
            train_frac=float(sp.get("train_frac", 0.70)),
            val_frac=float(sp.get("val_frac", 0.15)),
        )


@dataclass
class TrainingConfig:
    """Shared training budget / early-stopping knobs."""

    early_stop_patience: int = 20
    min_epochs: int = 10

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "TrainingConfig":
        tr = cfg.get("training", {})
        return cls(
            early_stop_patience=int(tr.get("early_stop_patience", 20)),
            min_epochs=int(tr.get("min_epochs", 10)),
        )


def resolve_split(world, cfg: Dict[str, Any], seed: int) -> TrajSplit:
    sc = SplitConfig.from_cfg(cfg)
    return build_traj_split(world, sc.train_frac, sc.val_frac, seed)


def trajs_at(world, indices: Sequence[int]) -> List:
    return [world.experts.trajs[int(i)] for i in indices]


def save_split_artifact(path: Path, split: TrajSplit, n_total: int, seed: int) -> None:
    payload = {
        "seed": seed,
        "n_total": n_total,
        "n_train": int(len(split.train_idx)),
        "n_val": int(len(split.val_idx)),
        "n_test": int(len(split.test_idx)),
        "train_idx": [int(i) for i in split.train_idx.tolist()],
        "val_idx": [int(i) for i in split.val_idx.tolist()],
        "test_idx": [int(i) for i in split.test_idx.tolist()],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def split_meta_dict(split: TrajSplit) -> Dict[str, Any]:
    return {
        "n_train_trajs": int(len(split.train_idx)),
        "n_val_trajs": int(len(split.val_idx)),
        "n_test_trajs": int(len(split.test_idx)),
    }
