"""Unified (state, action) supervision from existing expert CSV / ``GridWorld.experts``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _traj_indices_split(
    n_trajs: int, train_frac: float, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trajs)
    if n_trajs <= 1:
        return perm, perm.copy(), perm.copy()
    n_train = max(1, int(np.floor(n_trajs * train_frac)))
    n_val = max(1, int(np.ceil(n_trajs * val_frac))) if n_trajs > 2 else max(0, n_trajs - n_train)
    if n_train + n_val >= n_trajs:
        n_val = max(0, n_trajs - n_train - 1)
    if n_val == 0 and n_trajs > 1:
        n_train = max(1, n_trajs - 1)
        n_val = n_trajs - n_train
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    if len(test_idx) == 0:
        test_idx = val_idx.copy()
    return train_idx, val_idx, test_idx


def flatten_state_action_from_trajs(world, traj_list: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Map expert steps to rows of ``world.features_arr`` and discrete actions."""
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for traj in traj_list:
        for step in traj:
            if step is None or len(step) < 2:
                continue
            s, a = int(step[0]), int(step[1])
            if s not in world.state_fid:
                continue
            row = world.state_fid[s]
            xs.append(np.asarray(world.features_arr[row], dtype=np.float32))
            ys.append(int(a))
    if not xs:
        return np.zeros((0, world.features_arr.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


@dataclass
class TrajSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def build_traj_split(world, train_frac: float, val_frac: float, seed: int) -> TrajSplit:
    n = len(world.experts.trajs)
    train_idx, val_idx, test_idx = _traj_indices_split(n, train_frac, val_frac, seed)
    return TrajSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


class GridStateActionDataset(Dataset):
    """PyTorch dataset: environment feature vector -> discrete action (BC)."""

    def __init__(self, features: np.ndarray, actions: np.ndarray):
        self.x = torch.from_numpy(features).float()
        self.y = torch.from_numpy(actions).long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]
