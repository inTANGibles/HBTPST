"""
EMD / Wasserstein-style comparison of two SVF vectors over active grid states.

Uses **Manhattan distance on the 2D grid** as ground cost (states identified by
``world.fid_state``). Default transport is **Sinkhorn** (entropic regularization),
no extra pip dependency. If `POT` is installed, ``exact_emd`` uses ``ot.emd2``.

Recommended location: **benchmark** (evaluation), not inside ``DMEIRL`` core.
Optional hook from ``DMEIRL`` training: ``from benchmark.common.emd_metrics import ...``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from grid_world import grid_utils


def manhattan_cost_matrix(world) -> np.ndarray:
    """C[i,j] = |x_i-x_j|+|y_i-y_j| in grid coordinates (active states only)."""
    n = world.n_states_active
    C = np.zeros((n, n), dtype=np.float64)
    w = int(world.width)
    for i in range(n):
        si = world.fid_state[i]
        xi, yi = grid_utils.StateToCoord(si, w)
        for j in range(i + 1, n):
            sj = world.fid_state[j]
            xj, yj = grid_utils.StateToCoord(sj, w)
            d = abs(xi - xj) + abs(yi - yj)
            C[i, j] = d
            C[j, i] = d
    return C


def _normalize_positive_mass(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.maximum(v.astype(np.float64), 0.0)
    s = v.sum()
    if s < eps:
        return np.ones_like(v) / len(v)
    return v / s


def sinkhorn_ot_cost(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float = 0.03,
    n_iter: int = 300,
) -> float:
    """
    Balanced entropic OT cost <C, P> with Sinkhorn scaling.
    ``reg`` is blur strength (larger = easier / more diffuse transport).
    """
    n = C.shape[0]
    K = np.exp(-C / reg)
    K = np.maximum(K, 1e-300)
    u = np.ones(n, dtype=np.float64) / n
    v = np.ones(n, dtype=np.float64) / n
    for _ in range(n_iter):
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    return float(np.sum(P * C))


def exact_emd_cost_ot(C: np.ndarray, a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Exact discrete EMD / W_1 transport cost if ``pot`` (``ot``) is available."""
    try:
        import ot
    except ImportError:
        return None
    return float(ot.emd2(a, b, C))


def compare_svf_transport_distance(
    world,
    svf1: np.ndarray,
    svf2: np.ndarray,
    *,
    reg: float = 0.03,
    n_iter: int = 300,
    compute_exact_if_available: bool = True,
) -> dict:
    """
    Compare two SVF-like nonnegative vectors on active states.

    Returns dict with ``sinkhorn_cost``, optional ``exact_emd2``, and copies of
    ``mse_x100`` (same scale as ``metrics.compare_svf_mse``).
    """
    import torch
    import torch.nn as nn

    a = _normalize_positive_mass(np.asarray(svf1).ravel())
    b = _normalize_positive_mass(np.asarray(svf2).ravel())
    if a.size != b.size or a.size != world.n_states_active:
        raise ValueError(
            f"SVF length mismatch: got {a.size}, {b.size}, world.n_states_active={world.n_states_active}"
        )
    C = manhattan_cost_matrix(world)
    sk = sinkhorn_ot_cost(C, a, b, reg=reg, n_iter=n_iter)
    compare = nn.MSELoss()
    with torch.no_grad():
        mse_x100 = float(
            compare(
                torch.from_numpy(a.astype(np.float32)),
                torch.from_numpy(b.astype(np.float32)),
            )
            * 100.0
        )
    out = {
        "sinkhorn_transport_cost": sk,
        "sinkhorn_reg": reg,
        "sinkhorn_n_iter": n_iter,
        "svf_mse_x100": mse_x100,
    }
    if compute_exact_if_available:
        ex = exact_emd_cost_ot(C, a, b)
        if ex is not None:
            out["exact_emd2_pot"] = ex
    return out
