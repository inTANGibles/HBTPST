"""Metrics aligned with MEDIRL / ``DMEIRL`` (SVF comparison, optional action accuracy)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def expert_state_visitation_frequency(world) -> np.ndarray:
    """Same normalization as ``DMEIRL.StateVisitationFrequency`` (per-traj average of counts)."""
    svf = np.zeros(world.n_states_active, dtype=np.float64)
    trajs = world.experts.trajs
    for traj in trajs:
        for step in traj:
            s = int(step[0])
            if s not in world.state_fid:
                continue
            svf[world.state_fid[s]] += 1.0
    if len(trajs) == 0:
        return svf
    return svf / len(trajs)


def empirical_svf_from_state_action_trajs(world, trajs: Sequence) -> np.ndarray:
    """SVF from trajectories as list of (s,a,s') or (s,a) steps — uses state occupancy."""
    svf = np.zeros(world.n_states_active, dtype=np.float64)
    n = 0
    for traj in trajs:
        n += 1
        for step in traj:
            s = int(step[0])
            if s not in world.state_fid:
                continue
            svf[world.state_fid[s]] += 1.0
    if n == 0:
        return svf
    return svf / n


def compare_svf_mse(svf1: np.ndarray, svf2: np.ndarray) -> float:
    """Same scaling as ``DMEIRL.CompareSVF`` (MSE * 100)."""
    compare = nn.MSELoss()
    with torch.no_grad():
        t1 = torch.from_numpy(svf1.astype(np.float32))
        t2 = torch.from_numpy(svf2.astype(np.float32))
        return float(compare(t1, t2) * 100.0)


def policy_svf_mse_on_trajs(world, policy_probs: np.ndarray, expert_trajs) -> float:
    """MSE×100 between expert SVF on ``expert_trajs`` and expected SVF under policy."""
    svf_e = empirical_svf_from_state_action_trajs(world, expert_trajs)
    svf_exp = expected_state_visitation_frequency(world, policy_probs, expert_trajs=expert_trajs)
    return compare_svf_mse(svf_e, svf_exp)


def expected_state_visitation_frequency(
    world, policy_probs: np.ndarray, expert_trajs=None
) -> np.ndarray:
    """
    Expected state occupancy under ``policy_probs`` and ``world.dynamics_fid``,
    matching ``DMEIRL.DMEIRL.Expected_StateVisitationFrequency`` (empirical start
    distribution over expert trajectories, horizon ``traj_avg_length``, sum over time).
    """
    n = int(world.n_states_active)
    na = int(world.n_actions)
    pi = np.asarray(policy_probs, dtype=np.float64).reshape(n, na)
    rs = pi.sum(axis=1, keepdims=True)
    rs = np.maximum(rs, 1e-15)
    pi = pi / rs

    P = np.asarray(world.dynamics_fid, dtype=np.float64)
    # T[i, j] = sum_a pi[i,a] * P[i,a,j]
    T = np.einsum("sa,sat->st", pi, P)

    prob0 = np.zeros(n, dtype=np.float64)
    trajs = expert_trajs if expert_trajs is not None else world.experts.trajs
    for traj in trajs:
        if not traj:
            continue
        s0 = int(traj[0][0])
        if s0 not in world.state_fid:
            continue
        prob0[world.state_fid[s0]] += 1.0
    denom = float(len(trajs)) if trajs else 1.0
    prob0 = prob0 / max(denom, 1.0)

    H = max(1, int(getattr(world.experts, "traj_avg_length", 1)))
    mu = np.zeros((H, n), dtype=np.float64)
    mu[0] = prob0
    for t in range(1, H):
        mu[t] = mu[t - 1] @ T
    return mu.sum(axis=0)


def action_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def policy_action_accuracy(world, policy_probs: np.ndarray, trajs: Sequence) -> float:
    """Fraction of expert (s,a) pairs matching argmax_a policy(state). policy_probs: (n_active, n_actions)."""
    correct, total = 0, 0
    for traj in trajs:
        for step in traj:
            if len(step) < 2:
                continue
            s, a = int(step[0]), int(step[1])
            if s not in world.state_fid:
                continue
            idx = world.state_fid[s]
            if int(policy_probs[idx].argmax()) == a:
                correct += 1
            total += 1
    return float(correct / total) if total else 0.0


def pack_metrics(**kwargs) -> dict:
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, (float, int, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = v
    return out


def compare_svf_mse_and_transport(world, svf1: np.ndarray, svf2: np.ndarray, **kwargs) -> dict:
    """MSE×100 (same as training monitor) + Sinkhorn / optional exact EMD (see ``emd_metrics``)."""
    from benchmark.common import emd_metrics

    return emd_metrics.compare_svf_transport_distance(world, svf1, svf2, **kwargs)
