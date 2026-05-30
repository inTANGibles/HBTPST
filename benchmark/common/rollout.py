"""Rollout on the same ``GridWorld`` dynamics as ``GridWorld_trajGen.step`` (stochastic trans_prob)."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np


ActionFn = Callable[[int], int]


def sample_initial_state_from_experts(world, rng: np.random.Generator) -> int:
    traj = world.experts.trajs[int(rng.integers(0, len(world.experts.trajs)))]
    return int(traj[0][0])


def step_stochastic(world, state: int, action: int, rng: np.random.Generator) -> int:
    idx = world.state_fid[state]
    probs = world.dynamics_fid[idx, action, :].astype(np.float64)
    probs = probs / probs.sum()
    j = int(rng.choice(world.n_states_active, p=probs))
    return int(world.fid_state[j])


def rollout_trajs(
    world,
    action_fn: ActionFn,
    n_trajs: int,
    horizon: int,
    seed: int,
) -> List[List[Tuple[int, int, int]]]:
    rng = np.random.default_rng(seed)
    out: List[List[Tuple[int, int, int]]] = []
    for _ in range(n_trajs):
        s = sample_initial_state_from_experts(world, rng)
        traj = []
        for _t in range(horizon):
            a = int(action_fn(s))
            s2 = step_stochastic(world, s, a, rng)
            traj.append((s, a, s2))
            s = s2
        out.append(traj)
    return out


def rollout_trajs_matched(
    world,
    action_fn: ActionFn,
    expert_trajs: Sequence,
    seed: int,
) -> List[List[Tuple[int, int, int]]]:
    """One rollout per expert trajectory, same start state and horizon length."""
    rng = np.random.default_rng(seed)
    out: List[List[Tuple[int, int, int]]] = []
    for expert_traj in expert_trajs:
        if not expert_traj:
            continue
        s = int(expert_traj[0][0])
        if s not in world.state_fid:
            s = sample_initial_state_from_experts(world, rng)
        traj = []
        for _t in range(len(expert_traj)):
            a = int(action_fn(s))
            s2 = step_stochastic(world, s, a, rng)
            traj.append((s, a, s2))
            s = s2
        out.append(traj)
    return out
