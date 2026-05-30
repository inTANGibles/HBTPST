from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Small grid worlds spend more time in kernel launch / sync than in compute on GPU.
# Vectorized Bellman updates are fast on CPU; only use GPU above this threshold.
_GPU_VI_MIN_STATES = 4096


def _resolve_vi_device(rewards: torch.Tensor, n_states: int) -> torch.device:
    if rewards.is_cuda and n_states >= _GPU_VI_MIN_STATES:
        return rewards.device
    return torch.device("cpu")


def _dynamics_tensor(world, vi_device: torch.device) -> torch.Tensor:
    cache_key = (id(world), str(vi_device))
    cached = getattr(world, "_vi_dynamics_cache", None)
    if cached is None:
        world._vi_dynamics_cache = {}
        cached = world._vi_dynamics_cache
    if cache_key not in cached:
        cached[cache_key] = torch.from_numpy(world.dynamics_fid).float().to(vi_device)
    return cached[cache_key]


def _bellman_update(
    dynamics: torch.Tensor,
    rewards: torch.Tensor,
    discount: float,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """Vectorized one-step value iteration. dynamics: [S, A, S], rewards/V: [S]."""
    next_val = rewards + discount * V
    q = torch.einsum("sas,s->sa", dynamics, next_val)
    v_new = q.max(dim=1).values
    delta = float((v_new - V).abs().max().item())
    return v_new, delta


def _softmax_policy(
    dynamics: torch.Tensor,
    rewards: torch.Tensor,
    discount: float,
    V: torch.Tensor,
) -> torch.Tensor:
    next_val = rewards + discount * V
    q = torch.einsum("sas,s->sa", dynamics, next_val)
    q = q - q.max(dim=1, keepdim=True).values
    exps = torch.exp(q)
    return exps / exps.sum(dim=1, keepdim=True)


def value_iteration(
    threshold,
    world,
    rewards,
    discount=0.01,
    showInfo=False,
    demo=False,
):
    orig_device = rewards.device
    vi_device = _resolve_vi_device(rewards, world.n_states_active)
    rewards = rewards.detach().to(vi_device)
    dynamics = _dynamics_tensor(world, vi_device)
    V = torch.zeros(world.n_states_active, dtype=torch.float32, device=vi_device)

    with torch.no_grad():
        if demo:
            delta = np.inf
            while delta > threshold:
                V, delta = _bellman_update(dynamics, rewards, discount, V)
                if showInfo:
                    print(f"delta: {delta}")
        else:
            V, delta = _bellman_update(dynamics, rewards, discount, V)
            delta_last = delta
            total = int(max(0.0, (delta - threshold) * 1_000_000))
            with tqdm(total=total) as pbar:
                pbar.set_description("Value Iteration:")
                while delta > threshold:
                    V, delta = _bellman_update(dynamics, rewards, discount, V)
                    cut_d = threshold if delta < threshold else delta
                    step = delta_last - cut_d
                    pbar.update(int(step * 1_000_000))
                    delta_last = delta
                    if showInfo:
                        print(f"delta: {delta}")

        policy = _softmax_policy(dynamics, rewards, discount, V)

    return policy.to(orig_device)


def value_iteration_fullGrid(threshold, world, rewards, discount=0.01):
    orig_device = rewards.device
    vi_device = _resolve_vi_device(rewards, world.n_states)
    rewards = rewards.detach().to(vi_device)
    dynamics = torch.from_numpy(world.dynamics).float().to(vi_device)
    V = torch.zeros(world.n_states, dtype=torch.float32, device=vi_device)

    with torch.no_grad():
        delta = np.inf
        while delta > threshold:
            V, delta = _bellman_update(dynamics, rewards, discount, V)

        policy = _softmax_policy(dynamics, rewards, discount, V)

    return policy.to(orig_device)
