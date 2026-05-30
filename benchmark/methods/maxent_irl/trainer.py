"""Linear MaxEnt-style IRL: R(s)=w^T f(s), match expert SVF via the same VI + gradient signal as ``DMEIRL``."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from benchmark.common import evaluation as EV
from benchmark.common import metrics as M
from benchmark.common.early_stop import EarlyStopper
from benchmark.common.splits import TrainingConfig, resolve_split, save_split_artifact, trajs_at
from benchmark.common.utils import build_grid_world, save_json, set_seed

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _expert_svf_tensor(world, trajs) -> torch.Tensor:
    svf = torch.zeros(world.n_states_active, dtype=torch.float32, device=_device)
    for traj in trajs:
        for step in traj:
            s = int(step[0])
            if s not in world.state_fid:
                continue
            svf[world.state_fid[s]] += 1.0
    n = len(trajs)
    if n == 0:
        return svf
    return svf / n


def _expected_svf_tensor(world, policy: torch.Tensor, trajs) -> torch.Tensor:
    prob_initial = torch.zeros(world.n_states_active, dtype=torch.float32, device=_device)
    for traj in trajs:
        prob_initial[world.state_fid[int(traj[0][0])]] += 1.0
    prob_initial = prob_initial / max(1, len(trajs))
    dynamics = torch.from_numpy(world.dynamics_fid).float().to(_device)
    T = int(world.experts.traj_avg_length)
    mu = prob_initial.unsqueeze(0).repeat(T, 1)
    x = (policy[:, :, None] * dynamics).sum(1)
    for t in range(1, T):
        mu[t, :] = torch.matmul(mu[t - 1, :], x)
    return mu.sum(dim=0)


def train(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    from DMEIRL.value_iteration import value_iteration

    seed = int(config.get("seed", 110))
    set_seed(seed)
    world = build_grid_world(config)
    split = resolve_split(world, config, seed)
    train_trajs = trajs_at(world, split.train_idx)
    val_trajs = trajs_at(world, split.val_idx)

    out_dir = Path(output_dir or config["_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_split_artifact(out_dir / "traj_split.json", split, len(world.experts.trajs), seed)

    mx = config.get("maxent_irl", {})
    n_epochs = int(mx.get("n_epochs", 200))
    lr = float(mx.get("lr", 0.05))
    vi_demo = bool(mx.get("vi_demo", True))
    tcfg = TrainingConfig.from_cfg(config)
    stopper = EarlyStopper(tcfg.early_stop_patience, tcfg.min_epochs)

    F = torch.from_numpy(world.features_arr).float().to(_device)
    w = torch.zeros(F.shape[1], device=_device, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    expert_svf = _expert_svf_tensor(world, train_trajs)

    best_val = float("inf")
    best_w = w.detach().clone()
    best_reward = None
    trained_epochs = 0

    for ep in range(n_epochs):
        reward = (F @ w.unsqueeze(1)).squeeze(1)
        policy = value_iteration(0.001, world, reward.detach(), world.discount, demo=vi_demo)
        exp_svf = _expected_svf_tensor(world, policy, train_trajs)
        r_grad = expert_svf - exp_svf
        opt.zero_grad()
        reward.backward(-r_grad)
        opt.step()

        pol_np = policy.detach().cpu().numpy()
        val_mse = M.policy_svf_mse_on_trajs(world, pol_np, val_trajs)
        if val_mse < best_val:
            best_val = val_mse
            best_w = w.detach().clone()
            best_reward = reward.detach().clone()
        trained_epochs = ep + 1
        if stopper.step(val_mse, ep):
            break

    w = best_w
    with torch.no_grad():
        reward_final = (F @ w.unsqueeze(1)).squeeze(1) if best_reward is None else best_reward
    policy_final = value_iteration(0.001, world, reward_final, world.discount, demo=vi_demo)
    pol_np = policy_final.detach().cpu().numpy()
    np.save(out_dir / "policy.npy", pol_np)
    np.save(out_dir / "reward_map.npy", reward_final.detach().cpu().numpy())
    np.save(out_dir / "linear_weights.npy", w.detach().cpu().numpy())

    def action_fn(s: int) -> int:
        return int(pol_np[world.state_fid[int(s)]].argmax())

    gen = EV.rollout_on_split(world, action_fn, split, partition="test", seed=seed + 3)
    with open(out_dir / "generated_trajectories.pkl", "wb") as f:
        pickle.dump(gen, f)

    reward_np = reward_final.detach().cpu().numpy()
    extra = M.pack_metrics(
        best_val_svf_mse=best_val,
        stopped_epoch=stopper.stopped_epoch,
        trained_epochs=trained_epochs,
    )
    return EV.finalize_benchmark_run(
        world,
        split,
        gen,
        method="maxent_irl",
        output_dir=out_dir,
        extra=extra,
        learned_reward_active=reward_np,
        svf_transport_cfg=config.get("svf_transport"),
    )


def evaluate(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    from DMEIRL.value_iteration import value_iteration

    seed = int(config.get("seed", 110))
    set_seed(seed)
    world = build_grid_world(config)
    split = resolve_split(world, config, seed)
    out_dir = Path(output_dir or config.get("_output_dir", "."))
    w_np = np.load(out_dir / "linear_weights.npy")
    F = torch.from_numpy(world.features_arr.astype(np.float32)).to(_device)
    w = torch.from_numpy(w_np.astype(np.float32)).to(_device)
    reward = (F @ w.unsqueeze(1)).squeeze(1)
    vi_demo = bool(config.get("maxent_irl", {}).get("vi_demo", True))
    policy = value_iteration(0.001, world, reward, world.discount, demo=vi_demo)
    pol_np = policy.detach().cpu().numpy()

    def action_fn(s: int) -> int:
        return int(pol_np[world.state_fid[int(s)]].argmax())

    gen = EV.rollout_on_split(world, action_fn, split, partition="test", seed=seed + 9)
    m = EV.metrics_for_rollout(
        world,
        gen,
        trajs_at(world, split.test_idx),
        learned_reward_active=reward.detach().cpu().numpy(),
        svf_transport_cfg=config.get("svf_transport"),
    )
    save_json(out_dir / "metrics_eval.json", m)
    return {"metrics": m, "output_dir": str(out_dir)}


def predict_rollout(config: Dict[str, Any], output_dir: Optional[Path] = None):
    return evaluate(config, output_dir=output_dir)
