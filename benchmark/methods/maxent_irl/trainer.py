"""Linear MaxEnt-style IRL: R(s)=w^T f(s), match expert SVF via the same VI + gradient signal as ``DMEIRL``."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from benchmark.common import emd_metrics
from benchmark.common import metrics as M
from benchmark.common import paper_metrics as PM
from benchmark.common import rollout as R
from benchmark.common.utils import build_grid_world, set_seed
from benchmark.common.utils import save_json

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _expert_svf_tensor(world) -> torch.Tensor:
    svf = torch.zeros(world.n_states_active, dtype=torch.float32, device=_device)
    for traj in world.experts.trajs:
        for step in traj:
            s = int(step[0])
            if s not in world.state_fid:
                continue
            svf[world.state_fid[s]] += 1.0
    n = len(world.experts.trajs)
    if n == 0:
        return svf
    return svf / n


def _expected_svf_tensor(world, policy: torch.Tensor) -> torch.Tensor:
    """Port of ``DMEIRL.Expected_StateVisitationFrequency`` (tensor dynamics)."""
    prob_initial = torch.zeros(world.n_states_active, dtype=torch.float32, device=_device)
    for traj in world.experts.trajs:
        prob_initial[world.state_fid[int(traj[0][0])]] += 1.0
    prob_initial = prob_initial / max(1, len(world.experts.trajs))
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
    out_dir = Path(output_dir or config["_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mx = config.get("maxent_irl", {})
    n_epochs = int(mx.get("n_epochs", 80))
    lr = float(mx.get("lr", 0.05))
    vi_demo = bool(mx.get("vi_demo", True))

    F = torch.from_numpy(world.features_arr).float().to(_device)
    w = torch.zeros(F.shape[1], device=_device, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    expert_svf = _expert_svf_tensor(world)

    for _ in range(n_epochs):
        reward = (F @ w.unsqueeze(1)).squeeze(1)
        policy = value_iteration(0.001, world, reward.detach(), world.discount, demo=vi_demo)
        exp_svf = _expected_svf_tensor(world, policy)
        r_grad = expert_svf - exp_svf
        opt.zero_grad()
        reward.backward(-r_grad)
        opt.step()

    with torch.no_grad():
        reward_final = (F @ w.unsqueeze(1)).squeeze(1)
    policy_final = value_iteration(0.001, world, reward_final, world.discount, demo=vi_demo)
    pol_np = policy_final.detach().cpu().numpy()
    np.save(out_dir / "policy.npy", pol_np)
    np.save(out_dir / "reward_map.npy", reward_final.detach().cpu().numpy())

    ro = config.get("rollout", {})
    horizon = int(ro["horizon"]) if ro.get("horizon") is not None else int(world.experts.traj_avg_length)
    n_roll = int(ro["n_trajs"]) if ro.get("n_trajs") is not None else len(world.experts.trajs)

    def action_fn(s: int) -> int:
        return int(pol_np[world.state_fid[int(s)]].argmax())

    gen = R.rollout_trajs(world, action_fn, n_roll, horizon, seed=seed + 3)
    with open(out_dir / "generated_trajectories.pkl", "wb") as f:
        pickle.dump(gen, f)

    svf_exp = M.expert_state_visitation_frequency(world)
    svf_gen = M.empirical_svf_from_state_action_trajs(world, gen)
    svf_mse = M.compare_svf_mse(svf_exp, svf_gen)
    emd_opts = config.get("svf_transport", {})
    transport = emd_metrics.compare_svf_transport_distance(
        world,
        svf_exp,
        svf_gen,
        reg=float(emd_opts.get("sinkhorn_reg", 0.03)),
        n_iter=int(emd_opts.get("sinkhorn_iter", 300)),
    )
    m = M.pack_metrics(
        method="maxent_irl",
        svf_mse_rollout_vs_expert=svf_mse,
        sinkhorn_transport_cost=transport["sinkhorn_transport_cost"],
        sinkhorn_reg=transport["sinkhorn_reg"],
        n_epochs=n_epochs,
    )
    if "exact_emd2_pot" in transport:
        m["exact_emd2_pot"] = transport["exact_emd2_pot"]
    reward_np = reward_final.detach().cpu().numpy()
    m.update(
        PM.build_paper_metric_dict(
            world,
            gen,
            learned_reward_active=reward_np,
            svf_transport_cfg=config.get("svf_transport"),
        )
    )
    save_json(out_dir / "metrics.json", m)
    np.save(out_dir / "linear_weights.npy", w.detach().cpu().numpy())
    return {"metrics": m, "output_dir": str(out_dir)}


def evaluate(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    from DMEIRL.value_iteration import value_iteration

    seed = int(config.get("seed", 110))
    set_seed(seed)
    world = build_grid_world(config)
    out_dir = Path(output_dir or config.get("_output_dir", "."))
    w_np = np.load(out_dir / "linear_weights.npy")
    F = torch.from_numpy(world.features_arr.astype(np.float32)).to(_device)
    w = torch.from_numpy(w_np.astype(np.float32)).to(_device)
    reward = (F @ w.unsqueeze(1)).squeeze(1)
    vi_demo = bool(config.get("maxent_irl", {}).get("vi_demo", True))
    policy = value_iteration(0.001, world, reward, world.discount, demo=vi_demo)
    pol_np = policy.detach().cpu().numpy()
    ro = config.get("rollout", {})
    horizon = int(ro["horizon"]) if ro.get("horizon") is not None else int(world.experts.traj_avg_length)
    n_roll = int(ro["n_trajs"]) if ro.get("n_trajs") is not None else len(world.experts.trajs)

    def action_fn(s: int) -> int:
        return int(pol_np[world.state_fid[int(s)]].argmax())

    gen = R.rollout_trajs(world, action_fn, n_roll, horizon, seed=seed + 9)
    svf_exp = M.expert_state_visitation_frequency(world)
    svf_gen = M.empirical_svf_from_state_action_trajs(world, gen)
    svf_mse = M.compare_svf_mse(svf_exp, svf_gen)
    emd_opts = config.get("svf_transport", {})
    transport = emd_metrics.compare_svf_transport_distance(
        world,
        svf_exp,
        svf_gen,
        reg=float(emd_opts.get("sinkhorn_reg", 0.03)),
        n_iter=int(emd_opts.get("sinkhorn_iter", 300)),
    )
    m = M.pack_metrics(
        method="maxent_irl",
        svf_mse_rollout_vs_expert=svf_mse,
        sinkhorn_transport_cost=transport["sinkhorn_transport_cost"],
        sinkhorn_reg=transport["sinkhorn_reg"],
    )
    if "exact_emd2_pot" in transport:
        m["exact_emd2_pot"] = transport["exact_emd2_pot"]
    m.update(
        PM.build_paper_metric_dict(
            world,
            gen,
            learned_reward_active=reward.detach().cpu().numpy(),
            svf_transport_cfg=config.get("svf_transport"),
        )
    )
    save_json(out_dir / "metrics_eval.json", m)
    return {"metrics": m, "output_dir": str(out_dir)}


def predict_rollout(config: Dict[str, Any], output_dir: Optional[Path] = None):
    return evaluate(config, output_dir=output_dir)
