"""GAIL: policy pi(f(s)) vs discriminator D(f(s),a); same ``features_arr`` as BC/MEDIRL."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from benchmark.common import emd_metrics
from benchmark.common import metrics as M
from benchmark.common import paper_metrics as PM
from benchmark.common import rollout as R
from benchmark.common.utils import build_grid_world, save_json, set_seed

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyMLP(nn.Module):
    def __init__(self, n_in: int, n_actions: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers: List[nn.Module] = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscriminatorMLP(nn.Module):
    def __init__(self, n_in: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers: List[nn.Module] = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _expert_fa_tensors(world) -> Tuple[torch.Tensor, torch.Tensor]:
    feats: List[np.ndarray] = []
    acts: List[int] = []
    for traj in world.experts.trajs:
        for step in traj:
            if len(step) < 2:
                continue
            s, a = int(step[0]), int(step[1])
            if s not in world.state_fid:
                continue
            row = world.state_fid[s]
            feats.append(world.features_arr[row].astype(np.float32))
            acts.append(int(a))
    if not feats:
        raise RuntimeError("GAIL: no expert (s,a) pairs.")
    return torch.from_numpy(np.stack(feats, axis=0)).to(_device), torch.tensor(acts, dtype=torch.long, device=_device)


def _one_hot_a(a: torch.Tensor, n_actions: int) -> torch.Tensor:
    return F.one_hot(a, num_classes=n_actions).float()


def _rollout_stochastic_torch(
    world,
    policy: PolicyMLP,
    n_trajs: int,
    horizon: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[Tuple[int, int, int]]]]:
    rng = np.random.default_rng(seed)
    F_list: List[torch.Tensor] = []
    A_list: List[torch.Tensor] = []
    LP_list: List[torch.Tensor] = []
    raw: List[List[Tuple[int, int, int]]] = []
    policy.eval()
    for _ in range(n_trajs):
        traj_ex = world.experts.trajs[int(rng.integers(0, len(world.experts.trajs)))]
        s = int(traj_ex[0][0])
        row_traj: List[Tuple[int, int, int]] = []
        for _t in range(horizon):
            if s not in world.state_fid:
                s = int(world.fid_state[int(rng.integers(0, world.n_states_active))])
            idx = world.state_fid[s]
            f_t = torch.from_numpy(world.features_arr[idx].astype(np.float32)).to(_device)
            logits = policy(f_t.unsqueeze(0)).squeeze(0)
            dist = Categorical(logits=logits)
            a_t = dist.sample()
            logp = dist.log_prob(a_t)
            F_list.append(f_t)
            A_list.append(a_t)
            LP_list.append(logp)
            a_int = int(a_t.item())
            probs = world.dynamics_fid[idx, a_int, :].astype(np.float64)
            probs = probs / probs.sum()
            j = int(rng.choice(world.n_states_active, p=probs))
            s2 = int(world.fid_state[j])
            row_traj.append((s, a_int, s2))
            s = s2
        raw.append(row_traj)
    if not F_list:
        raise RuntimeError("GAIL rollout produced no steps.")
    return torch.stack(F_list, dim=0), torch.stack(A_list, dim=0), torch.stack(LP_list, dim=0), raw


def _disc_input(f: torch.Tensor, a: torch.Tensor, n_act: int) -> torch.Tensor:
    return torch.cat([f, _one_hot_a(a, n_act)], dim=1)


def _policy_matrix_argmax(policy: PolicyMLP, world) -> np.ndarray:
    policy.eval()
    with torch.no_grad():
        x = torch.from_numpy(world.features_arr.astype(np.float32)).to(_device)
        logits = policy(x)
        return torch.softmax(logits, dim=1).cpu().numpy()


def train(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    seed = int(config.get("seed", 110))
    set_seed(seed)
    world = build_grid_world(config)
    g = config.get("gail", {})
    n_feat = world.features_arr.shape[1]
    n_act = world.n_actions
    hidden_p = tuple(g.get("policy_hidden", [128, 128]))
    hidden_d = tuple(g.get("disc_hidden", [256, 256]))
    lr_pi = float(g.get("lr_policy", 3e-4))
    lr_d = float(g.get("lr_disc", 3e-4))
    n_iters = int(g.get("n_iters", 200))
    n_d = int(g.get("n_d_steps", 2))
    batch = int(g.get("batch_size", 512))
    roll_trajs = int(g.get("rollout_trajs", 32))
    roll_h = int(g.get("rollout_horizon", 24))
    gamma = float(g.get("gamma", 0.95))
    ent_coef = float(g.get("entropy_coef", 0.01))

    out_dir = Path(output_dir or config["_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = PolicyMLP(n_feat, n_act, hidden_p).to(_device)
    disc = DiscriminatorMLP(n_feat + n_act, hidden_d).to(_device)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=lr_pi)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()

    F_exp, A_exp = _expert_fa_tensors(world)
    n_exp = F_exp.shape[0]
    rng = np.random.default_rng(seed)

    for _ in range(n_iters):
        # --- Discriminator: expert vs policy rollout ---
        for _ in range(n_d):
            idx_e = torch.from_numpy(rng.integers(0, n_exp, size=min(batch, n_exp))).long().to(_device)
            fe, ae = F_exp[idx_e], A_exp[idx_e]
            xe = _disc_input(fe, ae, n_act)

            Fg, Ag, _, _ = _rollout_stochastic_torch(
                world, policy, roll_trajs, roll_h, seed=int(rng.integers(0, 2**31 - 1))
            )
            nf = Fg.shape[0]
            if nf == 0:
                continue
            take = min(batch, nf)
            idx_g = torch.from_numpy(rng.integers(0, nf, size=take)).long().to(_device)
            xg = _disc_input(Fg[idx_g], Ag[idx_g], n_act)

            m = min(xe.shape[0], xg.shape[0])
            xe, xg = xe[:m], xg[:m]

            opt_d.zero_grad()
            loss_d = bce(disc(xe), torch.ones(m, device=_device)) + bce(disc(xg), torch.zeros(m, device=_device))
            loss_d.backward()
            opt_d.step()

        # --- Policy: REINFORCE with log(D(s,a)) ---
        Fg, Ag, logp, _ = _rollout_stochastic_torch(
            world, policy, roll_trajs, roll_h, seed=int(rng.integers(0, 2**31 - 1))
        )
        xg = _disc_input(Fg, Ag, n_act)
        policy.train()
        opt_pi.zero_grad()
        with torch.no_grad():
            d_logits = disc(xg)
            rew = torch.log(torch.sigmoid(d_logits) + 1e-8)

        steps = logp.shape[0]
        rew = rew[:steps]
        logp = logp[:steps]
        if steps % roll_trajs != 0:
            roll_trajs_eff = 1
            H = steps
        else:
            roll_trajs_eff = roll_trajs
            H = steps // roll_trajs_eff
        rew_m = rew.view(roll_trajs_eff, H)
        Gm = torch.zeros_like(rew_m)
        for tr in range(roll_trajs_eff):
            for t in range(H - 1, -1, -1):
                Gm[tr, t] = rew_m[tr, t] + (gamma * Gm[tr, t + 1] if t + 1 < H else 0.0)
        adv = (Gm - Gm.mean()).reshape(-1)

        dist2 = Categorical(logits=policy(Fg[:steps]))
        ent = dist2.entropy().mean()
        loss_pi = -(logp * adv.detach()).mean() - ent_coef * ent
        loss_pi.backward()
        opt_pi.step()

    pol = _policy_matrix_argmax(policy, world)
    np.save(out_dir / "policy.npy", pol)
    torch.save(
        {"policy": policy.state_dict(), "disc": disc.state_dict(), "n_feat": n_feat, "n_act": n_act},
        out_dir / "gail.pt",
    )

    ro = config.get("rollout", {})
    horizon = int(ro["horizon"]) if ro.get("horizon") is not None else int(world.experts.traj_avg_length)
    n_roll = int(ro["n_trajs"]) if ro.get("n_trajs") is not None else len(world.experts.trajs)

    def action_fn(s: int) -> int:
        return int(pol[world.state_fid[int(s)]].argmax())

    gen = R.rollout_trajs(world, action_fn, n_roll, horizon, seed=seed + 77)
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
        method="gail",
        svf_mse_rollout_vs_expert=svf_mse,
        sinkhorn_transport_cost=transport["sinkhorn_transport_cost"],
        sinkhorn_reg=transport["sinkhorn_reg"],
        n_iters=n_iters,
    )
    if "exact_emd2_pot" in transport:
        m["exact_emd2_pot"] = transport["exact_emd2_pot"]
    m.update(
        PM.build_paper_metric_dict(
            world,
            gen,
            learned_reward_active=None,
            policy_active_probs=pol,
            svf_transport_cfg=config.get("svf_transport"),
        )
    )
    save_json(out_dir / "metrics.json", m)
    return {"metrics": m, "output_dir": str(out_dir)}


def evaluate(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    seed = int(config.get("seed", 110))
    set_seed(seed)
    world = build_grid_world(config)
    out_dir = Path(output_dir or config.get("_output_dir", "."))
    ck = torch.load(out_dir / "gail.pt", map_location=_device)
    g = config.get("gail", {})
    hidden_p = tuple(g.get("policy_hidden", [128, 128]))
    policy = PolicyMLP(int(ck["n_feat"]), int(ck["n_act"]), hidden_p).to(_device)
    policy.load_state_dict(ck["policy"])
    pol = _policy_matrix_argmax(policy, world)
    ro = config.get("rollout", {})
    horizon = int(ro["horizon"]) if ro.get("horizon") is not None else int(world.experts.traj_avg_length)
    n_roll = int(ro["n_trajs"]) if ro.get("n_trajs") is not None else len(world.experts.trajs)

    def action_fn(s: int) -> int:
        return int(pol[world.state_fid[int(s)]].argmax())

    gen = R.rollout_trajs(world, action_fn, n_roll, horizon, seed=seed + 78)
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
        method="gail",
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
            learned_reward_active=None,
            policy_active_probs=pol,
            svf_transport_cfg=config.get("svf_transport"),
        )
    )
    save_json(out_dir / "metrics_eval.json", m)
    return {"metrics": m, "output_dir": str(out_dir)}


def predict_rollout(config: Dict[str, Any], output_dir: Optional[Path] = None):
    return evaluate(config, output_dir=output_dir)
