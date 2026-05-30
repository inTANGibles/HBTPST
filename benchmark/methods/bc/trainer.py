"""Behavioral cloning: feature vector -> discrete action; rollout on ``GridWorld`` dynamics."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from benchmark.common import emd_metrics
from benchmark.common import metrics as M
from benchmark.common import paper_metrics as PM
from benchmark.common import rollout as R
from benchmark.common.dataset import GridStateActionDataset, build_traj_split, flatten_state_action_from_trajs
from benchmark.common.utils import build_grid_world, load_config, repo_root, save_json, set_seed

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BCNet(nn.Module):
    def __init__(self, n_in: int, n_actions: int = 5, hidden: Optional[tuple] = None):
        super().__init__()
        hidden = hidden or (128, 128)
        layers: List[nn.Module] = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _world_and_split(cfg: Dict[str, Any]):
    repo_root()
    world = build_grid_world(cfg)
    seed = int(cfg.get("seed", 0))
    bc = cfg.get("bc", {})
    split = build_traj_split(
        world,
        float(bc.get("train_frac", 0.8)),
        float(bc.get("val_frac", 0.1)),
        seed,
    )
    return world, split, bc, seed


def _make_loaders(world, split, bc: Dict[str, Any], seed: int):
    train_trajs = [world.experts.trajs[i] for i in split.train_idx]
    val_trajs = [world.experts.trajs[i] for i in split.val_idx]
    X_tr, y_tr = flatten_state_action_from_trajs(world, train_trajs)
    X_va, y_va = flatten_state_action_from_trajs(world, val_trajs)
    if X_va.shape[0] == 0:
        X_va, y_va = X_tr[: min(256, X_tr.shape[0])], y_tr[: min(256, y_tr.shape[0])]
    bs = int(bc.get("batch_size", 256))
    tr_loader = DataLoader(
        GridStateActionDataset(X_tr, y_tr),
        batch_size=bs,
        shuffle=True,
        drop_last=False,
    )
    va_loader = DataLoader(GridStateActionDataset(X_va, y_va), batch_size=bs, shuffle=False)
    return tr_loader, va_loader, X_tr.shape[1]


def _policy_matrix(model: BCNet, world) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(world.features_arr.astype(np.float32)).to(_device)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
    return p


def train(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    seed = int(config.get("seed", 110))
    set_seed(seed)
    world, split, bc, _ = _world_and_split(config)
    tr_loader, va_loader, n_in = _make_loaders(world, split, bc, seed)
    hidden = bc.get("hidden")
    if hidden is not None:
        hidden = tuple(int(x) for x in hidden)
    model = BCNet(n_in, n_actions=world.n_actions, hidden=hidden).to(_device)
    opt = torch.optim.Adam(model.parameters(), lr=float(bc.get("lr", 1e-3)))
    crit = nn.CrossEntropyLoss()
    epochs = int(bc.get("epochs", 30))

    out_dir = Path(output_dir) if output_dir is not None else Path(config["_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(_device), yb.to(_device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        correct, tot = 0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(_device), yb.to(_device)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum().item())
                tot += yb.shape[0]
        val_acc = correct / tot if tot else 0.0
        best_val = max(best_val, val_acc)

    pol = _policy_matrix(model, world)
    np.save(out_dir / "policy.npy", pol)
    torch.save(model.state_dict(), out_dir / "bc_model.pt")

    ro = config.get("rollout", {})
    h = ro.get("horizon")
    horizon = int(h) if h is not None else int(world.experts.traj_avg_length)
    nt = ro.get("n_trajs")
    n_roll = int(nt) if nt is not None else len(world.experts.trajs)

    def action_fn(s: int) -> int:
        idx = world.state_fid[int(s)]
        return int(pol[idx].argmax())

    gen = R.rollout_trajs(world, action_fn, n_roll, horizon, seed=seed + 1)
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
    bc_train_acc = M.policy_action_accuracy(world, pol, [world.experts.trajs[i] for i in split.train_idx])

    m = M.pack_metrics(
        method="bc",
        val_action_acc_best=best_val,
        train_action_acc_argmax=bc_train_acc,
        svf_mse_rollout_vs_expert=svf_mse,
        sinkhorn_transport_cost=transport["sinkhorn_transport_cost"],
        sinkhorn_reg=transport["sinkhorn_reg"],
        n_train_trajs=len(split.train_idx),
        n_val_trajs=len(split.val_idx),
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
    return {"metrics": m, "output_dir": str(out_dir), "policy_path": str(out_dir / "policy.npy")}


def evaluate(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    seed = int(config.get("seed", 110))
    set_seed(seed)
    world, split, bc, _ = _world_and_split(config)
    out_dir = Path(output_dir or config.get("_output_dir", "."))
    n_in = world.features_arr.shape[1]
    hidden = bc.get("hidden")
    if hidden is not None:
        hidden = tuple(int(x) for x in hidden)
    model = BCNet(n_in, n_actions=world.n_actions, hidden=hidden).to(_device)
    model.load_state_dict(torch.load(out_dir / "bc_model.pt", map_location=_device))
    pol = _policy_matrix(model, world)
    val_trajs = [world.experts.trajs[i] for i in split.val_idx]
    val_acc = M.policy_action_accuracy(world, pol, val_trajs)
    ro = config.get("rollout", {})
    h = ro.get("horizon")
    horizon = int(h) if h is not None else int(world.experts.traj_avg_length)
    nt = ro.get("n_trajs")
    n_roll = int(nt) if nt is not None else len(world.experts.trajs)

    def action_fn(s: int) -> int:
        return int(pol[world.state_fid[int(s)]].argmax())

    gen = R.rollout_trajs(world, action_fn, n_roll, horizon, seed=seed + 2)
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
        method="bc",
        val_action_acc_argmax=val_acc,
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


def main_train(config_path: str, output_dir: str) -> None:
    cfg = load_config(config_path)
    cfg["_output_dir"] = output_dir
    train(cfg, Path(output_dir))


if __name__ == "__main__":
    import sys

    main_train(sys.argv[1], sys.argv[2])
