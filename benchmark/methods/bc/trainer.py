"""Behavioral cloning: feature vector -> discrete action; rollout on ``GridWorld`` dynamics."""
from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from benchmark.common import evaluation as EV
from benchmark.common import metrics as M
from benchmark.common.dataset import GridStateActionDataset, flatten_state_action_from_trajs
from benchmark.common.early_stop import EarlyStopper
from benchmark.common.splits import TrainingConfig, resolve_split, save_split_artifact, split_meta_dict, trajs_at
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


def _policy_matrix(model: BCNet, world) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(world.features_arr.astype(np.float32)).to(_device)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
    return p


def _val_accuracy(model: BCNet, loader: DataLoader) -> float:
    model.eval()
    correct, tot = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(_device), yb.to(_device)
            pred = model(xb).argmax(dim=1)
            correct += int((pred == yb).sum().item())
            tot += yb.shape[0]
    return float(correct / tot) if tot else 0.0


def train(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    seed = int(config.get("seed", 110))
    set_seed(seed)
    world = build_grid_world(config)
    split = resolve_split(world, config, seed)
    bc = config.get("bc", {})
    tcfg = TrainingConfig.from_cfg(config)
    hidden = bc.get("hidden")
    if hidden is not None:
        hidden = tuple(int(x) for x in hidden)

    train_trajs = trajs_at(world, split.train_idx)
    val_trajs = trajs_at(world, split.val_idx)
    test_trajs = trajs_at(world, split.test_idx)
    X_tr, y_tr = flatten_state_action_from_trajs(world, train_trajs)
    X_va, y_va = flatten_state_action_from_trajs(world, val_trajs)
    if X_va.shape[0] == 0:
        X_va, y_va = X_tr[: min(256, X_tr.shape[0])], y_tr[: min(256, y_tr.shape[0])]

    bs = int(bc.get("batch_size", 256))
    tr_loader = DataLoader(GridStateActionDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    va_loader = DataLoader(GridStateActionDataset(X_va, y_va), batch_size=bs, shuffle=False)

    model = BCNet(X_tr.shape[1], n_actions=world.n_actions, hidden=hidden).to(_device)
    opt = torch.optim.Adam(model.parameters(), lr=float(bc.get("lr", 1e-3)))
    crit = nn.CrossEntropyLoss()
    epochs = int(bc.get("epochs", 100))
    stopper = EarlyStopper(tcfg.early_stop_patience, tcfg.min_epochs)

    out_dir = Path(output_dir) if output_dir is not None else Path(config["_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_split_artifact(out_dir / "traj_split.json", split, len(world.experts.trajs), seed)

    best_val = -1.0
    best_state = deepcopy(model.state_dict())
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(_device), yb.to(_device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        val_acc = _val_accuracy(model, va_loader)
        if val_acc >= best_val:
            best_val = val_acc
            best_state = deepcopy(model.state_dict())
        if stopper.step(-val_acc, ep):
            break

    model.load_state_dict(best_state)
    pol = _policy_matrix(model, world)
    np.save(out_dir / "policy.npy", pol)
    torch.save(model.state_dict(), out_dir / "bc_model.pt")

    def action_fn(s: int) -> int:
        return int(pol[world.state_fid[int(s)]].argmax())

    gen = EV.rollout_on_split(world, action_fn, split, partition="test", seed=seed + 1)
    with open(out_dir / "generated_trajectories.pkl", "wb") as f:
        pickle.dump(gen, f)

    extra = M.pack_metrics(
        val_action_acc_best=best_val,
        train_action_acc_argmax=M.policy_action_accuracy(world, pol, train_trajs),
        stopped_epoch=stopper.stopped_epoch,
        trained_epochs=ep + 1,
    )
    return EV.finalize_benchmark_run(
        world,
        split,
        gen,
        method="bc",
        output_dir=out_dir,
        extra=extra,
        policy_active_probs=pol,
        svf_transport_cfg=config.get("svf_transport"),
    )


def evaluate(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    seed = int(config.get("seed", 110))
    set_seed(seed)
    world = build_grid_world(config)
    split = resolve_split(world, config, seed)
    out_dir = Path(output_dir or config.get("_output_dir", "."))
    bc = config.get("bc", {})
    hidden = bc.get("hidden")
    if hidden is not None:
        hidden = tuple(int(x) for x in hidden)
    model = BCNet(world.features_arr.shape[1], n_actions=world.n_actions, hidden=hidden).to(_device)
    model.load_state_dict(torch.load(out_dir / "bc_model.pt", map_location=_device))
    pol = _policy_matrix(model, world)

    def action_fn(s: int) -> int:
        return int(pol[world.state_fid[int(s)]].argmax())

    gen = EV.rollout_on_split(world, action_fn, split, partition="test", seed=seed + 2)
    m = EV.metrics_for_rollout(
        world, gen, trajs_at(world, split.test_idx), policy_active_probs=pol,
        svf_transport_cfg=config.get("svf_transport"),
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
