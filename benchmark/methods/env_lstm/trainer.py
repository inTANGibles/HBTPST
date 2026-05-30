"""Environment-conditioned LSTM: sliding window of (norm_xy, f(s)) -> action. Same ``features_arr`` as BC/MEDIRL."""
from __future__ import annotations

import pickle
import pickle
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from benchmark.common import evaluation as EV
from benchmark.common import metrics as M
from benchmark.common.early_stop import EarlyStopper
from benchmark.common.splits import TrainingConfig, resolve_split, save_split_artifact, trajs_at
from benchmark.common.utils import build_grid_world, save_json, set_seed
from grid_world import grid_utils

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _xy_norm(world, state: int) -> Tuple[float, float]:
    x, y = grid_utils.StateToCoord(state, world.width)
    return float(x) / max(1, world.width - 1), float(y) / max(1, world.height - 1)


def _step_vec(world, state: int) -> np.ndarray:
    xn, yn = _xy_norm(world, state)
    idx = world.state_fid[int(state)]
    f = world.features_arr[idx].astype(np.float32)
    return np.concatenate([[xn, yn], f], axis=0)


def _build_windows(
    world, trajs: List, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """X: (N, L, D), y: (N,) action at time t (expert label)."""
    rows: List[np.ndarray] = []
    ys: List[int] = []
    d = 2 + world.features_arr.shape[1]
    for traj in trajs:
        if len(traj) < 1:
            continue
        states = [int(traj[t][0]) for t in range(len(traj))]
        actions = [int(traj[t][1]) for t in range(len(traj))]
        for t in range(len(traj)):
            win = np.zeros((seq_len, d), dtype=np.float32)
            t0 = t - seq_len + 1
            for k in range(seq_len):
                ti = t0 + k
                if ti < 0:
                    continue
                if int(states[ti]) not in world.state_fid:
                    continue
                win[k] = _step_vec(world, int(states[ti]))
            rows.append(win)
            ys.append(int(actions[t]))
    if not rows:
        return np.zeros((0, seq_len, d), np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(rows, axis=0), np.asarray(ys, dtype=np.int64)


class EnvLSTM(nn.Module):
    def __init__(self, step_dim: int, hidden: int, n_actions: int = 5, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(step_dim, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _policy_matrix_stationary(model: EnvLSTM, world, seq_len: int) -> np.ndarray:
    """Per active state: repeat same (xy,f(s)) for L steps (location-only marginal policy for metrics)."""
    n = world.n_states_active
    d = 2 + world.features_arr.shape[1]
    model.eval()
    pol = np.zeros((n, world.n_actions), dtype=np.float64)
    with torch.no_grad():
        for idx in range(n):
            s = int(world.fid_state[idx])
            v = _step_vec(world, s)
            win = np.tile(v, (seq_len, 1)).astype(np.float32)[None, ...]
            logits = model(torch.from_numpy(win).to(_device))
            pol[idx] = torch.softmax(logits, dim=1).cpu().numpy().ravel()
    return pol


def _rollout_env_lstm(
    world,
    model: EnvLSTM,
    seq_len: int,
    expert_trajs: List,
    seed: int,
) -> List[List[Tuple[int, int, int]]]:
    rng = np.random.default_rng(seed)
    d = 2 + world.features_arr.shape[1]
    model.eval()
    out: List[List[Tuple[int, int, int]]] = []
    with torch.no_grad():
        for expert_traj in expert_trajs:
            if not expert_traj:
                continue
            buf: Deque[np.ndarray] = deque(maxlen=seq_len)
            s = int(expert_traj[0][0])
            if s not in world.state_fid:
                s = int(world.fid_state[int(rng.integers(0, world.n_states_active))])
            traj = []
            for _t in range(len(expert_traj)):
                buf.append(_step_vec(world, s))
                win = np.zeros((1, seq_len, d), dtype=np.float32)
                nbuf = len(buf)
                for i, vv in enumerate(buf):
                    win[0, seq_len - nbuf + i] = vv
                logits = model(torch.from_numpy(win).to(_device))
                a = int(logits.argmax(dim=1).item())
                idx = world.state_fid[s]
                probs = world.dynamics_fid[idx, a, :].astype(np.float64)
                probs = probs / probs.sum()
                j = int(rng.choice(world.n_states_active, p=probs))
                s2 = int(world.fid_state[j])
                traj.append((s, a, s2))
                s = s2
            out.append(traj)
    return out


def train(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    seed = int(config.get("seed", 110))
    set_seed(seed)
    world = build_grid_world(config)
    split = resolve_split(world, config, seed)
    el = config.get("env_lstm", {})
    tcfg = TrainingConfig.from_cfg(config)
    seq_len = int(el.get("seq_len", 8))
    hidden = int(el.get("hidden", 128))
    n_layers = int(el.get("n_layers", 1))
    epochs = int(el.get("epochs", 80))
    lr = float(el.get("lr", 1e-3))
    bs = int(el.get("batch_size", 256))

    train_trajs = trajs_at(world, split.train_idx)
    val_trajs = trajs_at(world, split.val_idx)
    X_tr, y_tr = _build_windows(world, train_trajs, seq_len)
    X_va, y_va = _build_windows(world, val_trajs, seq_len)
    if X_tr.shape[0] == 0:
        raise RuntimeError("env_lstm: no training windows; check trajectories / seq_len.")

    out_dir = Path(output_dir or config["_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_split_artifact(out_dir / "traj_split.json", split, len(world.experts.trajs), seed)

    step_dim = X_tr.shape[2]
    model = EnvLSTM(step_dim, hidden, world.n_actions, n_layers).to(_device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    stopper = EarlyStopper(tcfg.early_stop_patience, tcfg.min_epochs)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=bs,
        shuffle=True,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=bs,
        shuffle=False,
    )

    best_val = -1.0
    best_state = deepcopy(model.state_dict())
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(_device), yb.to(_device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
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
        if val_acc >= best_val:
            best_val = val_acc
            best_state = deepcopy(model.state_dict())
        if stopper.step(-val_acc, ep):
            break

    model.load_state_dict(best_state)
    pol = _policy_matrix_stationary(model, world, seq_len)
    np.save(out_dir / "policy.npy", pol)
    torch.save({"model": model.state_dict(), "seq_len": seq_len}, out_dir / "env_lstm.pt")

    test_trajs = trajs_at(world, split.test_idx)
    gen = _rollout_env_lstm(world, model, seq_len, test_trajs, seed=seed + 11)
    with open(out_dir / "generated_trajectories.pkl", "wb") as f:
        pickle.dump(gen, f)

    extra = M.pack_metrics(
        val_action_acc_best=best_val,
        stopped_epoch=stopper.stopped_epoch,
        trained_epochs=ep + 1,
        seq_len=seq_len,
    )
    return EV.finalize_benchmark_run(
        world,
        split,
        gen,
        method="env_lstm",
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
    ck = torch.load(out_dir / "env_lstm.pt", map_location=_device)
    seq_len = int(ck.get("seq_len", config.get("env_lstm", {}).get("seq_len", 8)))
    el = config.get("env_lstm", {})
    hidden = int(el.get("hidden", 128))
    n_layers = int(el.get("n_layers", 1))
    step_dim = 2 + world.features_arr.shape[1]
    model = EnvLSTM(step_dim, hidden, world.n_actions, n_layers).to(_device)
    model.load_state_dict(ck["model"])
    pol = _policy_matrix_stationary(model, world, seq_len)
    test_trajs = trajs_at(world, split.test_idx)
    gen = _rollout_env_lstm(world, model, seq_len, test_trajs, seed=seed + 12)
    m = EV.metrics_for_rollout(
        world,
        gen,
        test_trajs,
        policy_active_probs=pol,
        svf_transport_cfg=config.get("svf_transport"),
    )
    save_json(out_dir / "metrics_eval.json", m)
    return {"metrics": m, "output_dir": str(out_dir)}


def predict_rollout(config: Dict[str, Any], output_dir: Optional[Path] = None):
    return evaluate(config, output_dir=output_dir)
