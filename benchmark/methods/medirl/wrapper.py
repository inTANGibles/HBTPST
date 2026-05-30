"""Thin wrapper around existing ``DMEIRL.DMEIRL`` — does not duplicate training logic."""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from benchmark.common import evaluation as EV
from benchmark.common import metrics as M
from benchmark.common.splits import TrainingConfig, resolve_split, save_split_artifact, trajs_at
from benchmark.common.utils import build_grid_world, save_json, set_seed

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    from DMEIRL.DeepMEIRL_FC import DMEIRL
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

    me = config.get("medirl", {})
    tcfg = TrainingConfig.from_cfg(config)
    layers = tuple(me.get("layers", (64, 128, 128, 64)))
    lr = float(me.get("lr", 5e-5))
    wd = float(me.get("weight_decay", 0.25))
    log_tag = str(me.get("log_tag", "benchmark_wrap"))
    log_dir = str(me.get("log_dir", "run_benchmark"))
    n_epochs = int(me.get("n_epochs", 500))
    vi_demo = bool(me.get("vi_demo", False))

    dme = DMEIRL(
        world,
        layers=layers,
        lr=lr,
        weight_decay=wd,
        log=log_tag,
        log_dir=log_dir,
        train_trajs=train_trajs,
    )
    best_reward, best_iter, rewards_hist, best_svf_mse = dme.train(
        n_epochs=n_epochs,
        save=True,
        demo=True,
        showInfo=False,
        val_trajs=val_trajs,
        early_stop_patience=tcfg.early_stop_patience,
        min_epochs=tcfg.min_epochs,
        select_on_val=True,
    )

    np_path = out_dir / "best_reward_active.npy"
    np.save(np_path, best_reward)

    reward_t = torch.from_numpy(np.asarray(best_reward, dtype=np.float32)).to(_device)
    policy = value_iteration(0.001, world, reward_t, world.discount, demo=vi_demo)
    pol_np = policy.detach().cpu().numpy()

    def action_fn(s: int) -> int:
        return int(pol_np[world.state_fid[int(s)]].argmax())

    gen = EV.rollout_on_split(world, action_fn, split, partition="test", seed=seed + 99)
    with open(out_dir / "generated_trajectories.pkl", "wb") as f:
        pickle.dump(gen, f)

    extra: Dict[str, Any] = {
        "best_iter": int(best_iter),
        "best_val_svf_mse": float(best_svf_mse),
        "dme_result_path": dme.result_path,
        "best_reward_shape": list(np.asarray(best_reward).shape),
    }
    result = EV.finalize_benchmark_run(
        world,
        split,
        gen,
        method="medirl",
        output_dir=out_dir,
        extra=extra,
        learned_reward_active=np.asarray(best_reward, dtype=np.float64).ravel(),
        svf_transport_cfg=config.get("svf_transport"),
    )
    res = Path(dme.result_path)
    if res.is_dir():
        for pat in ("*.pth", "*.npy"):
            for f in res.glob(pat):
                try:
                    shutil.copy2(f, out_dir / f.name)
                except OSError:
                    pass
    return result


def evaluate(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    from DMEIRL.value_iteration import value_iteration

    out_dir = Path(output_dir or config.get("_output_dir", "."))
    reward_path = out_dir / "best_reward_active.npy"
    if reward_path.is_file():
        seed = int(config.get("seed", 110))
        set_seed(seed)
        world = build_grid_world(config)
        split = resolve_split(world, config, seed)
        best_reward = np.load(reward_path)
        vi_demo = bool(config.get("medirl", {}).get("vi_demo", False))
        reward_t = torch.from_numpy(np.asarray(best_reward, dtype=np.float32)).to(_device)
        policy = value_iteration(0.001, world, reward_t, world.discount, demo=vi_demo)
        pol_np = policy.detach().cpu().numpy()

        def action_fn(s: int) -> int:
            return int(pol_np[world.state_fid[int(s)]].argmax())

        gen = EV.rollout_on_split(world, action_fn, split, partition="test", seed=seed + 100)
        mp = out_dir / "metrics.json"
        m = dict(json.loads(mp.read_text(encoding="utf-8"))) if mp.is_file() else {"method": "medirl"}
        m.update(
            EV.metrics_for_rollout(
                world,
                gen,
                trajs_at(world, split.test_idx),
                learned_reward_active=np.asarray(best_reward, dtype=np.float64).ravel(),
                svf_transport_cfg=config.get("svf_transport"),
            )
        )
        save_json(out_dir / "metrics_eval.json", m)
        return {"metrics": m, "output_dir": str(out_dir)}
    mp = out_dir / "metrics.json"
    if mp.is_file():
        m = json.loads(mp.read_text(encoding="utf-8"))
        return {"metrics": m, "output_dir": str(out_dir)}
    return {"metrics": {}, "output_dir": str(out_dir), "note": "Run train first or add evaluation script."}


def predict_rollout(config: Dict[str, Any], output_dir: Optional[Path] = None):
    return evaluate(config, output_dir=output_dir)
