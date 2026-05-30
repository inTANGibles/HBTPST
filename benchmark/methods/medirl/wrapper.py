"""Thin wrapper around existing ``DMEIRL.DMEIRL`` — does not duplicate training logic."""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from benchmark.common import paper_metrics as PM
from benchmark.common import rollout as R
from benchmark.common.utils import build_grid_world, save_json, set_seed

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    from DMEIRL.DeepMEIRL_FC import DMEIRL
    from DMEIRL.value_iteration import value_iteration

    set_seed(int(config.get("seed", 110)))
    world = build_grid_world(config)
    out_dir = Path(output_dir or config["_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    me = config.get("medirl", {})
    layers = tuple(me.get("layers", (64, 128, 128, 64)))
    lr = float(me.get("lr", 5e-5))
    wd = float(me.get("weight_decay", 0.25))
    log_tag = str(me.get("log_tag", "benchmark_wrap"))
    log_dir = str(me.get("log_dir", "run_benchmark"))
    n_epochs = int(me.get("n_epochs", 500))
    vi_demo = bool(me.get("vi_demo", False))

    dme = DMEIRL(world, layers=layers, lr=lr, weight_decay=wd, log=log_tag, log_dir=log_dir)
    best_reward, best_iter, rewards_hist, best_svf_mse = dme.train(
        n_epochs=n_epochs, save=True, demo=False, showInfo=False
    )

    np_path = out_dir / "best_reward_active.npy"
    np.save(np_path, best_reward)

    ro = config.get("rollout", {})
    horizon = int(ro["horizon"]) if ro.get("horizon") is not None else int(world.experts.traj_avg_length)
    n_roll = int(ro["n_trajs"]) if ro.get("n_trajs") is not None else len(world.experts.trajs)
    seed = int(config.get("seed", 110))

    reward_t = torch.from_numpy(np.asarray(best_reward, dtype=np.float32)).to(_device)
    policy = value_iteration(0.001, world, reward_t, world.discount, demo=vi_demo)
    pol_np = policy.detach().cpu().numpy()

    def action_fn(s: int) -> int:
        return int(pol_np[world.state_fid[int(s)]].argmax())

    gen = R.rollout_trajs(world, action_fn, n_roll, horizon, seed=seed + 99)
    with open(out_dir / "generated_trajectories.pkl", "wb") as f:
        pickle.dump(gen, f)

    m: Dict[str, Any] = {
        "method": "medirl",
        "best_iter": int(best_iter),
        "best_svf_mse": float(best_svf_mse),
        "dme_result_path": dme.result_path,
        "best_reward_shape": list(np.asarray(best_reward).shape),
    }
    m.update(
        PM.build_paper_metric_dict(
            world,
            gen,
            learned_reward_active=np.asarray(best_reward, dtype=np.float64).ravel(),
            svf_transport_cfg=config.get("svf_transport"),
        )
    )
    save_json(out_dir / "metrics.json", m)
    # Copy latest artifacts next to benchmark run for a single folder view
    res = Path(dme.result_path)
    if res.is_dir():
        for pat in ("*.pth", "*.npy"):
            for f in res.glob(pat):
                try:
                    shutil.copy2(f, out_dir / f.name)
                except OSError:
                    pass
    return {"metrics": m, "output_dir": str(out_dir), "dme_result_path": dme.result_path}


def evaluate(config: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Re-evaluate using saved reward vector if present; otherwise surface stored metrics."""
    from DMEIRL.value_iteration import value_iteration

    out_dir = Path(output_dir or config.get("_output_dir", "."))
    mp = out_dir / "metrics.json"
    reward_path = out_dir / "best_reward_active.npy"
    if reward_path.is_file():
        set_seed(int(config.get("seed", 110)))
        world = build_grid_world(config)
        best_reward = np.load(reward_path)
        ro = config.get("rollout", {})
        horizon = int(ro["horizon"]) if ro.get("horizon") is not None else int(world.experts.traj_avg_length)
        n_roll = int(ro["n_trajs"]) if ro.get("n_trajs") is not None else len(world.experts.trajs)
        seed = int(config.get("seed", 110))
        vi_demo = bool(config.get("medirl", {}).get("vi_demo", False))
        reward_t = torch.from_numpy(np.asarray(best_reward, dtype=np.float32)).to(_device)
        policy = value_iteration(0.001, world, reward_t, world.discount, demo=vi_demo)
        pol_np = policy.detach().cpu().numpy()

        def action_fn(s: int) -> int:
            return int(pol_np[world.state_fid[int(s)]].argmax())

        gen = R.rollout_trajs(world, action_fn, n_roll, horizon, seed=seed + 100)
        m = dict(json.loads(mp.read_text(encoding="utf-8"))) if mp.is_file() else {"method": "medirl"}
        m.update(
            PM.build_paper_metric_dict(
                world,
                gen,
                learned_reward_active=np.asarray(best_reward, dtype=np.float64).ravel(),
                svf_transport_cfg=config.get("svf_transport"),
            )
        )
        save_json(out_dir / "metrics_eval.json", m)
        return {"metrics": m, "output_dir": str(out_dir)}
    if mp.is_file():
        m = json.loads(mp.read_text(encoding="utf-8"))
        return {"metrics": m, "output_dir": str(out_dir)}
    return {"metrics": {}, "output_dir": str(out_dir), "note": "Run train first or add evaluation script."}


def predict_rollout(config: Dict[str, Any], output_dir: Optional[Path] = None):
    return evaluate(config, output_dir=output_dir)
