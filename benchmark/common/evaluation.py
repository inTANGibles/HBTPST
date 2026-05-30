"""Test-set rollout and paper metrics (same split protocol for every method)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from benchmark.common import emd_metrics
from benchmark.common import metrics as M
from benchmark.common import paper_metrics as PM
from benchmark.common import rollout as R
from benchmark.common.dataset import TrajSplit
from benchmark.common.splits import trajs_at, split_meta_dict
from benchmark.common.utils import save_json

ActionFn = Callable[[int], int]


def rollout_on_split(
    world,
    action_fn: ActionFn,
    split: TrajSplit,
    *,
    partition: str = "test",
    seed: int,
    use_matched_length: bool = True,
) -> List[List[Tuple[int, int, int]]]:
    if partition == "test":
        idx = split.test_idx
    elif partition == "val":
        idx = split.val_idx
    elif partition == "train":
        idx = split.train_idx
    else:
        raise ValueError(partition)
    expert_trajs = trajs_at(world, idx)
    if not expert_trajs:
        return []
    if use_matched_length:
        return R.rollout_trajs_matched(world, action_fn, expert_trajs, seed=seed)
    horizon = int(world.experts.traj_avg_length)
    return R.rollout_trajs(world, action_fn, len(expert_trajs), horizon, seed=seed)


def metrics_for_rollout(
    world,
    gen_trajs: Sequence,
    expert_trajs: Sequence,
    *,
    learned_reward_active=None,
    policy_active_probs=None,
    svf_transport_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    svf_e = M.empirical_svf_from_state_action_trajs(world, expert_trajs)
    svf_g = M.empirical_svf_from_state_action_trajs(world, gen_trajs)
    out: Dict[str, float] = {}
    if svf_g.sum() > 0 or svf_e.sum() > 0:
        out["svf_mse_rollout_vs_expert"] = M.compare_svf_mse(svf_e, svf_g)
    emd_opts = svf_transport_cfg or {}
    transport = emd_metrics.compare_svf_transport_distance(
        world,
        svf_e,
        svf_g,
        reg=float(emd_opts.get("sinkhorn_reg", 0.03)),
        n_iter=int(emd_opts.get("sinkhorn_iter", 300)),
    )
    out["sinkhorn_transport_cost"] = float(transport["sinkhorn_transport_cost"])
    out["sinkhorn_reg"] = float(transport["sinkhorn_reg"])
    if "exact_emd2_pot" in transport and transport["exact_emd2_pot"] is not None:
        out["exact_emd2_pot"] = float(transport["exact_emd2_pot"])
    out.update(
        PM.build_paper_metric_dict(
            world,
            gen_trajs,
            learned_reward_active=learned_reward_active,
            policy_active_probs=policy_active_probs,
            svf_transport_cfg=svf_transport_cfg,
            expert_trajs=expert_trajs,
        )
    )
    return out


def finalize_benchmark_run(
    world,
    split: TrajSplit,
    gen_trajs: Sequence,
    *,
    method: str,
    output_dir: Path,
    partition: str = "test",
    extra: Optional[Dict[str, Any]] = None,
    learned_reward_active=None,
    policy_active_probs=None,
    svf_transport_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    expert_trajs = trajs_at(world, split.test_idx if partition == "test" else split.val_idx)
    m = M.pack_metrics(method=method, eval_partition=partition, **split_meta_dict(split))
    m.update(
        metrics_for_rollout(
            world,
            gen_trajs,
            expert_trajs,
            learned_reward_active=learned_reward_active,
            policy_active_probs=policy_active_probs,
            svf_transport_cfg=svf_transport_cfg,
        )
    )
    if extra:
        m.update(extra)
    save_json(output_dir / "metrics.json", m)
    return {"metrics": m, "output_dir": str(output_dir)}
