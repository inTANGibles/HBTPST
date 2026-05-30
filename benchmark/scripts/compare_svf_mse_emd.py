#!/usr/bin/env python
"""
Compare expert SVF vs a second distribution using **MSE×100** (MEDIRL-style) and **transport cost** (Sinkhorn; optional exact EMD if POT installed).

Examples::

    python benchmark/scripts/compare_svf_mse_emd.py --config benchmark/configs/bc.yaml

    python benchmark/scripts/compare_svf_mse_emd.py --config benchmark/configs/bc.yaml \\
        --svf2 path/to/rollout_svf.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    root = _root()
    sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(description="SVF: MSE vs Sinkhorn/EMD comparison")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--svf1",
        type=str,
        default=None,
        help="npy vector length n_states_active (default: expert SVF from world)",
    )
    parser.add_argument(
        "--svf2",
        type=str,
        default=None,
        help="Second SVF npy (default: uniform over active states — sanity baseline)",
    )
    parser.add_argument("--sinkhorn_reg", type=float, default=0.03)
    parser.add_argument("--sinkhorn_iter", type=int, default=300)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    from benchmark.common import emd_metrics
    from benchmark.common.metrics import expert_state_visitation_frequency
    from benchmark.common.utils import build_grid_world, load_config

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / args.config
    cfg = load_config(cfg_path)
    world = build_grid_world(cfg)
    n = world.n_states_active

    if args.svf1:
        p = Path(args.svf1)
        if not p.is_absolute():
            p = root / p
        svf1 = np.load(p)
    else:
        svf1 = expert_state_visitation_frequency(world)

    if args.svf2:
        p = Path(args.svf2)
        if not p.is_absolute():
            p = root / p
        svf2 = np.load(p)
    else:
        svf2 = np.ones(n, dtype=np.float64) / n

    svf1 = np.asarray(svf1).ravel()
    svf2 = np.asarray(svf2).ravel()
    if svf1.size != n or svf2.size != n:
        raise SystemExit(f"Expected SVF length {n}, got {svf1.size} and {svf2.size}")

    out = emd_metrics.compare_svf_transport_distance(
        world,
        svf1,
        svf2,
        reg=args.sinkhorn_reg,
        n_iter=args.sinkhorn_iter,
    )
    print(json.dumps(out, indent=2))
    if args.out_json:
        outp = Path(args.out_json)
        if not outp.is_absolute():
            outp = root / outp
        outp.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
