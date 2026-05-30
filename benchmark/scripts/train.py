#!/usr/bin/env python
"""Unified training entry: ``python benchmark/scripts/train.py --method bc --config benchmark/configs/bc.yaml``."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["bc", "env_lstm", "maxent_irl", "gail", "medirl"])
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", default=None, help="Override auto output directory")
    args = parser.parse_args()

    from benchmark.common.utils import ensure_run_dir, load_config, save_json, set_seed
    from benchmark.methods import get_train_eval

    cfg_src = Path(args.config)
    if not cfg_src.is_absolute():
        cfg_src = root / args.config
    cfg = load_config(cfg_src)
    seed = int(cfg.get("seed", 110))
    set_seed(seed)

    out_root = root / cfg.get("output", {}).get("root", "benchmark/outputs")
    out_dir = Path(args.output_dir) if args.output_dir else ensure_run_dir(out_root, args.method, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_src, out_dir / "config_snapshot.yaml")
    save_json(out_dir / "run_meta.json", {"method": args.method, "seed": seed, "config_path": str(args.config)})

    cfg["_output_dir"] = str(out_dir)
    train_fn, _, _ = get_train_eval(args.method)
    result = train_fn(cfg, output_dir=out_dir)
    save_json(out_dir / "train_result.json", {k: v for k, v in result.items() if k != "metrics"})
    print("Done:", result.get("output_dir", out_dir))


if __name__ == "__main__":
    main()
