#!/usr/bin/env python
"""Evaluate a finished run folder (expects config_snapshot.yaml and method artifacts)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--run_dir", required=True, type=str, help="Path to a train.py output folder")
    args = parser.parse_args()

    from benchmark.common.utils import load_config, set_seed
    from benchmark.methods import get_train_eval

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    snap = run_dir / "config_snapshot.yaml"
    if not snap.is_file():
        raise FileNotFoundError(snap)
    cfg = load_config(snap)
    meta = run_dir / "run_meta.json"
    if meta.is_file():
        meta_d = json.loads(meta.read_text(encoding="utf-8"))
        set_seed(int(meta_d.get("seed", cfg.get("seed", 110))))
    cfg["_output_dir"] = str(run_dir)

    _, eval_fn, _ = get_train_eval(args.method)
    out = eval_fn(cfg, output_dir=run_dir)
    print(json.dumps(out.get("metrics", out), indent=2))


if __name__ == "__main__":
    main()
