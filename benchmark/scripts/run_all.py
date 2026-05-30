#!/usr/bin/env python
"""Run a subset of benchmark methods sequentially (BC + maxent_irl by default)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    root = _repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        default="bc,maxent_irl",
        help="Comma-separated: bc,env_lstm,maxent_irl,gail,medirl",
    )
    args = parser.parse_args()
    train_py = root / "benchmark" / "scripts" / "train.py"
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        cfg = root / "benchmark" / "configs" / f"{m}.yaml"
        if not cfg.is_file():
            print("Skip (no config):", cfg)
            continue
        cmd = [sys.executable, str(train_py), "--method", m, "--config", str(cfg)]
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(root))
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
