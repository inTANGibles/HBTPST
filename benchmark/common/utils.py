"""Paths, config loading, GridWorld construction (same contract as ``3_DMEIRL.py``)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_ROOT: Optional[Path] = None


def repo_root() -> Path:
    global _ROOT
    if _ROOT is None:
        _ROOT = Path(__file__).resolve().parents[2]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
    return _ROOT


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required for .yaml configs: pip install pyyaml") from e
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_grid_world(cfg: Dict[str, Any]):
    """Build ``GridWorld`` from benchmark config (mirrors ``3_DMEIRL.py``)."""
    repo_root()
    from grid_world.grid_world import GridWorld

    p = cfg.get("paths", {})
    grid = cfg.get("grid", {})
    expert = p.get("expert_csv")
    env_folder = p.get("env_img_folder")
    if not expert or not env_folder:
        raise KeyError("paths.expert_csv and paths.env_img_folder are required")

    root = repo_root()
    expert_path = expert if Path(expert).is_absolute() else str(root / expert)
    env_path = env_folder if Path(env_folder).is_absolute() else str(root / env_folder)

    return GridWorld(
        expert_traj_filePath=expert_path,
        environments_img_folderPath=env_path,
        width=int(grid.get("width", 40)),
        height=int(grid.get("height", 30)),
        discount=float(grid.get("discount", 0.95)),
        trans_prob=float(grid.get("trans_prob", 0.8)),
        traj_length_bias=int(grid.get("traj_length_bias", 0)),
    )


def ensure_run_dir(base: Path, method: str, seed: int) -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = base / f"{method}_{stamp}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
