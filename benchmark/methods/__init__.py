"""Registry for ``--method`` dispatch."""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional, Tuple

METHOD_MODULES: Dict[str, str] = {
    "bc": "benchmark.methods.bc.trainer",
    "env_lstm": "benchmark.methods.env_lstm.trainer",
    "maxent_irl": "benchmark.methods.maxent_irl.trainer",
    "gail": "benchmark.methods.gail.trainer",
    "medirl": "benchmark.methods.medirl.wrapper",
}


def load_method(method: str):
    if method not in METHOD_MODULES:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(METHOD_MODULES)}.")
    mod = importlib.import_module(METHOD_MODULES[method])
    return mod


def get_train_eval(method: str) -> Tuple[Callable, Callable, Callable]:
    mod = load_method(method)
    return mod.train, mod.evaluate, mod.predict_rollout
