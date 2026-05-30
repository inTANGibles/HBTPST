"""Paper-style benchmark metrics; only include what is computable from current data."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from benchmark.common import emd_metrics
from benchmark.common import metrics as M


def _normalize_prob(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.maximum(np.asarray(v, dtype=np.float64).ravel(), 0.0)
    s = v.sum()
    if s < eps:
        return np.ones_like(v) / len(v)
    return v / s


def svf_kl_symmetric(svf_expert: np.ndarray, svf_other: np.ndarray, eps: float = 1e-10) -> float:
    """Symmetric KL on occupancy: (KL(p||q)+KL(q||p))/2, lower is better."""
    p = _normalize_prob(svf_expert)
    q = _normalize_prob(svf_other)
    kl_pq = float(np.sum(p * np.log((p + eps) / (q + eps))))
    kl_qp = float(np.sum(q * np.log((q + eps) / (p + eps))))
    return 0.5 * (kl_pq + kl_qp)


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-15:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _real_reward_active(world) -> Optional[np.ndarray]:
    r = getattr(world, "real_reward_arr", None)
    if r is None:
        return None
    r = np.asarray(r, dtype=np.float64).ravel()
    if r.size == 0:
        return None
    return r


def _policy_from_reward_vi(world, reward_active: np.ndarray) -> np.ndarray:
    import torch
    from DMEIRL.value_iteration import value_iteration

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    r = torch.from_numpy(np.asarray(reward_active, dtype=np.float32)).to(device)
    pol = value_iteration(0.001, world, r, world.discount, demo=True)
    return pol.detach().cpu().numpy()


def svf_mean_abs_error(svf_expert: np.ndarray, svf_expected: np.ndarray) -> float:
    """Mean absolute deviation between expert SVF and expected SVF (lower is better)."""
    a = np.asarray(svf_expert, dtype=np.float64).ravel()
    b = np.asarray(svf_expected, dtype=np.float64).ravel()
    return float(np.mean(np.abs(a - b)))


def pearson_corr_np(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 2:
        return None
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if den < 1e-15:
        return None
    return float((a * b).sum() / den)


def feature_match_error(world, svf_expert: np.ndarray, svf_other: np.ndarray) -> float:
    """L2 norm of difference in expected feature vectors under two occupancy distributions."""
    F = np.asarray(world.features_arr, dtype=np.float64)
    p = _normalize_prob(svf_expert)
    q = _normalize_prob(svf_other)
    mu_e = p @ F
    mu_o = q @ F
    return float(np.linalg.norm(mu_e - mu_o))


def build_paper_metric_dict(
    world,
    gen_trajs: Sequence,
    learned_reward_active: Optional[np.ndarray] = None,
    policy_active_probs: Optional[np.ndarray] = None,
    svf_transport_cfg: Optional[Dict[str, Any]] = None,
    expert_trajs: Optional[Sequence] = None,
) -> Dict[str, float]:
    """
    Paper-style metrics. **R² / RMSE / EVD** compare expert empirical SVF to **expected SVF**
    under the current policy (same construction as ``DMEIRL.Expected_StateVisitationFrequency``):
    VI softmax from ``learned_reward_active`` when given, else ``policy_active_probs``.

    **Reward_Corr** (optional) still uses ``world.real_reward_arr`` vs learned reward when both exist.
    """
    out: Dict[str, float] = {}
    if expert_trajs is not None:
        svf_e = M.empirical_svf_from_state_action_trajs(world, expert_trajs)
    else:
        svf_e = M.expert_state_visitation_frequency(world)
    svf_g = M.empirical_svf_from_state_action_trajs(world, gen_trajs)
    if svf_g.sum() > 0 or svf_e.sum() > 0:
        out["SVF_KL"] = svf_kl_symmetric(svf_e, svf_g)
        out["Feat_Match_Err"] = feature_match_error(world, svf_e, svf_g)

    cfg = svf_transport_cfg or {}
    t = emd_metrics.compare_svf_transport_distance(
        world,
        svf_e,
        svf_g,
        reg=float(cfg.get("sinkhorn_reg", 0.03)),
        n_iter=int(cfg.get("sinkhorn_iter", 300)),
    )
    if "exact_emd2_pot" in t and t["exact_emd2_pot"] is not None:
        out["EMD"] = float(t["exact_emd2_pot"])
    else:
        out["EMD"] = float(t["sinkhorn_transport_cost"])

    n_act = int(world.n_states_active)
    pi_model: Optional[np.ndarray] = None
    if learned_reward_active is not None:
        rvec = np.asarray(learned_reward_active, dtype=np.float64).ravel()
        if rvec.size == n_act:
            pi_model = _policy_from_reward_vi(world, rvec)
    if pi_model is None and policy_active_probs is not None:
        pol = np.asarray(policy_active_probs, dtype=np.float64)
        if pol.shape == (n_act, int(world.n_actions)):
            pi_model = pol

    if pi_model is not None:
        svf_exp = M.expected_state_visitation_frequency(
            world, pi_model, expert_trajs=expert_trajs
        )
        out["R2"] = r2_np(svf_e, svf_exp)
        out["RMSE"] = rmse_np(svf_e, svf_exp)
        out["EVD"] = svf_mean_abs_error(svf_e, svf_exp)

    r_true = _real_reward_active(world)
    if (
        learned_reward_active is not None
        and r_true is not None
        and r_true.size == len(np.asarray(learned_reward_active).ravel())
    ):
        y_p = np.asarray(learned_reward_active, dtype=np.float64).ravel()
        c = pearson_corr_np(r_true, y_p)
        if c is not None:
            out["Reward_Corr"] = c

    return out


PAPER_METRIC_ORDER = ["R2", "RMSE", "SVF_KL", "EMD", "EVD", "Reward_Corr", "Feat_Match_Err"]

# Shown in batch table even when every method is missing (NaN), so the CSV matches the paper column set.
PAPER_TABLE_FORCE_COLUMNS = ("R2", "RMSE", "EVD")

PAPER_LABELS = {
    "R2": "R²↑ (SVF)",
    "RMSE": "RMSE↓ (SVF)",
    "SVF_KL": "SVF-KL↓",
    "EMD": "EMD↓",
    "EVD": "EVD↓ (SVF MAE)",
    "Reward_Corr": "Reward-Corr↑",
    "Feat_Match_Err": "Feat. Match Err↓",
}
