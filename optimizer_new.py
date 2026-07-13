"""Shared optimizer for Stage 4/5.

This module provides the interface expected by the Stage 5 notebook.
It is self-contained and avoids the malformed generated file issue.
"""

from __future__ import annotations

import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = PROJECT_DIR / "stage4_optimizer_config"

with (_CONFIG_DIR / "optimizer_constants.json").open(encoding="utf-8") as fh:
    _constants = json.load(fh)

_bounds_raw = pd.read_csv(_CONFIG_DIR / "weight_bounds.csv")
_groups_raw = pd.read_csv(_CONFIG_DIR / "combined_constraints.csv")

ASSETS = [a for a in _constants["asset_order"] if a in {
    "NIFTYBEES", "JUNIORBEES", "GOLDBEES", "LIQUIDBEES"}]
N_ASSETS = len(ASSETS)
CVaR_CONFIDENCE = float(_constants["cvar_confidence"])
SOLVER_ORDER = _constants["solver_order"]
OBJECTIVE_WEIGHTS = _constants["objective_weights"]
GOOD_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
TOL = 1e-6

_cost_bps = pd.DataFrame(_constants["transaction_cost_bps"]).T.loc[ASSETS, [
    "buy", "sell"]].astype(float)
BUY_COSTS = _cost_bps["buy"].to_numpy() / 10_000
SELL_COSTS = _cost_bps["sell"].to_numpy() / 10_000


def regime_bounds(regime: str):
    sub = _bounds_raw[_bounds_raw["regime"].eq(
        regime)].set_index("asset").loc[ASSETS]
    return sub["lower"].astype(float).to_numpy(), sub["upper"].astype(float).to_numpy()


def regime_group_rows(regime: str):
    rows = []
    for row in _groups_raw[_groups_raw["regime"].eq(regime)].to_dict("records"):
        assets = [a.strip()
                  for a in str(row["assets"]).split("|") if a.strip()]
        rows.append({
            "group": row["group"],
            "assets": assets,
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
        })
    return rows


def _coerce_scenarios(scenarios) -> np.ndarray:
    arr = np.asarray(scenarios, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("scenarios must be a 2D array-like")
    if arr.shape[1] != N_ASSETS:
        if arr.shape[1] < N_ASSETS:
            pad = np.zeros(
                (arr.shape[0], N_ASSETS - arr.shape[1]), dtype=float)
            arr = np.concatenate([arr, pad], axis=1)
        else:
            arr = arr[:, :N_ASSETS]
    return arr


def _project_to_bounds(weights, regime: str) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    lower, upper = regime_bounds(regime)
    w = np.clip(w, lower, upper)
    total = w.sum()
    if total <= 0:
        return np.ones(N_ASSETS) / N_ASSETS
    return w / total


def solve_regime_objective(regime: str, sigma, scenarios, previous_weights=None, reject_singular=False):
    lower, upper = regime_bounds(regime)
    prev = np.asarray(
        previous_weights, dtype=float).reshape(-1) if previous_weights is not None else None
    if prev is None or prev.size != N_ASSETS:
        prev = np.ones(N_ASSETS) / N_ASSETS

    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.shape != (N_ASSETS, N_ASSETS):
        sigma_arr = np.eye(N_ASSETS)
    sigma_arr = (sigma_arr + sigma_arr.T) / 2.0

    if reject_singular and np.linalg.matrix_rank(sigma_arr) < N_ASSETS:
        return {
            "weights": _project_to_bounds(prev, regime),
            "status": "fallback_singular",
            "solver": None,
            "fallback": True,
            "objective_value": float("nan"),
            "error": "singular_covariance",
        }

    scenarios_arr = _coerce_scenarios(scenarios)
    w = cp.Variable(N_ASSETS, name="w")
    tau = cp.Variable(name="tau")
    ret = scenarios_arr @ w

    constraints = [cp.sum(w) == 1.0, w >= lower, w <= upper]
    for row in regime_group_rows(regime):
        idx = [ASSETS.index(asset)
               for asset in row["assets"] if asset in ASSETS]
        if idx:
            group_sum = cp.sum(w[idx])
            constraints.append(group_sum >= row["lower"])
            constraints.append(group_sum <= row["upper"])

    cvar_penalty = tau + (1.0 / (1.0 - CVaR_CONFIDENCE)) * \
        cp.mean(cp.pos(-ret - tau))
    obj_weights = OBJECTIVE_WEIGHTS.get(regime, {})
    variance_weight = float(obj_weights.get("variance", 1.0))
    cvar_weight = float(obj_weights.get("cvar", 0.1))
    tc_weight = float(obj_weights.get("transaction_cost", 1.0))
    turnover_penalty = cp.sum(cp.abs(w - prev))
    objective = variance_weight * \
        cp.quad_form(w, sigma_arr) + cvar_weight * \
        cvar_penalty + tc_weight * turnover_penalty

    problem = cp.Problem(cp.Minimize(objective), constraints)

    last_error = None
    for solver_name in SOLVER_ORDER:
        try:
            problem.solve(solver=solver_name, verbose=False)
            if w.value is None or not np.isfinite(w.value).all():
                raise RuntimeError("solver returned invalid weights")
            if problem.status not in GOOD_STATUSES:
                raise RuntimeError(problem.status)
            w_val = np.asarray(w.value, dtype=float).reshape(-1)
            w_val = np.clip(w_val, lower, upper)
            total = w_val.sum()
            if total <= 0:
                raise RuntimeError("invalid weight total")
            return {
                "weights": w_val / total,
                "status": problem.status,
                "solver": solver_name,
                "fallback": False,
                "objective_value": float(objective.value),
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = repr(exc)
            continue

    return {
        "weights": _project_to_bounds(prev, regime),
        "status": "fallback_solver",
        "solver": None,
        "fallback": True,
        "objective_value": float("nan"),
        "error": last_error or "solver_failure",
    }


def validate_weights(regime: str, weights):
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != N_ASSETS:
        return False
    lower, upper = regime_bounds(regime)
    if not np.isfinite(w).all():
        return False
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        return False
    if np.any(w < lower - TOL) or np.any(w > upper + TOL):
        return False
    return True


def portfolio_metrics(weights, sigma, scenarios, previous_weights=None):
    w = np.asarray(weights, dtype=float).reshape(-1)
    scenarios_arr = _coerce_scenarios(scenarios)
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.shape != (N_ASSETS, N_ASSETS):
        sigma_arr = np.eye(N_ASSETS)
    sigma_arr = (sigma_arr + sigma_arr.T) / 2.0
    port_returns = scenarios_arr @ w
    return {
        "expected_return": float(port_returns.mean()),
        "volatility": float(np.sqrt(max(float(w @ sigma_arr @ w), 0.0))),
        "cvar": float(np.quantile(-port_returns, 0.05)),
    }


def covariance_health(sigma):
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.shape != (N_ASSETS, N_ASSETS):
        sigma_arr = np.eye(N_ASSETS)
    sigma_arr = (sigma_arr + sigma_arr.T) / 2.0
    eigs = np.linalg.eigvalsh(sigma_arr)
    return {
        "ok": bool(np.all(eigs > 0)),
        "min_eig": float(eigs.min()),
        "max_eig": float(eigs.max()),
        "condition_number": float(eigs.max() / max(eigs.min(), 1e-12)),
    }
