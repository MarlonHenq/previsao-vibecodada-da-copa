"""Component 2: Dixon-Coles goal rates and scoreline simulation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from features import time_decay_weights, zscore


def dc_tau(home_goals: int, away_goals: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_h * lambda_a * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_h * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_a * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def fit_fast(matches: pd.DataFrame, context: pd.DataFrame) -> dict[str, Any]:
    """Heuristic attack/defense from weighted goals relative to global average."""
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    weights = time_decay_weights(df["date"])

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    attack: dict[str, float] = {}
    defense: dict[str, float] = {}
    n_matches: dict[str, int] = {}

    global_gf: list[float] = []
    global_ga: list[float] = []

    for team in teams:
        mask_h = df["home_team"] == team
        mask_a = df["away_team"] == team
        gf = np.concatenate([
            df.loc[mask_h, "home_goals"].values,
            df.loc[mask_a, "away_goals"].values,
        ])
        ga = np.concatenate([
            df.loc[mask_h, "away_goals"].values,
            df.loc[mask_a, "home_goals"].values,
        ])
        w_h = weights[mask_h.values]
        w_a = weights[mask_a.values]
        w = np.concatenate([w_h, w_a]) if len(w_h) + len(w_a) else np.array([1.0])

        avg_gf = float(np.average(gf, weights=w)) if len(gf) else 1.2
        avg_ga = float(np.average(ga, weights=w)) if len(ga) else 1.2
        attack[team] = avg_gf
        defense[team] = avg_ga
        n_matches[team] = len(gf)
        global_gf.extend(gf.tolist())
        global_ga.extend(ga.tolist())

    mean_gf = float(np.mean(global_gf)) if global_gf else 1.35
    mean_ga = float(np.mean(global_ga)) if global_ga else 1.35

    ctx = context.set_index("team_name") if not context.empty else pd.DataFrame()
    team_params: dict[str, dict[str, float]] = {}

    for team in teams:
        struct = float(ctx.loc[team, "structural_strength"]) if team in ctx.index else 0.0
        # alpha: attack (higher → more goals). beta: defense strength (higher → fewer conceded)
        alpha = math.log(max(attack[team], 0.3) / mean_gf) + 0.05 * struct
        beta = -math.log(max(defense[team], 0.3) / mean_ga) + 0.03 * struct
        team_params[team] = {"alpha": alpha, "beta": beta}

    atk = pd.Series({t: team_params[t]["alpha"] for t in teams})
    defn = pd.Series({t: team_params[t]["beta"] for t in teams})
    atk_mean, def_mean = atk.mean(), defn.mean()
    for team in teams:
        team_params[team] = {
            "alpha": round(float(atk[team] - atk_mean), 4),
            "beta": round(float(defn[team] - def_mean), 4),
            "n_matches": n_matches[team],
        }

    return {
        "baseline": round(math.log(mean_gf), 4),
        "gamma": 0.27,
        "rho": -0.13,
        "teams": team_params,
    }


def lambdas(
    team_a: str,
    team_b: str,
    weights: dict[str, Any],
    *,
    team_a_home: bool = True,
    neutral: bool = False,
) -> tuple[float, float]:
    teams = weights.get("teams", {})
    baseline = weights.get("baseline", 0.22)
    gamma = weights.get("gamma", 0.27)

    a = teams.get(team_a, {"alpha": 0.0, "beta": 0.0})
    b = teams.get(team_b, {"alpha": 0.0, "beta": 0.0})

    if neutral:
        log_la = baseline + a["alpha"] - b["beta"]
        log_lb = baseline + b["alpha"] - a["beta"]
    elif team_a_home:
        log_la = baseline + a["alpha"] - b["beta"] + gamma
        log_lb = baseline + b["alpha"] - a["beta"]
    else:
        log_la = baseline + a["alpha"] - b["beta"]
        log_lb = baseline + b["alpha"] - a["beta"] + gamma

    return math.exp(np.clip(log_la, -2.0, 2.5)), math.exp(np.clip(log_lb, -2.0, 2.5))


def component_predict(
    team_a: str,
    team_b: str,
    weights: dict[str, Any],
    *,
    team_a_home: bool = True,
    neutral: bool = False,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    lh, la = lambdas(team_a, team_b, weights, team_a_home=team_a_home, neutral=neutral)
    rho = weights.get("rho", -0.13)

    p_home, p_draw, p_away = 0.0, 0.0, 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p = poisson_pmf(x, lh) * poisson_pmf(y, la) * dc_tau(x, y, lh, la, rho)
            if x > y:
                p_home += p
            elif x == y:
                p_draw += p
            else:
                p_away += p

    total = p_home + p_draw + p_away
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p_home / total, p_draw / total, p_away / total


def simulate_scorelines(
    lambda_a: float,
    lambda_b: float,
    rho: float,
    n_sims: int,
    rng: np.random.Generator,
    max_goals: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    goals_a = rng.poisson(lambda_a, size=n_sims)
    goals_b = rng.poisson(lambda_b, size=n_sims)
    goals_a = np.clip(goals_a, 0, max_goals)
    goals_b = np.clip(goals_b, 0, max_goals)
    return goals_a, goals_b
