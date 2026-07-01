"""Orchestrator: load data, train, predict, persist weights."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from components import dixon_coles, elo_model, hierarchical
from ensemble import ensemble_predict
from features import compute_elo

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MATCHES = DATA_DIR / "matches.csv"
DEFAULT_CONTEXT = DATA_DIR / "teams_context.csv"
DEFAULT_WEIGHTS = Path(__file__).parent / "model_weights.json"
DEFAULT_ALIASES = DATA_DIR / "team_aliases.csv"

MATCHES_COLUMNS = [
    "match_id", "date", "home_team", "away_team", "home_goals", "away_goals",
    "is_neutral", "tournament", "tournament_tier", "venue_country",
]
CONTEXT_COLUMNS = [
    "team_name", "confederation", "elo_rating", "elo_external", "fifa_rank",
    "squad_value_eur", "gdp_per_capita_usd", "population_millions",
    "world_cups_won", "wc_appearances", "culture_index", "structural_strength",
    "matches_played", "historical_win_rate", "recent_form", "is_host_2026",
]


def load_aliases(path: Path | None = None) -> dict[str, str]:
    path = path or DEFAULT_ALIASES
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["alias"], df["canonical"]))


def resolve_team(name: str, aliases: dict[str, str] | None = None) -> str:
    aliases = aliases or load_aliases()
    return aliases.get(name, name)


def load_matches(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_MATCHES
    df = pd.read_csv(path)
    missing = set(MATCHES_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"matches.csv missing columns: {missing}")
    return df


def load_teams_context(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_CONTEXT
    df = pd.read_csv(path)
    if "team_name" not in df.columns:
        raise ValueError("teams_context.csv must have team_name column")
    return df


def validate_team_coverage(matches: pd.DataFrame, context: pd.DataFrame) -> None:
    teams_in_matches = set(matches["home_team"]) | set(matches["away_team"])
    teams_in_context = set(context["team_name"])
    missing = teams_in_matches - teams_in_context
    if missing and len(missing) < len(teams_in_matches) * 0.5:
        pass  # bootstrap generates context for all match teams
    elif missing:
        sample = sorted(missing)[:10]
        raise ValueError(f"Teams in matches missing from context (sample): {sample}")


def train_fast(
    matches: pd.DataFrame,
    context: pd.DataFrame,
) -> dict[str, Any]:
    elo_ratings = compute_elo(matches)
    for _, row in context.iterrows():
        team = row["team_name"]
        if team not in elo_ratings:
            elo_ratings[team] = float(row.get("elo_rating", 1500))

    return {
        "version": "1.0.0",
        "model": "ensemble-v1-fast",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_window": {
            "since": str(matches["date"].min()),
            "until": str(matches["date"].max()),
            "n_matches": len(matches),
            "n_teams": len(set(matches["home_team"]) | set(matches["away_team"])),
        },
        "hyperparams": {
            "time_decay_half_life_days": 1825,
        },
        "components": {
            "elo": elo_model.fit_fast(elo_ratings),
            "dixon_coles": dixon_coles.fit_fast(matches, context),
            "hierarchical": hierarchical.fit_fast(matches, context),
        },
        "ensemble": {
            "weights": [0.333, 0.333, 0.334],
            "calibrator_tier": "tournament",
        },
        "structural_priors": {
            "w_elo": 0.70,
            "w_squad": 0.30,
            "context_weights": {"culture": 0.25, "log_gdp": 0.10},
        },
    }


def save_weights(weights: dict[str, Any], path: Path | None = None) -> Path:
    path = path or DEFAULT_WEIGHTS
    path.write_text(json.dumps(weights, indent=2, ensure_ascii=False))
    return path


def load_weights(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_WEIGHTS
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path}. Run `train` first.")
    return json.loads(path.read_text())


def predict_match(
    team_a: str,
    team_b: str,
    weights: dict[str, Any],
    *,
    n_sims: int = 50_000,
    neutral: bool = False,
    team_a_home: bool = True,
    tier: str = "tournament",
    seed: int = 42,
) -> dict[str, Any]:
    aliases = load_aliases()
    team_a = resolve_team(team_a, aliases)
    team_b = resolve_team(team_b, aliases)

    teams_dc = weights.get("components", {}).get("dixon_coles", {}).get("teams", {})
    if team_a not in teams_dc:
        raise KeyError(f"Team not in model: {team_a!r}")
    if team_b not in teams_dc:
        raise KeyError(f"Team not in model: {team_b!r}")

    p_a, p_draw, p_b = ensemble_predict(
        team_a, team_b, weights,
        team_a_home=team_a_home, neutral=neutral, tier=tier,  # type: ignore[arg-type]
    )

    dc = weights["components"]["dixon_coles"]
    lambda_a, lambda_b = dixon_coles.lambdas(
        team_a, team_b, dc, team_a_home=team_a_home, neutral=neutral,
    )
    rho = dc.get("rho", -0.13)
    rng = np.random.default_rng(seed)
    ga, gb = dixon_coles.simulate_scorelines(lambda_a, lambda_b, rho, n_sims, rng)

    outcomes = np.where(ga > gb, 0, np.where(ga == gb, 1, 2))
    sim_p_a = float(np.mean(outcomes == 0))
    sim_p_draw = float(np.mean(outcomes == 1))
    sim_p_b = float(np.mean(outcomes == 2))

    score_keys = [f"{a}-{b}" for a, b in zip(ga, gb)]
    unique, counts = np.unique(score_keys, return_counts=True)
    top_idx = np.argsort(counts)[::-1][:3]
    top_scores = [
        {"scoreline": unique[i], "probability": float(counts[i] / n_sims)}
        for i in top_idx
    ]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "neutral": neutral,
        "n_sims": n_sims,
        "ensemble": {
            "p_win_a": round(p_a, 4),
            "p_draw": round(p_draw, 4),
            "p_win_b": round(p_b, 4),
        },
        "monte_carlo": {
            "p_win_a": round(sim_p_a, 4),
            "p_draw": round(sim_p_draw, 4),
            "p_win_b": round(sim_p_b, 4),
        },
        "expected_goals": {
            "team_a": round(lambda_a, 3),
            "team_b": round(lambda_b, 3),
        },
        "top_scorelines": top_scores,
    }


def rank_teams(weights: dict[str, Any], top: int = 20, min_matches: int = 30) -> list[dict[str, Any]]:
    dc = weights.get("components", {}).get("dixon_coles", {}).get("teams", {})
    rows = []
    for team, p in dc.items():
        if p.get("n_matches", 0) < min_matches:
            continue
        strength = p.get("alpha", 0.0) + p.get("beta", 0.0)
        rows.append({"team": team, "strength": round(strength, 4), **p})
    rows.sort(key=lambda r: r["strength"], reverse=True)
    return rows[:top]
