"""Component 1: Elo-based W/D/L probabilities."""

from __future__ import annotations

import math
from typing import Any


def expected_score(elo_a: float, elo_b: float, hfa: float = 0.0) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a - hfa) / 400.0))


def predict_proba(
    elo_home: float,
    elo_away: float,
    *,
    hfa_points: float = 100.0,
    draw_rate: float = 0.22,
    neutral: bool = False,
) -> tuple[float, float, float]:
    hfa = 0.0 if neutral else hfa_points
    p_home_win = expected_score(elo_home, elo_away, hfa)
    p_draw = draw_rate
    p_away_win = max(0.0, 1.0 - p_home_win - p_draw)
    total = p_home_win + p_draw + p_away_win
    return p_home_win / total, p_draw / total, p_away_win / total


def fit_fast(
    elo_ratings: dict[str, float],
    hfa_points: float = 100.0,
    draw_rate: float = 0.22,
) -> dict[str, Any]:
    return {
        "hfa_points": hfa_points,
        "draw_rate": draw_rate,
        "ratings": {k: round(v, 1) for k, v in elo_ratings.items()},
    }


def component_predict(
    team_a: str,
    team_b: str,
    weights: dict[str, Any],
    *,
    team_a_home: bool = True,
    neutral: bool = False,
) -> tuple[float, float, float]:
    ratings = weights.get("ratings", {})
    default = 1500.0
    elo_a = ratings.get(team_a, default)
    elo_b = ratings.get(team_b, default)
    if neutral:
        return predict_proba(
            elo_a, elo_b,
            hfa_points=weights.get("hfa_points", 100.0),
            draw_rate=weights.get("draw_rate", 0.22),
            neutral=True,
        )
    if team_a_home:
        return predict_proba(
            elo_a, elo_b,
            hfa_points=weights.get("hfa_points", 100.0),
            draw_rate=weights.get("draw_rate", 0.22),
            neutral=False,
        )
    return predict_proba(
        elo_b, elo_a,
        hfa_points=weights.get("hfa_points", 100.0),
        draw_rate=weights.get("draw_rate", 0.22),
        neutral=False,
    )
