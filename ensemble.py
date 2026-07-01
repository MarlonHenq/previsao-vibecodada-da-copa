"""Uniform ensemble of Elo + Dixon-Coles + Hierarchical components."""

from __future__ import annotations

from typing import Any, Literal

from calibration import calibrate_proba
from components import dixon_coles, elo_model, hierarchical

Tier = Literal["friendly", "qualifier", "tournament"]


def combine_uniform(
    probs: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    if not probs:
        return 1 / 3, 1 / 3, 1 / 3
    h = sum(p[0] for p in probs) / len(probs)
    d = sum(p[1] for p in probs) / len(probs)
    a = sum(p[2] for p in probs) / len(probs)
    total = h + d + a
    return h / total, d / total, a / total


def ensemble_predict(
    team_a: str,
    team_b: str,
    weights: dict[str, Any],
    *,
    team_a_home: bool = True,
    neutral: bool = False,
    tier: Tier = "tournament",
) -> tuple[float, float, float]:
    components = weights.get("components", {})
    probs = []

    if "elo" in components:
        probs.append(
            elo_model.component_predict(
                team_a, team_b, components["elo"],
                team_a_home=team_a_home, neutral=neutral,
            )
        )
    if "dixon_coles" in components:
        probs.append(
            dixon_coles.component_predict(
                team_a, team_b, components["dixon_coles"],
                team_a_home=team_a_home, neutral=neutral,
            )
        )
    if "hierarchical" in components:
        probs.append(
            hierarchical.component_predict(
                team_a, team_b, components["hierarchical"],
                team_a_home=team_a_home, neutral=neutral,
            )
        )

    raw = combine_uniform(probs)
    w = weights.get("ensemble", {}).get("weights")
    if w and len(w) == 3 and len(probs) == 3:
        raw = (
            probs[0][0] * w[0] + probs[1][0] * w[1] + probs[2][0] * w[2],
            probs[0][1] * w[0] + probs[1][1] * w[1] + probs[2][1] * w[2],
            probs[0][2] * w[0] + probs[1][2] * w[1] + probs[2][2] * w[2],
        )
        t = sum(raw)
        raw = (raw[0] / t, raw[1] / t, raw[2] / t)

    cal_tier = weights.get("ensemble", {}).get("calibrator_tier", tier)
    return calibrate_proba(*raw, tier=cal_tier)
