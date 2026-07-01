"""Component 3: Hierarchical Bayesian (fast heuristic for Fase 1)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from components.dixon_coles import fit_fast as dc_fit_fast


CONFED_DEFAULT_OFFSETS = {
    "UEFA": 0.08,
    "CONMEBOL": 0.12,
    "CAF": -0.05,
    "AFC": -0.03,
    "CONCACAF": -0.02,
    "OFC": -0.15,
    "OTHER": 0.0,
}


def fit_fast(matches: pd.DataFrame, context: pd.DataFrame) -> dict[str, Any]:
    """Structural-prior shrunk attack/defense (mock for PyMC in Fase 3)."""
    dc = dc_fit_fast(matches, context)
    ctx = context.set_index("team_name")

    confed_offsets: dict[str, float] = {}
    if not context.empty:
        for conf in context["confederation"].unique():
            sub = context[context["confederation"] == conf]
            confed_offsets[conf] = CONFED_DEFAULT_OFFSETS.get(
                conf,
                float(sub["structural_strength"].mean()) if len(sub) else 0.0,
            )

    teams_out: dict[str, dict[str, float]] = {}
    for team, params in dc["teams"].items():
        conf = str(ctx.loc[team, "confederation"]) if team in ctx.index else "OTHER"
        offset = confed_offsets.get(conf, 0.0)
        struct = float(ctx.loc[team, "structural_strength"]) if team in ctx.index else 0.0
        shrink = 0.3  # partial pooling toward confederation mean
        teams_out[team] = {
            "alpha": round(params["alpha"] * (1 - shrink) + offset * shrink + 0.1 * struct, 4),
            "beta": round(params["beta"] * (1 - shrink) - offset * shrink * 0.5, 4),
        }

    return {
        "confed_offsets": {k: round(v, 4) for k, v in confed_offsets.items()},
        "teams": teams_out,
    }


def component_predict(
    team_a: str,
    team_b: str,
    weights: dict[str, Any],
    *,
    team_a_home: bool = True,
    neutral: bool = False,
) -> tuple[float, float, float]:
    from components.dixon_coles import component_predict as dc_predict

    dc_weights = {
        "baseline": 0.300,
        "gamma": 0.27,
        "rho": -0.13,
        "teams": weights.get("teams", {}),
    }
    return dc_predict(
        team_a, team_b, dc_weights, team_a_home=team_a_home, neutral=neutral
    )
