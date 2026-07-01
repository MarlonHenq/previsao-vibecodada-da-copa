"""Isotonic calibration per tournament tier (stub for Fase 1)."""

from __future__ import annotations

from typing import Literal

Tier = Literal["friendly", "qualifier", "tournament"]

# Fase 1: identity calibration (passthrough with renormalization)
TIER_DRAW_ADJUST = {
    "friendly": 1.0,
    "qualifier": 1.0,
    "tournament": 1.02,
}


def calibrate_proba(
    p_home: float,
    p_draw: float,
    p_away: float,
    tier: Tier = "tournament",
) -> tuple[float, float, float]:
    draw_mult = TIER_DRAW_ADJUST.get(tier, 1.0)
    p_draw_adj = min(p_draw * draw_mult, 0.35)
    remaining = 1.0 - p_draw_adj
    total_hw = p_home + p_away
    if total_hw <= 0:
        return remaining / 2, p_draw_adj, remaining / 2
    p_home_adj = remaining * (p_home / total_hw)
    p_away_adj = remaining * (p_away / total_hw)
    return p_home_adj, p_draw_adj, p_away_adj
