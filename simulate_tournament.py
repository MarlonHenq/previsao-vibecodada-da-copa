"""Monte Carlo simulation of WC 2026 knockout bracket from current state."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from components import dixon_coles
from model import DEFAULT_WEIGHTS, load_weights

DATA_DIR = Path(__file__).parent / "data"
BRACKET_PATH = DATA_DIR / "wc2026_bracket.json"

# Shootout winners not captured by 1-1 score in results.csv
PENALTY_WINNERS: dict[tuple[str, str], str] = {
    ("Germany", "Paraguay"): "Paraguay",
    ("Paraguay", "Germany"): "Paraguay",
    ("Netherlands", "Morocco"): "Morocco",
    ("Morocco", "Netherlands"): "Morocco",
}


def load_bracket(path: Path | None = None) -> dict[str, Any]:
    path = path or BRACKET_PATH
    return json.loads(path.read_text())


def _match_teams(defn: dict[str, Any], winners: dict[str, str]) -> tuple[str, str, bool]:
    if "home" in defn and "away" in defn:
        home, away = defn["home"], defn["away"]
    else:
        home = winners[defn["home_from"]]
        away = winners[defn["away_from"]]
    neutral = defn.get("neutral", True)
    return home, away, neutral


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    weights: dict[str, Any],
    rng: np.random.Generator,
    *,
    neutral: bool = True,
) -> str:
    """Single knockout tie: 90' + ET + pens if needed."""
    dc = weights["components"]["dixon_coles"]
    la, lb = dixon_coles.lambdas(
        team_a, team_b, dc,
        team_a_home=not neutral,
        neutral=neutral,
    )
    ga = int(rng.poisson(la))
    gb = int(rng.poisson(lb))

    if ga > gb:
        return team_a
    if gb > ga:
        return team_b

    ga += int(rng.poisson(la * 0.33))
    gb += int(rng.poisson(lb * 0.33))
    if ga > gb:
        return team_a
    if gb > ga:
        return team_b

    ratings = weights["components"]["elo"]["ratings"]
    ea = ratings.get(team_a, 1500.0)
    eb = ratings.get(team_b, 1500.0)
    hfa = 0.0 if neutral else 100.0
    p_a = 1.0 / (1.0 + 10 ** ((eb - ea - hfa) / 400.0))
    return team_a if rng.random() < p_a else team_b


def run_single_tournament(
    bracket: dict[str, Any],
    weights: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    winners: dict[str, str] = {}
    path: list[dict[str, str]] = []

    for m in bracket["matches"]:
        mid = m["id"]
        if m.get("winner"):
            winners[mid] = m["winner"]
            continue

        if "home" in m and "away" in m:
            home, away = m["home"], m["away"]
            neutral = m.get("neutral", True)
        else:
            home, away, neutral = _match_teams(m, winners)

        winner = simulate_knockout_match(
            home, away, weights, rng, neutral=neutral,
        )
        winners[mid] = winner
        path.append({
            "id": mid,
            "round": m["round"],
            "home": home,
            "away": away,
            "winner": winner,
        })

    return {
        "champion": winners["M104"],
        "finalists": [winners["M101"], winners["M102"]],
        "semifinalists": [winners["M101"], winners["M102"]],
        "winners": winners,
        "simulated_matches": path,
    }


def _finalists_from_winners(winners: dict[str, str]) -> tuple[str, str]:
    champ = winners["M104"]
    a, b = winners["M101"], winners["M102"]
    runner_up = b if champ == a else a
    return champ, runner_up


def simulate_tournament(
    weights: dict[str, Any] | None = None,
    n_sims: int = 20_000,
    seed: int = 42,
    bracket_path: Path | None = None,
) -> dict[str, Any]:
    weights = weights or load_weights()
    bracket = load_bracket(bracket_path)
    rng = np.random.default_rng(seed)

    champion_ctr: Counter[str] = Counter()
    finalist_ctr: Counter[str] = Counter()
    semifinal_ctr: Counter[str] = Counter()
    r16_ctr: Counter[str] = Counter()

    for _ in range(n_sims):
        result = run_single_tournament(bracket, weights, rng)
        w = result["winners"]
        champ, finalist = _finalists_from_winners(w)

        champion_ctr[champ] += 1
        finalist_ctr[champ] += 1
        finalist_ctr[finalist] += 1
        for t in (w["M101"], w["M102"]):
            semifinal_ctr[t] += 1
        for mid in (f"M{id}" for id in range(89, 97)):
            if mid in w:
                r16_ctr[w[mid]] += 1

    def top_table(counter: Counter[str], label: str, top: int = 15) -> list[dict]:
        return [
            {"team": team, "count": cnt, "probability": round(cnt / n_sims, 4)}
            for team, cnt in counter.most_common(top)
        ]

    # One sample path for display
    sample_rng = np.random.default_rng(seed + 1)
    sample = run_single_tournament(bracket, weights, sample_rng)

    completed = [
        m for m in bracket["matches"]
        if m.get("winner") and m["round"] == "r32"
    ]
    pending_r32 = [
        m for m in bracket["matches"]
        if not m.get("winner") and m["round"] == "r32"
    ]

    return {
        "n_sims": n_sims,
        "as_of": bracket.get("as_of"),
        "completed_r32": len(completed),
        "pending_r32": len(pending_r32),
        "champion": top_table(champion_ctr, "champion", 20),
        "finalist": top_table(finalist_ctr, "finalist", 20),
        "semifinal": top_table(semifinal_ctr, "semifinal", 20),
        "round_of_16": top_table(r16_ctr, "r16", 24),
        "sample_champion": sample["champion"],
        "sample_final": _finalists_from_winners(sample["winners"]),
        "sample_path": sample["simulated_matches"],
        "known_winners": {
            m["id"]: m["winner"]
            for m in bracket["matches"]
            if m.get("winner")
        },
    }
