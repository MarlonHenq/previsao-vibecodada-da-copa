"""Feature engineering: Elo, structural index, time decay, culture."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HALF_LIFE_DAYS = 1825  # 5 years for international football


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def tournament_importance(tournament: str) -> float:
    """Elo K-factor style weights (used for Elo updates, not DC goal model)."""
    name = (tournament or "").lower()
    if "world cup" in name and "qualification" not in name:
        return 1.0
    if "qualification" in name or "qualifying" in name:
        return 0.65
    return 0.30


def time_decay_weights(dates: pd.Series, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> np.ndarray:
    ref = pd.to_datetime(dates).max()
    ages = (ref - pd.to_datetime(dates)).dt.days.astype(float)
    return np.exp(-math.log(2) * ages / half_life_days)


def compute_elo(
    matches: pd.DataFrame,
    k: float = 40.0,
    hfa: float = 100.0,
    initial: float = 1500.0,
) -> dict[str, float]:
    """World-Football-Elo style ratings computed match-by-match."""
    ratings: dict[str, float] = {}
    df = matches.sort_values("date").copy()

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        ratings.setdefault(home, initial)
        ratings.setdefault(away, initial)

        h_elo = ratings[home]
        a_elo = ratings[away]
        hfa_adj = 0.0 if row.get("is_neutral", 0) else hfa
        expected_home = 1.0 / (1.0 + 10 ** ((a_elo - h_elo - hfa_adj) / 400.0))

        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        if hg > ag:
            actual_home = 1.0
        elif hg < ag:
            actual_home = 0.0
        else:
            actual_home = 0.5

        margin = abs(hg - ag)
        mov_mult = math.log(max(margin, 1) + 1) * (2.2 / (0.001 * abs(h_elo - a_elo) + 2.2))
        imp = tournament_importance(str(row.get("tournament", "")))
        delta = k * imp * mov_mult * (actual_home - expected_home)

        ratings[home] += delta
        ratings[away] -= delta

    return ratings


def count_wc_appearances(matches: pd.DataFrame) -> dict[str, int]:
    wc = matches[
        matches["tournament"].str.contains("FIFA World Cup", case=False, na=False)
        & ~matches["tournament"].str.contains("qualification", case=False, na=False)
    ]
    teams: dict[str, int] = {}
    for _, row in wc.iterrows():
        for t in (row["home_team"], row["away_team"]):
            teams[t] = teams.get(t, 0) + 1
    return teams


def recent_form(matches: pd.DataFrame, n: int = 20, half_life_days: float = 365.0) -> dict[str, float]:
    """Weighted goal difference over last N matches per team."""
    form: dict[str, float] = {}
    df = matches.sort_values("date").copy()
    weights = time_decay_weights(df["date"], half_life_days)

    for team in pd.unique(pd.concat([df["home_team"], df["away_team"]])):
        mask = (df["home_team"] == team) | (df["away_team"] == team)
        team_df = df.loc[mask].tail(n)
        if team_df.empty:
            form[team] = 0.0
            continue
        w = time_decay_weights(team_df["date"], half_life_days)
        gd = []
        for _, row in team_df.iterrows():
            if row["home_team"] == team:
                gd.append(int(row["home_goals"]) - int(row["away_goals"]))
            else:
                gd.append(int(row["away_goals"]) - int(row["home_goals"]))
        form[team] = float(np.average(gd, weights=w)) if w.sum() > 0 else 0.0
    return form


def gdp_term(log_gdp: float) -> float:
    """Inverted-U GDP contribution (playmobil-style)."""
    return 0.15 * (log_gdp - 4.3) - 0.05 * max(0.0, log_gdp - 4.8)


def structural_strength(
    elo_z: float,
    squad_z: float,
    culture: float,
    log_gdp: float,
    log_pop: float,
    is_host: int = 0,
    w_elo: float = 0.70,
    w_squad: float = 0.30,
) -> float:
    anchor = w_elo * elo_z + w_squad * squad_z
    s_raw = (
        anchor
        + gdp_term(log_gdp)
        + 0.10 * (log_pop - 1.3)
        + 0.35 * is_host
        + 0.25 * culture
        + 0.15 * (log_pop * culture)
    )
    return float(s_raw)


def build_teams_context(
    matches: pd.DataFrame,
    confederations_path: Path,
    socio_path: Path,
    wc_won_path: Path | None = None,
    external_elo: dict[str, float] | None = None,
) -> pd.DataFrame:
    confed = pd.read_csv(confederations_path)
    socio = pd.read_csv(socio_path)
    wc_won = pd.read_csv(wc_won_path) if wc_won_path and wc_won_path.exists() else pd.DataFrame()

    all_teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    elo = compute_elo(matches)
    wc_apps = count_wc_appearances(matches)
    form = recent_form(matches)

    # Long-run Elo proxy: same as current for mock; full history in Fase 2
    elo_series = pd.Series(elo)
    elo_z_map = zscore(elo_series).to_dict()
    wc_apps_series = pd.Series({t: wc_apps.get(t, 0) for t in all_teams})
    wc_z = zscore(wc_apps_series).to_dict()
    culture_map = {
        t: 0.5 * elo_z_map.get(t, 0.0) + 0.5 * wc_z.get(t, 0.0) for t in all_teams
    }

    squad_log_z: dict[str, float] = {}
    if not socio.empty and "squad_value_eur" in socio.columns:
        log_vals = socio.set_index("team_name")["squad_value_eur"].apply(
            lambda x: math.log10(max(float(x), 1.0))
        )
        squad_log_z = zscore(log_vals).to_dict()

    rows = []
    for team in all_teams:
        conf_row = confed[confed["team_name"] == team]
        confederation = conf_row["confederation"].iloc[0] if len(conf_row) else "OTHER"

        socio_row = socio[socio["team_name"] == team]
        if len(socio_row):
            sr = socio_row.iloc[0]
            fifa_rank = sr.get("fifa_rank", np.nan)
            gdp = sr.get("gdp_per_capita_usd", np.nan)
            pop = sr.get("population_millions", np.nan)
            squad = sr.get("squad_value_eur", np.nan)
            is_host = int(sr.get("is_host_2026", 0))
        else:
            conf_med = socio.merge(confed, on="team_name", how="inner")
            conf_med = conf_med[conf_med["confederation"] == confederation]
            fifa_rank = conf_med["fifa_rank"].median() if len(conf_med) else 100
            gdp = conf_med["gdp_per_capita_usd"].median() if len(conf_med) else 5000.0
            pop = conf_med["population_millions"].median() if len(conf_med) else 10.0
            squad = conf_med["squad_value_eur"].median() if len(conf_med) else 50000000.0
            is_host = 0

        wc_w = 0
        if len(wc_won) and team in wc_won["team_name"].values:
            wc_w = int(wc_won.loc[wc_won["team_name"] == team, "world_cups_won"].iloc[0])

        team_matches = matches[
            (matches["home_team"] == team) | (matches["away_team"] == team)
        ]
        wins = 0
        for _, m in team_matches.iterrows():
            if m["home_team"] == team:
                if m["home_goals"] > m["away_goals"]:
                    wins += 1
            elif m["away_goals"] > m["home_goals"]:
                wins += 1
        n = len(team_matches)
        win_rate = wins / n if n else 0.0

        gf, ga = [], []
        for _, m in team_matches.iterrows():
            if m["home_team"] == team:
                gf.append(m["home_goals"])
                ga.append(m["away_goals"])
            else:
                gf.append(m["away_goals"])
                ga.append(m["home_goals"])

        log_gdp = math.log10(max(float(gdp), 1.0))
        log_pop = math.log10(max(float(pop), 0.1) * 1e6)
        squad_z = squad_log_z.get(team, 0.0)

        struct = structural_strength(
            elo_z_map.get(team, 0.0),
            squad_z,
            culture_map.get(team, 0.0),
            log_gdp,
            log_pop,
            is_host,
        )

        rows.append(
            {
                "team_name": team,
                "confederation": confederation,
                "elo_rating": round(elo.get(team, 1500.0), 1),
                "elo_external": round((external_elo or {}).get(team, elo.get(team, 1500.0)), 1),
                "fifa_rank": int(fifa_rank) if not pd.isna(fifa_rank) else 100,
                "squad_value_eur": float(squad) if not pd.isna(squad) else 0.0,
                "gdp_per_capita_usd": float(gdp),
                "population_millions": float(pop),
                "world_cups_won": wc_w,
                "wc_appearances": wc_apps.get(team, 0),
                "culture_index": round(culture_map.get(team, 0.0), 3),
                "structural_strength": round(struct, 3),
                "matches_played": n,
                "historical_win_rate": round(win_rate, 3),
                "avg_goals_for": round(float(np.mean(gf)) if gf else 0.0, 3),
                "avg_goals_against": round(float(np.mean(ga)) if ga else 0.0, 3),
                "recent_form": round(form.get(team, 0.0), 3),
                "is_host_2026": is_host,
            }
        )

    return pd.DataFrame(rows).sort_values("elo_rating", ascending=False)
