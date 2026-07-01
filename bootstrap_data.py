"""Download and normalize multi-source football data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

from features import build_teams_context

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PATCHES_PATH = DATA_DIR / "wc2026_patches.csv"

MARTJ42_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
ELO_TSV_URL = "https://www.eloratings.net/World.tsv"

MATCHES_COLUMNS = [
    "match_id",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "is_neutral",
    "tournament",
    "tournament_tier",
    "venue_country",
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def tournament_tier(tournament: str) -> str:
    name = (tournament or "").lower()
    if "world cup" in name and "qualification" not in name:
        return "tournament"
    if any(
        kw in name
        for kw in (
            "euro",
            "copa america",
            "copa américa",
            "africa cup",
            "asian cup",
            "gold cup",
            "nations league",
            "confederations cup",
            "continental",
        )
    ):
        return "tournament"
    if "qualification" in name or "qualifying" in name:
        return "qualifier"
    return "friendly"


def download_martj42(dest: Path | None = None) -> Path:
    dest = dest or RAW_DIR / "results.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(MARTJ42_URL, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def fetch_elo_ratings() -> dict[str, dict]:
    """Parse eloratings.net World.tsv into {team_name: stats}."""
    try:
        resp = requests.get(ELO_TSV_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return {}

    ratings: dict[str, dict] = {}
    for line in resp.text.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            elo = float(parts[3])
        except ValueError:
            continue
        code = parts[2].strip()
        ratings[code] = {
            "elo_external": elo,
            "total_games": int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else None,
            "wins": int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else None,
            "losses": int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else None,
        }
    return ratings


# ISO3 codes from eloratings.net → martj42 team names
ELO_CODE_TO_TEAM: dict[str, str] = {
    "BRA": "Brazil",
    "ARG": "Argentina",
    "FRA": "France",
    "ENG": "England",
    "BEL": "Belgium",
    "POR": "Portugal",
    "NED": "Netherlands",
    "ESP": "Spain",
    "ITA": "Italy",
    "CRO": "Croatia",
    "URU": "Uruguay",
    "COL": "Colombia",
    "GER": "Germany",
    "USA": "USA",
    "MEX": "Mexico",
    "MAR": "Morocco",
    "SUI": "Switzerland",
    "JPN": "Japan",
    "SEN": "Senegal",
    "IRN": "Iran",
    "DEN": "Denmark",
    "KOR": "Korea Republic",
    "AUS": "Australia",
    "AUT": "Austria",
    "TUR": "Turkey",
    "CAN": "Canada",
    "UKR": "Ukraine",
    "ECU": "Ecuador",
    "SCO": "Scotland",
    "SRB": "Serbia",
    "PAR": "Paraguay",
    "TUN": "Tunisia",
    "CHI": "Chile",
    "POL": "Poland",
    "WAL": "Wales",
    "ALG": "Algeria",
    "EGY": "Egypt",
    "NOR": "Norway",
    "PAN": "Panama",
    "PER": "Peru",
    "SVK": "Slovakia",
    "QAT": "Qatar",
    "CZE": "Czech Republic",
    "CRC": "Costa Rica",
    "NGA": "Nigeria",
    "IRL": "Republic of Ireland",
    "HUN": "Hungary",
    "CIV": "Ivory Coast",
    "KSA": "Saudi Arabia",
    "CMR": "Cameroon",
    "GHA": "Ghana",
    "BOL": "Bolivia",
    "VEN": "Venezuela",
    "IRQ": "Iraq",
    "RUS": "Russia",
    "NZL": "New Zealand",
    "RSA": "South Africa",
}


def elo_by_team_name(fetch: bool = True) -> dict[str, float]:
    if not fetch:
        return {}
    raw = fetch_elo_ratings()
    out: dict[str, float] = {}
    for code, stats in raw.items():
        team = ELO_CODE_TO_TEAM.get(code)
        if team:
            out[team] = stats["elo_external"]
    return out


def merge_result_patches(raw: pd.DataFrame, patches_path: Path | None = None) -> pd.DataFrame:
    """Apply manual result patches (e.g. latest WC 2026 knockouts not yet in martj42)."""
    patches_path = patches_path or PATCHES_PATH
    if not patches_path.exists():
        return raw

    out = raw.copy()
    patches = pd.read_csv(patches_path)
    for _, row in patches.iterrows():
        mask = (
            (out["date"].astype(str) == str(row["date"]))
            & (out["home_team"] == row["home_team"])
            & (out["away_team"] == row["away_team"])
            & (out["tournament"] == row["tournament"])
        )
        if mask.any():
            out.loc[mask, "home_score"] = row["home_score"]
            out.loc[mask, "away_score"] = row["away_score"]
            if "neutral" in row and "neutral" in out.columns:
                out.loc[mask, "neutral"] = row["neutral"]
        else:
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out


def load_raw_results(apply_patches: bool = True) -> pd.DataFrame:
    raw_path = RAW_DIR / "results.csv"
    if not raw_path.exists():
        download_martj42()
    raw = pd.read_csv(raw_path)
    if apply_patches:
        raw = merge_result_patches(raw)
        raw.to_csv(raw_path, index=False)
    return raw


def wc2026_summary(matches: pd.DataFrame, cutoff: str = "2026-06-30") -> dict:
    wc = matches[
        (matches["tournament"] == "FIFA World Cup")
        & (matches["date"] >= "2026-06-11")
        & (matches["date"] <= cutoff)
    ]
    return {
        "n_matches": len(wc),
        "first_date": wc["date"].min() if len(wc) else None,
        "last_date": wc["date"].max() if len(wc) else None,
        "group_stage": int((wc["date"] <= "2026-06-27").sum()),
        "knockout": int((wc["date"] >= "2026-06-28").sum()),
    }


def normalize_matches(raw_path: Path | None = None, since: str = "1990-01-01", raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw_df is not None:
        df = raw_df.copy()
    else:
        df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])
    if since:
        df = df[df["date"] >= pd.Timestamp(since)]

    df = df.rename(
        columns={
            "home_score": "home_goals",
            "away_score": "away_goals",
            "country": "venue_country",
        }
    )
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["is_neutral"] = df["neutral"].astype(int) if "neutral" in df.columns else 0
    df["tournament_tier"] = df["tournament"].apply(tournament_tier)
    df["match_id"] = df.apply(
        lambda r: f"{r['date'].strftime('%Y-%m-%d')}_{_slug(r['home_team'])}_{_slug(r['away_team'])}",
        axis=1,
    )

    out = df[
        [
            "match_id",
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "is_neutral",
            "tournament",
            "tournament_tier",
            "venue_country",
        ]
    ].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def run_bootstrap(
    since: str = "1990-01-01",
    fetch_elo: bool = True,
    matches_out: Path | None = None,
    context_out: Path | None = None,
    refresh: bool = True,
    apply_patches: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches_out = matches_out or DATA_DIR / "matches.csv"
    context_out = context_out or DATA_DIR / "teams_context.csv"

    if refresh:
        download_martj42()
    raw = load_raw_results(apply_patches=apply_patches)
    matches = normalize_matches(raw_df=raw, since=since)
    matches.to_csv(matches_out, index=False)

    external_elo = elo_by_team_name(fetch=fetch_elo)
    context = build_teams_context(
        matches,
        confederations_path=DATA_DIR / "confederations.csv",
        socio_path=DATA_DIR / "socio_economic.csv",
        wc_won_path=DATA_DIR / "world_cups_won.csv",
        external_elo=external_elo,
    )
    context.to_csv(context_out, index=False)
    return matches, context


def sync_wc2026(
    cutoff: str = "2026-06-30",
    fetch_elo: bool = True,
    retrain: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Refresh martj42, patch WC 2026 results through cutoff, rebuild CSVs."""
    download_martj42()
    raw = load_raw_results(apply_patches=True)

    # Drop unplayed WC 2026 fixtures (NaN scores) after group stage
    wc_future = (
        (raw["tournament"] == "FIFA World Cup")
        & (raw["date"].astype(str) > cutoff)
    )
    wc_nan = (
        (raw["tournament"] == "FIFA World Cup")
        & (raw["date"].astype(str) <= cutoff)
        & (raw["home_score"].isna() | raw["away_score"].isna())
    )
    raw = raw[~wc_future & ~wc_nan].copy()
    raw.to_csv(RAW_DIR / "results.csv", index=False)

    matches = normalize_matches(raw_df=raw, since="1990-01-01")
    matches.to_csv(DATA_DIR / "matches.csv", index=False)

    external_elo = elo_by_team_name(fetch=fetch_elo)
    context = build_teams_context(
        matches,
        confederations_path=DATA_DIR / "confederations.csv",
        socio_path=DATA_DIR / "socio_economic.csv",
        wc_won_path=DATA_DIR / "world_cups_won.csv",
        external_elo=external_elo,
    )
    context.to_csv(DATA_DIR / "teams_context.csv", index=False)

    summary = wc2026_summary(matches, cutoff=cutoff)
    summary["cutoff"] = cutoff
    return matches, context, summary
