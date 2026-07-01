#!/usr/bin/env python3
"""Bayesian Football Prediction Engine — CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from simulate_tournament import simulate_tournament
from bootstrap_data import run_bootstrap, sync_wc2026
from model import (
    DEFAULT_CONTEXT,
    DEFAULT_MATCHES,
    DEFAULT_WEIGHTS,
    load_matches,
    load_teams_context,
    load_weights,
    predict_match,
    rank_teams,
    save_weights,
    train_fast,
    validate_team_coverage,
)

app = typer.Typer(
    name="prever-copa",
    help="Motor Preditivo de Futebol Bayesiano — ensemble Elo + Dixon-Coles + Hierárquico",
)
console = Console()


@app.command("bootstrap-data")
def bootstrap_data(
    since: str = typer.Option("1990-01-01", help="Include matches from this date"),
    fetch_elo: bool = typer.Option(True, "--fetch-elo/--no-fetch-elo"),
    full_history: bool = typer.Option(False, "--full-history", help="Use all matches since 1872"),
):
    """Download martj42 results and build matches.csv + teams_context.csv."""
    if full_history:
        since = "1872-01-01"
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Downloading and normalizing data...", total=None)
        matches, context = run_bootstrap(since=since, fetch_elo=fetch_elo)
        progress.update(task, description="Done.")

    console.print(Panel(
        f"[green]Bootstrap complete[/green]\n"
        f"Matches: {len(matches):,} → {DEFAULT_MATCHES}\n"
        f"Teams: {len(context):,} → {DEFAULT_CONTEXT}",
        title="bootstrap-data",
    ))


@app.command("sync-wc2026")
def sync_wc2026_cmd(
    cutoff: str = typer.Option("2026-06-30", help="Include WC 2026 matches through this date (YYYY-MM-DD)"),
    fetch_elo: bool = typer.Option(True, "--fetch-elo/--no-fetch-elo"),
    retrain: bool = typer.Option(True, "--retrain/--no-retrain", help="Retrain model after sync"),
):
    """Download latest results and ingest FIFA World Cup 2026 matches through cutoff."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Syncing WC 2026 from martj42 + patches...", total=None)
        matches, context, summary = sync_wc2026(cutoff=cutoff, fetch_elo=fetch_elo)

        if retrain:
            progress.add_task("Retraining ensemble with updated data...", total=None)
            weights = train_fast(matches, context)
            save_weights(weights, DEFAULT_WEIGHTS)

    lines = [
        "[green]Copa 2026 sincronizada[/green]",
        f"Jogos da Copa até {cutoff}: [bold]{summary['n_matches']}[/bold]",
        f"  • Fase de grupos: {summary['group_stage']}",
        f"  • Mata-mata (R32): {summary['knockout']}",
        f"  • Período: {summary['first_date']} → {summary['last_date']}",
        f"Total matches.csv: {len(matches):,}",
        f"Total teams_context.csv: {len(context):,}",
    ]
    if retrain:
        lines.append(f"Modelo retreinado → {DEFAULT_WEIGHTS}")
    lines.append(
        "\n[dim]Nota: México vs Equador (30/06 21h ET) não incluído — jogo ainda não disputado.[/dim]"
    )
    console.print(Panel("\n".join(lines), title="sync-wc2026"))


@app.command()
def train(
    matches_path: Path = typer.Option(DEFAULT_MATCHES, "--matches"),
    context_path: Path = typer.Option(DEFAULT_CONTEXT, "--context"),
    output: Path = typer.Option(DEFAULT_WEIGHTS, "--output"),
    fast: bool = typer.Option(True, "--fast/--full", help="Fast heuristic train (Fase 1)"),
):
    """Train ensemble and save model_weights.json."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Loading data...", total=None)
        matches = load_matches(matches_path)
        context = load_teams_context(context_path)
        validate_team_coverage(matches, context)

        progress.add_task("Training ensemble...", total=None)
        if fast:
            weights = train_fast(matches, context)
        else:
            weights = train_fast(matches, context)  # Fase 2+: real MCMC
        save_weights(weights, output)

    table = Table(title=f"Top teams by strength (α + β) — saved to {output}")
    table.add_column("Rank", style="dim")
    table.add_column("Team")
    table.add_column("Strength", justify="right")
    table.add_column("α", justify="right")
    table.add_column("β", justify="right")

    for i, row in enumerate(rank_teams(weights, top=15), 1):
        dc = weights["components"]["dixon_coles"]["teams"].get(row["team"], {})
        table.add_row(
            str(i), row["team"],
            f"{row['strength']:.3f}",
            f"{dc.get('alpha', 0):.3f}",
            f"{dc.get('beta', 0):.3f}",
        )
    console.print(table)


@app.command()
def predict(
    team_a: str = typer.Option(..., "--team-a"),
    team_b: str = typer.Option(..., "--team-b"),
    weights_path: Path = typer.Option(DEFAULT_WEIGHTS, "--weights"),
    sims: int = typer.Option(50_000, "--sims"),
    neutral: bool = typer.Option(False, "--neutral", help="Neutral venue (e.g. World Cup)"),
    seed: int = typer.Option(42, "--seed"),
):
    """Predict match outcome with ensemble + Monte Carlo simulation."""
    try:
        weights = load_weights(weights_path)
        result = predict_match(
            team_a, team_b, weights,
            n_sims=sims, neutral=neutral, seed=seed,
        )
    except (KeyError, FileNotFoundError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    ens = result["ensemble"]
    mc = result["monte_carlo"]
    xg = result["expected_goals"]

    console.print(Panel(
        f"[bold]{result['team_a']}[/bold] vs [bold]{result['team_b']}[/bold]"
        + (" [dim](neutral)[/dim]" if neutral else ""),
        title="Match prediction",
    ))

    prob_table = Table(title="Probabilities")
    prob_table.add_column("Outcome")
    prob_table.add_column("Ensemble", justify="right")
    prob_table.add_column(f"Monte Carlo ({sims:,})", justify="right")
    prob_table.add_row(
        f"Win {result['team_a']}",
        f"{ens['p_win_a']:.1%}",
        f"{mc['p_win_a']:.1%}",
    )
    prob_table.add_row("Draw", f"{ens['p_draw']:.1%}", f"{mc['p_draw']:.1%}")
    prob_table.add_row(
        f"Win {result['team_b']}",
        f"{ens['p_win_b']:.1%}",
        f"{mc['p_win_b']:.1%}",
    )
    console.print(prob_table)

    xg_table = Table(title="Expected goals (λ)")
    xg_table.add_column("Team")
    xg_table.add_column("λ", justify="right")
    xg_table.add_row(result["team_a"], f"{xg['team_a']:.2f}")
    xg_table.add_row(result["team_b"], f"{xg['team_b']:.2f}")
    console.print(xg_table)

    score_table = Table(title="Top 3 exact scorelines")
    score_table.add_column("Scoreline")
    score_table.add_column("Probability", justify="right")
    for s in result["top_scorelines"]:
        score_table.add_row(s["scoreline"], f"{s['probability']:.1%}")
    console.print(score_table)


@app.command()
def rank(
    weights_path: Path = typer.Option(DEFAULT_WEIGHTS, "--weights"),
    top: int = typer.Option(20, "--top"),
):
    """Global team ranking by attack − defense strength."""
    weights = load_weights(weights_path)
    rows = rank_teams(weights, top=top)

    table = Table(title=f"Global ranking (top {top})")
    table.add_column("#", style="dim")
    table.add_column("Team")
    table.add_column("Strength", justify="right")
    table.add_column("α", justify="right")
    table.add_column("β", justify="right")

    for i, row in enumerate(rows, 1):
        table.add_row(
            str(i), row["team"],
            f"{row['strength']:.3f}",
            f"{row.get('alpha', 0):.3f}",
            f"{row.get('beta', 0):.3f}",
        )
    console.print(table)


@app.command("simulate-tournament")
def simulate_tournament_cmd(
    weights_path: Path = typer.Option(DEFAULT_WEIGHTS, "--weights"),
    sims: int = typer.Option(20_000, "--sims", help="Number of full tournament simulations"),
    seed: int = typer.Option(42, "--seed"),
    top: int = typer.Option(15, "--top", help="Rows in probability tables"),
):
    """Simulate the rest of WC 2026 knockout from current results (Monte Carlo)."""
    weights = load_weights(weights_path)
    result = simulate_tournament(weights=weights, n_sims=sims, seed=seed)

    console.print(Panel(
        f"[bold]Copa 2026 — simulação do mata-mata[/bold]\n"
        f"Estado em: {result['as_of']} | {result['completed_r32']}/16 R32 decididos | "
        f"{result['pending_r32']} R32 restantes\n"
        f"{sims:,} torneios simulados",
        title="simulate-tournament",
    ))

    if result["known_winners"]:
        known = Table(title="Resultados já conhecidos (R32)")
        known.add_column("Jogo")
        known.add_column("Vencedor", style="green")
        for mid, team in sorted(result["known_winners"].items()):
            known.add_row(mid, team)
        console.print(known)

    champ = Table(title=f"Probabilidade de CAMPEÃO (top {top})")
    champ.add_column("#", style="dim")
    champ.add_column("Seleção")
    champ.add_column("Prob.", justify="right")
    for i, row in enumerate(result["champion"][:top], 1):
        champ.add_row(str(i), row["team"], f"{row['probability']:.1%}")
    console.print(champ)

    final = Table(title=f"Probabilidade de chegar à FINAL (top {top})")
    final.add_column("#", style="dim")
    final.add_column("Seleção")
    final.add_column("Prob.", justify="right")
    for i, row in enumerate(result["finalist"][:top], 1):
        final.add_row(str(i), row["team"], f"{row['probability']:.1%}")
    console.print(final)

    sf = Table(title=f"Probabilidade de SEMIFINAL (top {top})")
    sf.add_column("#", style="dim")
    sf.add_column("Seleção")
    sf.add_column("Prob.", justify="right")
    for i, row in enumerate(result["semifinal"][:top], 1):
        sf.add_row(str(i), row["team"], f"{row['probability']:.1%}")
    console.print(sf)

    c, r = result["sample_final"]
    console.print(Panel(
        f"[bold green]{c}[/bold green] campeão na simulação de exemplo\n"
        f"Finalista: {r}\n"
        f"(1 entre {sims:,} cenários — veja as tabelas acima para probabilidades)",
        title="Exemplo de final simulada",
    ))


@app.command()
def backtest(
    since: str = typer.Option("2014-01-01", "--since"),
):
    """Walk-forward backtest (Fase 3 stub)."""
    result = run_backtest(since=since)
    console.print(Panel(
        f"[yellow]{result['message']}[/yellow]\nSince: {result['since']}",
        title="backtest",
    ))


if __name__ == "__main__":
    app()
