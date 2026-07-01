"""Walk-forward backtest (stub for Fase 3)."""

from __future__ import annotations

from typing import Any


def run_backtest(since: str = "2014-01-01") -> dict[str, Any]:
    return {
        "status": "stub",
        "message": "Walk-forward backtest will be implemented in Fase 3.",
        "since": since,
        "metrics": {
            "brier": None,
            "log_loss": None,
            "ece": None,
        },
    }
