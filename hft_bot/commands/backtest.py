"""Backtest command – versão simplificada e mais robusta."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from hft_bot.backtesting.engine import BacktestEngine
from hft_bot.core.config import ConfigProvider


# --------------------------------------------------------------------------- #
# Função principal chamada pelo CLI
# --------------------------------------------------------------------------- #
async def run_backtest(
    data_file: str,
    config_file: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    output_file: str = "backtest_results.json",
    show_plot: bool = True,
    walk_forward: bool = False,        # <‑‑ mantidos p/ compatibilidade
    window_test: int = 30,
) -> Dict[str, Any]:
    """Carrega dados, filtra período e delega ao BacktestEngine."""

    # ------------------------------------------------------------------ #
    # 1) Configuração do bot
    # ------------------------------------------------------------------ #
    config = ConfigProvider(config_file).load()

    # ------------------------------------------------------------------ #
    # 2) Carregamento do CSV
    # ------------------------------------------------------------------ #
    # Aceitamos tanto 'timestamp' quanto 'open_time'
    try:
        df = pd.read_csv(
            data_file,
            parse_dates=["timestamp"],
        )
        time_col = "timestamp"
    except ValueError:
        # coluna 'timestamp' não existe; tentamos 'open_time'
        df = pd.read_csv(
            data_file,
            parse_dates=["open_time"],
        ).rename(columns={"open_time": "timestamp"})
        time_col = "timestamp"

    # Se vier com timezone ( +00:00 ), removemos para evitar comparações
    if df[time_col].dt.tz is not None:
        df[time_col] = df[time_col].dt.tz_convert(None)

    # ------------------------------------------------------------------ #
    # 3) Filtro de datas
    # ------------------------------------------------------------------ #
    if start_date is not None:
        df = df[df[time_col] >= start_date]
    if end_date is not None:
        df = df[df[time_col] <= end_date]

    # ------------------------------------------------------------------ #
    # 4) Executa back‑test
    # ------------------------------------------------------------------ #
    engine = BacktestEngine(config)
    results = await engine.run(df)

    # ------------------------------------------------------------------ #
    # 5) Persistência e opcionalmente gráfico
    # ------------------------------------------------------------------ #
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, default=str)

    if show_plot:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot(results["equity_curve"])
            plt.title("Equity Curve")
            plt.xlabel("Time (bar)")
            plt.ylabel("Equity")
            plt.grid(True)
            plt.show()
        except Exception:  # noqa: BLE001
            pass

    return results
