# hft_bot/optimization/optimizer.py
"""
ParameterOptimizer – busca de hiper‑parâmetros com Optuna.

• Estratégia: ma_crossover (médias móveis).
• Métrica‑alvo: maximizar Sharpe (ou total_return / min. drawdown).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd


# --------------------------------------------------------------------------- #
# Configuração                                                                
# --------------------------------------------------------------------------- #
@dataclass
class MAConfig:
    fast: int
    slow: int


# --------------------------------------------------------------------------- #
# Métricas básicas                                                            
# --------------------------------------------------------------------------- #
def calc_metrics(equity: np.ndarray) -> Dict[str, float]:
    total_return = float(equity[-1] / equity[0] - 1)

    # amostragem diária simples (1440 candles de 1 min ≈ 1 dia)
    daily = equity[::1440]
    if len(daily) >= 2:
        daily_ret = daily[1:] / daily[:-1] - 1
        sharpe = float(daily_ret.mean() / (daily_ret.std() + 1e-12) * np.sqrt(252))
    else:
        sharpe = 0.0

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())

    return {"total_return": total_return, "sharpe": sharpe, "max_drawdown": max_dd}


# --------------------------------------------------------------------------- #
# Back‑test muito simples para crossover                                      
# --------------------------------------------------------------------------- #
def backtest_ma(df: pd.DataFrame, cfg: MAConfig) -> Dict[str, float]:
    price = df["close"].to_numpy(float)

    sma_fast = pd.Series(price).rolling(cfg.fast).mean().to_numpy(float)
    sma_slow = pd.Series(price).rolling(cfg.slow).mean().to_numpy(float)

    equity = [1.0]
    position = 0  # 0 = flat, 1 = comprado
    entry_price = 0.0

    for i, p in enumerate(price):
        if np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]):
            equity.append(equity[-1])
            continue

        if position == 0 and sma_fast[i] > sma_slow[i]:
            position = 1
            entry_price = p
        elif position == 1 and sma_fast[i] < sma_slow[i]:
            equity[-1] *= p / entry_price
            position = 0

        equity.append(equity[-1])

    return calc_metrics(np.array(equity[1:]))


# --------------------------------------------------------------------------- #
# Classe principal                                                            
# --------------------------------------------------------------------------- #
class ParameterOptimizer:
    def __init__(self, data_file: str, metric: str = "sharpe") -> None:
        self.metric = metric.lower()
        self.df = self._load_data(Path(data_file))

    @staticmethod
    def _load_data(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        return df

    # ---------- função objetivo do Optuna ---------- #
    def _objective(self, trial: optuna.Trial) -> float:
        fast = trial.suggest_int("fast", 5, 60)
        slow = trial.suggest_int("slow", 70, 300)

        if fast >= slow:
            raise optuna.TrialPruned()

        metrics = backtest_ma(self.df, MAConfig(fast, slow))

        if self.metric == "sharpe":
            return -metrics["sharpe"]          # maximizar Sharpe
        if self.metric == "total_return":
            return -metrics["total_return"]    # maximizar retorno
        if self.metric == "max_drawdown":
            return metrics["max_drawdown"]     # minimizar drawdown
        raise ValueError(f"Métrica desconhecida: {self.metric}")

    # ---------- API chamada pelo CLI ---------- #
    async def optimize_async(self, n_trials: int = 100, n_jobs: int = 1) -> Dict[str, Any]:
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

        best = study.best_params
        best["best_metric"] = (
            -study.best_value if self.metric in {"sharpe", "total_return"} else study.best_value
        )
        return best
