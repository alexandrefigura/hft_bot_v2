# hft_bot/backtesting/engine.py
"""BacktestEngine — SMA crossover parametrizado via YAML.

Usa média móvel curta/longa definidas em `strategy_params`:
  fast: 30
  slow: 120

Estratégia:
• Se SMA‑fast cruza ACIMA da SMA‑slow → compra 100 % do capital
• Se cruza ABAIXO → zera posição
• Fator de taxa simples (0,04 % na entrada e na saída)
"""

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

getcontext().prec = 28           # mais precisão nos decimais


# --------------------------------------------------------------------------- #
#  Estruturas auxiliares
# --------------------------------------------------------------------------- #
@dataclass
class Trade:
    entry: Decimal
    exit: Decimal
    side: str          # sempre LONG aqui

    @property
    def pnl_percent(self) -> float:
        return float((self.exit / self.entry) - 1)


@dataclass
class Position:
    qty: Decimal
    entry_price: Decimal


# --------------------------------------------------------------------------- #
#  Motor de back‑test
# --------------------------------------------------------------------------- #
class BacktestEngine:
    FEE = Decimal("0.0004")

    def __init__(self,
                 cfg: Any | None = None,
                 *,
                 config: Any | None = None,      # aceita “config=”
                 initial_capital: float | Decimal = 1_000):

        # ⇢ agora self.cfg é preenchido venha de onde vier
        self.cfg = config or cfg or {}
        self.capital = Decimal(str(initial_capital))

        # --- parâmetros da estratégia (ou defaults 30/120) ---
        params = (self.cfg.get("strategy_params") or {})
        self.fast = int(params.get("fast", 30))
        self.slow = int(params.get("slow", 120))

        if self.fast >= self.slow:
            raise ValueError("fast precisa ser < slow (ex.: 30/120)")

    # -------------------- API principal chamada pelo CLI ------------------- #
    async def run(
        self,
        df: pd.DataFrame,
        *_,                     # ignora posicionais adicionais
        **__,                   # ignora keyword‑args como strategy_cls
    ) -> Dict[str, Any]:
        df = df.copy()
        self._validate_columns(df)

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        df["sma_fast"] = df["close"].rolling(self.fast).mean()
        df["sma_slow"] = df["close"].rolling(self.slow).mean()

        equity_curve: List[Decimal] = []
        trades: List[Trade] = []
        cash: Decimal = self.capital
        pos: Optional[Position] = None

        for ts, row in df.iterrows():
            price = Decimal(str(row["close"]))
            sma_f = row["sma_fast"]
            sma_s = row["sma_slow"]

            # ainda sem SMAs populadas
            if np.isnan(sma_f) or np.isnan(sma_s):
                equity_curve.append(cash)
                continue

            # ---- lógica de sinal ---- #
            if (sma_f > sma_s) and (pos is None):
                # BUY
                qty = (cash * (1 - self.FEE)) / price
                pos = Position(qty=qty, entry_price=price)
                cash = Decimal("0")

            elif (sma_f < sma_s) and (pos is not None):
                # SELL
                exit_cash = pos.qty * price * (1 - self.FEE)
                trades.append(Trade(entry=pos.entry_price, exit=price, side="LONG"))
                cash = exit_cash
                pos = None

            # ---- marcação a mercado ---- #
            equity = cash + (pos.qty * price if pos else Decimal("0"))
            equity_curve.append(equity)

        # fecha posição no último candle, se houver
        if pos is not None:
            final_price = Decimal(str(df.iloc[-1]["close"]))
            cash = pos.qty * final_price * (1 - self.FEE)
            trades.append(Trade(entry=pos.entry_price, exit=final_price, side="LONG"))
            equity_curve[-1] = cash

        # ---------------- métricas ---------------- #
        eq = np.array([float(e) for e in equity_curve])
        total_return = float(eq[-1] / eq[0] - 1)

        daily_ret = np.diff(np.log(eq[::1440] + 1e-9))  # 1440 min (1 dia) em série 1‑min
        sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-12)) * np.sqrt(252) if len(daily_ret) else 0

        running_max = np.maximum.accumulate(eq)
        max_dd = float(((eq - running_max) / running_max).min())

        wins = sum(1 for t in trades if t.pnl_percent > 0)
        win_rate = wins / len(trades) if trades else 0.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "equity_curve": eq.tolist(),
        }

    # ---------------- helpers ----------------- #
    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        # aceita 'open_time' como alias de 'timestamp'
        # e 'price' como alias de 'close'
        if "timestamp" not in df.columns and "open_time" in df.columns:
            df.rename(columns={"open_time": "timestamp"}, inplace=True)
        if "close" not in df.columns and "price" in df.columns:
            df.rename(columns={"price": "close"}, inplace=True)

        missing = {"timestamp", "close"} - set(df.columns)
        if missing:
            raise ValueError(
                "CSV precisa ter colunas: timestamp, close "
                f"(faltando: {', '.join(missing)})"
            )