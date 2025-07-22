#!/usr/bin/env python3
"""
Gera dataset supervisionado (features ➜ best_strategy) para treinar
um selector de estratégias.

Exemplo:
    python scripts\\build_training_set.py data\\BTCUSDT_15m.csv ^
           --out datasets\\selector_15m.csv ^
           --window 30 --step 7 --metric sharpe_ratio
"""
from __future__ import annotations

import argparse
import asyncio
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas_ta as ta

from hft_bot.backtesting.engine import BacktestEngine

# --------------------------------------------------------------------------- #
# Mapas de parâmetros das variantes de SMA‑crossover
# --------------------------------------------------------------------------- #
STRATEGIES: Dict[str, Dict[str, int]] = {
    "sma_20_50":   {"fast": 20,  "slow": 50},
    "sma_30_120":  {"fast": 30,  "slow": 120},
    "sma_50_200":  {"fast": 50,  "slow": 200},
    "sma_10_30":   {"fast": 10,  "slow": 30},
    "sma_15_60":   {"fast": 15,  "slow": 60},
}

# --------------------------------------------------------------------------- #
# Funções auxiliares
# --------------------------------------------------------------------------- #
def _prepare_price_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns and "open_time" in df.columns:
        df = df.rename(columns={"open_time": "timestamp"})
    if "close" not in df.columns and "price" in df.columns:
        df = df.rename(columns={"price": "close"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    if "close" not in df.columns:
        raise ValueError("CSV precisa conter coluna 'close'.")
    return df


def slice_df(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index < end)]


def compute_features(df):
    pct = df["close"].pct_change().dropna()
    last = df["close"].iloc[-1]

    # indicadores pandas‑ta
    rsi14  = ta.rsi(df["close"], length=14).iloc[-1]
    atr14  = ta.atr(df["high"], df["low"], df["close"], length=14).iloc[-1]
    macd   = ta.macd(df["close"]).iloc[-1, 0]           # coluna MACD
    slope  = ta.linreg(df["close"], length=20).iloc[-1] # slope correto

    sma200 = df["close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else np.nan

    return {
        "ret_1d":  float(last / df["close"].iloc[-1440] - 1) if len(df) > 1440 else 0.0,
        "vol_1d":  float(pct[-1440:].std()) if len(pct) > 1440 else float(pct.std()),
        "vol_1h":  float(pct[-60:].std())   if len(pct) > 60   else float(pct.std()),
        "sma_ratio": float(last / sma200) if not np.isnan(sma200) else 1.0,
        "rsi14":  float(rsi14),
        "atr14":  float(atr14),
        "macd":   float(macd),
        "slope_20": float(slope),
        "dow":    df.index[-1].dayofweek,
        "hour":   df.index[-1].hour,
    }


async def evaluate_strategy(df_idx: pd.DataFrame,
                            params: Dict[str, int],
                            metric: str) -> float:
    """Executa o motor; recoloca o índice como coluna 'timestamp'."""
    df = df_idx.reset_index().rename_axis(None, axis=1)   # <-- ajuste crucial
    cfg = {"strategy_params": params}
    eng = BacktestEngine(config=cfg, initial_capital=1_000)
    res = await eng.run(df)
    return res[metric]


async def build_dataset(data_csv: str,
                        out_csv: str,
                        window_days: int,
                        step_days: int,
                        metric: str):
    price_df = _prepare_price_df(data_csv)

    rows: List[Dict[str, float]] = []
    window = timedelta(days=window_days)
    step = timedelta(days=step_days)

    t_start = price_df.index.min() + window
    t_end   = price_df.index.max()
    total_windows = int((t_end - t_start) / step) + 1

    cur = t_start
    pbar = tqdm(total=total_windows, desc="janelas", unit="window")

    while cur <= t_end:
        win_df = slice_df(price_df, cur - window, cur)

        if len(win_df) < 300:
            cur += step
            pbar.update(1)
            continue

        # avalia todas as variantes
        scores = {
            name: await evaluate_strategy(win_df, params, metric)
            for name, params in STRATEGIES.items()
        }
        best = max(scores, key=scores.get)

        rows.append({
            "row_id": uuid.uuid4().hex,
            "ts_end": cur,
            **compute_features(win_df),
            "best_strategy": best,
        })
        print(f"✓ janela até {cur:%Y-%m-%d} — melhor: {best:<10} "
              f"({len(rows):3d} amostras)")

        cur += step
        pbar.update(1)

    pbar.close()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nDataset salvo em {out_csv} ({len(rows)} linhas)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gera dataset para StrategySelector (SMA‑crossover)")
    p.add_argument("data_csv", help="CSV de preços (timestamp, close)")
    p.add_argument("--out", default="datasets/selector_training.csv")
    p.add_argument("--window", type=int, default=5, help="dias")
    p.add_argument("--step",   type=int, default=2, help="dias")
    p.add_argument("--metric",
                   choices=["sharpe_ratio", "total_return"],
                   default="sharpe_ratio")
    # 🔽 NOVAS FLAGS
    p.add_argument("--fast", default="", help="ex.: 5,10,15,20")
    p.add_argument("--slow", default="", help="ex.: 30,60,120,200")
    return p.parse_args()

# ------------- GERAR LISTA DE ESTRATÉGIAS --------------
def build_param_grid(args: argparse.Namespace) -> dict[str, dict]:
    """Retorna dict {nome: {'fast': X, 'slow': Y}}"""
    # se usuário passar --fast/--slow = gera grade, senão defaults
    if args.fast and args.slow:
        fasts = [int(x) for x in args.fast.split(",")]
        slows = [int(x) for x in args.slow.split(",")]
        combos = ((f, s) for f, s in product(fasts, slows) if f < s)
    else:  # defaults
        combos = [(10, 30), (20, 50), (30, 120), (50, 200)]

    out: dict[str, dict] = {}
    for f, s in combos:
        out[f"sma_{f}_{s}"] = {"fast": f, "slow": s}
    return out


def main() -> None:
    args = parse_args()
    asyncio.run(build_dataset(args.data_csv, args.out,
                              args.window, args.step, args.metric))


if __name__ == "__main__":
    main()
