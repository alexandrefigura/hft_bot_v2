#!/usr/bin/env python3
"""
Dataset Builder v4 – geração de amostras supervisionadas para o *Strategy Selector*
=================================================================================

Principais melhorias
-------------------
1. **Estratégias parametrizadas por CSV** – passe quantas quiser via ``--param-file``
2. **Features dirigidas por YAML**        – ligue/desligue indicadores sem tocar no código
3. **Compatível com vários time‑frames** – defina ``--timeframe 1min|5min|...``
4. **Métricas Sharpe ou Retorno Total**  – escolha em ``--metric``
5. **Remoção automática de baixa variância** – evita lixo no dataset

Exemplo de uso (CMD/PowerShell)::

    python scripts\build_training_set_v4.py data\BTCUSDT_1m.csv \
           --window 10 --step 2 --timeframe 1min --metric sharpe_ratio \
           --param-file params_ma.csv \
           --features-config features.yaml \
           --out datasets\selector_1m_v4.csv

Arquivos auxiliares
~~~~~~~~~~~~~~~~~~~
``params_ma.csv`` (exemplo)::

    name,fast,slow
    ma_20_50,20,50
    ma_30_120,30,120
    ma_50_200,50,200

``features.yaml`` (exemplo completo encontra‑se em ``--create-features-config``)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from hft_bot.backtesting.engine import BacktestEngine

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Time‑frame helpers
# ----------------------------------------------------------------------------
TF_MIN = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1440,
}

def periods_to_candles(periods: int, timeframe: str) -> int:
    return max(1, periods // TF_MIN[timeframe])

# ----------------------------------------------------------------------------
# Feature configuration via YAML
# ----------------------------------------------------------------------------
@dataclass
class FeatureBlock:
    params: Dict[str, Any]
    enabled: bool = True

@dataclass
class FeatureConfig:
    returns: Dict[str, FeatureBlock] = field(default_factory=dict)
    volatility: Dict[str, FeatureBlock] = field(default_factory=dict)
    momentum: Dict[str, FeatureBlock] = field(default_factory=dict)
    trend: Dict[str, FeatureBlock] = field(default_factory=dict)
    volume: Dict[str, FeatureBlock] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "FeatureConfig":
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            return cls(**{k: {kk: FeatureBlock(**vv) for kk, vv in v.items()} for k, v in data.items()})
        log.warning("Arquivo de configuração não encontrado – usando default simplificado")
        # fallback mínimo
        base = {
            "returns": {"1h": FeatureBlock({"periods": 60})},
            "volatility": {"atr": FeatureBlock({"length": 14})},
            "momentum": {"rsi": FeatureBlock({"length": 14})},
            "trend": {"sma_50": FeatureBlock({"length": 50}), "sma_200": FeatureBlock({"length": 200})},
        }
        return cls(**base)

# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------

def _safe(fn, *args, default=np.nan, **kwargs):
    try:
        out = fn(*args, **kwargs)
        if isinstance(out, (pd.Series, pd.DataFrame)):
            return out.iloc[-1]
        return out
    except Exception as e:
        log.debug("Indicator error: %s", e)
        return default


def make_features(df: pd.DataFrame, cfg: FeatureConfig, timeframe: str) -> Dict[str, float]:
    closing = df["close"]
    high, low, vol = df["high"], df["low"], df["volume"]
    feats: Dict[str, float] = {}

    # returns
    for name, blk in cfg.returns.items():
        if not blk.enabled:
            continue
        pcs = closing.pct_change().dropna()
        n = periods_to_candles(blk.params["periods"], timeframe)
        feats[f"ret_{name}"] = float(pcs[-n:].sum()) if len(pcs) >= n else float(pcs.sum())

    # volatility
    for name, blk in cfg.volatility.items():
        if not blk.enabled:
            continue
        if name == "atr":
            feats[f"atr_{blk.params['length']}"] = _safe(ta.atr, high, low, closing, length=blk.params["length"])
        elif name.startswith("std"):
            l = blk.params["length"]
            feats[f"std_{l}"] = float(closing.rolling(l).std().iloc[-1])

    # momentum
    for name, blk in cfg.momentum.items():
        if not blk.enabled:
            continue
        if name == "rsi":
            feats[f"rsi_{blk.params['length']}"] = _safe(ta.rsi, closing, length=blk.params["length"])
        elif name == "roc":
            feats[f"roc_{blk.params['length']}"] = _safe(ta.roc, closing, length=blk.params["length"])

    # trend
    for name, blk in cfg.trend.items():
        if not blk.enabled:
            continue
        if name.startswith("sma"):
            l = blk.params["length"]
            sma = closing.rolling(l).mean().iloc[-1]
            feats[f"{name}_rel"] = float(closing.iloc[-1] / sma) if sma != 0 else 1.0

    # volume
    for name, blk in cfg.volume.items():
        if not blk.enabled:
            continue
        if name == "obv":
            obv = _safe(ta.obv, closing, vol)
            obv_mean = _safe(ta.obv, closing, vol).rolling(20).mean()
            obv_mean = obv_mean.iloc[-1] if not obv_mean.empty else np.nan
            feats["obv_norm"] = float(obv / obv_mean) if obv_mean not in (0, np.nan) else 1.0

    # drop NaNs
    return {k: v for k, v in feats.items() if not pd.isna(v)}

# ----------------------------------------------------------------------------
# Strategy evaluation
# ----------------------------------------------------------------------------
async def backtest(df: pd.DataFrame, params: Dict[str, Any], metric: str) -> float:
    try:
        res = await BacktestEngine({"strategy_params": params}).run(df)
        return res.get(metric, -np.inf)
    except Exception as e:
        log.debug("Backtest error: %s", e)
        return -np.inf

# ----------------------------------------------------------------------------
# Core builder
# ----------------------------------------------------------------------------
async def build_dataset(
    data_csv: str,
    out_csv: str,
    window: int,
    step: int,
    metric: str,
    param_file: str,
    feat_cfg_path: Optional[str],
    timeframe: str,
    min_var: float,
):
    df_price = pd.read_csv(data_csv, parse_dates=["timestamp"], index_col="timestamp")
    log.info("%s candles de %s a %s", len(df_price), df_price.index.min(), df_price.index.max())

    cfg = FeatureConfig.from_yaml(feat_cfg_path)

    # carrega strategies
    df_params = pd.read_csv(param_file)
    strategies = {row["name"]: row.drop("name").to_dict() for _, row in df_params.iterrows()}
    log.info("%d estratégias carregadas", len(strategies))

    rows: List[Dict[str, Any]] = []
    win_delta, step_delta = timedelta(days=window), timedelta(days=step)
    cur = df_price.index.min() + win_delta
    total = int((df_price.index.max() - cur) / step_delta) + 1
    pbar = tqdm(total=total, desc="Janelas")

    while cur <= df_price.index.max():
        win_df = df_price[cur - win_delta : cur]
        if len(win_df) < 300:
            cur += step_delta; pbar.update(1); continue

        feats = make_features(win_df, cfg, timeframe)
        if not feats:
            cur += step_delta; pbar.update(1); continue

        results = {n: s for n, s in zip(strategies, await asyncio.gather(*[backtest(win_df.copy(), p, metric) for p in strategies.values()]))}
        valid = {k: v for k, v in results.items() if v != -np.inf}
        if not valid:
            cur += step_delta; pbar.update(1); continue
        best = max(valid, key=valid.get)

        rows.append({
            "row_id": uuid.uuid4().hex,
            "ts_end": cur,
            **feats,
            "best_strategy": best,
            f"best_{metric}": valid[best],
        })
        if len(rows) % 50 == 0:
            log.info("%d linhas – última janela %s best=%s %.3f", len(rows), cur.date(), best, valid[best])
        cur += step_delta; pbar.update(1)

    pbar.close()
    df_out = pd.DataFrame(rows)

    # remove baixa variância
    if len(df_out) > 2 and min_var > 0:
        feat_cols = [c for c in df_out.columns if c not in ("row_id", "ts_end", "best_strategy") and not c.startswith("best_")]
        std = np.var(StandardScaler().fit_transform(df_out[feat_cols]), axis=0)
        low = [c for c, v in zip(feat_cols, std) if v < min_var]
        if low:
            log.info("Removendo %d features baixa var: %s", len(low), low)
            df_out.drop(columns=low, inplace=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    log.info("Dataset salvo em %s (%d linhas, %d colunas)", out_csv, len(df_out), df_out.shape[1])
    log.info("Distribuição: %s", df_out["best_strategy"].value_counts().to_dict())

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def default_yaml(path: str):
    """Gera YAML de referência."""
    samp = {
        "returns": {
            "1h": {"periods": 60, "enabled": True},
            "1d": {"periods": 1440, "enabled": True},
        },
        "volatility": {"atr": {"length": 14, "enabled": True}},
        "momentum": {"rsi": {"length": 14, "enabled": True}},
        "trend": {"sma_50": {"length": 50, "enabled": True}, "sma_200": {"length": 200, "enabled": True}},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(samp, f, sort_keys=False)
    log.info("Exemplo salvo em %s", path)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser("Dataset Builder v4")
    p.add_argument("csv_in", help="CSV de preços")
    p.add_argument("--out", required=True, help="CSV de saída")
    p.add_argument("--window", type=int, default=10, help="Janela em dias")
    p.add_argument("--step", type=int, default=2, help="Passo em dias")
    p.add_argument("--metric", choices=["sharpe_ratio", "total_return"], default="sharpe_ratio")
    p.add_argument("--param-file", required=True, help="CSV de parâmetros das estratégias")
    p.add_argument("--features-config", help="YAML de features")
    p.add_argument("--create-features-config", help="Cria YAML exemplo e sai")
    p.add_argument("--timeframe", choices=list(TF_MIN), default="1min")
    p.add_argument("--min-variance", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse()
    if args.create_features_config:
        default_yaml(args.create_features_config)
        return
    asyncio.run(
        build_dataset(
            data_csv=args.csv_in,
            out_csv=args.out,
            window=args.window,
            step=args.step,
            metric=args.metric,
            param_file=args.param_file,
            feat_cfg_path=args.features_config,
            timeframe=args.timeframe,
            min_var=args.min_variance,
        )
    )


if __name__ == "__main__":
    main()
