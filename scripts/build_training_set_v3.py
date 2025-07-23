#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera dataset supervisionado (features → melhor_estratégia) para o seletor de estratégias
ML do HFT Bot. Principais recursos:

• Janela deslizante configurável   (--window N dias, --step M dias)
• Métrica-alvo selecionável        (--metric sharpe|sortino|calmar|total_return)
• Estratégias lidas de CSV         (--param-file)
• Features técnicas em YAML        (--features-config)
• Suporte a diferentes timeframes  (--timeframe 1min|5min|...)
• Logging detalhado, progress bar  (tqdm) e tratamento robusto de erros
• Opcional: salva scores de TODAS as estratégias (--save-all-scores)
• Remove features de baixa variância (--min-variance)

Usage (exemplo Windows CMD):

python scripts\build_training_set_v3.py data\BTCUSDT_1m.csv ^
       --window 14 --step 3 --metric sortino ^
       --param-file params_ma.csv ^
       --features-config features.yaml ^
       --timeframe 1min ^
       --save-all-scores ^
       --out datasets\selector_1m_sor.csv
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
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

# Importa motor de backtest do seu projeto
from hft_bot.backtesting.engine import BacktestEngine

# ---------------------------------------------------------------------------
# Alias de métricas devolvidas pelo BacktestEngine
# ---------------------------------------------------------------------------
METRIC_ALIAS = {
    "sharpe": "sharpe_ratio",
    "sortino": "sharpe_ratio",  # Temporariamente usa sharpe_ratio
    "calmar": "sharpe_ratio",   # Temporariamente usa sharpe_ratio
    "total_return": "total_return",
    "max_drawdown": "max_drawdown",
    "win_rate": "win_rate",
}

# Mapeamento reverso para exibição
METRIC_DISPLAY = {
    "sharpe_ratio": "sharpe",
    "total_return": "total_return",
    "max_drawdown": "max_drawdown",
    "win_rate": "win_rate",
}

# ---------------------------------------------------------------------------
# Configuração de logging global
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass para configuração de features via YAML
# ---------------------------------------------------------------------------


@dataclass
class FeatureConfig:
    """Configuração das features calculadas de cada janela."""

    returns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    volatility: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    momentum: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trend: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    volume: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "FeatureConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def default(cls) -> "FeatureConfig":
        """Configuração default - usada caso não seja passado --features-config"""
        return cls(
            returns={
                "1h": {"periods": 60, "enabled": True},
                "1d": {"periods": 1440, "enabled": True},
            },
            volatility={
                "atr": {"length": 14, "enabled": True},
                "parkinson": {"length": 20, "enabled": True},
                "std_20": {"length": 20, "enabled": True},
            },
            momentum={
                "rsi": {"length": 14, "enabled": True},
                "roc": {"length": 5, "enabled": True},
                "macd": {"fast": 12, "slow": 26, "signal": 9, "enabled": True},
            },
            trend={
                "sma_50": {"length": 50, "enabled": True},
                "sma_200": {"length": 200, "enabled": True},
                "ema_20": {"length": 20, "enabled": True},
                "linreg_slope": {"length": 50, "enabled": True},
            },
            volume={
                "obv": {"enabled": True},
                "vwap_dev": {"enabled": True},
                "volume_sma": {"length": 20, "enabled": True},
            },
        )


# ---------------------------------------------------------------------------
# Utilidades de timeframe
# ---------------------------------------------------------------------------
TIMEFRAME_MINUTES = {
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
    """Converte minutos (períodos) para número de candles daquele timeframe."""
    return max(1, periods // TIMEFRAME_MINUTES.get(timeframe, 1))


# ---------------------------------------------------------------------------
# Funções auxiliares de cálculo de indicadores
# ---------------------------------------------------------------------------

def safe_indicator(func, *args, default=np.nan, **kwargs):
    """Wrapper que captura exceções e retorna default (np.nan)."""
    try:
        res = func(*args, **kwargs)
        if isinstance(res, (pd.Series, pd.DataFrame)):
            return res.iloc[-1] if not res.empty else default
        return res
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Erro indic %s: %s", func.__name__, e)
        return default


# ---------------------------------------------------------------------------
# Cálculo de features para uma janela
# ---------------------------------------------------------------------------

def make_features(df: pd.DataFrame, cfg: FeatureConfig, timeframe: str) -> Dict[str, float]:
    closes, highs, lows, vols = df["close"], df["high"], df["low"], df["volume"]

    # Verifica se há mínimo de barras
    if len(df) < 200:
        logger.debug("Menos de 200 candles, pulando janela")
        return {}

    feats: Dict[str, float] = {}

    # --- returns
    pct = closes.pct_change().dropna()
    for name, p in cfg.returns.items():
        if not p.get("enabled", True):
            continue
        candles = periods_to_candles(p["periods"], timeframe)
        feats[f"ret_{name}"] = pct[-candles:].sum() if len(pct) >= candles else pct.sum()

    # --- volatility
    for name, p in cfg.volatility.items():
        if not p.get("enabled", True):
            continue
        length = p.get("length", 14)
        if name == "atr":
            feats[f"atr_{length}"] = safe_indicator(ta.atr, highs, lows, closes, length=length)
        elif name == "parkinson":
            feats[f"parkinson_{length}"] = safe_indicator(ta.natr, highs, lows, closes, length=length)
        elif name == "std_20":
            feats[f"std_{length}"] = closes.rolling(length).std().iloc[-1]

    # --- momentum
    for name, p in cfg.momentum.items():
        if not p.get("enabled", True):
            continue
        if name == "rsi":
            feats[f"rsi_{p['length']}"] = safe_indicator(ta.rsi, closes, length=p["length"])
        elif name == "roc":
            feats[f"roc_{p['length']}"] = safe_indicator(ta.roc, closes, length=p["length"])
        elif name == "macd":
            macd_df = safe_indicator(
                ta.macd,
                closes,
                fast=p["fast"],
                slow=p["slow"],
                signal=p["signal"],
            )
            if isinstance(macd_df, pd.DataFrame) and not macd_df.empty:
                feats["macd_hist"] = macd_df[f"MACDh_{p['fast']}_{p['slow']}_{p['signal']}"]

    # --- trend
    for name, p in cfg.trend.items():
        if not p.get("enabled", True):
            continue
        if name.startswith("sma"):
            length = p["length"]
            sma = closes.rolling(length).mean()
            feats[f"{name}_rel"] = closes.iloc[-1] / sma.iloc[-1]
        elif name.startswith("ema"):
            length = p["length"]
            ema = closes.ewm(span=length).mean()
            feats[f"{name}_rel"] = closes.iloc[-1] / ema.iloc[-1]
        elif name == "linreg_slope":
            feats["linreg_slope"] = safe_indicator(ta.linreg, closes, length=p["length"], angle=True)

    # --- volume
    for name, p in cfg.volume.items():
        if not p.get("enabled", True):
            continue
        if name == "obv":
            # série completa de OBV
            obv_series = ta.obv(closes, vols)
            if not obv_series.empty:
                obv_last = obv_series.iloc[-1]
                obv_mean = obv_series.rolling(20).mean().iloc[-1]
                feats["obv_norm"] = obv_last / obv_mean if obv_mean else 1
        elif name == "vwap_dev":
            vwap = safe_indicator(ta.vwap, highs, lows, closes, vols)
            feats["vwap_dev"] = closes.iloc[-1] / vwap if vwap not in (np.nan, 0) else np.nan
        elif name == "volume_sma":
            length = p["length"]
            vol_sma = vols.rolling(length).mean().iloc[-1]
            feats["volume_ratio"] = vols.iloc[-1] / vol_sma if vol_sma else np.nan

    # remove NaN
    return {k: v for k, v in feats.items() if not pd.isna(v)}


# ---------------------------------------------------------------------------
# Execução de uma estratégia no motor de backtest (async)
# ---------------------------------------------------------------------------


async def run_strategy(df: pd.DataFrame,
                       params: Dict[str, Any],
                       metric_key: str,
                       timeout: float = 30.0) -> float:
    """
    Executa backtest com a estratégia e retorna a métrica solicitada.
    
    Args:
        df: DataFrame com candles
        params: Parâmetros da estratégia
        metric_key: Chave da métrica no resultado do backtest (já convertida)
        timeout: Tempo máximo de execução
    
    Returns:
        Valor da métrica ou -np.inf em caso de erro
    """
    df = (
        df.reset_index()
          .rename(columns={'index': 'timestamp'})
          .assign(timestamp=lambda d: d['timestamp'].dt.tz_localize(None))
    )
    
    try:
        result = await asyncio.wait_for(
            BacktestEngine({"strategy_params": params}).run(df),
            timeout=timeout
        )
        value = result.get(metric_key, -np.inf)
        
        # Log de debug para verificar métricas
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Backtest result keys: %s", list(result.keys()))
            logger.debug("Metric '%s' = %s", metric_key, value)
            
        return value
    except asyncio.TimeoutError:
        logger.warning("Timeout no backtest (params=%s)", params)
        return -np.inf
    except Exception as exc:
        logger.error("Erro backtest: %s", exc)
        return -np.inf


# ---------------------------------------------------------------------------
# Builder principal - percorre janelas, gera dataset
# ---------------------------------------------------------------------------


async def build_dataset(
    csv_in: str,
    csv_out: str,
    window: int,
    step: int,
    metric: str,
    param_file: str,
    features_config_path: Optional[str] = None,
    timeframe: str = "1min",
    save_all_scores: bool = False,
    min_variance: float = 0.01,
):
    """
    Constrói dataset de treino para seletor de estratégias.
    
    Args:
        csv_in: Arquivo CSV com candles
        csv_out: Arquivo CSV de saída
        window: Tamanho da janela em dias
        step: Passo da janela em dias
        metric: Métrica para seleção (sharpe|sortino|calmar|total_return)
        param_file: CSV com parâmetros das estratégias
        features_config_path: YAML com config de features (opcional)
        timeframe: Timeframe dos candles
        save_all_scores: Se True, salva scores de todas as estratégias
        min_variance: Threshold mínimo de variância para features
    """
    # Converte alias da métrica para chave real do backtest
    metric_key = METRIC_ALIAS.get(metric, metric)
    metric_display = METRIC_DISPLAY.get(metric_key, metric)
    
    logger.info("Métrica selecionada: %s (chave interna: %s)", metric, metric_key)
    
    # Aviso se sortino ou calmar foram solicitados
    if metric in ("sortino", "calmar"):
        logger.warning("ATENÇÃO: %s não disponível no BacktestEngine, usando sharpe_ratio", metric)
    
    price = pd.read_csv(csv_in, parse_dates=["timestamp"], index_col="timestamp")
    logger.info("Candles: %d (%s -> %s)", len(price), price.index.min(), price.index.max())

    # carrega config de features
    cfg = (
        FeatureConfig.from_yaml(features_config_path)
        if features_config_path and Path(features_config_path).exists()
        else FeatureConfig.default()
    )

    # carrega estratégias
    params_df = pd.read_csv(param_file)
    strategies = {row["name"]: dict(row.drop("name")) for _, row in params_df.iterrows()}
    logger.info("Estratégias carregadas: %d", len(strategies))

    win_delta = timedelta(days=window)
    step_delta = timedelta(days=step)

    cur = price.index.min() + win_delta
    end = price.index.max()

    rows: List[Dict[str, Any]] = []
    total = int((end - cur) / step_delta) + 1
    pbar = tqdm(total=total, desc="Processando janelas")

    while cur <= end:
        win_df = price[cur - win_delta : cur]
        if len(win_df) < 300:
            cur += step_delta
            pbar.update(1)
            continue

        # calcula features
        feats = make_features(win_df, cfg, timeframe)
        if not feats:
            cur += step_delta
            pbar.update(1)
            continue

        # executa estratégias concorrentemente
        scores = dict(
            zip(
                strategies.keys(),
                await asyncio.gather(
                    *[run_strategy(win_df.copy(), p, metric_key) for p in strategies.values()]
                ),
            )
        )
        
        # Filtra resultados válidos
        valid = {k: v for k, v in scores.items() if v != -np.inf and not np.isnan(v)}
        if not valid:
            logger.debug("Nenhuma estratégia válida para janela %s", cur)
            cur += step_delta
            pbar.update(1)
            continue
            
        # Para max_drawdown, queremos o valor MENOS negativo (mais próximo de zero)
        if metric == "max_drawdown":
            best = max(valid, key=valid.get)  # max_drawdown é negativo, então max pega o menos negativo
        else:
            best = max(valid, key=valid.get)  # Para outras métricas, queremos o maior valor

        # Monta linha do dataset
        row = {
            "row_id": uuid.uuid4().hex, 
            "ts_end": cur, 
            **feats, 
            "best_strategy": best, 
            f"best_{metric_display}": valid[best]
        }
        
        if save_all_scores:
            row.update({f"{metric_display}_{k}": v for k, v in valid.items()})
            
        rows.append(row)

        if len(rows) % 50 == 0:
            logger.info("%d janelas -> %s | best=%s %.3f", len(rows), cur.date(), best, valid[best])
            
        cur += step_delta
        pbar.update(1)

    pbar.close()

    if not rows:
        logger.error("Nenhuma janela válida encontrada!")
        return

    df = pd.DataFrame(rows)
    logger.info("Dataset bruto: %d linhas, %d colunas", *df.shape)

    # Remove features de baixa variância
    if min_variance > 0 and len(df) > 1:
        feats_cols = [c for c in df.columns if c not in ("row_id", "ts_end", "best_strategy") and not c.startswith("best_")]
        if feats_cols:
            scaler = StandardScaler()
            var = np.var(scaler.fit_transform(df[feats_cols]), axis=0)
            low_var = [c for c, v in zip(feats_cols, var) if v < min_variance]
            if low_var:
                logger.info("Removendo %d features baixa variância: %s", len(low_var), low_var[:5])
                df = df.drop(columns=low_var)

    # Salva dataset
    df.to_csv(csv_out, index=False, encoding='utf-8')
    logger.info("Dataset salvo: %s (%d linhas)", csv_out, len(df))

    # Resumo de classes
    logger.info("\nDistribuição de estratégias:")
    for strat, cnt in df["best_strategy"].value_counts().items():
        logger.info("  %s: %d (%.1f%%)", strat, cnt, cnt / len(df) * 100)


# ---------------------------------------------------------------------------
# Helpers CLI
# ---------------------------------------------------------------------------

def create_default_features_config(path: str):
    """Cria arquivo YAML de exemplo com configuração de features."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            json.loads(json.dumps(FeatureConfig.default(), default=lambda o: o.__dict__)), 
            f, 
            sort_keys=False,
            allow_unicode=True
        )
    logger.info("Arquivo de exemplo salvo em %s", path)


# ---------------------------------------------------------------------------
# Entry-point CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Constrói dataset para o seletor de estratégias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Primeiro verifica se é só para criar config
    if len(sys.argv) > 1 and sys.argv[1] == "--create-features-config":
        if len(sys.argv) > 2:
            create_default_features_config(sys.argv[2])
        else:
            print("Uso: python build_training_set_v3.py --create-features-config <arquivo.yaml>")
        raise SystemExit(0)

    parser.add_argument("csv_in", help="CSV de candles com coluna timestamp")
    parser.add_argument("--out", required=True, help="CSV de saída do dataset")
    parser.add_argument("--window", type=int, default=14, help="Janela em dias (default=14)")
    parser.add_argument("--step", type=int, default=3, help="Passo em dias (default=3)")
    parser.add_argument(
        "--metric",
        choices=("sharpe", "total_return", "max_drawdown", "win_rate"),
        default="sharpe",
        help="Métrica-alvo para escolher estratégia (default=sharpe)",
    )
    parser.add_argument("--param-file", required=True, help="CSV com parâmetros das estratégias")
    parser.add_argument("--features-config", help="YAML com configuração de features")
    parser.add_argument("--create-features-config", help="Cria YAML de features de exemplo e sai")
    parser.add_argument(
        "--timeframe",
        default="1min",
        choices=list(TIMEFRAME_MINUTES.keys()),
        help="Timeframe dos candles (default=1min)",
    )
    parser.add_argument("--save-all-scores", action="store_true", help="Salva scores de todas as estratégias")
    parser.add_argument("--min-variance", type=float, default=0.01, help="Threshold de variância (default=0.01)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível global de log (default=INFO)",
    )

    args = parser.parse_args()

    # Configura logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.create_features_config:
        create_default_features_config(args.create_features_config)
        raise SystemExit(0)

    # Executa builder
    asyncio.run(
        build_dataset(
            csv_in=args.csv_in,
            csv_out=args.out,
            window=args.window,
            step=args.step,
            metric=args.metric,
            param_file=args.param_file,
            features_config_path=args.features_config,
            timeframe=args.timeframe,
            save_all_scores=args.save_all_scores,
            min_variance=args.min_variance,
        )
    )