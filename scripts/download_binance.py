#!/usr/bin/env python3
"""
download_binance.py – dumpa velas (klines) da Binance para CSV, com logs e barra
de progresso.

Compatível com python‑binance 1.x (≥ 1.0.17).  Não requer credenciais para dados
históricos; se você passar --testnet ele usará a rede de testes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import pandas as pd
from binance import AsyncClient
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuração de logging
# --------------------------------------------------------------------------- #
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(format=LOG_FMT, datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Intervalos aceitos
# --------------------------------------------------------------------------- #
INTERVAL_MAP = {
    "1m": AsyncClient.KLINE_INTERVAL_1MINUTE,
    "3m": AsyncClient.KLINE_INTERVAL_3MINUTE,
    "5m": AsyncClient.KLINE_INTERVAL_5MINUTE,
    "15m": AsyncClient.KLINE_INTERVAL_15MINUTE,
    "30m": AsyncClient.KLINE_INTERVAL_30MINUTE,
    "1h": AsyncClient.KLINE_INTERVAL_1HOUR,
    "2h": AsyncClient.KLINE_INTERVAL_2HOUR,
    "4h": AsyncClient.KLINE_INTERVAL_4HOUR,
    "6h": AsyncClient.KLINE_INTERVAL_6HOUR,
    "8h": AsyncClient.KLINE_INTERVAL_8HOUR,
    "12h": AsyncClient.KLINE_INTERVAL_12HOUR,
    "1d": AsyncClient.KLINE_INTERVAL_1DAY,
}


def interval_to_binance(interval: str) -> str:
    try:
        return INTERVAL_MAP[interval]
    except KeyError as exc:
        raise ValueError(f"Intervalo não suportado: {interval}") from exc


# --------------------------------------------------------------------------- #
#  Função principal de download
# --------------------------------------------------------------------------- #
async def download_klines(
    symbol: str,
    interval_str: str,
    start_date: datetime,
    end_date: datetime | None,
    out_path: Path,
    *,
    testnet: bool = False,
) -> None:
    """Baixa todas as velas disponíveis e grava em ``out_path``."""

    client = await AsyncClient.create(testnet=testnet)
    interval = interval_to_binance(interval_str)

    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000) if end_date else None

    logger.info(
        "Iniciando download: %s %s de %s até %s",
        symbol,
        interval_str,
        start_date.date(),
        end_date.date() if end_date else "agora",
    )

    all_klines: list[list[Any]] = []
    pbar = tqdm(total=0, unit="kline", unit_scale=True, dynamic_ncols=True)

    try:
        while True:
            klines: List[list] = await client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ms,
                endTime=end_ms,
                limit=1000,
            )

            if not klines:  # terminou
                break

            all_klines.extend(klines)
            pbar.update(len(klines))

            # próximo lote começa 1 ms após a última vela obtida
            start_ms = klines[-1][0] + 1

            # Evita rate limit agressivo
            await asyncio.sleep(0.2)

    finally:
        pbar.close()
        await client.close_connection()

    # Se nada baixado, avisa e sai
    if not all_klines:
        logger.warning("Nenhum candle retornado – verifique datas e símbolo.")
        return

    header = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(all_klines, columns=header)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.to_csv(out_path, index=False)

    logger.info("✅ Salvo em %s (%d linhas)", out_path, len(df))


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ivals = list(INTERVAL_MAP.keys())
    parser = argparse.ArgumentParser(description="Baixa klines e grava em CSV")
    parser.add_argument("symbol", help="Par de moedas, ex.: BTCUSDT")
    parser.add_argument("interval", choices=ivals, help="Intervalo de velas")
    parser.add_argument("--start", required=True, help="AAAA-MM-DD data inicial")
    parser.add_argument("--end", help="AAAA-MM-DD data final (padrão = hoje)")
    parser.add_argument(
        "--out",
        help="Arquivo CSV de saída (default: ./data/<SYMBOL>_<INTERVAL>.csv)",
    )
    parser.add_argument("--testnet", action="store_true", help="Usa a testnet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # datas
    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemExit(f"Data --start inválida: {args.start}") from exc

    if args.end:
        try:
            end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise SystemExit(f"Data --end inválida: {args.end}") from exc
    else:
        end_dt = None

    # saída
    default_name = f"{args.symbol.upper()}_{args.interval}.csv"
    out_path = Path(args.out) if args.out else Path("data") / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # executa
    asyncio.run(
        download_klines(
            symbol=args.symbol.upper(),
            interval_str=args.interval,
            start_date=start_dt,
            end_date=end_dt,
            out_path=out_path,
            testnet=args.testnet,
        )
    )


if __name__ == "__main__":
    main()
