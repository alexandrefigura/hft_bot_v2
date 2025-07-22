"""Logging estruturado e logger de trades."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

# --------------------------------------------------------------------------- #
#  Logger estruturado
# --------------------------------------------------------------------------- #
class StructuredLogger(logging.LoggerAdapter):
    """Encapsula logger padrão e adiciona campos extras em cada mensagem."""

    def __init__(self, name: str, extra: Dict[str, Any] | None = None):
        super().__init__(logging.getLogger(name), extra or {})
        # Formato simples JSON‑line
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = [handler]

    def process(self, msg, kwargs):
        record = {
            "ts": datetime.utcnow().isoformat(),
            "msg": msg,
            **self.extra,
        }
        return json.dumps(record, default=str), kwargs


# --------------------------------------------------------------------------- #
#  Logger de trades (grava um JSONL por dia)
# --------------------------------------------------------------------------- #
class TradingLogger:
    """Grava operações abertas/fechadas em arquivos JSONL."""

    def __init__(self, logger: StructuredLogger, base_dir: str | Path = "logs/trades"):
        self.logger = logger
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _file(self) -> Path:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return self.base_path / f"{today}.jsonl"

    async def log_trade_opened(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        signal_strength: float,
        signal_confidence: float,
        reason: str,
    ):
        await self._append(
            {
                "event": "OPEN",
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "qty": quantity,
                "price": price,
                "strength": signal_strength,
                "confidence": signal_confidence,
                "reason": reason,
                "ts": datetime.utcnow().isoformat(),
            }
        )

    async def log_trade_closed(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        reason: str,
    ):
        await self._append(
            {
                "event": "CLOSE",
                "trade_id": trade_id,
                "exit_price": exit_price,
                "pnl": pnl,
                "reason": reason,
                "ts": datetime.utcnow().isoformat(),
            }
        )

    async def _append(self, obj: Dict[str, Any]):
        path = self._file()
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(obj, default=str) + "
")
        # também manda para o logger de console
        self.logger.info(obj),
