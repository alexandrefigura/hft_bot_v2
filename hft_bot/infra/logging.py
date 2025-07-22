"""Structured logging helpers for the bot."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import structlog
except ImportError:  # fallback caso structlog não esteja disponível
    structlog = None


# -----------------------------------------------------------------------------
# StructuredLogger – usa structlog se existir, senão o logging padrão
# -----------------------------------------------------------------------------
class StructuredLogger:
    """Wrapper simples para criar logs estruturados JSON."""

    def __init__(self, name: str) -> None:
        if structlog:
            structlog.configure(
                wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
                processors=[
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ],
            )
            self._logger = structlog.get_logger(name)
        else:
            # logging padrão como fallback
            self._logger = logging.getLogger(name)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    # Proxy métodos comuns
    def info(self, msg: str, **kwargs: Any) -> None:  # noqa: D401
        self._logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._logger.error(msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._logger.debug(msg, **kwargs)

    def bind(self, **kwargs: Any):  # só disponível em structlog
        if structlog:
            return self._logger.bind(**kwargs)
        return self


# -----------------------------------------------------------------------------
# TradingLogger – grava cada trade em formato JSON‑lines
# -----------------------------------------------------------------------------
class TradingLogger:
    """Mantém um arquivo .jsonl com cada trade aberto/fechado."""

    def __init__(self, base_logger: StructuredLogger, folder: str = "logs/trades") -> None:
        self.log = base_logger
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

        self.open_file = self.folder / "trades_opened.jsonl"
        self.close_file = self.folder / "trades_closed.jsonl"

    # ---------------------- API usada pelo bot ---------------------- #
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
    ) -> None:
        record = {
            "ts": datetime.utcnow().isoformat(),
            "event": "OPEN",
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": quantity,
            "price": price,
            "signal_strength": signal_strength,
            "signal_confidence": signal_confidence,
            "reason": reason,
        }
        self._append(self.open_file, record)
        self.log.info("Trade opened", **record)

    async def log_trade_closed(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        reason: str,
    ) -> None:
        record = {
            "ts": datetime.utcnow().isoformat(),
            "event": "CLOSE",
            "trade_id": trade_id,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
        }
        self._append(self.close_file, record)
        self.log.info("Trade closed", **record)

    # ------------------------- helpers ------------------------- #
    def _append(self, path: Path, obj: Dict[str, Any]) -> None:
        """Acrescenta uma linha JSON ao arquivo sem quebrar encoding."""
        # Abrir sempre em modo texto UTF‑8 explicitamente
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(obj, default=str) + "\n")
