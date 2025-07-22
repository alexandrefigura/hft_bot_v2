"""Coletor de métricas e servidor HTTP (health, metrics, status)"""

import asyncio
import time
from typing import Dict, Any

from aiohttp import web
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# --------------------------------------------------------------------------- #
#  Métricas Prometheus (pode estender depois)
# --------------------------------------------------------------------------- #
DECISION_LATENCY = Histogram(
    "hftbot_decision_latency_seconds",
    "Tempo gasto para tomar decisões de trade",
)
ORDER_LATENCY = Histogram(
    "hftbot_order_latency_seconds",
    "Tempo de execução de ordens",
)
TRADE_COUNTER = Counter(
    "hftbot_trades_total",
    "Quantidade de trades executados",
)
ERROR_COUNTER = Counter(
    "hftbot_errors_total",
    "Quantidade de erros",
    ["type"],
)
USDT_BALANCE = Gauge(
    "hftbot_balance_usdt",
    "Saldo em USDT",
)
DRAW_DOWN = Gauge(
    "hftbot_drawdown_ratio",
    "Drawdown atual (0‑1)",
)


class MetricsCollector:
    """Camada fina para o bot reportar métricas."""

    # Exemplo de wrappers usados em bot.py
    decision_latency = DECISION_LATENCY
    order_latency = ORDER_LATENCY

    def record_decision_time(self, seconds: float) -> None:
        self.decision_latency.observe(seconds)

    def record_order_time(self, seconds: float) -> None:
        self.order_latency.observe(seconds)

    def record_trade(self, **kwargs) -> None:  # kwargs ignorados por enquanto
        TRADE_COUNTER.inc()

    def record_error(self, err_type: str, _msg: str = "") -> None:
        ERROR_COUNTER.labels(type=err_type).inc()

    def update_balance_metrics(self, balances: Dict[str, float]) -> None:
        if "USDT" in balances:
            USDT_BALANCE.set(balances["USDT"])

    def update_risk_metrics(self, risk: Dict[str, Any]) -> None:
        if "current_drawdown" in risk:
            DRAW_DOWN.set(risk["current_drawdown"])

    # Métodos utilitários usados em bot.py
    def get_summary(self) -> Dict[str, Any]:
        return {
            "trades": TRADE_COUNTER._value.get(),
            "errors": ERROR_COUNTER._value.get(),
            "drawdown": DRAW_DOWN._value.get(),
        }


# --------------------------------------------------------------------------- #
#  Servidor HTTP assíncrono
# --------------------------------------------------------------------------- #
class MetricsServer:
    """Servidor leve em aiohttp para /health, /metrics e /status."""

    def __init__(self, port: int = 8080) -> None:
        self.port = port
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/metrics", self._prom_metrics)
        app.router.add_get("/status", self._status)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    # ------------------------- Handlers ------------------------- #
    async def _health(self, _request: web.Request) -> web.Response:
        """Usado pelo Docker HEALTHCHECK."""
        return web.json_response({"status": "ok"})

    async def _prom_metrics(self, _request: web.Request) -> web.Response:
        """Endpoint consumido pelo Prometheus."""
        data = generate_latest()
        return web.Response(body=data, content_type=CONTENT_TYPE_LATEST)

    async def _status(self, _request: web.Request) -> web.Response:
        """Resumo simples para 'hft status'."""
        return web.json_response(
            {
                "status": "running",
                "uptime": int(time.time()),  # placeholder
                "total_trades": TRADE_COUNTER._value.get(),
                "drawdown": DRAW_DOWN._value.get(),
            }
        )
