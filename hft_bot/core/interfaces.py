"""Interfaces (ABCs) que definem os contratos básicos do bot.

Você pode estender ou trocar as implementações concretas (exchanges,
feeds de dados, persistência etc.) mantendo a mesma assinatura.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class ExchangeInterface(abc.ABC):
    """Contrato para qualquer exchange (Binance, Paper Trade etc.)."""

    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def disconnect(self) -> None: ...

    @abc.abstractmethod
    async def buy(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "LIMIT",
    ) -> Dict[str, Any]: ...

    @abc.abstractmethod
    async def sell(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "LIMIT",
    ) -> Dict[str, Any]: ...

    @abc.abstractmethod
    async def get_balance(self, asset: str) -> float: ...

    @abc.abstractmethod
    async def cancel_order(self, symbol: str, order_id: int) -> bool: ...

    @abc.abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]: ...

    @abc.abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, float]: ...

    @abc.abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 10) -> Dict[str, Any]: ...


class DataFeedInterface(abc.ABC):
    """(Reservado) Stream/consulta de dados de mercado/históricos."""

    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def disconnect(self) -> None: ...

    @abc.abstractmethod
    async def subscribe(self, symbol: str) -> None: ...

    @abc.abstractmethod
    async def get_latest(self, symbol: str) -> Dict[str, Any]: ...


class PersistenceInterface(abc.ABC):
    """Armazenamento/recuperação de estado, logs, snapshots."""

    @abc.abstractmethod
    async def start_sync(self) -> None: ...

    @abc.abstractmethod
    async def stop_sync(self) -> None: ...

    @abc.abstractmethod
    async def save_state(self, key: str, data: Dict[str, Any]) -> None: ...

    @abc.abstractmethod
    async def load_state(self, key: str) -> Optional[Dict[str, Any]]: ...


class AlertingInterface(abc.ABC):
    """Envio de alertas (e‑mail, Slack, Telegram…)."""

    @abc.abstractmethod
    async def send_alert(self, message: str, severity: str = "info") -> None: ...
