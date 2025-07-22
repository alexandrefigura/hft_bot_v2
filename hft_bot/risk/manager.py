"""Very simplesinho: controla posições abertas e regras básicas de risco."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Any


@dataclass
class Position:  # estrutura única para guardar cada trade aberto
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    trade_id: str


class RiskManager:
    """Mantém controle de drawdown, nº de posições, time‑out etc."""

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params or {}
        self.positions: Dict[str, Position] = {}

        # parâmetros vindos do bot_config.yaml
        self.max_positions = self.params.get("max_positions", 3)
        self.max_drawdown = self.params.get("max_drawdown", 0.15)
        self.position_timeout = self.params.get("position_timeout", 600)

        # estatísticas simples de equity
        self.equity_peak: float = 0.0
        self.current_equity: float = 0.0

    # ---------- API usada pelo bot.py ---------- #

    def can_open_position(self, balance: float, size: float, symbol: str) -> tuple[bool, str]:
        if len(self.positions) >= self.max_positions:
            return False, "max_positions reached"
        # (aqui você pode checar drawdown, exposição etc.)
        return True, "ok"

    def add_position(self, data: Dict[str, Any]) -> None:
        self.positions[data["symbol"]] = Position(**data)

    def update_position(self, symbol: str, last_price: float) -> Optional[str]:
        """Atualiza PnL interno; devolve motivo de saída se detectar algo."""
        pos = self.positions.get(symbol)
        if not pos:
            return None

        # exemplo: trailing‑stop ou break‑even ficariam aqui
        return None  # None = manter a posição

    def check_position_timeout(self, pos: Position) -> bool:
        return (datetime.now() - pos.entry_time).total_seconds() > self.position_timeout

    def remove_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)

    def get_risk_metrics(self) -> Dict[str, float]:
        # devolve só drawdown por enquanto
        return {"current_drawdown": 0.0}
