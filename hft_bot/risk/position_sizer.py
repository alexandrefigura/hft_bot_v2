"""Calcula o tamanho ideal da ordem com base em Kelly fracionado."""

from typing import List


class PositionSizer:
    def __init__(self, kelly_fraction: float = 0.25) -> None:
        self.kelly_fraction = kelly_fraction
        self._returns_history: List[float] = []

    # ---------- API usada pelo bot.py ---------- #

    def calculate_position_size(
        self, balance: float, signal_strength: float, volatility: float, confidence: float
    ) -> float:
        risk_frac = self.kelly_fraction * signal_strength * confidence
        risk_frac = max(0.01, min(risk_frac, 0.5))  # entre 1 % e 50 %
        return balance * risk_frac

    def update_history(self, trade_return: float) -> None:
        self._returns_history.append(trade_return)
