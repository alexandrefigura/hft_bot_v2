"""Estratégias básicas e objeto Signal usados pelo HFT Bot."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Any, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Estrutura de saída que o restante do bot espera
# --------------------------------------------------------------------------- #

@dataclass
class Signal:
    direction: str          # 'BUY' | 'SELL' | 'HOLD'
    strength: float         # 0‑1 (quão forte é o sinal)
    confidence: float       # 0‑1 (quão confiável é o sinal)
    indicators: Dict[str, Any]
    reason: str             # texto curto explicando o motivo


# --------------------------------------------------------------------------- #
# Classe‑base — pode servir para estratégias futuras
# --------------------------------------------------------------------------- #

class BaseStrategy:
    """API mínima para uma estratégia assíncrona."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}

    async def analyze(self, market_data: Dict[str, Any]) -> Signal:  # noqa: D401
        """Gerar um sinal de trading. Deve ser sobreposta."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Estratégia 1 – Mean Reversion simples
# --------------------------------------------------------------------------- #

class MeanReversionAdaptive(BaseStrategy):
    """Compra quando o preço está X % abaixo da média; vende quando acima."""

    async def analyze(self, market_data: Dict[str, Any]) -> Signal:
        prices: Sequence[float] = market_data["prices"]
        if len(prices) < 20:
            return Signal("HOLD", 0.0, 0.0, {}, "histórico insuficiente")

        sma_period = int(self.params.get("sma_period", 20))
        threshold = float(self.params.get("deviation", 0.002))  # 0,2 %

        sma = np.mean(prices[-sma_period:])
        price = prices[-1]
        deviation = (price - sma) / sma

        indicators = {"sma": sma, "deviation": deviation}

        if deviation <= -threshold:            # abaixo da média → BUY
            strength = min(abs(deviation) / threshold, 1.0)
            return Signal("BUY", strength, 0.6, indicators, "price below SMA")
        elif deviation >= threshold:           # acima da média → SELL
            strength = min(abs(deviation) / threshold, 1.0)
            return Signal("SELL", strength, 0.6, indicators, "price above SMA")
        else:
            return Signal("HOLD", 0.0, 0.5, indicators, "within band")


# --------------------------------------------------------------------------- #
# Estratégia 2 – Momentum simples
# --------------------------------------------------------------------------- #

class MomentumStrategy(BaseStrategy):
    """Compra se a tendência de curto prazo for positiva, vende se negativa."""

    async def analyze(self, market_data: Dict[str, Any]) -> Signal:
        prices: Sequence[float] = market_data["prices"]
        if len(prices) < 15:
            return Signal("HOLD", 0.0, 0.0, {}, "histórico insuficiente")

        lookback = int(self.params.get("lookback", 10))
        momentum = (prices[-1] - prices[-lookback]) / prices[-lookback]

        indicators = {"momentum": momentum}

        if momentum > 0:
            strength = min(momentum * 50, 1.0)         # escala simples
            confidence = 0.55 + strength * 0.4         # 0.55‑0.95
            return Signal("BUY", strength, confidence, indicators, "positive momentum")
        elif momentum < 0:
            strength = min(abs(momentum) * 50, 1.0)
            confidence = 0.55 + strength * 0.4
            return Signal("SELL", strength, confidence, indicators, "negative momentum")
        else:
            return Signal("HOLD", 0.0, 0.5, indicators, "flat momentum")


# Exporta nomes para o import externo
__all__ = [
    "Signal",
    "BaseStrategy",
    "MeanReversionAdaptive",
    "MomentumStrategy",
    "ma_crossover",
]
