# hft_bot/strategies/simple.py
"""Estratégias‑stub só para o bot rodar."""

import random
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Signal:
    direction: str         # BUY, SELL, HOLD
    strength: float        # 0–1
    confidence: float      # 0–1
    indicators: Dict[str, float]
    reason: str = ""

class BaseStrategy:
    def __init__(self, _params: Dict[str, Any] | None = None) -> None:
        self.params = _params or {}

    async def analyze(self, _market: Dict[str, Any]) -> Signal:  # noqa: D401
        raise NotImplementedError

class MeanReversionAdaptive(BaseStrategy):
    async def analyze(self, _market: Dict[str, Any]) -> Signal:
        # sai BUY/SELL/HOLD aleatório só para testes
        direction = random.choice(["BUY", "SELL", "HOLD"])
        return Signal(
            direction=direction,
            strength=random.uniform(0.4, 0.9),
            confidence=random.uniform(0.5, 0.9),
            indicators={"dummy": 0.0},
            reason="stub‑mean‑reversion",
        )

class MomentumStrategy(BaseStrategy):
    async def analyze(self, _market: Dict[str, Any]) -> Signal:
        direction = random.choice(["BUY", "SELL", "HOLD"])
        return Signal(
            direction=direction,
            strength=random.uniform(0.4, 0.9),
            confidence=random.uniform(0.5, 0.9),
            indicators={"dummy": 0.0},
            reason="stub‑momentum",
        )
