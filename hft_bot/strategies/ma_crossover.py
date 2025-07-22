# hft_bot/strategies/ma_crossover.py
from hft_bot.strategies import BaseStrategy, Signal
import numpy as np

class MovingAverageCrossover(BaseStrategy):
    async def analyze(self, market_data):
        prices = market_data["prices"]
        fast = int(self.params.get("fast", 50))
        slow = int(self.params.get("slow", 200))
        if len(prices) < slow:
            return Signal("HOLD", 0, 0, {}, "histórico insuficiente")

        sma_fast = np.mean(prices[-fast:])
        sma_slow = np.mean(prices[-slow:])
        if sma_fast > sma_slow:
            return Signal("BUY", 1.0, 0.6, {}, "sma crossover up")
        elif sma_fast < sma_slow:
            return Signal("SELL", 1.0, 0.6, {}, "sma crossover down")
        return Signal("HOLD", 0, 0.5, {}, "sem sinal")
