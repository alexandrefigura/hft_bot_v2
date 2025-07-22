"""HFT Bot - Enterprise Ready High Frequency Trading System"""

__version__ = "2.0.0"
__author__ = "Alexandre Figura"
__email__ = "alexandre_figura@hotmail.com"

# Public API
__all__ = [
    "HFTBot",
    "BotConfig",
    "ExchangeInterface",
    "RiskManager",
    "MarketAnalyzer",
]

from hft_bot.bot import HFTBot
from hft_bot.core.config import BotConfig
from hft_bot.core.interfaces import ExchangeInterface
from hft_bot.risk.manager import RiskManager
from hft_bot.analytics.market import MarketAnalyzer
