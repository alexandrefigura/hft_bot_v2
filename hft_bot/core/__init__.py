"""Core module - interfaces, config, and exceptions"""

__all__ = [
    "BotConfig",
    "ConfigProvider",
    "ExchangeInterface",
    "DataFeedInterface",
    "PersistenceInterface",
    "AlertingInterface",
    "HFTBotException",
    "ConfigurationError",
    "ExchangeError",
    "InsufficientBalanceError",
    "OrderError",
    "RiskLimitError",
    "DataError",
]

from hft_bot.core.config import BotConfig, ConfigProvider
from hft_bot.core.interfaces import (
    ExchangeInterface,
    DataFeedInterface,
    PersistenceInterface,
    AlertingInterface,
)
from hft_bot.core.exceptions import (
    HFTBotException,
    ConfigurationError,
    ExchangeError,
    InsufficientBalanceError,
    OrderError,
    RiskLimitError,
    DataError,
)
