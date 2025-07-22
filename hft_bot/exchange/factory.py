"""Exchange factory for creating exchange instances"""

from typing import Optional

from hft_bot.core.config import BotConfig
from hft_bot.core.interfaces import ExchangeInterface
from hft_bot.exchange.binance_exchange import BinanceExchange
from hft_bot.exchange.paper_exchange import PaperExchange


def create_exchange(config: BotConfig) -> ExchangeInterface:
    """Create exchange instance based on configuration"""
    if config.paper_trading:
        base_asset = config.symbol.replace('USDT', '')
        return PaperExchange(
            initial_balances={
                'USDT': config.initial_capital,
                base_asset: 0
            },
            commission_rate=config.trading_params.commission_rate
        )
    else:
        if not config.api_key or not config.api_secret:
            raise ValueError("API key and secret required for live trading")
            
        return BinanceExchange(
            api_key=config.api_key,
            api_secret=config.api_secret,
            testnet=config.testnet
        )
