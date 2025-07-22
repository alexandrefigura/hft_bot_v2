"""Run command implementation"""

import asyncio
import logging
from typing import Optional

from hft_bot.bot import HFTBot

logger = logging.getLogger(__name__)


async def run_bot(config_file: str, paper_trading: bool = True, 
                 testnet: bool = False, strategy: Optional[str] = None,
                 log_level: str = "INFO"):
    """Run the trading bot"""
    logger.info(f"Starting bot with config: {config_file}")
    
    # Create and run bot
    bot = HFTBot(config_file, log_level)
    
    # Override config if needed
    if paper_trading:
        bot.config.paper_trading = True
    if testnet:
        bot.config.testnet = True
    if strategy:
        bot.config.strategy = strategy
        
    # Run bot
    await bot.run()
