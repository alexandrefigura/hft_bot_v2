"""Backtest command implementation"""

import json
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime

from hft_bot.core.config import ConfigProvider
from hft_bot.backtesting.engine import BacktestEngine


async def run_backtest(data_file: str, config_file: str,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      output_file: str = "backtest_results.json",
                      show_plot: bool = True,
                      walk_forward: bool = False,
                      window_test: int = 30) -> Dict[str, Any]:
    """Run backtesting"""
    # Load configuration
    config = ConfigProvider(config_file).load()
    
    # Load data
    data = pd.read_csv(data_file, parse_dates=['timestamp'], index_col='timestamp')
    
    # Filter by date if specified
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]
    
    # Run backtest
    engine = BacktestEngine(config)
    results = await engine.run(data)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Plot if requested
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            plt.plot(results['equity_curve'])
            plt.title('Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.show()
        except ImportError:
            pass
    
    return results
