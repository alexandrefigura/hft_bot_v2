"""Optimization command implementation"""

from typing import Dict, Any, Optional

from hft_bot.optimization.optimizer import ParameterOptimizer


async def run_optimization(data_file: str, n_trials: int = 100,
                         metric: str = "sharpe", n_jobs: int = -1,
                         study_name: Optional[str] = None) -> Dict[str, Any]:
    """Run parameter optimization"""
    optimizer = ParameterOptimizer(data_file, metric)
    best_params = await optimizer.optimize_async(n_trials, n_jobs)
    
    return best_params
