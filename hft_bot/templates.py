"""
hft_bot.templates
Contém o YAML‑padrão usado por `hft init --no-interactive`.
"""

DEFAULT_CONFIG = """\
# HFT Bot Configuration (default gerado pelo comando init)
symbol: BTCUSDT
paper_trading: true
initial_capital: 1000.0

trading_params:
  decision_interval: 0.5
  min_signal_confluences: 2
  gross_take_profit: 0.0008
  gross_stop_loss: 0.0004
  max_position_size: 500
  kelly_fraction: 0.25

risk:
  max_drawdown: 0.15
  max_positions: 3
  max_exposure: 0.5
  daily_loss_limit: 0.05
  position_timeout: 600

alerts:
  email:
    enabled: false

strategy: ma_crossover
strategy_params:
  fast: 30
  slow: 120
"""
