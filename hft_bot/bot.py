"""Main bot orchestration"""

import asyncio
import time
import logging
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime

from hft_bot.core.config import ConfigProvider, BotConfig
from hft_bot.core.exceptions import RiskLimitError, OrderError
from hft_bot.exchange.factory import create_exchange
from hft_bot.strategies import MeanReversionAdaptive, MomentumStrategy
from hft_bot.risk.manager import RiskManager
from hft_bot.risk.position_sizer import PositionSizer
from hft_bot.infra.logging import StructuredLogger, TradingLogger
from hft_bot.infra.metrics import MetricsCollector, MetricsServer
from hft_bot.infra.alerts import AlertManager
from hft_bot.infra.persistence import StatePersistence

logger = logging.getLogger(__name__)


class HFTBot:
    """Main HFT Bot implementation"""
    
    def __init__(self, config_path: str, log_level: str = "INFO"):
        # Configuration
        self.config_provider = ConfigProvider(config_path)
        self.config = self.config_provider.load()
        
        # Logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = StructuredLogger("hft_bot")
        self.trade_logger = TradingLogger(self.logger, "logs/trades")
        
        # Core components
        self.exchange = create_exchange(self.config)
        self._setup_strategy()
        self.risk_manager = RiskManager(self.config.risk.model_dump())
        self.position_sizer = PositionSizer(
            kelly_fraction=self.config.trading_params.kelly_fraction
        )
        
        # Infrastructure
        self.metrics = MetricsCollector()
        self.metrics_server = MetricsServer(port=8080)
        self.alerts = AlertManager(self.config.alerts.model_dump())
        self.persistence = StatePersistence()
        
        # State
        self.running = False
        self.last_signal_time = 0
        self.consecutive_errors = 0
        self.price_history = []
        self.volume_history = []
        
    def _setup_strategy(self):
        """Setup trading strategy based on config"""
        strategy_name = self.config.strategy
        strategy_params = self.config.strategy_params or {}
        
        if strategy_name == "mean_reversion_adaptive":
            self.strategy = MeanReversionAdaptive(strategy_params)
        elif strategy_name == "momentum":
            self.strategy = MomentumStrategy(strategy_params)
        else:
            # Default to mean reversion
            self.strategy = MeanReversionAdaptive(strategy_params)
            logger.warning(f"Unknown strategy {strategy_name}, using mean_reversion_adaptive")
    
    async def run(self):
        """Main bot loop"""
        logger.info("Starting HFT Bot")
        
        try:
            # Connect to exchange
            await self.exchange.connect()
            await self.metrics_server.start()
            await self.persistence.start_sync()
            
            # Load previous state if exists
            state = await self.persistence.load_state("bot_state")
            if state:
                self._restore_state(state)
            
            self.running = True
            logger.info(f"Bot started - Symbol: {self.config.symbol}, Strategy: {self.config.strategy}")
            
            # Main trading loop
            while self.running:
                loop_start = time.time()
                
                try:
                    await self._trading_iteration()
                    self.consecutive_errors = 0
                    
                except Exception as e:
                    self.consecutive_errors += 1
                    logger.error(f"Trading iteration error: {e}")
                    self.metrics.record_error("trading_iteration", str(e))
                    
                    if self.consecutive_errors > 5:
                        await self.alerts.send_alert(
                            f"Too many consecutive errors: {self.consecutive_errors}",
                            severity="error"
                        )
                        await asyncio.sleep(60)  # Back off
                
                # Rate limiting
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.config.trading_params.decision_interval - loop_duration)
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Critical error: {e}")
            await self.alerts.send_alert(
                f"Bot crashed: {str(e)}",
                severity="critical"
            )
        finally:
            await self.shutdown()
    
    async def _trading_iteration(self):
        """Single trading iteration"""
        start_time = time.time()
        
        # Get market data
        ticker = await self.exchange.get_ticker(self.config.symbol)
        orderbook = await self.exchange.get_orderbook(self.config.symbol)
        
        # Update price history
        self.price_history.append(ticker['last'])
        self.volume_history.append(ticker['volume'])
        
        # Keep limited history
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]
            self.volume_history = self.volume_history[-1000:]
        
        # Prepare market data for strategy
        market_data = {
            'prices': np.array(self.price_history),
            'volumes': np.array(self.volume_history),
            'bid': ticker['bid'],
            'ask': ticker['ask'],
            'spread': ticker['ask'] - ticker['bid'],
            'orderbook': orderbook
        }
        
        # Get trading signal
        signal = await self.strategy.analyze(market_data)
        
        # Record decision time
        decision_time = time.time() - start_time
        self.metrics.record_decision_time(decision_time)
        
        # Check if we should act on signal
        if self._should_trade(signal):
            await self._execute_signal(signal, ticker)
        
        # Update positions
        await self._update_positions(ticker['last'])
        
        # Update metrics
        await self._update_metrics()
        
        # Save state periodically
        if int(time.time()) % 60 == 0:  # Every minute
            await self._save_state()
    
    def _should_trade(self, signal) -> bool:
        """Check if we should act on the signal"""
        # Check signal quality
        if signal.direction == 'HOLD':
            return False
            
        if signal.strength < self.config.trading_params.min_signal_strength:
            return False
            
        if signal.confidence < 0.5:
            return False
        
        # Rate limiting
        time_since_last = time.time() - self.last_signal_time
        if time_since_last < 5:  # Minimum 5 seconds between trades
            return False
            
        return True
    
    async def _execute_signal(self, signal, ticker):
        """Execute trading signal"""
        try:
            # Get current balance
            balance = await self.exchange.get_balance('USDT')
            
            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                balance=balance,
                signal_strength=signal.strength,
                volatility=signal.indicators.get('volatility', 0.01),
                confidence=signal.confidence
            )
            
            # Risk checks
            can_trade, reason = self.risk_manager.can_open_position(
                balance, position_size, self.config.symbol
            )
            
            if not can_trade:
                logger.info(f"Trade rejected by risk manager: {reason}")
                return
            
            # Execute order
            order_start = time.time()
            
            if signal.direction == 'BUY':
                order = await self.exchange.buy(
                    self.config.symbol,
                    position_size / ticker['ask'],  # Convert USDT to asset quantity
                    ticker['ask']
                )
            else:  # SELL
                base_asset = self.config.symbol.replace('USDT', '')
                asset_balance = await self.exchange.get_balance(base_asset)
                
                if asset_balance > 0:
                    order = await self.exchange.sell(
                        self.config.symbol,
                        asset_balance,
                        ticker['bid']
                    )
                else:
                    logger.warning("No position to sell")
                    return
            
            # Record order execution time
            order_time = time.time() - order_start
            self.metrics.record_order_time(order_time)
            
            # Log trade
            trade_id = f"{self.config.symbol}_{int(time.time())}"
            await self.trade_logger.log_trade_opened(
                trade_id=trade_id,
                symbol=self.config.symbol,
                side=signal.direction,
                quantity=order['quantity'],
                price=order['price'],
                signal_strength=signal.strength,
                signal_confidence=signal.confidence,
                reason=signal.reason
            )
            
            # Update risk manager
            self.risk_manager.add_position({
                'symbol': self.config.symbol,
                'size': order['quantity'],
                'entry_price': order['price'],
                'entry_time': datetime.now(),
                'trade_id': trade_id
            })
            
            # Record metrics
            self.metrics.record_trade(
                symbol=self.config.symbol,
                side=signal.direction,
                size=order['quantity'],
                pnl=0  # Will be updated when position closes
            )
            
            # Update last signal time
            self.last_signal_time = time.time()
            
            logger.info(f"Executed {signal.direction} order: {order}")
            
        except OrderError as e:
            logger.error(f"Order execution failed: {e}")
            self.metrics.record_error("order_execution", str(e))
        except Exception as e:
            logger.error(f"Unexpected error in signal execution: {e}")
            self.metrics.record_error("signal_execution", str(e))
    
    async def _update_positions(self, current_price: float):
        """Update and manage open positions"""
        positions_to_close = []
        
        for symbol, position in self.risk_manager.positions.items():
            # Update position with current price
            exit_signal = self.risk_manager.update_position(symbol, current_price)
            
            # Check for exit signals
            if exit_signal:
                positions_to_close.append((position, exit_signal))
                continue
            
            # Check position timeout
            if self.risk_manager.check_position_timeout(position):
                positions_to_close.append((position, "TIMEOUT"))
                continue
            
            # Check fixed take profit/stop loss
            pnl_percent = (current_price - position.entry_price) / position.entry_price
            
            if pnl_percent >= self.config.trading_params.gross_take_profit:
                positions_to_close.append((position, "TAKE_PROFIT"))
            elif pnl_percent <= -self.config.trading_params.gross_stop_loss:
                positions_to_close.append((position, "STOP_LOSS"))
        
        # Close positions
        for position, reason in positions_to_close:
            await self._close_position(position, current_price, reason)
    
    async def _close_position(self, position, exit_price: float, reason: str):
        """Close a position"""
        try:
            # Execute sell order
            order = await self.exchange.sell(
                position.symbol,
                position.size,
                exit_price
            )
            
            # Calculate PnL
            pnl = (exit_price - position.entry_price) * position.size
            
            # Log trade closure
            await self.trade_logger.log_trade_closed(
                trade_id=position.trade_id,
                exit_price=exit_price,
                pnl=pnl,
                reason=reason
            )
            
            # Update position sizer with result
            trade_return = pnl / (position.entry_price * position.size)
            self.position_sizer.update_history(trade_return)
            
            # Remove from risk manager
            self.risk_manager.remove_position(position.symbol)
            
            # Update metrics
            self.metrics.record_trade(
                symbol=position.symbol,
                side='SELL',
                size=position.size,
                pnl=pnl
            )
            
            logger.info(f"Closed position: {position.symbol} for {reason}, PnL: {pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            self.metrics.record_error("position_close", str(e))
    
    async def _update_metrics(self):
        """Update monitoring metrics"""
        try:
            # Get balances
            usdt_balance = await self.exchange.get_balance('USDT')
            base_asset = self.config.symbol.replace('USDT', '')
            asset_balance = await self.exchange.get_balance(base_asset)
            
            # Update balance metrics
            self.metrics.update_balance_metrics({
                'USDT': usdt_balance,
                base_asset: asset_balance
            })
            
            # Update risk metrics
            risk_metrics = self.risk_manager.get_risk_metrics()
            self.metrics.update_risk_metrics(risk_metrics)
            
            # Update position counts
            self.metrics.update_position_metrics({
                self.config.symbol: len(self.risk_manager.positions)
            })
            
            # Check for alerts
            alert_breaches = self.alerts.check_thresholds({
                'drawdown': risk_metrics['current_drawdown'],
                'latency_ms': self.metrics.decision_latency._sum.get() * 1000  # Convert to ms
            })
            
            for breach in alert_breaches:
                await self.alerts.send_alert(
                    f"Threshold breach: {breach['metric']} = {breach['value']:.2f} (limit: {breach['threshold']})",
                    severity=breach['severity']
                )
                
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    async def _save_state(self):
        """Save bot state for recovery"""
        state = {
            'price_history': self.price_history[-100:],  # Last 100 prices
            'volume_history': self.volume_history[-100:],
            'positions': {
                symbol: {
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'entry_time': pos.entry_time.isoformat()
                }
                for symbol, pos in self.risk_manager.positions.items()
            },
            'metrics': self.metrics.get_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        await self.persistence.save_state("bot_state", state)
    
    def _restore_state(self, state: Dict[str, Any]):
        """Restore bot state from saved data"""
        self.price_history = state.get('price_history', [])
        self.volume_history = state.get('volume_history', [])
        
        # Restore positions
        for symbol, pos_data in state.get('positions', {}).items():
            self.risk_manager.add_position({
                'symbol': symbol,
                'size': pos_data['size'],
                'entry_price': pos_data['entry_price'],
                'entry_time': datetime.fromisoformat(pos_data['entry_time'])
            })
        
        logger.info(f"Restored state from {state.get('timestamp', 'unknown')}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down bot")
        self.running = False
        
        # Close all positions
        for position in list(self.risk_manager.positions.values()):
            ticker = await self.exchange.get_ticker(position.symbol)
            await self._close_position(position, ticker['last'], "SHUTDOWN")
        
        # Save final state
        await self._save_state()
        
        # Disconnect services
        await self.exchange.disconnect()
        await self.metrics_server.stop()
        await self.persistence.stop_sync()
        
        # Send shutdown alert
        await self.alerts.send_alert(
            "Bot shutdown completed",
            severity="info"
        )
        
        logger.info("Shutdown complete")
