"""Paper trading exchange for testing"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import random

from hft_bot.core.interfaces import ExchangeInterface
from hft_bot.core.exceptions import InsufficientBalanceError, OrderError

logger = logging.getLogger(__name__)


class PaperExchange(ExchangeInterface):
    """Paper trading exchange for testing without real money"""
    
    def __init__(self, initial_balances: Optional[Dict[str, float]] = None,
                 commission_rate: float = 0.001):
        self.balances = defaultdict(float, initial_balances or {'USDT': 10000})
        self.commission_rate = commission_rate
        self.orders = []
        self.order_id_counter = 1000
        self.trades = []
        self.current_prices = {}
        
    async def connect(self) -> None:
        """Mock connection"""
        logger.info("Connected to paper exchange")
        
    async def disconnect(self) -> None:
        """Mock disconnection"""
        logger.info("Disconnected from paper exchange")
        
    async def buy(self, symbol: str, quantity: float, price: Optional[float] = None,
                  order_type: str = 'LIMIT') -> Dict[str, Any]:
        """Execute buy order"""
        base_asset = symbol.replace('USDT', '')
        
        # Use provided price or simulate market price
        if not price:
            price = self._get_market_price(symbol)
            
        total_cost = quantity * price
        commission = total_cost * self.commission_rate
        total_with_commission = total_cost + commission
        
        # Check balance
        if self.balances['USDT'] < total_with_commission:
            raise InsufficientBalanceError(
                required=total_with_commission,
                available=self.balances['USDT'],
                asset='USDT'
            )
            
        # Execute trade
        self.balances['USDT'] -= total_with_commission
        self.balances[base_asset] += quantity
        
        order = {
            'orderId': self.order_id_counter,
            'symbol': symbol,
            'side': 'BUY',
            'status': 'FILLED',
            'price': price,
            'quantity': quantity,
            'commission': commission,
            'timestamp': datetime.now()
        }
        
        self.order_id_counter += 1
        self.orders.append(order)
        self.trades.append(order)
        
        logger.info(f"Paper BUY: {quantity} {base_asset} @ {price} USDT")
        
        return order
        
    async def sell(self, symbol: str, quantity: float, price: Optional[float] = None,
                   order_type: str = 'LIMIT') -> Dict[str, Any]:
        """Execute sell order"""
        base_asset = symbol.replace('USDT', '')
        
        # Check balance
        if self.balances[base_asset] < quantity:
            raise InsufficientBalanceError(
                required=quantity,
                available=self.balances[base_asset],
                asset=base_asset
            )
            
        # Use provided price or simulate market price
        if not price:
            price = self._get_market_price(symbol)
            
        total_value = quantity * price
        commission = total_value * self.commission_rate
        total_after_commission = total_value - commission
        
        # Execute trade
        self.balances[base_asset] -= quantity
        self.balances['USDT'] += total_after_commission
        
        order = {
            'orderId': self.order_id_counter,
            'symbol': symbol,
            'side': 'SELL',
            'status': 'FILLED',
            'price': price,
            'quantity': quantity,
            'commission': commission,
            'timestamp': datetime.now()
        }
        
        self.order_id_counter += 1
        self.orders.append(order)
        self.trades.append(order)
        
        logger.info(f"Paper SELL: {quantity} {base_asset} @ {price} USDT")
        
        return order
        
    async def get_balance(self, asset: str) -> float:
        """Get available balance for an asset"""
        return self.balances.get(asset, 0.0)
        
    async def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel an open order (always returns True for paper trading)"""
        return True
        
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of open orders (empty for paper trading as orders fill instantly)"""
        return []
        
    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Get current ticker data"""
        price = self._get_market_price(symbol)
        return {
            'bid': price * 0.9999,
            'ask': price * 1.0001,
            'last': price,
            'volume': random.uniform(1000, 10000)
        }
        
    async def get_orderbook(self, symbol: str, limit: int = 10) -> Dict[str, List]:
        """Get orderbook data"""
        price = self._get_market_price(symbol)
        
        # Generate fake orderbook
        bids = []
        asks = []
        
        for i in range(limit):
            bid_price = price * (1 - 0.0001 * (i + 1))
            ask_price = price * (1 + 0.0001 * (i + 1))
            
            bid_size = random.uniform(0.1, 10)
            ask_size = random.uniform(0.1, 10)
            
            bids.append((bid_price, bid_size))
            asks.append((ask_price, ask_size))
            
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now()
        }
        
    def _get_market_price(self, symbol: str) -> float:
        """Simulate market price"""
        # Use stored price or generate random price
        if symbol not in self.current_prices:
            if symbol == 'BTCUSDT':
                self.current_prices[symbol] = 50000
            elif symbol == 'ETHUSDT':
                self.current_prices[symbol] = 3000
            else:
                self.current_prices[symbol] = 100
                
        # Add small random variation
        variation = random.uniform(-0.001, 0.001)
        self.current_prices[symbol] *= (1 + variation)
        
        return self.current_prices[symbol]
        
    async def _deduct_costs(self, costs: float) -> None:
        """Deduct additional trading costs"""
        self.balances['USDT'] -= costs
        
    def set_price(self, symbol: str, price: float) -> None:
        """Set price for testing"""
        self.current_prices[symbol] = price
        
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get all trades executed"""
        return self.trades.copy()
        
    def get_total_commission_paid(self) -> float:
        """Get total commission paid"""
        return sum(trade.get('commission', 0) for trade in self.trades)
        
    def reset(self, initial_balances: Optional[Dict[str, float]] = None) -> None:
        """Reset paper exchange to initial state"""
        self.balances = defaultdict(float, initial_balances or {'USDT': 10000})
        self.orders = []
        self.trades = []
        self.order_id_counter = 1000
        self.current_prices = {}
