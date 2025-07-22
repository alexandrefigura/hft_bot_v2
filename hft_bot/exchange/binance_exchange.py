"""Binance exchange implementation"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

from hft_bot.core.interfaces import ExchangeInterface
from hft_bot.core.exceptions import ExchangeError, InsufficientBalanceError, OrderError

logger = logging.getLogger(__name__)


class BinanceExchange(ExchangeInterface):
    """Binance exchange implementation with async support"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client: Optional[AsyncClient] = None
        self.socket_manager: Optional[BinanceSocketManager] = None
        
    async def connect(self) -> None:
        """Connect to Binance"""
        try:
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            self.socket_manager = BinanceSocketManager(self.client)
            logger.info(f"Connected to Binance {'testnet' if self.testnet else 'mainnet'}")
        except Exception as e:
            raise ExchangeError(f"Failed to connect to Binance: {e}")
            
    async def disconnect(self) -> None:
        """Disconnect from Binance"""
        if self.client:
            await self.client.close_connection()
            
    async def buy(self, symbol: str, quantity: float, price: Optional[float] = None,
                  order_type: str = 'LIMIT') -> Dict[str, Any]:
        """Execute buy order"""
        if not self.client:
            raise ExchangeError("Not connected to exchange")
            
        try:
            params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': order_type,
                'quantity': quantity
            }
            
            if order_type == 'LIMIT' and price:
                params['price'] = price
                params['timeInForce'] = 'GTC'
                
            order = await self.client.create_order(**params)
            
            return {
                'orderId': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'status': order['status'],
                'price': float(order.get('price', 0)),
                'quantity': float(order['executedQty']),
                'timestamp': datetime.now()
            }
            
        except BinanceAPIException as e:
            if e.code == -2010:  # Insufficient balance
                raise InsufficientBalanceError(
                    required=quantity * (price or 0),
                    available=0,  # Would need to fetch actual balance
                    asset='USDT'
                )
            else:
                raise OrderError(f"Buy order failed: {e}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error: {e}")
            
    async def sell(self, symbol: str, quantity: float, price: Optional[float] = None,
                   order_type: str = 'LIMIT') -> Dict[str, Any]:
        """Execute sell order"""
        if not self.client:
            raise ExchangeError("Not connected to exchange")
            
        try:
            params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': order_type,
                'quantity': quantity
            }
            
            if order_type == 'LIMIT' and price:
                params['price'] = price
                params['timeInForce'] = 'GTC'
                
            order = await self.client.create_order(**params)
            
            return {
                'orderId': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'status': order['status'],
                'price': float(order.get('price', 0)),
                'quantity': float(order['executedQty']),
                'timestamp': datetime.now()
            }
            
        except BinanceAPIException as e:
            if e.code == -2010:
                raise InsufficientBalanceError(
                    required=quantity,
                    available=0,
                    asset=symbol.replace('USDT', '')
                )
            else:
                raise OrderError(f"Sell order failed: {e}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error: {e}")
            
    async def get_balance(self, asset: str) -> float:
        """Get available balance for an asset"""
        if not self.client:
            raise ExchangeError("Not connected to exchange")
            
        try:
            account = await self.client.get_account()
            
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
                    
            return 0.0
            
        except Exception as e:
            raise ExchangeError(f"Failed to get balance: {e}")
            
    async def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel an open order"""
        if not self.client:
            raise ExchangeError("Not connected to exchange")
            
        try:
            result = await self.client.cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            return result['status'] == 'CANCELED'
            
        except BinanceAPIException as e:
            if e.code == -2011:  # Order not found
                return False
            raise OrderError(f"Failed to cancel order: {e}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error: {e}")
            
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of open orders"""
        if not self.client:
            raise ExchangeError("Not connected to exchange")
            
        try:
            if symbol:
                orders = await self.client.get_open_orders(symbol=symbol)
            else:
                orders = await self.client.get_open_orders()
                
            return [
                {
                    'orderId': order['orderId'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'price': float(order['price']),
                    'quantity': float(order['origQty']),
                    'filled': float(order['executedQty']),
                    'status': order['status'],
                    'timestamp': datetime.fromtimestamp(order['time'] / 1000)
                }
                for order in orders
            ]
            
        except Exception as e:
            raise ExchangeError(f"Failed to get open orders: {e}")
            
    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Get current ticker data"""
        if not self.client:
            raise ExchangeError("Not connected to exchange")
            
        try:
            ticker = await self.client.get_ticker(symbol=symbol)
            
            return {
                'bid': float(ticker['bidPrice']),
                'ask': float(ticker['askPrice']),
                'last': float(ticker['lastPrice']),
                'volume': float(ticker['volume'])
            }
            
        except Exception as e:
            raise ExchangeError(f"Failed to get ticker: {e}")
            
    async def get_orderbook(self, symbol: str, limit: int = 10) -> Dict[str, List]:
        """Get orderbook data"""
        if not self.client:
            raise ExchangeError("Not connected to exchange")
            
        try:
            orderbook = await self.client.get_order_book(
                symbol=symbol,
                limit=limit
            )
            
            return {
                'bids': [(float(price), float(qty)) for price, qty in orderbook['bids']],
                'asks': [(float(price), float(qty)) for price, qty in orderbook['asks']],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            raise ExchangeError(f"Failed to get orderbook: {e}")
