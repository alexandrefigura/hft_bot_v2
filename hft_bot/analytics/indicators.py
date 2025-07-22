"""Technical indicators implementation"""

import numpy as np
from typing import Optional


class TechnicalIndicators:
    """Technical indicators for trading analysis"""
    
    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return float(prices[-1]) if len(prices) > 0 else 0.0
            
        k = 2 / (period + 1)
        ema = prices[-period]
        
        for p in prices[-period+1:]:
            ema = p * k + ema * (1 - k)
            
        return float(ema)
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices[-(period+1):])
        ups = deltas.clip(min=0)
        downs = -deltas.clip(max=0)
        
        avg_gain = np.mean(ups) if len(ups) > 0 else 1e-9
        avg_loss = np.mean(downs) if len(downs) > 0 else 1e-9
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    @staticmethod
    def calculate_momentum(prices: np.ndarray, period: int = 10) -> float:
        """Calculate price momentum"""
        if len(prices) < period + 1:
            return 0.0
            
        return float((prices[-1] - prices[-period]) / prices[-period])
    
    @staticmethod
    def calculate_bollinger_position(price: float, prices: np.ndarray, 
                                   period: int = 20, stds: float = 2) -> float:
        """Calculate position within Bollinger Bands (-1 to 1)"""
        if len(prices) < period:
            return 0.0
            
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        if std == 0:
            return 0.0
            
        upper = sma + stds * std
        lower = sma - stds * std
        
        # Normalize position between -1 and 1
        position = 2 * (price - lower) / (upper - lower) - 1
        
        return float(np.clip(position, -1, 1))
    
    @staticmethod
    def calculate_volatility(prices: np.ndarray, lookback: int = 20) -> float:
        """Calculate price volatility using log returns"""
        if len(prices) < lookback + 1:
            return 0.0
            
        # Use log returns for better statistical properties
        log_prices = np.log(prices[-lookback:])
        returns = np.diff(log_prices)
        
        return float(np.std(returns))
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(high) < period or len(low) < period or len(close) < period:
            return 0.0
            
        # Calculate true ranges
        tr_list = []
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr = max(hl, hc, lc)
            tr_list.append(tr)
            
        if not tr_list:
            return 0.0
            
        # Calculate ATR
        atr = np.mean(tr_list[-period:])
        
        return float(atr)
    
    @staticmethod
    def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> tuple:
        """Calculate MACD and signal line"""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
            
        # Calculate EMAs
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        # For simplicity, we'll use the last 'signal' MACD values
        macd_values = []
        for i in range(signal):
            idx = -(signal - i)
            if idx == 0:
                ema_f = ema_fast
                ema_s = ema_slow
            else:
                ema_f = TechnicalIndicators.calculate_ema(prices[:idx], fast)
                ema_s = TechnicalIndicators.calculate_ema(prices[:idx], slow)
            macd_values.append(ema_f - ema_s)
            
        signal_line = np.mean(macd_values)
        
        # MACD histogram
        histogram = macd_line - signal_line
        
        return float(macd_line), float(signal_line), float(histogram)
