"""Market analysis and regime detection"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging

from hft_bot.analytics.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Advanced market analysis"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        
    def detect_regime(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Detect current market regime"""
        if len(prices) < 50:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'volatility': 0.0,
                'trend_strength': 0.0
            }
            
        # Calculate metrics
        volatility = self.indicators.calculate_volatility(prices)
        
        # Trend detection using multiple timeframes
        ema_short = self.indicators.calculate_ema(prices, 10)
        ema_medium = self.indicators.calculate_ema(prices, 20)
        ema_long = self.indicators.calculate_ema(prices, 50)
        
        current_price = prices[-1]
        
        # Determine trend
        if ema_short > ema_medium > ema_long:
            trend = 'bullish'
            trend_strength = (ema_short - ema_long) / ema_long
        elif ema_short < ema_medium < ema_long:
            trend = 'bearish'
            trend_strength = (ema_long - ema_short) / ema_long
        else:
            trend = 'ranging'
            trend_strength = abs(ema_short - ema_long) / ema_long
            
        # Determine regime
        if volatility > 0.02:  # High volatility threshold
            regime = 'volatile'
        elif trend == 'ranging' and volatility < 0.01:
            regime = 'ranging'
        elif trend in ['bullish', 'bearish'] and trend_strength > 0.01:
            regime = 'trending'
        else:
            regime = 'choppy'
            
        # Calculate confidence
        confidence = self._calculate_regime_confidence(
            prices, volumes, volatility, trend_strength
        )
        
        return {
            'regime': regime,
            'trend': trend,
            'confidence': confidence,
            'volatility': volatility,
            'trend_strength': trend_strength
        }
        
    def _calculate_regime_confidence(self, prices: np.ndarray, volumes: np.ndarray,
                                   volatility: float, trend_strength: float) -> float:
        """Calculate confidence in regime detection"""
        # Volume consistency
        vol_mean = np.mean(volumes[-20:])
        vol_std = np.std(volumes[-20:])
        vol_consistency = 1 - (vol_std / (vol_mean + 1e-9))
        
        # Price action consistency
        price_changes = np.diff(prices[-20:])
        consecutive_moves = 0
        
        for i in range(1, len(price_changes)):
            if np.sign(price_changes[i]) == np.sign(price_changes[i-1]):
                consecutive_moves += 1
                
        price_consistency = consecutive_moves / len(price_changes)
        
        # Combine factors
        confidence = (
            vol_consistency * 0.3 +
            price_consistency * 0.3 +
            min(trend_strength * 10, 1.0) * 0.4
        )
        
        return float(np.clip(confidence, 0, 1))
        
    def detect_support_resistance(self, prices: np.ndarray, 
                                lookback: int = 100) -> Dict[str, List[float]]:
        """Detect support and resistance levels"""
        if len(prices) < lookback:
            return {'support': [], 'resistance': []}
            
        # Use recent price data
        recent_prices = prices[-lookback:]
        
        # Find local minima and maxima
        support_levels = []
        resistance_levels = []
        
        for i in range(2, len(recent_prices) - 2):
            # Local minimum (support)
            if (recent_prices[i] < recent_prices[i-1] and 
                recent_prices[i] < recent_prices[i-2] and
                recent_prices[i] < recent_prices[i+1] and 
                recent_prices[i] < recent_prices[i+2]):
                support_levels.append(float(recent_prices[i]))
                
            # Local maximum (resistance)
            if (recent_prices[i] > recent_prices[i-1] and 
                recent_prices[i] > recent_prices[i-2] and
                recent_prices[i] > recent_prices[i+1] and 
                recent_prices[i] > recent_prices[i+2]):
                resistance_levels.append(float(recent_prices[i]))
                
        # Cluster nearby levels
        support_levels = self._cluster_levels(support_levels)
        resistance_levels = self._cluster_levels(resistance_levels)
        
        return {
            'support': support_levels[:3],  # Top 3 support levels
            'resistance': resistance_levels[:3]  # Top 3 resistance levels
        }
        
    def _cluster_levels(self, levels: List[float], threshold: float = 0.002) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
            
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Cluster nearby levels
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]
                
        if current_cluster:
            clusters.append(current_cluster)
            
        # Return average of each cluster, sorted by frequency
        clustered_levels = []
        for cluster in clusters:
            avg_level = np.mean(cluster)
            weight = len(cluster)
            clustered_levels.append((avg_level, weight))
            
        # Sort by weight (frequency) descending
        clustered_levels.sort(key=lambda x: x[1], reverse=True)
        
        return [level for level, _ in clustered_levels]
