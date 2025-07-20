"""
Multi-Strategy Engine for Trading System
======================================

Orchestrates multiple trading strategies and combines their signals.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from abc import ABC, abstractmethod

from .temporal_analyzer import TradingPairTemporalAnalyzer, TemporalAnalysisResult
from .pair_manager import PairManager, PairData


class StrategyType(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    REGIME = "regime"


class SignalStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


@dataclass
class StrategySignal:
    """Individual strategy signal."""
    strategy_type: StrategyType
    symbol: str
    direction: float  # -1.0 to 1.0 (bearish to bullish)
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    timeframe: str
    timestamp: datetime
    metadata: Dict[str, Any]
    

@dataclass
class AggregatedSignal:
    """Aggregated signal from multiple strategies."""
    symbol: str
    overall_direction: float  # -1.0 to 1.0
    overall_confidence: float  # 0.0 to 1.0
    strategy_consensus: float  # How well strategies agree
    contributing_strategies: List[StrategySignal]
    temporal_analysis: Optional[TemporalAnalysisResult]
    timestamp: datetime
    

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, strategy_type: StrategyType, weight: float = 1.0):
        self.strategy_type = strategy_type
        self.weight = weight
        self.enabled = True
        
    @abstractmethod
    def analyze(self, pair_data: PairData, context: Dict[str, Any]) -> List[StrategySignal]:
        """Analyze pair data and generate signals."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if strategy is enabled."""
        return self.enabled
    
    def set_enabled(self, enabled: bool):
        """Enable or disable strategy."""
        self.enabled = enabled
        
    def set_weight(self, weight: float):
        """Set strategy weight."""
        self.weight = max(0.0, min(2.0, weight))


class TechnicalAnalysisStrategy(BaseStrategy):
    """Technical analysis strategy using indicators."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(StrategyType.TECHNICAL, weight)
        
    def analyze(self, pair_data: PairData, context: Dict[str, Any]) -> List[StrategySignal]:
        """Analyze using technical indicators."""
        signals = []
        data = pair_data.data
        
        if len(data) < 50:  # Need sufficient data
            return signals
        
        try:
            # Moving average signals
            ma_signal = self._analyze_moving_averages(data, pair_data.symbol)
            if ma_signal:
                signals.append(ma_signal)
            
            # RSI signals
            rsi_signal = self._analyze_rsi(data, pair_data.symbol)
            if rsi_signal:
                signals.append(rsi_signal)
            
            # MACD signals
            macd_signal = self._analyze_macd(data, pair_data.symbol)
            if macd_signal:
                signals.append(macd_signal)
            
            # Bollinger Bands signals
            bb_signal = self._analyze_bollinger_bands(data, pair_data.symbol)
            if bb_signal:
                signals.append(bb_signal)
                
        except Exception as e:
            logging.error(f"Technical analysis error for {pair_data.symbol}: {e}")
        
        return signals
    
    def _analyze_moving_averages(self, data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Analyze moving average crossovers."""
        try:
            # Calculate MAs
            ma_short = data['close'].rolling(window=20).mean()
            ma_long = data['close'].rolling(window=50).mean()
            
            # Current values
            current_short = ma_short.iloc[-1]
            current_long = ma_long.iloc[-1]
            prev_short = ma_short.iloc[-2]
            prev_long = ma_long.iloc[-2]
            
            # Direction and strength
            if current_short > current_long:
                direction = 0.6 if prev_short <= prev_long else 0.3  # Stronger on crossover
                strength = SignalStrength.STRONG if prev_short <= prev_long else SignalStrength.MODERATE
            else:
                direction = -0.6 if prev_short >= prev_long else -0.3
                strength = SignalStrength.STRONG if prev_short >= prev_long else SignalStrength.MODERATE
            
            # Distance between MAs as confidence
            ma_distance = abs(current_short - current_long) / current_long
            confidence = min(0.9, ma_distance * 50)
            
            return StrategySignal(
                strategy_type=StrategyType.TECHNICAL,
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timeframe='1d',
                timestamp=datetime.now(),
                metadata={
                    'indicator': 'moving_average',
                    'ma_short': float(current_short),
                    'ma_long': float(current_long),
                    'crossover': prev_short <= prev_long if direction > 0 else prev_short >= prev_long
                }
            )
            
        except Exception as e:
            logging.error(f"MA analysis error: {e}")
            return None
    
    def _analyze_rsi(self, data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Analyze RSI signals."""
        try:
            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            # Generate signal based on RSI levels
            if current_rsi > 80:
                direction = -0.7  # Overbought - bearish
                strength = SignalStrength.STRONG
                confidence = (current_rsi - 80) / 20
            elif current_rsi < 20:
                direction = 0.7  # Oversold - bullish
                strength = SignalStrength.STRONG
                confidence = (20 - current_rsi) / 20
            elif current_rsi > 70:
                direction = -0.4
                strength = SignalStrength.MODERATE
                confidence = (current_rsi - 70) / 10
            elif current_rsi < 30:
                direction = 0.4
                strength = SignalStrength.MODERATE
                confidence = (30 - current_rsi) / 10
            else:
                return None  # Neutral zone
            
            return StrategySignal(
                strategy_type=StrategyType.TECHNICAL,
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=min(0.9, confidence),
                timeframe='1d',
                timestamp=datetime.now(),
                metadata={
                    'indicator': 'rsi',
                    'rsi_value': float(current_rsi),
                    'signal_type': 'overbought' if direction < 0 else 'oversold'
                }
            )
            
        except Exception as e:
            logging.error(f"RSI analysis error: {e}")
            return None
    
    def _analyze_macd(self, data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Analyze MACD signals."""
        try:
            # Calculate MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            current_hist = histogram.iloc[-1]
            prev_hist = histogram.iloc[-2]
            
            # Signal based on histogram and crossovers
            if current_hist > 0 and prev_hist <= 0:
                direction = 0.6  # Bullish crossover
                strength = SignalStrength.STRONG
                confidence = 0.8
            elif current_hist < 0 and prev_hist >= 0:
                direction = -0.6  # Bearish crossover
                strength = SignalStrength.STRONG
                confidence = 0.8
            elif current_hist > 0:
                direction = 0.3  # Above zero
                strength = SignalStrength.WEAK
                confidence = 0.4
            else:
                direction = -0.3  # Below zero
                strength = SignalStrength.WEAK
                confidence = 0.4
            
            return StrategySignal(
                strategy_type=StrategyType.TECHNICAL,
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timeframe='1d',
                timestamp=datetime.now(),
                metadata={
                    'indicator': 'macd',
                    'macd_line': float(macd_line.iloc[-1]),
                    'signal_line': float(signal_line.iloc[-1]),
                    'histogram': float(current_hist),
                    'crossover': abs(current_hist - prev_hist) > abs(current_hist) * 0.1
                }
            )
            
        except Exception as e:
            logging.error(f"MACD analysis error: {e}")
            return None
    
    def _analyze_bollinger_bands(self, data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Analyze Bollinger Bands signals."""
        try:
            # Calculate Bollinger Bands
            window = 20
            rolling_mean = data['close'].rolling(window).mean()
            rolling_std = data['close'].rolling(window).std()
            
            upper_band = rolling_mean + (rolling_std * 2)
            lower_band = rolling_mean - (rolling_std * 2)
            
            current_price = data['close'].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_middle = rolling_mean.iloc[-1]
            
            # Position relative to bands
            band_width = current_upper - current_lower
            position = (current_price - current_lower) / band_width
            
            # Generate signals
            if position > 0.9:
                direction = -0.5  # Near upper band - bearish
                strength = SignalStrength.MODERATE
                confidence = (position - 0.9) * 10
            elif position < 0.1:
                direction = 0.5  # Near lower band - bullish
                strength = SignalStrength.MODERATE
                confidence = (0.1 - position) * 10
            else:
                # Trend signal based on position relative to middle
                direction = (position - 0.5) * 0.4
                strength = SignalStrength.WEAK
                confidence = abs(position - 0.5) * 2
            
            return StrategySignal(
                strategy_type=StrategyType.TECHNICAL,
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=min(0.9, confidence),
                timeframe='1d',
                timestamp=datetime.now(),
                metadata={
                    'indicator': 'bollinger_bands',
                    'current_price': float(current_price),
                    'upper_band': float(current_upper),
                    'lower_band': float(current_lower),
                    'band_position': float(position),
                    'squeeze': band_width / current_middle < 0.1
                }
            )
            
        except Exception as e:
            logging.error(f"Bollinger Bands analysis error: {e}")
            return None


class MomentumAnalysisStrategy(BaseStrategy):
    """Momentum-based analysis strategy."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(StrategyType.MOMENTUM, weight)
        
    def analyze(self, pair_data: PairData, context: Dict[str, Any]) -> List[StrategySignal]:
        """Analyze momentum indicators."""
        signals = []
        data = pair_data.data
        
        if len(data) < 30:
            return signals
        
        try:
            # Price momentum
            momentum_signal = self._analyze_price_momentum(data, pair_data.symbol)
            if momentum_signal:
                signals.append(momentum_signal)
            
            # Volume momentum
            if 'volume' in data.columns:
                volume_signal = self._analyze_volume_momentum(data, pair_data.symbol)
                if volume_signal:
                    signals.append(volume_signal)
                    
        except Exception as e:
            logging.error(f"Momentum analysis error for {pair_data.symbol}: {e}")
        
        return signals
    
    def _analyze_price_momentum(self, data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Analyze price momentum."""
        try:
            # Calculate rate of change over different periods
            roc_5 = data['close'].pct_change(5)
            roc_10 = data['close'].pct_change(10)
            roc_20 = data['close'].pct_change(20)
            
            current_roc_5 = roc_5.iloc[-1]
            current_roc_10 = roc_10.iloc[-1]
            current_roc_20 = roc_20.iloc[-1]
            
            # Weighted momentum score
            momentum_score = (current_roc_5 * 0.5 + current_roc_10 * 0.3 + current_roc_20 * 0.2)
            
            # Direction and strength
            direction = np.tanh(momentum_score * 20)  # Scale and bound between -1 and 1
            
            # Strength based on magnitude
            magnitude = abs(momentum_score)
            if magnitude > 0.05:
                strength = SignalStrength.STRONG
            elif magnitude > 0.02:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Confidence based on consistency across timeframes
            consistency = 1.0 - np.std([current_roc_5, current_roc_10, current_roc_20]) / (magnitude + 1e-6)
            confidence = max(0.1, min(0.9, consistency))
            
            return StrategySignal(
                strategy_type=StrategyType.MOMENTUM,
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timeframe='1d',
                timestamp=datetime.now(),
                metadata={
                    'momentum_score': float(momentum_score),
                    'roc_5': float(current_roc_5),
                    'roc_10': float(current_roc_10),
                    'roc_20': float(current_roc_20),
                    'consistency': float(consistency)
                }
            )
            
        except Exception as e:
            logging.error(f"Price momentum analysis error: {e}")
            return None
    
    def _analyze_volume_momentum(self, data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Analyze volume momentum."""
        try:
            # Volume moving averages
            vol_ma_short = data['volume'].rolling(10).mean()
            vol_ma_long = data['volume'].rolling(30).mean()
            
            current_vol = data['volume'].iloc[-1]
            current_vol_ma_short = vol_ma_short.iloc[-1]
            current_vol_ma_long = vol_ma_long.iloc[-1]
            
            # Volume relative to averages
            vol_ratio_short = current_vol / current_vol_ma_short
            vol_ratio_long = current_vol / current_vol_ma_long
            
            # Price change
            price_change = data['close'].pct_change().iloc[-1]
            
            # Volume confirms price movement
            if vol_ratio_short > 1.5 and vol_ratio_long > 1.2:
                # High volume - confirms direction
                direction = np.sign(price_change) * 0.6
                strength = SignalStrength.STRONG
                confidence = min(0.9, (vol_ratio_short - 1) * 0.5)
            elif vol_ratio_short > 1.2:
                # Moderate volume
                direction = np.sign(price_change) * 0.4
                strength = SignalStrength.MODERATE
                confidence = min(0.7, (vol_ratio_short - 1) * 0.8)
            else:
                # Low volume - weak signal
                direction = np.sign(price_change) * 0.2
                strength = SignalStrength.WEAK
                confidence = 0.3
            
            return StrategySignal(
                strategy_type=StrategyType.MOMENTUM,
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timeframe='1d',
                timestamp=datetime.now(),
                metadata={
                    'current_volume': float(current_vol),
                    'vol_ratio_short': float(vol_ratio_short),
                    'vol_ratio_long': float(vol_ratio_long),
                    'price_change': float(price_change),
                    'volume_confirmation': vol_ratio_short > 1.2
                }
            )
            
        except Exception as e:
            logging.error(f"Volume momentum analysis error: {e}")
            return None


class VolatilityAnalysisStrategy(BaseStrategy):
    """Volatility-based analysis strategy."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(StrategyType.VOLATILITY, weight)
        
    def analyze(self, pair_data: PairData, context: Dict[str, Any]) -> List[StrategySignal]:
        """Analyze volatility patterns."""
        signals = []
        data = pair_data.data
        
        if len(data) < 30:
            return signals
        
        try:
            volatility_signal = self._analyze_volatility_regime(data, pair_data.symbol)
            if volatility_signal:
                signals.append(volatility_signal)
                
        except Exception as e:
            logging.error(f"Volatility analysis error for {pair_data.symbol}: {e}")
        
        return signals
    
    def _analyze_volatility_regime(self, data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Analyze volatility regime changes."""
        try:
            # Calculate volatility measures
            returns = data['close'].pct_change().dropna()
            
            # Rolling volatility
            vol_short = returns.rolling(10).std()
            vol_long = returns.rolling(30).std()
            
            current_vol_short = vol_short.iloc[-1]
            current_vol_long = vol_long.iloc[-1]
            
            # Volatility ratio
            vol_ratio = current_vol_short / current_vol_long
            
            # ATR (Average True Range)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = tr.rolling(14).mean()
            current_atr = atr.iloc[-1]
            
            # Generate signal based on volatility regime
            if vol_ratio > 1.5:
                # High volatility - potential breakout
                direction = 0.0  # Neutral direction, but high opportunity
                strength = SignalStrength.STRONG
                confidence = min(0.9, (vol_ratio - 1) * 0.6)
                signal_type = 'breakout_potential'
            elif vol_ratio < 0.7:
                # Low volatility - potential compression
                direction = 0.0
                strength = SignalStrength.MODERATE
                confidence = min(0.8, (1 - vol_ratio) * 1.4)
                signal_type = 'compression'
            else:
                # Normal volatility
                direction = 0.0
                strength = SignalStrength.WEAK
                confidence = 0.3
                signal_type = 'normal'
            
            return StrategySignal(
                strategy_type=StrategyType.VOLATILITY,
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timeframe='1d',
                timestamp=datetime.now(),
                metadata={
                    'vol_ratio': float(vol_ratio),
                    'current_vol_short': float(current_vol_short),
                    'current_vol_long': float(current_vol_long),
                    'current_atr': float(current_atr),
                    'signal_type': signal_type
                }
            )
            
        except Exception as e:
            logging.error(f"Volatility regime analysis error: {e}")
            return None


class StrategyEngine:
    """Main strategy orchestration engine."""
    
    def __init__(
        self,
        pair_manager: PairManager,
        temporal_analyzer: Optional[TradingPairTemporalAnalyzer] = None
    ):
        self.pair_manager = pair_manager
        self.temporal_analyzer = temporal_analyzer
        
        # Initialize strategies
        self.strategies: Dict[StrategyType, BaseStrategy] = {
            StrategyType.TECHNICAL: TechnicalAnalysisStrategy(weight=1.2),
            StrategyType.MOMENTUM: MomentumAnalysisStrategy(weight=1.0),
            StrategyType.VOLATILITY: VolatilityAnalysisStrategy(weight=0.8)
        }
        
        # Strategy weights for aggregation
        self.strategy_weights = {
            StrategyType.TECHNICAL: 0.3,
            StrategyType.MOMENTUM: 0.25,
            StrategyType.VOLATILITY: 0.15,
            StrategyType.FUNDAMENTAL: 0.15,
            StrategyType.SENTIMENT: 0.1,
            StrategyType.CORRELATION: 0.05
        }
        
        # Signal history for learning
        self.signal_history: List[AggregatedSignal] = []
        
    def add_strategy(self, strategy: BaseStrategy):
        """Add a custom strategy."""
        self.strategies[strategy.strategy_type] = strategy
        logging.info(f"Added strategy: {strategy.strategy_type.value}")
    
    def remove_strategy(self, strategy_type: StrategyType):
        """Remove a strategy."""
        if strategy_type in self.strategies:
            del self.strategies[strategy_type]
            logging.info(f"Removed strategy: {strategy_type.value}")
    
    def set_strategy_weight(self, strategy_type: StrategyType, weight: float):
        """Set weight for strategy aggregation."""
        self.strategy_weights[strategy_type] = max(0.0, min(1.0, weight))
    
    async def analyze_pair(
        self,
        symbol: str,
        timeframe: str = '1d',
        period: str = '1y',
        include_temporal: bool = True
    ) -> Optional[AggregatedSignal]:
        """Comprehensive analysis of a trading pair."""
        
        # Get pair data
        pair_data = await self.pair_manager.get_pair_data(symbol, timeframe, period)
        if not pair_data:
            logging.error(f"Could not get data for {symbol}")
            return None
        
        # Collect signals from all enabled strategies
        all_signals = []
        context = {'timeframe': timeframe, 'period': period}
        
        for strategy_type, strategy in self.strategies.items():
            if strategy.is_enabled():
                try:
                    signals = strategy.analyze(pair_data, context)
                    all_signals.extend(signals)
                except Exception as e:
                    logging.error(f"Strategy {strategy_type.value} failed for {symbol}: {e}")
        
        if not all_signals:
            logging.warning(f"No signals generated for {symbol}")
            return None
        
        # Temporal analysis
        temporal_result = None
        if include_temporal and self.temporal_analyzer:
            try:
                temporal_result = self.temporal_analyzer.analyze_pair(
                    pair_data.data, symbol, f"strategy_engine_{timeframe}"
                )
            except Exception as e:
                logging.error(f"Temporal analysis failed for {symbol}: {e}")
        
        # Aggregate signals
        aggregated_signal = self._aggregate_signals(symbol, all_signals, temporal_result)
        
        # Store in history
        self.signal_history.append(aggregated_signal)
        
        return aggregated_signal
    
    def _aggregate_signals(
        self,
        symbol: str,
        signals: List[StrategySignal],
        temporal_result: Optional[TemporalAnalysisResult]
    ) -> AggregatedSignal:
        """Aggregate signals from multiple strategies."""
        
        if not signals:
            return AggregatedSignal(
                symbol=symbol,
                overall_direction=0.0,
                overall_confidence=0.0,
                strategy_consensus=0.0,
                contributing_strategies=[],
                temporal_analysis=temporal_result,
                timestamp=datetime.now()
            )
        
        # Calculate weighted direction and confidence
        total_weight = 0.0
        weighted_direction = 0.0
        weighted_confidence = 0.0
        
        for signal in signals:
            strategy_weight = self.strategy_weights.get(signal.strategy_type, 0.1)
            signal_weight = strategy_weight * signal.confidence * signal.strength.value
            
            weighted_direction += signal.direction * signal_weight
            weighted_confidence += signal.confidence * signal_weight
            total_weight += signal_weight
        
        if total_weight > 0:
            overall_direction = weighted_direction / total_weight
            overall_confidence = weighted_confidence / total_weight
        else:
            overall_direction = 0.0
            overall_confidence = 0.0
        
        # Calculate consensus (agreement between strategies)
        directions = [s.direction for s in signals]
        consensus = 1.0 - (np.std(directions) / (np.mean(np.abs(directions)) + 1e-6))
        consensus = max(0.0, min(1.0, consensus))
        
        # Incorporate temporal analysis
        if temporal_result:
            temporal_confidence = temporal_result.confidence_score
            
            # Blend with temporal predictions if available
            if temporal_result.trend_prediction:
                # Use 1d prediction as primary
                temporal_direction = temporal_result.trend_prediction.get('1d', 0.0)
                
                # Weight temporal analysis based on its confidence
                temporal_weight = temporal_confidence * 0.3  # 30% max weight
                total_analysis_weight = 1.0 + temporal_weight
                
                overall_direction = (overall_direction + temporal_direction * temporal_weight) / total_analysis_weight
                overall_confidence = (overall_confidence + temporal_confidence * temporal_weight) / total_analysis_weight
        
        return AggregatedSignal(
            symbol=symbol,
            overall_direction=overall_direction,
            overall_confidence=overall_confidence,
            strategy_consensus=consensus,
            contributing_strategies=signals,
            temporal_analysis=temporal_result,
            timestamp=datetime.now()
        )
    
    async def analyze_multiple_pairs(
        self,
        symbols: List[str],
        timeframe: str = '1d',
        period: str = '1y'
    ) -> Dict[str, AggregatedSignal]:
        """Analyze multiple pairs concurrently."""
        results = {}
        
        # For now, analyze sequentially (could be made async)
        for symbol in symbols:
            signal = await self.analyze_pair(symbol, timeframe, period)
            if signal:
                results[symbol] = signal
        
        logging.info(f"Analyzed {len(results)} pairs successfully")
        return results
    
    def get_top_opportunities(
        self,
        signals: Dict[str, AggregatedSignal],
        min_confidence: float = 0.6,
        limit: int = 10
    ) -> List[AggregatedSignal]:
        """Get top trading opportunities."""
        
        # Filter by minimum confidence
        filtered_signals = [
            signal for signal in signals.values()
            if signal.overall_confidence >= min_confidence
        ]
        
        # Sort by combined score (direction strength * confidence * consensus)
        def score_signal(signal: AggregatedSignal) -> float:
            return (
                abs(signal.overall_direction) * 
                signal.overall_confidence * 
                signal.strategy_consensus
            )
        
        top_signals = sorted(filtered_signals, key=score_signal, reverse=True)
        
        return top_signals[:limit]
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance statistics for strategies."""
        performance = {}
        
        for strategy_type, strategy in self.strategies.items():
            # Count signals generated
            type_signals = [
                signal for aggregated in self.signal_history
                for signal in aggregated.contributing_strategies
                if signal.strategy_type == strategy_type
            ]
            
            if type_signals:
                avg_confidence = np.mean([s.confidence for s in type_signals])
                avg_strength = np.mean([s.strength.value for s in type_signals])
                signal_count = len(type_signals)
            else:
                avg_confidence = 0.0
                avg_strength = 0.0
                signal_count = 0
            
            performance[strategy_type.value] = {
                'signal_count': signal_count,
                'avg_confidence': avg_confidence,
                'avg_strength': avg_strength,
                'enabled': strategy.is_enabled(),
                'weight': strategy.weight
            }
        
        return performance
    
    def optimize_strategy_weights(self, performance_data: Optional[Dict] = None):
        """Optimize strategy weights based on performance."""
        # Placeholder for weight optimization logic
        # Could be based on historical accuracy, Sharpe ratio, etc.
        pass