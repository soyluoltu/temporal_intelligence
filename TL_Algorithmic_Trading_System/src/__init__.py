"""
TL Algorithmic Trading System
============================

A comprehensive trading system built on the Temporal Intelligence Framework.
Provides multi-strategy analysis, temporal pattern recognition, and sophisticated
prediction capabilities for various trading pairs.

Author: Temporal Intelligence Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Temporal Intelligence Team"

from .core.temporal_analyzer import TradingPairTemporalAnalyzer, TemporalAnalysisResult
from .core.pair_manager import PairManager, PairData, PairInfo, PairType
from .core.strategy_engine import StrategyEngine, AggregatedSignal, StrategySignal
from .prediction.timeframe_predictor import TimeframePredictionEngine, MultiTimeframePrediction

__all__ = [
    "TradingPairTemporalAnalyzer",
    "TemporalAnalysisResult", 
    "PairManager",
    "PairData",
    "PairInfo",
    "PairType",
    "StrategyEngine",
    "AggregatedSignal",
    "StrategySignal",
    "TimeframePredictionEngine",
    "MultiTimeframePrediction"
]