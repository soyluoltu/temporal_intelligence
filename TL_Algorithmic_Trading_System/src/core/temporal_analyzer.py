"""
Temporal Analyzer for Trading Pairs
=================================

Core temporal learning engine that integrates with the Temporal Intelligence Framework
to analyze trading pairs using temporal patterns and predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import sys
from pathlib import Path

# Add temporal intelligence framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.temporal_system import TemporalIntelligenceSystem
from core.emergent_behavior import ConstraintMode
from validation.model_validator import ValidationResult


@dataclass
class TemporalAnalysisResult:
    """Temporal analysis result for a trading pair."""
    pair: str
    timestamp: datetime
    temporal_patterns: Dict[str, Any]
    trend_prediction: Dict[str, float]
    confidence_score: float
    risk_assessment: Dict[str, float]
    key_levels: Dict[str, List[float]]
    emergent_signals: List[Dict[str, Any]]


class TradingPairTemporalAnalyzer:
    """
    Temporal analyzer specialized for trading pair analysis.
    Integrates Temporal Intelligence Framework with financial data.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        sequence_length: int = 100,
        prediction_horizons: List[str] = None,
        constraint_mode: ConstraintMode = ConstraintMode.ADAPTIVE
    ):
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons or ['1h', '4h', '1d', '1w', '1m']
        
        # Initialize temporal intelligence system
        self.temporal_system = TemporalIntelligenceSystem(
            d_model=d_model,
            n_heads=8,
            hebbian_hidden=d_model // 2,
            learning_rate=0.001,
            validation_threshold=0.4
        )
        self.temporal_system.set_constraint_mode(constraint_mode)
        
        # Technical indicator processors
        self.price_encoder = PriceDataEncoder(d_model)
        self.pattern_detector = TemporalPatternDetector(d_model)
        self.trend_predictor = TrendPredictor(d_model)
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector(d_model)
        
        # Analysis history for temporal learning
        self.analysis_history = []
        
    def analyze_pair(
        self,
        pair_data: pd.DataFrame,
        pair_name: str,
        context: Optional[str] = None
    ) -> TemporalAnalysisResult:
        """
        Comprehensive temporal analysis of a trading pair.
        
        Args:
            pair_data: DataFrame with OHLCV data
            pair_name: Trading pair identifier (e.g., 'EUR/USD')
            context: Optional context for the analysis
            
        Returns:
            TemporalAnalysisResult with comprehensive analysis
        """
        timestamp = datetime.now()
        
        # 1. Encode price data to temporal features
        encoded_features = self._encode_price_data(pair_data)
        
        # 2. Apply temporal intelligence system
        temporal_results = self.temporal_system(
            encoded_features,
            context=f"{pair_name}_{context or 'analysis'}_{timestamp.isoformat()}"
        )
        
        # 3. Extract temporal patterns
        temporal_patterns = self._extract_temporal_patterns(temporal_results)
        
        # 4. Generate trend predictions
        trend_prediction = self._predict_trends(encoded_features, temporal_results)
        
        # 5. Calculate confidence and risk
        confidence_score = self._calculate_confidence(temporal_results)
        risk_assessment = self._assess_risk(pair_data, temporal_results)
        
        # 6. Identify key levels
        key_levels = self._identify_key_levels(pair_data, temporal_results)
        
        # 7. Detect emergent signals
        emergent_signals = self._detect_emergent_signals(temporal_results)
        
        # Create analysis result
        result = TemporalAnalysisResult(
            pair=pair_name,
            timestamp=timestamp,
            temporal_patterns=temporal_patterns,
            trend_prediction=trend_prediction,
            confidence_score=confidence_score,
            risk_assessment=risk_assessment,
            key_levels=key_levels,
            emergent_signals=emergent_signals
        )
        
        # Store for temporal learning
        self.analysis_history.append(result)
        
        return result
    
    def _encode_price_data(self, pair_data: pd.DataFrame) -> torch.Tensor:
        """Convert price data to temporal features."""
        return self.price_encoder.encode(pair_data, self.sequence_length)
    
    def _extract_temporal_patterns(self, temporal_results: Dict) -> Dict[str, Any]:
        """Extract temporal patterns from system results."""
        patterns = {}
        
        # Attention patterns
        if 'attention_output' in temporal_results:
            attention_weights = temporal_results['attention_output']['attention_weights']
            patterns['attention_focus'] = self._analyze_attention_patterns(attention_weights)
        
        # Hebbian learning patterns
        if 'hebbian_output' in temporal_results:
            hebbian_weights = temporal_results['hebbian_output']['weights']
            patterns['connection_strength'] = self._analyze_hebbian_patterns(hebbian_weights)
        
        # Memory patterns
        if 'memory_stats' in temporal_results:
            patterns['memory_consolidation'] = temporal_results['memory_stats']
        
        # Behavioral patterns
        if 'behavior_analysis' in temporal_results:
            patterns['emergent_behavior'] = temporal_results['behavior_analysis']
        
        return patterns
    
    def _predict_trends(
        self,
        encoded_features: torch.Tensor,
        temporal_results: Dict
    ) -> Dict[str, float]:
        """Generate trend predictions for different horizons."""
        predictions = {}
        
        for horizon in self.prediction_horizons:
            # Use temporal system's learned representations
            if 'processed_sequence' in temporal_results:
                processed_seq = temporal_results['processed_sequence']
                prediction = self.trend_predictor.predict(processed_seq, horizon)
                predictions[horizon] = prediction
        
        return predictions
    
    def _calculate_confidence(self, temporal_results: Dict) -> float:
        """Calculate overall confidence score."""
        confidence_factors = []
        
        # Validation confidence
        if 'validation' in temporal_results:
            validation_score = temporal_results['validation']['metrics']['overall_score']
            confidence_factors.append(validation_score)
        
        # Attention consistency
        if 'attention_output' in temporal_results:
            attention_entropy = temporal_results['attention_output'].get('attention_entropy', 0.5)
            attention_confidence = 1.0 - attention_entropy
            confidence_factors.append(attention_confidence)
        
        # Memory consistency
        if 'memory_stats' in temporal_results:
            memory_stability = temporal_results['memory_stats'].get('consolidation_rate', 0.5)
            confidence_factors.append(memory_stability)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _assess_risk(self, pair_data: pd.DataFrame, temporal_results: Dict) -> Dict[str, float]:
        """Assess various risk factors."""
        risk_assessment = {}
        
        # Volatility risk
        if len(pair_data) > 20:
            returns = pair_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            risk_assessment['volatility_risk'] = min(volatility * 10, 1.0)  # Normalize
        
        # Temporal pattern instability
        if 'behavior_analysis' in temporal_results:
            novelty_score = temporal_results['behavior_analysis'].get('novelty_score', 0)
            risk_assessment['pattern_instability'] = novelty_score
        
        # Validation uncertainty
        if 'validation' in temporal_results:
            validation_score = temporal_results['validation']['metrics']['overall_score']
            risk_assessment['prediction_uncertainty'] = 1.0 - validation_score
        
        return risk_assessment
    
    def _identify_key_levels(
        self,
        pair_data: pd.DataFrame,
        temporal_results: Dict
    ) -> Dict[str, List[float]]:
        """Identify key support and resistance levels."""
        key_levels = {'support': [], 'resistance': []}
        
        if len(pair_data) < 20:
            return key_levels
        
        # Use attention patterns to identify significant price levels
        if 'attention_output' in temporal_results:
            attention_weights = temporal_results['attention_output']['attention_weights']
            
            # Find peaks in attention corresponding to price levels
            recent_prices = pair_data['close'].tail(self.sequence_length).values
            
            # Simple support/resistance detection
            highs = pair_data['high'].tail(50)
            lows = pair_data['low'].tail(50)
            
            # Resistance levels (local maxima)
            for i in range(2, len(highs) - 2):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    if highs.iloc[i] > highs.iloc[i-2] and highs.iloc[i] > highs.iloc[i+2]:
                        key_levels['resistance'].append(float(highs.iloc[i]))
            
            # Support levels (local minima)
            for i in range(2, len(lows) - 2):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    if lows.iloc[i] < lows.iloc[i-2] and lows.iloc[i] < lows.iloc[i+2]:
                        key_levels['support'].append(float(lows.iloc[i]))
        
        # Sort and limit to top levels
        key_levels['resistance'] = sorted(key_levels['resistance'], reverse=True)[:5]
        key_levels['support'] = sorted(key_levels['support'], reverse=True)[:5]
        
        return key_levels
    
    def _detect_emergent_signals(self, temporal_results: Dict) -> List[Dict[str, Any]]:
        """Detect emergent trading signals."""
        signals = []
        
        if 'behavior_analysis' in temporal_results:
            behavior = temporal_results['behavior_analysis']
            
            # High novelty signal
            novelty_score = behavior.get('novelty_score', 0)
            if novelty_score > 0.7:
                signals.append({
                    'type': 'pattern_breakout',
                    'strength': novelty_score,
                    'description': 'New pattern detected - potential breakout'
                })
            
            # Quarantine decisions
            quarantine = behavior.get('quarantine_decision', {})
            if quarantine.get('action') == 'validate':
                signals.append({
                    'type': 'validation_required',
                    'strength': 0.8,
                    'description': 'Pattern requires validation - caution advised'
                })
        
        # Memory consolidation signal
        if 'memory_stats' in temporal_results:
            memory_stats = temporal_results['memory_stats']
            if memory_stats.get('consolidation_rate', 0) > 0.8:
                signals.append({
                    'type': 'pattern_confirmation',
                    'strength': memory_stats['consolidation_rate'],
                    'description': 'Pattern strongly confirmed by memory system'
                })
        
        return signals
    
    def _analyze_attention_patterns(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention weight patterns."""
        # Convert to numpy for analysis
        weights = attention_weights.detach().cpu().numpy()
        
        return {
            'max_attention_position': int(np.argmax(weights)),
            'attention_distribution': weights.tolist(),
            'focus_concentration': float(np.max(weights)),
            'attention_entropy': float(-np.sum(weights * np.log(weights + 1e-8)))
        }
    
    def _analyze_hebbian_patterns(self, hebbian_weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze Hebbian connection patterns."""
        weights = hebbian_weights.detach().cpu().numpy()
        
        return {
            'connection_density': float(np.mean(np.abs(weights))),
            'max_connection_strength': float(np.max(np.abs(weights))),
            'weight_distribution_std': float(np.std(weights))
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return self.temporal_system.get_system_statistics()
    
    def set_constraint_mode(self, mode: ConstraintMode):
        """Set constraint mode for temporal system."""
        self.temporal_system.set_constraint_mode(mode)
    
    def consolidate_memory(self):
        """Trigger memory consolidation."""
        self.temporal_system.consolidate_memory()


class PriceDataEncoder(nn.Module):
    """Encodes price data to temporal features."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Technical indicator calculators
        self.technical_features = nn.Linear(20, d_model // 2)  # Technical indicators
        self.price_features = nn.Linear(4, d_model // 2)      # OHLC features
        
    def encode(self, pair_data: pd.DataFrame, sequence_length: int) -> torch.Tensor:
        """Encode price data to temporal features."""
        if len(pair_data) < sequence_length:
            # Pad if necessary
            padding_needed = sequence_length - len(pair_data)
            pair_data = pd.concat([pair_data.iloc[:1].copy() for _ in range(padding_needed)] + [pair_data])
        
        # Take last sequence_length rows
        data = pair_data.tail(sequence_length).copy()
        
        # Calculate technical indicators
        technical_features = self._calculate_technical_indicators(data)
        
        # Normalize OHLC data
        ohlc = data[['open', 'high', 'low', 'close']].values
        ohlc_normalized = (ohlc - ohlc.mean(axis=0)) / (ohlc.std(axis=0) + 1e-8)
        
        # Convert to tensors
        tech_tensor = torch.FloatTensor(technical_features)
        ohlc_tensor = torch.FloatTensor(ohlc_normalized)
        
        # Process through networks
        tech_encoded = self.technical_features(tech_tensor)
        price_encoded = self.price_features(ohlc_tensor)
        
        # Combine features
        combined = torch.cat([tech_encoded, price_encoded], dim=-1)
        
        # Add batch dimension
        return combined.unsqueeze(0)
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate basic technical indicators."""
        indicators = np.zeros((len(data), 20))
        
        if len(data) < 20:
            return indicators
        
        # Simple moving averages
        sma_5 = data['close'].rolling(5).mean()
        sma_10 = data['close'].rolling(10).mean()
        sma_20 = data['close'].rolling(20).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        
        # Bollinger Bands
        bb_middle = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # Fill indicators array
        indicators[:, 0] = (data['close'] / sma_5).fillna(1).values
        indicators[:, 1] = (data['close'] / sma_10).fillna(1).values
        indicators[:, 2] = (data['close'] / sma_20).fillna(1).values
        indicators[:, 3] = (rsi / 100).fillna(0.5).values
        indicators[:, 4] = ((macd - macd.min()) / (macd.max() - macd.min() + 1e-8)).fillna(0.5).values
        indicators[:, 5] = ((data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)).fillna(0.5).values
        
        # Volume indicators
        if 'volume' in data.columns:
            vol_sma = data['volume'].rolling(20).mean()
            indicators[:, 6] = (data['volume'] / vol_sma).fillna(1).values
        
        # Price position indicators
        indicators[:, 7] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)).fillna(0.5).values
        
        # Volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(10).std()
        indicators[:, 8] = (volatility / volatility.max()).fillna(0).values
        
        # Fill remaining with noise for now
        for i in range(9, 20):
            indicators[:, i] = np.random.normal(0, 0.1, len(data))
        
        return indicators


class TemporalPatternDetector:
    """Detects temporal patterns in trading data."""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        
    def detect_patterns(self, features: torch.Tensor) -> Dict[str, Any]:
        """Detect temporal patterns in features."""
        # Placeholder for pattern detection logic
        return {
            'trend_patterns': [],
            'cycle_patterns': [],
            'breakout_patterns': []
        }


class TrendPredictor(nn.Module):
    """Predicts trends for different time horizons."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Prediction heads for different horizons
        self.horizon_predictors = nn.ModuleDict({
            '1h': nn.Linear(d_model, 3),    # up, down, sideways
            '4h': nn.Linear(d_model, 3),
            '1d': nn.Linear(d_model, 3),
            '1w': nn.Linear(d_model, 3),
            '1m': nn.Linear(d_model, 3)
        })
        
    def predict(self, processed_sequence: torch.Tensor, horizon: str) -> float:
        """Predict trend direction for given horizon."""
        if horizon not in self.horizon_predictors:
            return 0.0
        
        # Use last timestep for prediction
        last_features = processed_sequence[:, -1, :]  # [batch, features]
        
        # Get prediction probabilities
        logits = self.horizon_predictors[horizon](last_features)
        probs = torch.softmax(logits, dim=-1)
        
        # Convert to directional score: -1 (down) to +1 (up)
        # probs[0] = up, probs[1] = down, probs[2] = sideways
        directional_score = probs[0, 0] - probs[0, 1]  # up_prob - down_prob
        
        return float(directional_score)


class MarketRegimeDetector:
    """Detects market regime changes."""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        
    def detect_regime(self, features: torch.Tensor) -> str:
        """Detect current market regime."""
        # Placeholder for regime detection
        regimes = ['trending', 'ranging', 'volatile', 'calm']
        return np.random.choice(regimes)