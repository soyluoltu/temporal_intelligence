"""
Multi-Timeframe Prediction Engine
===============================

Generates predictions across multiple time horizons using temporal intelligence.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..core.temporal_analyzer import TemporalAnalysisResult
from ..core.strategy_engine import AggregatedSignal


class PredictionHorizon(Enum):
    VERY_SHORT = "1h"      # 1 hour
    SHORT = "4h"           # 4 hours  
    MEDIUM = "1d"          # 1 day
    LONG = "1w"            # 1 week
    VERY_LONG = "1M"       # 1 month


class PredictionType(Enum):
    DIRECTION = "direction"         # Up/Down/Sideways
    PRICE_TARGET = "price_target"   # Specific price levels
    VOLATILITY = "volatility"       # Expected volatility
    REGIME = "regime"              # Market regime


@dataclass
class PredictionTarget:
    """Single prediction target."""
    target_type: PredictionType
    horizon: PredictionHorizon
    value: float
    confidence: float
    probability_distribution: Optional[Dict[str, float]] = None
    

@dataclass
class MultiTimeframePrediction:
    """Multi-timeframe prediction result."""
    symbol: str
    timestamp: datetime
    predictions: Dict[PredictionHorizon, List[PredictionTarget]]
    overall_outlook: str  # "bullish", "bearish", "neutral"
    confidence_score: float
    key_drivers: List[str]
    risk_factors: List[str]
    

class DirectionPredictor(nn.Module):
    """Neural network for directional predictions."""
    
    def __init__(self, input_size: int = 256, hidden_size: int = 128):
        super().__init__()
        self.input_size = input_size
        
        self.direction_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 3)  # Up, Down, Sideways
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning direction probabilities and confidence."""
        direction_logits = self.direction_head(x)
        direction_probs = torch.softmax(direction_logits, dim=-1)
        
        confidence = self.confidence_head(x)
        
        return direction_probs, confidence


class PriceTargetPredictor(nn.Module):
    """Neural network for price target predictions."""
    
    def __init__(self, input_size: int = 256, hidden_size: int = 128):
        super().__init__()
        
        self.price_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # Relative price change
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning price change and volatility."""
        price_change = self.price_head(x)
        volatility = self.volatility_head(x)
        
        return price_change, volatility


class TimeframePredictionEngine:
    """Main engine for multi-timeframe predictions."""
    
    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        
        # Prediction models for different horizons
        self.direction_predictors = {
            horizon: DirectionPredictor(d_model) 
            for horizon in PredictionHorizon
        }
        
        self.price_predictors = {
            horizon: PriceTargetPredictor(d_model)
            for horizon in PredictionHorizon
        }
        
        # Volatility forecaster
        self.volatility_forecaster = VolatilityForecaster(d_model)
        
        # Prediction history for learning
        self.prediction_history: List[MultiTimeframePrediction] = []
        
    def predict(
        self,
        temporal_analysis: TemporalAnalysisResult,
        strategy_signal: AggregatedSignal,
        current_price: float,
        pair_data: pd.DataFrame
    ) -> MultiTimeframePrediction:
        """Generate comprehensive multi-timeframe predictions."""
        
        # Extract features for prediction
        features = self._extract_prediction_features(
            temporal_analysis, strategy_signal, pair_data
        )
        
        # Generate predictions for each horizon
        all_predictions = {}
        
        for horizon in PredictionHorizon:
            horizon_predictions = []
            
            # Direction prediction
            direction_pred = self._predict_direction(features, horizon, current_price)
            if direction_pred:
                horizon_predictions.append(direction_pred)
            
            # Price target prediction
            price_pred = self._predict_price_target(features, horizon, current_price)
            if price_pred:
                horizon_predictions.append(price_pred)
            
            # Volatility prediction
            vol_pred = self._predict_volatility(features, horizon, pair_data)
            if vol_pred:
                horizon_predictions.append(vol_pred)
            
            all_predictions[horizon] = horizon_predictions
        
        # Determine overall outlook
        overall_outlook = self._determine_overall_outlook(all_predictions)
        
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(all_predictions)
        
        # Identify key drivers
        key_drivers = self._identify_key_drivers(temporal_analysis, strategy_signal)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(temporal_analysis, strategy_signal)
        
        prediction = MultiTimeframePrediction(
            symbol=temporal_analysis.pair,
            timestamp=datetime.now(),
            predictions=all_predictions,
            overall_outlook=overall_outlook,
            confidence_score=confidence_score,
            key_drivers=key_drivers,
            risk_factors=risk_factors
        )
        
        # Store prediction
        self.prediction_history.append(prediction)
        
        return prediction
    
    def _extract_prediction_features(
        self,
        temporal_analysis: TemporalAnalysisResult,
        strategy_signal: AggregatedSignal,
        pair_data: pd.DataFrame
    ) -> torch.Tensor:
        """Extract features for prediction models."""
        
        feature_vector = []
        
        # Strategy signal features
        feature_vector.extend([
            strategy_signal.overall_direction,
            strategy_signal.overall_confidence,
            strategy_signal.strategy_consensus,
            len(strategy_signal.contributing_strategies) / 10.0  # Normalize
        ])
        
        # Temporal analysis features
        if temporal_analysis.temporal_patterns:
            patterns = temporal_analysis.temporal_patterns
            
            # Attention patterns
            if 'attention_focus' in patterns:
                attention = patterns['attention_focus']
                feature_vector.extend([
                    attention.get('focus_concentration', 0.5),
                    attention.get('attention_entropy', 0.5)
                ])
            else:
                feature_vector.extend([0.5, 0.5])
            
            # Connection strength
            if 'connection_strength' in patterns:
                connections = patterns['connection_strength']
                feature_vector.extend([
                    connections.get('connection_density', 0.5),
                    connections.get('max_connection_strength', 0.5)
                ])
            else:
                feature_vector.extend([0.5, 0.5])
        else:
            feature_vector.extend([0.5, 0.5, 0.5, 0.5])
        
        # Risk assessment features
        risk_assessment = temporal_analysis.risk_assessment
        feature_vector.extend([
            risk_assessment.get('volatility_risk', 0.5),
            risk_assessment.get('pattern_instability', 0.5),
            risk_assessment.get('prediction_uncertainty', 0.5)
        ])
        
        # Emergent signals features
        emergent_count = len(temporal_analysis.emergent_signals) / 5.0  # Normalize
        feature_vector.append(min(1.0, emergent_count))
        
        # Market data features
        if len(pair_data) >= 20:
            recent_data = pair_data.tail(20)
            
            # Price statistics
            returns = recent_data['close'].pct_change().dropna()
            feature_vector.extend([
                float(returns.mean()),
                float(returns.std()),
                float(returns.skew() if len(returns) > 2 else 0),
                float(returns.kurtosis() if len(returns) > 3 else 0)
            ])
            
            # Volume statistics
            if 'volume' in recent_data.columns:
                vol_ma = recent_data['volume'].mean()
                current_vol = recent_data['volume'].iloc[-1]
                feature_vector.append(float(current_vol / vol_ma) if vol_ma > 0 else 1.0)
            else:
                feature_vector.append(1.0)
        else:
            feature_vector.extend([0.0, 0.02, 0.0, 0.0, 1.0])
        
        # Pad to target size
        while len(feature_vector) < self.d_model:
            feature_vector.append(0.0)
        
        # Truncate if too long
        feature_vector = feature_vector[:self.d_model]
        
        return torch.FloatTensor(feature_vector).unsqueeze(0)
    
    def _predict_direction(
        self,
        features: torch.Tensor,
        horizon: PredictionHorizon,
        current_price: float
    ) -> Optional[PredictionTarget]:
        """Predict price direction for given horizon."""
        
        try:
            model = self.direction_predictors[horizon]
            
            with torch.no_grad():
                direction_probs, confidence = model(features)
                
                # Convert probabilities to direction score
                up_prob = direction_probs[0, 0].item()
                down_prob = direction_probs[0, 1].item()
                sideways_prob = direction_probs[0, 2].item()
                
                # Direction score: -1 (bearish) to +1 (bullish)
                direction_score = up_prob - down_prob
                
                # Confidence from model
                model_confidence = confidence[0, 0].item()
                
                # Adjust confidence based on probability distribution
                max_prob = max(up_prob, down_prob, sideways_prob)
                prob_confidence = (max_prob - 0.333) / 0.667  # Normalize from uniform
                
                final_confidence = (model_confidence + prob_confidence) / 2
                
                return PredictionTarget(
                    target_type=PredictionType.DIRECTION,
                    horizon=horizon,
                    value=direction_score,
                    confidence=final_confidence,
                    probability_distribution={
                        'bullish': up_prob,
                        'bearish': down_prob,
                        'neutral': sideways_prob
                    }
                )
                
        except Exception as e:
            logging.error(f"Direction prediction error for {horizon.value}: {e}")
            return None
    
    def _predict_price_target(
        self,
        features: torch.Tensor,
        horizon: PredictionHorizon,
        current_price: float
    ) -> Optional[PredictionTarget]:
        """Predict price target for given horizon."""
        
        try:
            model = self.price_predictors[horizon]
            
            with torch.no_grad():
                price_change, volatility = model(features)
                
                relative_change = price_change[0, 0].item()
                predicted_volatility = volatility[0, 0].item()
                
                # Calculate target price
                target_price = current_price * (1 + relative_change)
                
                # Confidence based on volatility (lower vol = higher confidence)
                vol_confidence = 1.0 / (1.0 + predicted_volatility * 10)
                
                # Bound confidence
                confidence = max(0.1, min(0.9, vol_confidence))
                
                return PredictionTarget(
                    target_type=PredictionType.PRICE_TARGET,
                    horizon=horizon,
                    value=target_price,
                    confidence=confidence,
                    probability_distribution={
                        'expected_change': relative_change,
                        'volatility': predicted_volatility
                    }
                )
                
        except Exception as e:
            logging.error(f"Price target prediction error for {horizon.value}: {e}")
            return None
    
    def _predict_volatility(
        self,
        features: torch.Tensor,
        horizon: PredictionHorizon,
        pair_data: pd.DataFrame
    ) -> Optional[PredictionTarget]:
        """Predict volatility for given horizon."""
        
        try:
            # Use volatility forecaster
            vol_forecast = self.volatility_forecaster.forecast(features, horizon, pair_data)
            
            if vol_forecast:
                return PredictionTarget(
                    target_type=PredictionType.VOLATILITY,
                    horizon=horizon,
                    value=vol_forecast['volatility'],
                    confidence=vol_forecast['confidence']
                )
            
        except Exception as e:
            logging.error(f"Volatility prediction error for {horizon.value}: {e}")
            
        return None
    
    def _determine_overall_outlook(
        self,
        predictions: Dict[PredictionHorizon, List[PredictionTarget]]
    ) -> str:
        """Determine overall market outlook."""
        
        direction_scores = []
        confidences = []
        
        for horizon_preds in predictions.values():
            for pred in horizon_preds:
                if pred.target_type == PredictionType.DIRECTION:
                    direction_scores.append(pred.value)
                    confidences.append(pred.confidence)
        
        if not direction_scores:
            return "neutral"
        
        # Weighted average direction
        weighted_direction = np.average(direction_scores, weights=confidences)
        
        if weighted_direction > 0.2:
            return "bullish"
        elif weighted_direction < -0.2:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_overall_confidence(
        self,
        predictions: Dict[PredictionHorizon, List[PredictionTarget]]
    ) -> float:
        """Calculate overall confidence score."""
        
        all_confidences = []
        
        for horizon_preds in predictions.values():
            for pred in horizon_preds:
                all_confidences.append(pred.confidence)
        
        if not all_confidences:
            return 0.5
        
        return np.mean(all_confidences)
    
    def _identify_key_drivers(
        self,
        temporal_analysis: TemporalAnalysisResult,
        strategy_signal: AggregatedSignal
    ) -> List[str]:
        """Identify key drivers for the prediction."""
        
        drivers = []
        
        # Strategy drivers
        strong_strategies = [
            s for s in strategy_signal.contributing_strategies
            if s.confidence > 0.7
        ]
        
        for strategy in strong_strategies:
            drivers.append(f"{strategy.strategy_type.value}_signal")
        
        # Temporal drivers
        if temporal_analysis.emergent_signals:
            for signal in temporal_analysis.emergent_signals:
                if signal.get('strength', 0) > 0.7:
                    drivers.append(signal.get('type', 'emergent_pattern'))
        
        # Pattern drivers
        patterns = temporal_analysis.temporal_patterns
        if patterns:
            if patterns.get('attention_focus', {}).get('focus_concentration', 0) > 0.8:
                drivers.append('strong_attention_focus')
            
            if patterns.get('connection_strength', {}).get('connection_density', 0) > 0.7:
                drivers.append('strong_temporal_connections')
        
        return drivers[:5]  # Limit to top 5
    
    def _identify_risk_factors(
        self,
        temporal_analysis: TemporalAnalysisResult,
        strategy_signal: AggregatedSignal
    ) -> List[str]:
        """Identify risk factors for the prediction."""
        
        risks = []
        
        # Low consensus risk
        if strategy_signal.strategy_consensus < 0.5:
            risks.append('low_strategy_consensus')
        
        # High volatility risk
        vol_risk = temporal_analysis.risk_assessment.get('volatility_risk', 0)
        if vol_risk > 0.7:
            risks.append('high_volatility')
        
        # Pattern instability risk
        instability = temporal_analysis.risk_assessment.get('pattern_instability', 0)
        if instability > 0.7:
            risks.append('unstable_patterns')
        
        # Prediction uncertainty risk
        uncertainty = temporal_analysis.risk_assessment.get('prediction_uncertainty', 0)
        if uncertainty > 0.6:
            risks.append('high_prediction_uncertainty')
        
        # Emergent behavior risk
        novel_signals = [
            s for s in temporal_analysis.emergent_signals
            if s.get('type') == 'pattern_breakout'
        ]
        if novel_signals:
            risks.append('potential_pattern_breakout')
        
        return risks[:5]  # Limit to top 5
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get statistics about prediction performance."""
        
        if not self.prediction_history:
            return {'total_predictions': 0}
        
        stats = {
            'total_predictions': len(self.prediction_history),
            'avg_confidence': np.mean([p.confidence_score for p in self.prediction_history]),
            'outlook_distribution': {},
            'horizon_coverage': {}
        }
        
        # Outlook distribution
        outlooks = [p.overall_outlook for p in self.prediction_history]
        for outlook in ['bullish', 'bearish', 'neutral']:
            stats['outlook_distribution'][outlook] = outlooks.count(outlook) / len(outlooks)
        
        # Horizon coverage
        for horizon in PredictionHorizon:
            horizon_count = sum(
                1 for p in self.prediction_history 
                if horizon in p.predictions and p.predictions[horizon]
            )
            stats['horizon_coverage'][horizon.value] = horizon_count / len(self.prediction_history)
        
        return stats


class VolatilityForecaster:
    """Specialized volatility forecasting model."""
    
    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        
        # Simple GARCH-like model
        self.volatility_model = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive output
        )
        
    def forecast(
        self,
        features: torch.Tensor,
        horizon: PredictionHorizon,
        pair_data: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """Forecast volatility for given horizon."""
        
        try:
            # Historical volatility baseline
            if len(pair_data) >= 30:
                returns = pair_data['close'].pct_change().dropna()
                hist_vol = returns.rolling(20).std().iloc[-1]
            else:
                hist_vol = 0.02  # Default 2% daily volatility
            
            # Model prediction
            with torch.no_grad():
                vol_pred = self.volatility_model(features)
                predicted_vol = vol_pred[0, 0].item()
            
            # Combine with historical
            combined_vol = (hist_vol * 0.7 + predicted_vol * 0.3)
            
            # Adjust for horizon
            horizon_multipliers = {
                PredictionHorizon.VERY_SHORT: 0.3,
                PredictionHorizon.SHORT: 0.6,
                PredictionHorizon.MEDIUM: 1.0,
                PredictionHorizon.LONG: 1.5,
                PredictionHorizon.VERY_LONG: 2.0
            }
            
            adjusted_vol = combined_vol * horizon_multipliers.get(horizon, 1.0)
            
            # Confidence based on historical volatility stability
            vol_stability = 1.0 / (1.0 + returns.rolling(20).std().std() * 100)
            confidence = max(0.2, min(0.9, vol_stability))
            
            return {
                'volatility': adjusted_vol,
                'confidence': confidence,
                'historical_vol': hist_vol,
                'predicted_vol': predicted_vol
            }
            
        except Exception as e:
            logging.error(f"Volatility forecasting error: {e}")
            return None


class ScenarioAnalysisEngine:
    """Generates scenario-based predictions."""
    
    def __init__(self):
        self.scenarios = ['bull_case', 'base_case', 'bear_case']
        
    def generate_scenarios(
        self,
        prediction: MultiTimeframePrediction,
        current_price: float
    ) -> Dict[str, Dict[str, Any]]:
        """Generate bull, base, and bear case scenarios."""
        
        scenarios = {}
        
        # Get base prediction for medium term
        medium_preds = prediction.predictions.get(PredictionHorizon.MEDIUM, [])
        
        base_direction = 0.0
        base_target = current_price
        base_volatility = 0.02
        
        for pred in medium_preds:
            if pred.target_type == PredictionType.DIRECTION:
                base_direction = pred.value
            elif pred.target_type == PredictionType.PRICE_TARGET:
                base_target = pred.value
            elif pred.target_type == PredictionType.VOLATILITY:
                base_volatility = pred.value
        
        # Bull case: Optimistic scenario
        bull_multiplier = 1.5
        scenarios['bull_case'] = {
            'description': 'Optimistic scenario with strong momentum',
            'price_target': current_price * (1 + abs(base_direction) * bull_multiplier),
            'probability': 0.25,
            'key_catalysts': ['momentum_acceleration', 'positive_sentiment'],
            'timeline': '1-2 weeks'
        }
        
        # Base case: Most likely scenario
        scenarios['base_case'] = {
            'description': 'Most likely outcome based on current analysis',
            'price_target': base_target,
            'probability': 0.5,
            'key_catalysts': prediction.key_drivers,
            'timeline': '1-4 weeks'
        }
        
        # Bear case: Pessimistic scenario
        bear_multiplier = 1.3
        scenarios['bear_case'] = {
            'description': 'Pessimistic scenario with adverse conditions',
            'price_target': current_price * (1 - abs(base_direction) * bear_multiplier),
            'probability': 0.25,
            'key_catalysts': prediction.risk_factors + ['market_stress'],
            'timeline': '1-3 weeks'
        }
        
        return scenarios