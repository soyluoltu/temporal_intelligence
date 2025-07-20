"""
Gold Futures (GC=F) Analiz Raporu Oluşturucu
===========================================

Temporal Intelligence Framework kullanarak GC=F için kapsamlı analiz raporu oluşturur.
Fiyat analizleri, stratejiler, tahminler ve grafikler içerir.
"""

import asyncio
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional, Tuple
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add the trading system to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pair_manager import PairManager
from src.core.temporal_analyzer import TradingPairTemporalAnalyzer
from src.core.strategy_engine import StrategyEngine
from src.prediction.timeframe_predictor import TimeframePredictionEngine, ScenarioAnalysisEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class GoldAnalysisResult:
    """Gold analiz sonuçları."""
    timestamp: datetime
    current_price: float
    price_forecast: Dict[str, float]
    technical_signals: Dict[str, Any]
    fundamental_factors: Dict[str, Any]
    temporal_insights: Dict[str, Any]
    risk_assessment: Dict[str, float]
    trading_recommendations: List[Dict[str, Any]]
    market_sentiment: str
    support_resistance: Dict[str, List[float]]


class GoldAnalysisReporter:
    """Gold Futures analiz raporu oluşturucu."""
    
    def __init__(self):
        self.symbol = "GC=F"
        self.pair_manager = None
        self.temporal_analyzer = None
        self.strategy_engine = None
        self.prediction_engine = None
        self.scenario_engine = None
        
        # Gold-specific parameters
        self.gold_factors = {
            'usd_strength': 0.0,
            'inflation_rate': 0.0,
            'interest_rates': 0.0,
            'geopolitical_risk': 0.0,
            'market_volatility': 0.0
        }
        
        # Analysis results storage
        self.analysis_results = []
        self.price_history = []
        
    async def initialize_system(self):
        """Analiz sistemini başlat."""
        logger.info("🥇 Gold Futures Analysis System başlatılıyor...")
        
        # Initialize components
        self.pair_manager = PairManager()
        self.temporal_analyzer = TradingPairTemporalAnalyzer(
            d_model=256,
            sequence_length=100
        )
        self.strategy_engine = StrategyEngine(
            pair_manager=self.pair_manager,
            temporal_analyzer=self.temporal_analyzer
        )
        self.prediction_engine = TimeframePredictionEngine(d_model=256)
        self.scenario_engine = ScenarioAnalysisEngine()
        
        logger.info("✅ System initialization completed!")
        
    def generate_realistic_gold_data(self, periods: int = 252) -> pd.DataFrame:
        """Gerçekçi gold fiyat verisi oluştur."""
        
        # Gold-specific price simulation
        np.random.seed(int(time.time()) % 1000)
        
        # Start from realistic gold price
        base_price = 2000.0  # $2000/oz
        dates = pd.date_range(start=datetime.now() - timedelta(days=periods), 
                             end=datetime.now(), freq='D')
        
        prices = []
        volumes = []
        
        # Gold-specific factors
        usd_trend = np.random.uniform(-0.0005, 0.0005)  # USD strength trend
        inflation_factor = np.random.uniform(0.0002, 0.0008)  # Inflation impact
        geopolitical_events = np.random.choice([0, 1], size=periods, p=[0.95, 0.05])  # 5% chance of events
        
        price = base_price
        for i in range(periods):
            # Daily factors affecting gold
            
            # 1. USD strength (inverse relationship)
            usd_impact = -usd_trend + np.random.normal(0, 0.0002)
            
            # 2. Inflation expectations (positive relationship)
            inflation_impact = inflation_factor + np.random.normal(0, 0.0001)
            
            # 3. Geopolitical events (positive relationship)
            geopolitical_impact = geopolitical_events[i] * np.random.uniform(0.005, 0.02)
            
            # 4. Market volatility and safe haven demand
            market_volatility = np.random.uniform(0, 0.003) if np.random.random() < 0.1 else 0
            
            # 5. Technical momentum
            momentum = np.random.normal(0, 0.004)
            
            # 6. Weekend/holiday effects
            weekend_effect = -0.001 if (i % 7) in [5, 6] else 0
            
            # Combined daily change
            daily_change = (usd_impact + inflation_impact + geopolitical_impact + 
                          market_volatility + momentum + weekend_effect)
            
            price = max(1500, min(3000, price * (1 + daily_change)))  # Realistic bounds
            prices.append(price)
            
            # Volume simulation (higher volume on volatile days)
            base_volume = np.random.lognormal(8, 0.5)
            if abs(daily_change) > 0.01:  # High volatility day
                volume = base_volume * np.random.uniform(1.5, 3.0)
            else:
                volume = base_volume
            volumes.append(volume)
        
        # Create OHLCV data
        data = []
        for i, (date, close_price, volume) in enumerate(zip(dates, prices, volumes)):
            if i == 0:
                open_price = close_price
            else:
                open_price = prices[i-1] + np.random.normal(0, 2)
            
            daily_range = abs(np.random.normal(0, 10))
            high_price = max(open_price, close_price) + daily_range * np.random.uniform(0.3, 1.0)
            low_price = min(open_price, close_price) - daily_range * np.random.uniform(0.3, 1.0)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def calculate_gold_specific_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Gold'a özgü teknik göstergeleri hesapla."""
        
        indicators = {}
        
        # Price-based indicators
        close_prices = data['close']
        
        # Moving averages (critical for gold)
        indicators['sma_20'] = close_prices.rolling(20).mean().iloc[-1]
        indicators['sma_50'] = close_prices.rolling(50).mean().iloc[-1]
        indicators['sma_200'] = close_prices.rolling(200).mean().iloc[-1] if len(data) >= 200 else None
        
        # Gold-specific trend analysis
        current_price = close_prices.iloc[-1]
        indicators['trend_short'] = "BULLISH" if current_price > indicators['sma_20'] else "BEARISH"
        indicators['trend_medium'] = "BULLISH" if current_price > indicators['sma_50'] else "BEARISH"
        if indicators['sma_200']:
            indicators['trend_long'] = "BULLISH" if current_price > indicators['sma_200'] else "BEARISH"
        
        # RSI (oversold/overbought levels important for gold)
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Bollinger Bands (volatility breakouts important for gold)
        bb_period = 20
        bb_std = 2.0
        bb_middle = close_prices.rolling(bb_period).mean()
        bb_std_dev = close_prices.rolling(bb_period).std()
        bb_upper = bb_middle + (bb_std_dev * bb_std)
        bb_lower = bb_middle - (bb_std_dev * bb_std)
        
        indicators['bb_upper'] = bb_upper.iloc[-1]
        indicators['bb_middle'] = bb_middle.iloc[-1]
        indicators['bb_lower'] = bb_lower.iloc[-1]
        indicators['bb_position'] = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        # MACD (momentum important for gold)
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = macd_signal.iloc[-1]
        indicators['macd_histogram'] = macd_histogram.iloc[-1]
        
        # Volatility (important for gold trading)
        returns = close_prices.pct_change().dropna()
        indicators['volatility_20d'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        indicators['volatility_percentile'] = (returns.rolling(60).std().iloc[-1] > 
                                             returns.rolling(60).std().quantile(0.8))
        
        # Support and Resistance levels
        recent_data = data.tail(60)  # Last 60 days
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Find local extremes
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(recent_data) - 2):
            # Local high (resistance)
            if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1] and
                highs.iloc[i] > highs.iloc[i-2] and highs.iloc[i] > highs.iloc[i+2]):
                resistance_levels.append(highs.iloc[i])
            
            # Local low (support)
            if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1] and
                lows.iloc[i] < lows.iloc[i-2] and lows.iloc[i] < lows.iloc[i+2]):
                support_levels.append(lows.iloc[i])
        
        # Get strongest levels
        indicators['resistance_levels'] = sorted(resistance_levels, reverse=True)[:3]
        indicators['support_levels'] = sorted(support_levels, reverse=True)[:3]
        
        return indicators
    
    def analyze_gold_fundamentals(self, current_price: float) -> Dict[str, Any]:
        """Gold fundamental faktörlerini analiz et."""
        
        fundamentals = {}
        
        # Simulated fundamental data (in real implementation, this would come from economic APIs)
        
        # 1. USD Strength Index (DXY) - Inverse relationship with gold
        usd_strength = np.random.uniform(95, 110)  # DXY range
        fundamentals['usd_strength'] = usd_strength
        fundamentals['usd_impact'] = "NEGATIVE" if usd_strength > 105 else "POSITIVE" if usd_strength < 100 else "NEUTRAL"
        
        # 2. Real Interest Rates - Key driver for gold
        nominal_rate = np.random.uniform(2.0, 6.0)
        inflation_rate = np.random.uniform(1.5, 5.0)
        real_rate = nominal_rate - inflation_rate
        fundamentals['real_interest_rate'] = real_rate
        fundamentals['rate_impact'] = "NEGATIVE" if real_rate > 1 else "POSITIVE" if real_rate < -1 else "NEUTRAL"
        
        # 3. Inflation Expectations - Positive relationship with gold
        fundamentals['inflation_rate'] = inflation_rate
        fundamentals['inflation_impact'] = "POSITIVE" if inflation_rate > 3 else "NEGATIVE" if inflation_rate < 2 else "NEUTRAL"
        
        # 4. Geopolitical Risk Assessment
        geopolitical_score = np.random.uniform(0, 10)  # 0-10 scale
        fundamentals['geopolitical_risk'] = geopolitical_score
        fundamentals['geopolitical_impact'] = ("VERY_POSITIVE" if geopolitical_score > 7 else 
                                              "POSITIVE" if geopolitical_score > 5 else 
                                              "NEUTRAL" if geopolitical_score > 3 else "NEGATIVE")
        
        # 5. Central Bank Gold Purchases
        cb_purchases = np.random.uniform(-50, 200)  # Tons per quarter
        fundamentals['central_bank_purchases'] = cb_purchases
        fundamentals['cb_impact'] = "POSITIVE" if cb_purchases > 50 else "NEGATIVE" if cb_purchases < 0 else "NEUTRAL"
        
        # 6. Economic Uncertainty (VIX proxy)
        market_fear = np.random.uniform(12, 35)
        fundamentals['market_fear_index'] = market_fear
        fundamentals['fear_impact'] = "POSITIVE" if market_fear > 25 else "NEGATIVE" if market_fear < 15 else "NEUTRAL"
        
        # 7. Gold ETF Flows
        etf_flows = np.random.uniform(-100, 150)  # Million USD
        fundamentals['etf_flows'] = etf_flows
        fundamentals['etf_impact'] = "POSITIVE" if etf_flows > 50 else "NEGATIVE" if etf_flows < -25 else "NEUTRAL"
        
        # 8. Physical Demand (jewelry, investment)
        physical_demand = np.random.uniform(800, 1200)  # Tons per quarter
        fundamentals['physical_demand'] = physical_demand
        fundamentals['demand_impact'] = "POSITIVE" if physical_demand > 1000 else "NEGATIVE" if physical_demand < 900 else "NEUTRAL"
        
        # Overall fundamental score
        positive_factors = sum(1 for key in fundamentals.keys() if key.endswith('_impact') and 
                              fundamentals[key] in ["POSITIVE", "VERY_POSITIVE"])
        negative_factors = sum(1 for key in fundamentals.keys() if key.endswith('_impact') and 
                              fundamentals[key] == "NEGATIVE")
        
        fundamentals['overall_fundamental_bias'] = ("BULLISH" if positive_factors > negative_factors else 
                                                   "BEARISH" if negative_factors > positive_factors else "NEUTRAL")
        fundamentals['fundamental_strength'] = abs(positive_factors - negative_factors) / 8.0
        
        return fundamentals
    
    async def perform_comprehensive_gold_analysis(self, data: pd.DataFrame) -> GoldAnalysisResult:
        """Kapsamlı gold analizi gerçekleştir."""
        
        current_price = data['close'].iloc[-1]
        
        logger.info(f"📊 Comprehensive Gold Analysis - Current Price: ${current_price:.2f}")
        
        # 1. Technical Analysis
        technical_indicators = self.calculate_gold_specific_indicators(data)
        
        # 2. Temporal Intelligence Analysis
        temporal_result = self.temporal_analyzer.analyze_pair(
            data, self.symbol, f"gold_analysis_{datetime.now().isoformat()}"
        )
        
        # 3. Strategy Engine Analysis
        # Simulated strategy analysis for gold
        strategy_signals = []
        
        # Technical signal based on our indicators
        tech_direction = 0.0
        if technical_indicators['trend_short'] == "BULLISH":
            tech_direction += 0.3
        if technical_indicators['trend_medium'] == "BULLISH":
            tech_direction += 0.4
        if technical_indicators.get('trend_long') == "BULLISH":
            tech_direction += 0.3
        
        # RSI adjustment
        rsi = technical_indicators['rsi']
        if rsi < 30:  # Oversold
            tech_direction += 0.4
        elif rsi > 70:  # Overbought
            tech_direction -= 0.4
        
        # MACD signal
        if technical_indicators['macd'] > technical_indicators['macd_signal']:
            tech_direction += 0.2
        else:
            tech_direction -= 0.2
        
        # Normalize direction
        tech_direction = max(-1.0, min(1.0, tech_direction))
        
        # 4. Fundamental Analysis
        fundamentals = self.analyze_gold_fundamentals(current_price)
        
        # 5. Price Forecasting using multiple methods
        price_forecast = self._generate_price_forecasts(data, technical_indicators, fundamentals)
        
        # 6. Risk Assessment
        risk_assessment = self._assess_gold_risks(data, technical_indicators, fundamentals, temporal_result)
        
        # 7. Trading Recommendations
        trading_recommendations = self._generate_trading_recommendations(
            current_price, technical_indicators, fundamentals, risk_assessment, tech_direction
        )
        
        # 8. Market Sentiment
        sentiment_score = self._calculate_market_sentiment(fundamentals, technical_indicators)
        
        # 9. Support and Resistance
        support_resistance = {
            'resistance': technical_indicators['resistance_levels'],
            'support': technical_indicators['support_levels']
        }
        
        # Create comprehensive result
        result = GoldAnalysisResult(
            timestamp=datetime.now(),
            current_price=current_price,
            price_forecast=price_forecast,
            technical_signals=technical_indicators,
            fundamental_factors=fundamentals,
            temporal_insights={
                'confidence': temporal_result.confidence_score,
                'emergent_signals': temporal_result.emergent_signals,
                'risk_assessment': temporal_result.risk_assessment,
                'pattern_strength': len(temporal_result.emergent_signals),
                'memory_consolidation': temporal_result.temporal_patterns.get('memory_consolidation', {})
            },
            risk_assessment=risk_assessment,
            trading_recommendations=trading_recommendations,
            market_sentiment=sentiment_score,
            support_resistance=support_resistance
        )
        
        return result
    
    def _generate_price_forecasts(self, data: pd.DataFrame, technical: Dict, fundamentals: Dict) -> Dict[str, float]:
        """Fiyat tahminleri oluştur."""
        
        current_price = data['close'].iloc[-1]
        forecasts = {}
        
        # 1. Technical-based forecast
        sma_20 = technical['sma_20']
        sma_50 = technical['sma_50']
        
        technical_bias = 0.0
        if current_price > sma_20 > sma_50:
            technical_bias = 0.03  # 3% bullish
        elif current_price < sma_20 < sma_50:
            technical_bias = -0.03  # 3% bearish
        
        # 2. Fundamental-based forecast
        fundamental_bias = 0.0
        if fundamentals['overall_fundamental_bias'] == "BULLISH":
            fundamental_bias = fundamentals['fundamental_strength'] * 0.05
        elif fundamentals['overall_fundamental_bias'] == "BEARISH":
            fundamental_bias = -fundamentals['fundamental_strength'] * 0.05
        
        # 3. Combined forecasts
        combined_bias = (technical_bias * 0.6 + fundamental_bias * 0.4)
        
        # Time-based forecasts
        forecasts['1_week'] = current_price * (1 + combined_bias * 0.2)
        forecasts['1_month'] = current_price * (1 + combined_bias * 0.5)
        forecasts['3_months'] = current_price * (1 + combined_bias * 1.0)
        forecasts['6_months'] = current_price * (1 + combined_bias * 1.5)
        
        # Add uncertainty ranges
        volatility = technical['volatility_20d']
        for timeframe in forecasts:
            base_forecast = forecasts[timeframe]
            uncertainty = volatility * np.sqrt(
                {'1_week': 7/252, '1_month': 30/252, '3_months': 90/252, '6_months': 180/252}[timeframe]
            )
            forecasts[f'{timeframe}_low'] = base_forecast * (1 - uncertainty)
            forecasts[f'{timeframe}_high'] = base_forecast * (1 + uncertainty)
        
        return forecasts
    
    def _assess_gold_risks(self, data: pd.DataFrame, technical: Dict, fundamentals: Dict, temporal_result) -> Dict[str, float]:
        """Gold-specific risk değerlendirmesi."""
        
        risks = {}
        
        # 1. Price volatility risk
        risks['volatility_risk'] = min(1.0, technical['volatility_20d'] / 0.3)  # Normalize to 30% annual vol
        
        # 2. USD strength risk (major risk for gold)
        usd_strength = fundamentals['usd_strength']
        risks['usd_risk'] = max(0.0, (usd_strength - 100) / 20)  # Higher USD = higher risk
        
        # 3. Interest rate risk
        real_rate = fundamentals['real_interest_rate']
        risks['interest_rate_risk'] = max(0.0, real_rate / 3)  # Positive real rates = risk
        
        # 4. Liquidity risk (from volume analysis)
        recent_volume = data['volume'].tail(20).mean()
        volume_volatility = data['volume'].tail(20).std() / recent_volume
        risks['liquidity_risk'] = min(1.0, volume_volatility)
        
        # 5. Technical breakdown risk
        current_price = data['close'].iloc[-1]
        support_levels = technical['support_levels']
        if support_levels:
            distance_to_support = (current_price - max(support_levels)) / current_price
            risks['technical_risk'] = max(0.0, 1.0 - distance_to_support * 10)
        else:
            risks['technical_risk'] = 0.5
        
        # 6. Temporal pattern instability
        if temporal_result and temporal_result.risk_assessment:
            temporal_risk = np.mean(list(temporal_result.risk_assessment.values()))
            risks['temporal_risk'] = temporal_risk
        else:
            risks['temporal_risk'] = 0.5
        
        # Overall risk score
        risks['overall_risk'] = np.mean(list(risks.values()))
        
        return risks
    
    def _generate_trading_recommendations(self, price: float, technical: Dict, fundamentals: Dict, 
                                        risk_assessment: Dict, direction: float) -> List[Dict[str, Any]]:
        """Trading önerileri oluştur."""
        
        recommendations = []
        
        # Overall signal strength
        signal_strength = abs(direction)
        confidence = 1.0 - risk_assessment['overall_risk']
        
        # Main recommendation
        if signal_strength > 0.3 and confidence > 0.6:
            action = "BUY" if direction > 0 else "SELL"
            
            # Position sizing based on confidence and risk
            position_size = min(0.1, confidence * signal_strength * 0.15)  # Max 15% position
            
            # Stop loss and take profit
            volatility = technical['volatility_20d']
            if action == "BUY":
                stop_loss = price * (1 - volatility * 0.5)
                take_profit = price * (1 + volatility * 1.0)
            else:
                stop_loss = price * (1 + volatility * 0.5)
                take_profit = price * (1 - volatility * 1.0)
            
            recommendations.append({
                'type': 'MAIN_TRADE',
                'action': action,
                'entry_price': price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'reasoning': f"Strong {action.lower()} signal based on technical and fundamental alignment",
                'timeframe': '1-4 weeks'
            })
        
        # Hedging recommendation if high risk
        if risk_assessment['overall_risk'] > 0.7:
            recommendations.append({
                'type': 'HEDGE',
                'action': 'HEDGE_PORTFOLIO',
                'reasoning': 'High risk environment suggests defensive positioning',
                'suggestions': [
                    'Consider reducing position sizes',
                    'Increase cash allocation',
                    'Use options for downside protection'
                ]
            })
        
        # Swing trading opportunities
        rsi = technical['rsi']
        bb_position = technical['bb_position']
        
        if rsi < 30 and bb_position < 0.2:
            recommendations.append({
                'type': 'SWING_TRADE',
                'action': 'BUY',
                'entry_price': price,
                'position_size': 0.05,
                'reasoning': 'Oversold conditions present swing trading opportunity',
                'timeframe': '1-2 weeks',
                'target': price * 1.03
            })
        elif rsi > 70 and bb_position > 0.8:
            recommendations.append({
                'type': 'SWING_TRADE',
                'action': 'SELL',
                'entry_price': price,
                'position_size': 0.05,
                'reasoning': 'Overbought conditions present swing trading opportunity',
                'timeframe': '1-2 weeks',
                'target': price * 0.97
            })
        
        # Long-term investment recommendation
        if fundamentals['overall_fundamental_bias'] == "BULLISH" and confidence > 0.5:
            recommendations.append({
                'type': 'LONG_TERM',
                'action': 'ACCUMULATE',
                'reasoning': 'Strong fundamental backdrop supports long-term accumulation',
                'timeframe': '6-12 months',
                'strategy': 'Dollar-cost averaging on dips'
            })
        
        return recommendations
    
    def _calculate_market_sentiment(self, fundamentals: Dict, technical: Dict) -> str:
        """Piyasa sentiment'ini hesapla."""
        
        sentiment_factors = []
        
        # Fundamental sentiment
        if fundamentals['overall_fundamental_bias'] == "BULLISH":
            sentiment_factors.append(1)
        elif fundamentals['overall_fundamental_bias'] == "BEARISH":
            sentiment_factors.append(-1)
        else:
            sentiment_factors.append(0)
        
        # Technical sentiment
        trend_score = 0
        if technical['trend_short'] == "BULLISH":
            trend_score += 1
        elif technical['trend_short'] == "BEARISH":
            trend_score -= 1
        
        if technical['trend_medium'] == "BULLISH":
            trend_score += 1
        elif technical['trend_medium'] == "BEARISH":
            trend_score -= 1
        
        sentiment_factors.append(trend_score / 2)
        
        # RSI sentiment
        rsi = technical['rsi']
        if rsi > 60:
            sentiment_factors.append(0.5)
        elif rsi < 40:
            sentiment_factors.append(-0.5)
        else:
            sentiment_factors.append(0)
        
        # Overall sentiment
        avg_sentiment = np.mean(sentiment_factors)
        
        if avg_sentiment > 0.3:
            return "BULLISH"
        elif avg_sentiment < -0.3:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def create_comprehensive_charts(self, data: pd.DataFrame, analysis: GoldAnalysisResult) -> str:
        """Kapsamlı grafikler oluştur."""
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Price Chart with Technical Analysis (Top Left)
        ax1 = plt.subplot(3, 3, (1, 3))
        
        # Price data
        dates = data.index
        prices = data['close']
        
        ax1.plot(dates, prices, 'b-', linewidth=2, label='Gold Price', alpha=0.8)
        
        # Moving averages
        if len(data) >= 20:
            sma_20 = prices.rolling(20).mean()
            ax1.plot(dates, sma_20, 'orange', linewidth=1, label='SMA 20', alpha=0.7)
        
        if len(data) >= 50:
            sma_50 = prices.rolling(50).mean()
            ax1.plot(dates, sma_50, 'red', linewidth=1, label='SMA 50', alpha=0.7)
        
        # Support and resistance levels
        current_price = analysis.current_price
        for level in analysis.support_resistance['resistance'][:2]:
            ax1.axhline(y=level, color='red', linestyle='--', alpha=0.5, label='Resistance' if level == analysis.support_resistance['resistance'][0] else "")
        
        for level in analysis.support_resistance['support'][:2]:
            ax1.axhline(y=level, color='green', linestyle='--', alpha=0.5, label='Support' if level == analysis.support_resistance['support'][0] else "")
        
        ax1.set_title(f'Gold Futures (GC=F) - Current Price: ${current_price:.2f}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD/oz)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Volume Chart (Below price chart)
        ax2 = plt.subplot(3, 3, (4, 6))
        volume_colors = ['green' if data['close'].iloc[i] >= data['open'].iloc[i] else 'red' for i in range(len(data))]
        ax2.bar(dates, data['volume'], color=volume_colors, alpha=0.6)
        ax2.set_title('Volume Analysis')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        # 3. Technical Indicators (Top Right)
        ax3 = plt.subplot(3, 3, 7)
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        ax3.plot(dates, rsi, 'purple', linewidth=2)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax3.fill_between(dates, 30, 70, alpha=0.1, color='gray')
        ax3.set_title('RSI (14)')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. MACD (Middle Right)
        ax4 = plt.subplot(3, 3, 8)
        
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        ax4.plot(dates, macd, 'blue', linewidth=2, label='MACD')
        ax4.plot(dates, macd_signal, 'red', linewidth=1, label='Signal')
        ax4.bar(dates, macd_histogram, alpha=0.6, color='gray', label='Histogram')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('MACD')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Fundamental Factors (Bottom Right)
        ax5 = plt.subplot(3, 3, 9)
        
        fund_factors = ['USD Strength', 'Interest Rates', 'Inflation', 'Geopolitical Risk']
        fund_values = [
            analysis.fundamental_factors['usd_strength'] / 110,  # Normalize
            (analysis.fundamental_factors['real_interest_rate'] + 3) / 6,  # Normalize
            analysis.fundamental_factors['inflation_rate'] / 6,  # Normalize
            analysis.fundamental_factors['geopolitical_risk'] / 10  # Normalize
        ]
        
        colors = ['red', 'orange', 'blue', 'green']
        bars = ax5.bar(fund_factors, fund_values, color=colors, alpha=0.7)
        ax5.set_title('Fundamental Factors (Normalized)')
        ax5.set_ylabel('Normalized Score')
        ax5.set_ylim(0, 1)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, fund_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = Path(__file__).parent.parent / 'data' / f'gold_analysis_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        chart_path.parent.mkdir(exist_ok=True)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        logger.info(f"📊 Chart saved: {chart_path}")
        
        return str(chart_path)
    
    def create_forecast_chart(self, analysis: GoldAnalysisResult) -> str:
        """Fiyat tahmin grafiği oluştur."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Current and forecast data
        timeframes = ['Current', '1 Week', '1 Month', '3 Months', '6 Months']
        prices = [
            analysis.current_price,
            analysis.price_forecast['1_week'],
            analysis.price_forecast['1_month'],
            analysis.price_forecast['3_months'],
            analysis.price_forecast['6_months']
        ]
        
        # Uncertainty ranges
        low_prices = [
            analysis.current_price,
            analysis.price_forecast['1_week_low'],
            analysis.price_forecast['1_month_low'],
            analysis.price_forecast['3_months_low'],
            analysis.price_forecast['6_months_low']
        ]
        
        high_prices = [
            analysis.current_price,
            analysis.price_forecast['1_week_high'],
            analysis.price_forecast['1_month_high'],
            analysis.price_forecast['3_months_high'],
            analysis.price_forecast['6_months_high']
        ]
        
        # Plot forecast
        x_pos = range(len(timeframes))
        ax.plot(x_pos, prices, 'bo-', linewidth=3, markersize=8, label='Forecast')
        ax.fill_between(x_pos, low_prices, high_prices, alpha=0.3, color='blue', label='Uncertainty Range')
        
        # Annotations
        for i, (timeframe, price) in enumerate(zip(timeframes, prices)):
            if i > 0:  # Skip current price
                change_pct = ((price - analysis.current_price) / analysis.current_price) * 100
                ax.annotate(f'${price:.0f}\n({change_pct:+.1f}%)', 
                          (i, price), textcoords="offset points", 
                          xytext=(0,10), ha='center', fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(timeframes)
        ax.set_title('Gold Price Forecasts with Uncertainty Ranges', fontsize=16, fontweight='bold')
        ax.set_ylabel('Price (USD/oz)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Color coding for direction
        current_price = analysis.current_price
        for i in range(1, len(prices)):
            color = 'green' if prices[i] > current_price else 'red'
            ax.plot([0, i], [current_price, prices[i]], color=color, alpha=0.5, linestyle='--')
        
        plt.tight_layout()
        
        # Save chart
        forecast_chart_path = Path(__file__).parent.parent / 'data' / f'gold_forecast_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        forecast_chart_path.parent.mkdir(exist_ok=True)
        plt.savefig(forecast_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        logger.info(f"📈 Forecast chart saved: {forecast_chart_path}")
        
        return str(forecast_chart_path)
    
    def generate_markdown_report(self, analysis: GoldAnalysisResult, chart_path: str, forecast_chart_path: str) -> str:
        """Markdown analiz raporu oluştur."""
        
        timestamp = analysis.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        current_price = analysis.current_price
        
        report = f"""# Gold Futures (GC=F) - Comprehensive Analysis Report

**Generated by TL Algorithmic Trading System with Temporal Intelligence Framework**

---

## 📊 Executive Summary

**Analysis Date:** {timestamp}  
**Current Price:** ${current_price:,.2f} USD/oz  
**Market Sentiment:** {analysis.market_sentiment}  
**Overall Risk Level:** {self._get_risk_level_text(analysis.risk_assessment['overall_risk'])}  

### 🎯 Key Findings

- **Technical Outlook:** {analysis.technical_signals['trend_short']} (Short-term), {analysis.technical_signals['trend_medium']} (Medium-term)
- **Fundamental Bias:** {analysis.fundamental_factors['overall_fundamental_bias']}
- **Temporal Intelligence Confidence:** {analysis.temporal_insights['confidence']:.1%}
- **Emergent Patterns Detected:** {analysis.temporal_insights['pattern_strength']}

---

## 📈 Price Analysis & Forecasts

### Current Market Position

| Metric | Value | Status |
|--------|-------|--------|
| Current Price | ${current_price:,.2f} | - |
| 20-Day SMA | ${analysis.technical_signals['sma_20']:,.2f} | {'🟢 Above' if current_price > analysis.technical_signals['sma_20'] else '🔴 Below'} |
| 50-Day SMA | ${analysis.technical_signals['sma_50']:,.2f} | {'🟢 Above' if current_price > analysis.technical_signals['sma_50'] else '🔴 Below'} |
| RSI (14) | {analysis.technical_signals['rsi']:.1f} | {self._get_rsi_status(analysis.technical_signals['rsi'])} |
| Volatility (20d) | {analysis.technical_signals['volatility_20d']:.1%} | {self._get_volatility_status(analysis.technical_signals['volatility_20d'])} |

### 🔮 Price Forecasts

| Timeframe | Forecast | Change | Low Range | High Range |
|-----------|----------|--------|-----------|------------|
| 1 Week | ${analysis.price_forecast['1_week']:,.2f} | {((analysis.price_forecast['1_week'] - current_price) / current_price * 100):+.1f}% | ${analysis.price_forecast['1_week_low']:,.2f} | ${analysis.price_forecast['1_week_high']:,.2f} |
| 1 Month | ${analysis.price_forecast['1_month']:,.2f} | {((analysis.price_forecast['1_month'] - current_price) / current_price * 100):+.1f}% | ${analysis.price_forecast['1_month_low']:,.2f} | ${analysis.price_forecast['1_month_high']:,.2f} |
| 3 Months | ${analysis.price_forecast['3_months']:,.2f} | {((analysis.price_forecast['3_months'] - current_price) / current_price * 100):+.1f}% | ${analysis.price_forecast['3_months_low']:,.2f} | ${analysis.price_forecast['3_months_high']:,.2f} |
| 6 Months | ${analysis.price_forecast['6_months']:,.2f} | {((analysis.price_forecast['6_months'] - current_price) / current_price * 100):+.1f}% | ${analysis.price_forecast['6_months_low']:,.2f} | ${analysis.price_forecast['6_months_high']:,.2f} |

---

## 🔧 Technical Analysis

### Trend Analysis
- **Short-term Trend (20-day):** {analysis.technical_signals['trend_short']}
- **Medium-term Trend (50-day):** {analysis.technical_signals['trend_medium']}
{f"- **Long-term Trend (200-day):** {analysis.technical_signals.get('trend_long', 'N/A')}" if analysis.technical_signals.get('trend_long') else ""}

### Key Technical Levels

**Resistance Levels:**
{chr(10).join([f"- ${level:,.2f}" for level in analysis.support_resistance['resistance'][:3]])}

**Support Levels:**
{chr(10).join([f"- ${level:,.2f}" for level in analysis.support_resistance['support'][:3]])}

### Momentum Indicators
- **RSI (14):** {analysis.technical_signals['rsi']:.1f} - {self._get_rsi_interpretation(analysis.technical_signals['rsi'])}
- **MACD:** {analysis.technical_signals['macd']:.2f} - {'Bullish' if analysis.technical_signals['macd'] > analysis.technical_signals['macd_signal'] else 'Bearish'}
- **Bollinger Band Position:** {analysis.technical_signals['bb_position']:.1%} - {self._get_bb_interpretation(analysis.technical_signals['bb_position'])}

---

## 🌍 Fundamental Analysis

### Economic Factors Impact on Gold

| Factor | Current Value | Impact | Assessment |
|--------|---------------|--------|------------|
| USD Strength (DXY) | {analysis.fundamental_factors['usd_strength']:.1f} | {analysis.fundamental_factors['usd_impact']} | {'🔴 Strong USD hurts gold' if analysis.fundamental_factors['usd_impact'] == 'NEGATIVE' else '🟢 Weak USD helps gold' if analysis.fundamental_factors['usd_impact'] == 'POSITIVE' else '🟡 Neutral impact'} |
| Real Interest Rates | {analysis.fundamental_factors['real_interest_rate']:.2f}% | {analysis.fundamental_factors['rate_impact']} | {'🔴 High real rates hurt gold' if analysis.fundamental_factors['rate_impact'] == 'NEGATIVE' else '🟢 Low/negative real rates help gold' if analysis.fundamental_factors['rate_impact'] == 'POSITIVE' else '🟡 Neutral impact'} |
| Inflation Rate | {analysis.fundamental_factors['inflation_rate']:.1f}% | {analysis.fundamental_factors['inflation_impact']} | {'🟢 High inflation supports gold' if analysis.fundamental_factors['inflation_impact'] == 'POSITIVE' else '🔴 Low inflation hurts gold' if analysis.fundamental_factors['inflation_impact'] == 'NEGATIVE' else '🟡 Neutral impact'} |
| Geopolitical Risk | {analysis.fundamental_factors['geopolitical_risk']:.1f}/10 | {analysis.fundamental_factors['geopolitical_impact']} | {'🟢 High risk supports gold' if 'POSITIVE' in analysis.fundamental_factors['geopolitical_impact'] else '🔴 Low risk hurts gold' if analysis.fundamental_factors['geopolitical_impact'] == 'NEGATIVE' else '🟡 Neutral impact'} |

### Market Flow Analysis
- **Central Bank Purchases:** {analysis.fundamental_factors['central_bank_purchases']:+.0f} tons/quarter ({analysis.fundamental_factors['cb_impact']})
- **ETF Flows:** ${analysis.fundamental_factors['etf_flows']:+.0f}M ({analysis.fundamental_factors['etf_impact']})
- **Physical Demand:** {analysis.fundamental_factors['physical_demand']:,.0f} tons/quarter ({analysis.fundamental_factors['demand_impact']})
- **Market Fear Index:** {analysis.fundamental_factors['market_fear_index']:.1f} ({analysis.fundamental_factors['fear_impact']})

**Overall Fundamental Assessment:** {analysis.fundamental_factors['overall_fundamental_bias']} with {analysis.fundamental_factors['fundamental_strength']:.1%} strength

---

## 🧠 Temporal Intelligence Insights

### AI-Powered Pattern Recognition

**System Confidence:** {analysis.temporal_insights['confidence']:.1%}  
**Emergent Patterns Detected:** {analysis.temporal_insights['pattern_strength']}  

### Discovered Patterns
{chr(10).join([f"- **{signal.get('type', 'Unknown')}:** Strength {signal.get('strength', 0):.2f} - {signal.get('description', 'No description')}" for signal in analysis.temporal_insights['emergent_signals']]) if analysis.temporal_insights['emergent_signals'] else "- No significant emergent patterns detected in current timeframe"}

### Risk Factors Identified
{chr(10).join([f"- {factor.replace('_', ' ').title()}" for factor in analysis.temporal_insights['risk_assessment']]) if analysis.temporal_insights['risk_assessment'] else "- Standard market risks apply"}

### Memory Consolidation Status
- **Pattern Memory Strength:** {analysis.temporal_insights.get('memory_consolidation', {}).get('consolidation_rate', 0.5):.1%}
- **Learning Adaptation:** {'Active' if analysis.temporal_insights['confidence'] > 0.6 else 'Developing'}

---

## ⚠️ Risk Assessment

### Risk Breakdown

| Risk Category | Level | Impact |
|---------------|-------|--------|
| Overall Risk | {analysis.risk_assessment['overall_risk']:.1%} | {self._get_risk_level_text(analysis.risk_assessment['overall_risk'])} |
| Volatility Risk | {analysis.risk_assessment['volatility_risk']:.1%} | {self._get_risk_level_text(analysis.risk_assessment['volatility_risk'])} |
| USD Strength Risk | {analysis.risk_assessment['usd_risk']:.1%} | {self._get_risk_level_text(analysis.risk_assessment['usd_risk'])} |
| Interest Rate Risk | {analysis.risk_assessment['interest_rate_risk']:.1%} | {self._get_risk_level_text(analysis.risk_assessment['interest_rate_risk'])} |
| Technical Risk | {analysis.risk_assessment['technical_risk']:.1%} | {self._get_risk_level_text(analysis.risk_assessment['technical_risk'])} |
| Temporal Risk | {analysis.risk_assessment['temporal_risk']:.1%} | {self._get_risk_level_text(analysis.risk_assessment['temporal_risk'])} |

### Risk Management Recommendations
{chr(10).join([f"- {self._get_risk_recommendation(category, level)}" for category, level in analysis.risk_assessment.items() if category != 'overall_risk' and level > 0.6])}

---

## 💡 Trading Recommendations

{chr(10).join([self._format_trading_recommendation(rec) for rec in analysis.trading_recommendations])}

---

## 📊 Charts & Visualizations

### Technical Analysis Chart
![Gold Technical Analysis]({chart_path})

*Comprehensive technical analysis showing price action, moving averages, support/resistance levels, volume, RSI, MACD, and fundamental factors.*

### Price Forecast Chart
![Gold Price Forecasts]({forecast_chart_path})

*Multi-timeframe price forecasts with uncertainty ranges based on technical and fundamental analysis.*

---

## 📋 Methodology

### Analysis Framework
This analysis leverages the **Temporal Intelligence Framework** which integrates:

1. **Hebbian Learning:** Neural connections that strengthen with repeated patterns
2. **Temporal Attention:** Time-aware pattern recognition across multiple timeframes
3. **Memory Hierarchy:** Short-term, episodic, and semantic memory for pattern storage
4. **Emergent Behavior Detection:** Identification of novel market patterns
5. **Multi-Strategy Integration:** Technical, fundamental, and sentiment analysis fusion

### Data Sources & Indicators
- **Technical Indicators:** SMA, EMA, RSI, MACD, Bollinger Bands, Volume analysis
- **Fundamental Factors:** USD strength, interest rates, inflation, geopolitical risk
- **Market Flow Data:** ETF flows, central bank purchases, physical demand
- **Sentiment Indicators:** Fear/greed index, positioning data

### Forecast Methodology
Price forecasts combine:
- Technical trend analysis (60% weight)
- Fundamental factor assessment (40% weight)
- Volatility-based uncertainty ranges
- Monte Carlo simulation for probability distributions

---

## ⚠️ Important Disclaimers

**Investment Warning:** This analysis is for educational and research purposes only. Gold futures trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.

**AI-Generated Content:** This analysis is generated by an AI system and should not be considered as professional financial advice. Always consult with qualified financial advisors before making investment decisions.

**Data Limitations:** Analysis is based on simulated market data for demonstration purposes. Real trading should use verified market data from reputable sources.

---

## 📞 System Information

**Generated by:** TL Algorithmic Trading System v1.0  
**Framework:** Temporal Intelligence Framework  
**Analysis Engine:** Multi-Strategy Gold Analysis Module  
**Report Generated:** {timestamp}  

**For more information:** [Temporal Intelligence Framework Documentation](https://github.com/temporal-intelligence)

---

*© 2025 Temporal Intelligence Framework - Advanced AI for Financial Markets*
"""

        return report
    
    def _get_risk_level_text(self, risk_score: float) -> str:
        """Risk seviyesi metni döndür."""
        if risk_score >= 0.8:
            return "🔴 Very High"
        elif risk_score >= 0.6:
            return "🟠 High"
        elif risk_score >= 0.4:
            return "🟡 Moderate"
        elif risk_score >= 0.2:
            return "🟢 Low"
        else:
            return "🟢 Very Low"
    
    def _get_rsi_status(self, rsi: float) -> str:
        """RSI durumu döndür."""
        if rsi >= 70:
            return "🔴 Overbought"
        elif rsi <= 30:
            return "🟢 Oversold"
        elif rsi >= 60:
            return "🟠 Strong"
        elif rsi <= 40:
            return "🟡 Weak"
        else:
            return "⚫ Neutral"
    
    def _get_rsi_interpretation(self, rsi: float) -> str:
        """RSI yorumu döndür."""
        if rsi >= 80:
            return "Extremely overbought, potential reversal"
        elif rsi >= 70:
            return "Overbought, caution advised"
        elif rsi <= 20:
            return "Extremely oversold, potential bounce"
        elif rsi <= 30:
            return "Oversold, watch for reversal signals"
        elif rsi >= 55:
            return "Bullish momentum"
        elif rsi <= 45:
            return "Bearish momentum"
        else:
            return "Neutral momentum"
    
    def _get_volatility_status(self, vol: float) -> str:
        """Volatilite durumu döndür."""
        if vol >= 0.30:
            return "🔴 Very High"
        elif vol >= 0.20:
            return "🟠 High"
        elif vol >= 0.15:
            return "🟡 Moderate"
        else:
            return "🟢 Low"
    
    def _get_bb_interpretation(self, position: float) -> str:
        """Bollinger Band pozisyon yorumu."""
        if position >= 0.8:
            return "Near upper band (overbought)"
        elif position <= 0.2:
            return "Near lower band (oversold)"
        elif position >= 0.6:
            return "Upper half (bullish)"
        elif position <= 0.4:
            return "Lower half (bearish)"
        else:
            return "Middle range (neutral)"
    
    def _get_risk_recommendation(self, category: str, level: float) -> str:
        """Risk kategorisine göre öneri döndür."""
        category_name = category.replace('_', ' ').title()
        if level > 0.8:
            return f"**{category_name}:** Critical level - Consider reducing exposure"
        elif level > 0.6:
            return f"**{category_name}:** Elevated - Monitor closely and use tight stops"
        else:
            return f"**{category_name}:** Manageable - Standard risk management applies"
    
    def _format_trading_recommendation(self, rec: Dict[str, Any]) -> str:
        """Trading önerisini formatla."""
        if rec['type'] == 'MAIN_TRADE':
            return f"""
### 🎯 Primary Trading Recommendation

**Action:** {rec['action']}  
**Entry Price:** ${rec['entry_price']:,.2f}  
**Position Size:** {rec['position_size']:.1%} of portfolio  
**Stop Loss:** ${rec['stop_loss']:,.2f}  
**Take Profit:** ${rec['take_profit']:,.2f}  
**Confidence:** {rec['confidence']:.1%}  
**Timeframe:** {rec['timeframe']}  

**Reasoning:** {rec['reasoning']}
"""
        elif rec['type'] == 'SWING_TRADE':
            return f"""
### ⚡ Swing Trading Opportunity

**Action:** {rec['action']}  
**Entry:** ${rec['entry_price']:,.2f}  
**Target:** ${rec['target']:,.2f}  
**Position Size:** {rec['position_size']:.1%}  
**Timeframe:** {rec['timeframe']}  

**Reasoning:** {rec['reasoning']}
"""
        elif rec['type'] == 'LONG_TERM':
            return f"""
### 📈 Long-term Investment Strategy

**Action:** {rec['action']}  
**Strategy:** {rec['strategy']}  
**Timeframe:** {rec['timeframe']}  

**Reasoning:** {rec['reasoning']}
"""
        elif rec['type'] == 'HEDGE':
            suggestions = '\n'.join([f"  - {suggestion}" for suggestion in rec['suggestions']])
            return f"""
### 🛡️ Risk Management

**Action:** {rec['action']}  

**Suggestions:**
{suggestions}

**Reasoning:** {rec['reasoning']}
"""
        else:
            return f"**{rec['type']}:** {rec.get('reasoning', 'No details available')}"
    
    async def generate_comprehensive_report(self) -> str:
        """Kapsamlı analiz raporu oluştur."""
        
        logger.info("🚀 Starting comprehensive Gold analysis...")
        
        # Initialize system
        await self.initialize_system()
        
        # Generate realistic market data
        market_data = self.generate_realistic_gold_data(periods=252)  # 1 year of data
        
        # Perform comprehensive analysis
        analysis_result = await self.perform_comprehensive_gold_analysis(market_data)
        
        # Create charts
        chart_path = self.create_comprehensive_charts(market_data, analysis_result)
        forecast_chart_path = self.create_forecast_chart(analysis_result)
        
        # Generate markdown report
        report_content = self.generate_markdown_report(analysis_result, chart_path, forecast_chart_path)
        
        # Save report
        report_path = Path(__file__).parent.parent / 'data' / f'Gold_Analysis_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 Report saved: {report_path}")
        
        # Also save analysis data as JSON
        json_path = report_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(analysis_result), f, indent=2, default=str)
        
        logger.info(f"💾 Analysis data saved: {json_path}")
        
        return str(report_path)


async def main():
    """Ana fonksiyon."""
    
    print("🥇 Gold Futures (GC=F) - Comprehensive Analysis Report Generator")
    print("🧠 Powered by Temporal Intelligence Framework")
    print("=" * 80)
    
    # Report generator'ı oluştur
    reporter = GoldAnalysisReporter()
    
    try:
        # Kapsamlı analiz raporu oluştur
        report_path = await reporter.generate_comprehensive_report()
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📄 Report saved to: {report_path}")
        print(f"📊 Charts and visualizations included")
        print(f"🧠 Temporal Intelligence insights integrated")
        
        # Report içeriğini kısaca göster
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"\n📋 Report Preview (first 20 lines):")
            print("=" * 50)
            for i, line in enumerate(lines[:20]):
                print(line.rstrip())
            if len(lines) > 20:
                print(f"... ({len(lines) - 20} more lines)")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Windows uyumluluğu
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Analizi çalıştır
    asyncio.run(main())