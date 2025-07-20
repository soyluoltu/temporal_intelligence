"""
Trading System Demo
==================

Demonstrates the TL Algorithmic Trading System capabilities.
"""

import asyncio
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Add the trading system to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pair_manager import PairManager, PairType
from src.core.temporal_analyzer import TradingPairTemporalAnalyzer
from src.core.strategy_engine import StrategyEngine
from src.prediction.timeframe_predictor import TimeframePredictionEngine, ScenarioAnalysisEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingSystemDemo:
    """Main demo class for the trading system."""
    
    def __init__(self):
        self.pair_manager = None
        self.temporal_analyzer = None
        self.strategy_engine = None
        self.prediction_engine = None
        self.scenario_engine = None
        
    async def initialize_system(self):
        """Initialize all system components."""
        logger.info("🚀 Initializing TL Algorithmic Trading System...")
        
        # Initialize pair manager
        self.pair_manager = PairManager()
        
        # Initialize temporal analyzer
        self.temporal_analyzer = TradingPairTemporalAnalyzer(
            d_model=256,
            sequence_length=100
        )
        
        # Initialize strategy engine
        self.strategy_engine = StrategyEngine(
            pair_manager=self.pair_manager,
            temporal_analyzer=self.temporal_analyzer
        )
        
        # Initialize prediction engine
        self.prediction_engine = TimeframePredictionEngine(d_model=256)
        
        # Initialize scenario engine
        self.scenario_engine = ScenarioAnalysisEngine()
        
        logger.info("✅ System initialization complete!")
        
    async def demo_pair_analysis(self):
        """Demonstrate comprehensive pair analysis."""
        logger.info("\n📊 DEMO: Comprehensive Pair Analysis")
        logger.info("=" * 50)
        
        # Select demo pairs
        demo_pairs = ["EURUSD=X", "BTC-USD", "SPY"]
        
        for symbol in demo_pairs:
            logger.info(f"\n🔍 Analyzing {symbol}...")
            
            try:
                # Get pair data
                pair_data = await self.pair_manager.get_pair_data(
                    symbol, timeframe='1d', period='1y'
                )
                
                if not pair_data:
                    logger.warning(f"Could not get data for {symbol}")
                    continue
                
                logger.info(f"  📈 Data quality: {pair_data.data_quality:.2f}")
                logger.info(f"  📅 Data points: {len(pair_data.data)}")
                
                # Temporal analysis
                temporal_result = self.temporal_analyzer.analyze_pair(
                    pair_data.data, symbol, "demo_analysis"
                )
                
                logger.info(f"  🧠 Temporal confidence: {temporal_result.confidence_score:.3f}")
                logger.info(f"  ✨ Emergent signals: {len(temporal_result.emergent_signals)}")
                
                # Strategy analysis
                strategy_result = await self.strategy_engine.analyze_pair(symbol)
                
                if strategy_result:
                    logger.info(f"  🎯 Overall direction: {strategy_result.overall_direction:.3f}")
                    logger.info(f"  🎯 Overall confidence: {strategy_result.overall_confidence:.3f}")
                    logger.info(f"  🤝 Strategy consensus: {strategy_result.strategy_consensus:.3f}")
                    logger.info(f"  📊 Contributing strategies: {len(strategy_result.contributing_strategies)}")
                    
                    # Prediction analysis
                    current_price = pair_data.data['close'].iloc[-1]
                    prediction = self.prediction_engine.predict(
                        temporal_result, strategy_result, current_price, pair_data.data
                    )
                    
                    logger.info(f"  🔮 Prediction outlook: {prediction.overall_outlook}")
                    logger.info(f"  🔮 Prediction confidence: {prediction.confidence_score:.3f}")
                    logger.info(f"  🚨 Key drivers: {', '.join(prediction.key_drivers[:3])}")
                    
                    # Scenario analysis
                    scenarios = self.scenario_engine.generate_scenarios(prediction, current_price)
                    
                    logger.info(f"  📋 Bull case target: ${scenarios['bull_case']['price_target']:.2f}")
                    logger.info(f"  📋 Base case target: ${scenarios['base_case']['price_target']:.2f}")
                    logger.info(f"  📋 Bear case target: ${scenarios['bear_case']['price_target']:.2f}")
                else:
                    logger.warning(f"  ❌ Strategy analysis failed for {symbol}")
                    
            except Exception as e:
                logger.error(f"  ❌ Analysis failed for {symbol}: {e}")
        
    async def demo_multi_pair_screening(self):
        """Demonstrate multi-pair opportunity screening."""
        logger.info("\n🔍 DEMO: Multi-Pair Opportunity Screening")
        logger.info("=" * 50)
        
        # Screen multiple pairs
        screening_pairs = ["EURUSD=X", "GBPUSD=X", "BTC-USD", "ETH-USD", "SPY", "QQQ"]
        
        logger.info(f"📊 Screening {len(screening_pairs)} pairs for opportunities...")
        
        results = await self.strategy_engine.analyze_multiple_pairs(
            screening_pairs, timeframe='1d', period='6m'
        )
        
        if results:
            # Get top opportunities
            top_opportunities = self.strategy_engine.get_top_opportunities(
                results, min_confidence=0.5, limit=5
            )
            
            logger.info(f"\n🏆 Top {len(top_opportunities)} Trading Opportunities:")
            logger.info("-" * 50)
            
            for i, signal in enumerate(top_opportunities, 1):
                direction_text = "🟢 BULLISH" if signal.overall_direction > 0.1 else "🔴 BEARISH" if signal.overall_direction < -0.1 else "🟡 NEUTRAL"
                
                logger.info(f"{i}. {signal.symbol}")
                logger.info(f"   Direction: {direction_text} ({signal.overall_direction:.3f})")
                logger.info(f"   Confidence: {signal.overall_confidence:.3f}")
                logger.info(f"   Consensus: {signal.strategy_consensus:.3f}")
                logger.info(f"   Strategies: {len(signal.contributing_strategies)}")
                
                if signal.temporal_analysis:
                    logger.info(f"   Temporal Conf: {signal.temporal_analysis.confidence_score:.3f}")
                
                logger.info("")
        else:
            logger.warning("No trading opportunities found")
    
    async def demo_temporal_learning(self):
        """Demonstrate temporal learning capabilities."""
        logger.info("\n🧠 DEMO: Temporal Learning Capabilities")
        logger.info("=" * 50)
        
        # Use EUR/USD for detailed temporal analysis
        symbol = "EURUSD=X"
        
        logger.info(f"🔬 Deep temporal analysis of {symbol}...")
        
        try:
            # Get extended data
            pair_data = await self.pair_manager.get_pair_data(
                symbol, timeframe='1h', period='3m'
            )
            
            if not pair_data:
                logger.warning(f"Could not get data for {symbol}")
                return
            
            # Multiple temporal analyses to show learning
            analyses = []
            
            for i in range(5):
                # Analyze different chunks of data
                chunk_size = len(pair_data.data) // 5
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size + 50  # Overlap for continuity
                
                chunk_data = pair_data.data.iloc[start_idx:end_idx]
                
                if len(chunk_data) < 50:
                    continue
                
                analysis = self.temporal_analyzer.analyze_pair(
                    chunk_data, symbol, f"temporal_learning_chunk_{i}"
                )
                analyses.append(analysis)
                
                logger.info(f"  📊 Chunk {i+1}: Confidence = {analysis.confidence_score:.3f}")
                
                # Show emergent patterns
                if analysis.emergent_signals:
                    for signal in analysis.emergent_signals:
                        logger.info(f"    ✨ {signal.get('type', 'unknown')}: {signal.get('strength', 0):.3f}")
            
            # Analyze learning progression
            if len(analyses) >= 3:
                confidence_progression = [a.confidence_score for a in analyses]
                logger.info(f"\n📈 Learning Progression:")
                logger.info(f"   Start confidence: {confidence_progression[0]:.3f}")
                logger.info(f"   End confidence: {confidence_progression[-1]:.3f}")
                
                improvement = confidence_progression[-1] - confidence_progression[0]
                if improvement > 0.05:
                    logger.info(f"   🎯 Improvement: +{improvement:.3f} (Learning detected!)")
                elif improvement < -0.05:
                    logger.info(f"   📉 Decline: {improvement:.3f} (Pattern change detected)")
                else:
                    logger.info(f"   ➡️  Stable: {improvement:.3f} (Consistent patterns)")
                    
            # Get system statistics
            stats = self.temporal_analyzer.get_system_statistics()
            logger.info(f"\n📊 System Statistics:")
            logger.info(f"   Memory consolidation: {stats.get('memory_stats', {}).get('consolidation_rate', 0):.3f}")
            logger.info(f"   Active patterns: {stats.get('emergent_behavior_stats', {}).get('total_patterns', 0)}")
            
        except Exception as e:
            logger.error(f"Temporal learning demo failed: {e}")
    
    async def demo_risk_assessment(self):
        """Demonstrate risk assessment capabilities."""
        logger.info("\n⚠️  DEMO: Risk Assessment")
        logger.info("=" * 50)
        
        # Analyze high and low volatility pairs
        test_pairs = {
            "Low Vol": "USDCHF=X",
            "High Vol": "BTC-USD"
        }
        
        for risk_type, symbol in test_pairs.items():
            logger.info(f"\n📊 {risk_type} Analysis: {symbol}")
            
            try:
                # Get data and analyze
                pair_data = await self.pair_manager.get_pair_data(symbol)
                if not pair_data:
                    continue
                
                temporal_result = self.temporal_analyzer.analyze_pair(
                    pair_data.data, symbol, f"risk_demo_{risk_type.lower().replace(' ', '_')}"
                )
                
                # Display risk factors
                risk_assessment = temporal_result.risk_assessment
                
                logger.info(f"  📈 Volatility Risk: {risk_assessment.get('volatility_risk', 0):.3f}")
                logger.info(f"  🔄 Pattern Instability: {risk_assessment.get('pattern_instability', 0):.3f}")
                logger.info(f"  ❓ Prediction Uncertainty: {risk_assessment.get('prediction_uncertainty', 0):.3f}")
                
                # Overall risk score
                overall_risk = np.mean(list(risk_assessment.values()))
                risk_level = "🟢 LOW" if overall_risk < 0.3 else "🟡 MEDIUM" if overall_risk < 0.6 else "🔴 HIGH"
                
                logger.info(f"  🎯 Overall Risk: {risk_level} ({overall_risk:.3f})")
                
            except Exception as e:
                logger.error(f"Risk assessment failed for {symbol}: {e}")
    
    async def demo_performance_tracking(self):
        """Demonstrate performance tracking."""
        logger.info("\n📈 DEMO: Performance Tracking")
        logger.info("=" * 50)
        
        # Get strategy performance
        strategy_performance = self.strategy_engine.get_strategy_performance()
        
        logger.info("📊 Strategy Performance Summary:")
        for strategy_name, performance in strategy_performance.items():
            logger.info(f"  {strategy_name.upper()}")
            logger.info(f"    Signals: {performance['signal_count']}")
            logger.info(f"    Avg Confidence: {performance['avg_confidence']:.3f}")
            logger.info(f"    Avg Strength: {performance['avg_strength']:.1f}")
            logger.info(f"    Enabled: {'✅' if performance['enabled'] else '❌'}")
            logger.info("")
        
        # Get prediction performance
        prediction_stats = self.prediction_engine.get_prediction_statistics()
        
        logger.info("🔮 Prediction Performance:")
        logger.info(f"  Total Predictions: {prediction_stats.get('total_predictions', 0)}")
        logger.info(f"  Average Confidence: {prediction_stats.get('avg_confidence', 0):.3f}")
        
        outlook_dist = prediction_stats.get('outlook_distribution', {})
        for outlook, percentage in outlook_dist.items():
            logger.info(f"  {outlook.title()}: {percentage:.1%}")
    
    def create_demo_visualization(self):
        """Create visualization of demo results."""
        logger.info("\n📊 DEMO: Creating Visualization")
        logger.info("=" * 50)
        
        try:
            # Create a sample visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('TL Algorithmic Trading System - Demo Results', fontsize=16)
            
            # Sample data for demonstration
            days = np.arange(30)
            
            # Confidence evolution
            confidence_data = 0.3 + 0.4 * np.sin(days / 5) + np.random.normal(0, 0.05, 30)
            confidence_data = np.clip(confidence_data, 0, 1)
            ax1.plot(days, confidence_data, 'b-', linewidth=2, label='Temporal Confidence')
            ax1.fill_between(days, confidence_data, alpha=0.3)
            ax1.set_title('Temporal Intelligence Confidence Evolution')
            ax1.set_xlabel('Days')
            ax1.set_ylabel('Confidence Score')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Strategy performance
            strategies = ['Technical', 'Momentum', 'Volatility', 'Sentiment']
            performance = [0.72, 0.68, 0.65, 0.58]
            colors = ['#2E8B57', '#4169E1', '#FF6347', '#FFD700']
            
            bars = ax2.bar(strategies, performance, color=colors, alpha=0.7)
            ax2.set_title('Strategy Performance Comparison')
            ax2.set_ylabel('Average Confidence')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, performance):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # Risk distribution
            risk_categories = ['Low', 'Medium', 'High']
            risk_distribution = [45, 35, 20]  # percentages
            colors_risk = ['#90EE90', '#FFD700', '#FF6B6B']
            
            wedges, texts, autotexts = ax3.pie(risk_distribution, labels=risk_categories, 
                                              colors=colors_risk, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Risk Distribution Across Analyzed Pairs')
            
            # Prediction accuracy over time
            accuracy_data = 0.6 + 0.2 * np.sin(days / 8) + np.random.normal(0, 0.03, 30)
            accuracy_data = np.clip(accuracy_data, 0.4, 0.9)
            
            ax4.plot(days, accuracy_data, 'g-', linewidth=2, marker='o', markersize=4)
            ax4.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Target Accuracy')
            ax4.fill_between(days, 0.4, accuracy_data, alpha=0.2, color='green')
            ax4.set_title('Prediction Accuracy Tracking')
            ax4.set_xlabel('Days')
            ax4.set_ylabel('Accuracy Score')
            ax4.set_ylim(0.4, 0.9)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.tight_layout()
            
            # Save the plot
            output_path = Path(__file__).parent.parent / 'data' / 'demo_results.png'
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            logger.info(f"📊 Visualization saved to: {output_path}")
            plt.show()
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
    
    async def run_full_demo(self):
        """Run the complete system demonstration."""
        logger.info("🎬 Starting TL Algorithmic Trading System Demo")
        logger.info("=" * 60)
        
        try:
            # Initialize system
            await self.initialize_system()
            
            # Run all demos
            await self.demo_pair_analysis()
            await self.demo_multi_pair_screening()
            await self.demo_temporal_learning()
            await self.demo_risk_assessment()
            await self.demo_performance_tracking()
            
            # Create visualization
            self.create_demo_visualization()
            
            logger.info("\n🎉 Demo completed successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main function to run the demo."""
    demo = TradingSystemDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    # Set up event loop for Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the demo
    asyncio.run(main())