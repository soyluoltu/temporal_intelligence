"""
Sürekli Pair Monitörü ve Alım-Satım Karar Sistemi
=================================================

Bir trading pair'ini sürekli olarak izler, temporal intelligence framework'ü
kullanarak gerçek zamanlı analizler yapar ve alım-satım kararları verir.
"""

import asyncio
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import deque

# Add the trading system to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pair_manager import PairManager, PairType
from src.core.temporal_analyzer import TradingPairTemporalAnalyzer
from src.core.strategy_engine import StrategyEngine
from src.prediction.timeframe_predictor import TimeframePredictionEngine, ScenarioAnalysisEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """Alım-satım kararı veri yapısı."""
    timestamp: datetime
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    price: float
    reasoning: List[str]
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    position_size: float  # Önerilen pozisyon büyüklüğü (%)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    temporal_signals: Dict[str, Any] = None
    strategy_breakdown: Dict[str, float] = None


@dataclass
class MarketState:
    """Piyasa durumu bilgisi."""
    timestamp: datetime
    price: float
    volatility: float
    trend: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    momentum: float
    volume_profile: str  # "HIGH", "NORMAL", "LOW"
    temporal_confidence: float
    emergent_patterns: List[Dict[str, Any]]


class ContinuousPairMonitor:
    """Sürekli pair izleme ve analiz sistemi."""
    
    def __init__(
        self,
        symbol: str = "EURUSD=X",
        update_interval: int = 300,  # 5 dakika
        analysis_window: int = 100,   # Son 100 veri noktası
        decision_threshold: float = 0.6  # Karar alma için minimum güven
    ):
        self.symbol = symbol
        self.update_interval = update_interval
        self.analysis_window = analysis_window
        self.decision_threshold = decision_threshold
        
        # Sistem bileşenleri
        self.pair_manager = None
        self.temporal_analyzer = None
        self.strategy_engine = None
        self.prediction_engine = None
        self.scenario_engine = None
        
        # Veri depolama
        self.price_history = deque(maxlen=1000)
        self.market_states = deque(maxlen=200)
        self.trading_decisions = deque(maxlen=50)
        self.analysis_history = deque(maxlen=100)
        
        # Performans takibi
        self.total_decisions = 0
        self.correct_decisions = 0
        self.portfolio_value = 10000.0  # Başlangıç sermayesi
        self.current_position = 0.0  # Mevcut pozisyon (-1 to 1)
        self.last_price = None
        
        # Monitoring durumu
        self.is_monitoring = False
        self.start_time = None
        
    async def initialize_system(self):
        """Sistem bileşenlerini başlat."""
        logger.info(f"🚀 Sürekli Pair Monitörü başlatılıyor - {self.symbol}")
        logger.info(f"📊 Güncelleme aralığı: {self.update_interval} saniye")
        logger.info(f"🎯 Karar eşiği: {self.decision_threshold}")
        
        # Pair manager'ı başlat
        self.pair_manager = PairManager()
        
        # Temporal analyzer'ı başlat
        self.temporal_analyzer = TradingPairTemporalAnalyzer(
            d_model=256,
            sequence_length=self.analysis_window
        )
        
        # Strategy engine'i başlat
        self.strategy_engine = StrategyEngine(
            pair_manager=self.pair_manager,
            temporal_analyzer=self.temporal_analyzer
        )
        
        # Prediction engine'i başlat
        self.prediction_engine = TimeframePredictionEngine(d_model=256)
        self.scenario_engine = ScenarioAnalysisEngine()
        
        # Pair'i aktif et
        if self.pair_manager.activate_pair(self.symbol):
            logger.info(f"✅ {self.symbol} pair'i aktif edildi")
        else:
            logger.warning(f"⚠️ {self.symbol} pair'i aktif edilemedi")
        
        logger.info("✅ Sistem başlatma tamamlandı!")
        
    async def collect_market_data(self) -> Optional[pd.DataFrame]:
        """Piyasa verilerini topla."""
        try:
            # Son verileri al
            pair_data = await self.pair_manager.get_pair_data(
                self.symbol, timeframe='1h', period='1mo'
            )
            
            if not pair_data or pair_data.data.empty:
                logger.error(f"❌ {self.symbol} için veri alınamadı")
                return None
            
            # Veri kalitesini kontrol et
            if pair_data.data_quality < 0.7:
                logger.warning(f"⚠️ Düşük veri kalitesi: {pair_data.data_quality:.2f}")
            
            return pair_data.data
            
        except Exception as e:
            logger.error(f"❌ Veri toplama hatası: {e}")
            return None
    
    def calculate_market_state(self, data: pd.DataFrame) -> MarketState:
        """Mevcut piyasa durumunu hesapla."""
        try:
            current_price = data['close'].iloc[-1]
            
            # Volatilite hesapla
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Trend belirle
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
            
            if current_price > sma_20 > sma_50:
                trend = "BULLISH"
            elif current_price < sma_20 < sma_50:
                trend = "BEARISH"
            else:
                trend = "SIDEWAYS"
            
            # Momentum hesapla
            momentum = data['close'].pct_change(5).iloc[-1] if len(data) >= 5 else 0
            
            # Volume profili
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio > 1.5:
                    volume_profile = "HIGH"
                elif volume_ratio < 0.7:
                    volume_profile = "LOW"
                else:
                    volume_profile = "NORMAL"
            else:
                volume_profile = "NORMAL"
            
            return MarketState(
                timestamp=datetime.now(),
                price=current_price,
                volatility=volatility,
                trend=trend,
                momentum=momentum,
                volume_profile=volume_profile,
                temporal_confidence=0.0,  # Will be updated after temporal analysis
                emergent_patterns=[]
            )
            
        except Exception as e:
            logger.error(f"❌ Piyasa durumu hesaplama hatası: {e}")
            return MarketState(
                timestamp=datetime.now(),
                price=0.0,
                volatility=0.0,
                trend="SIDEWAYS",
                momentum=0.0,
                volume_profile="NORMAL",
                temporal_confidence=0.0,
                emergent_patterns=[]
            )
    
    async def perform_comprehensive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Kapsamlı piyasa analizi yap."""
        try:
            # Temporal analiz
            temporal_result = self.temporal_analyzer.analyze_pair(
                data, self.symbol, f"continuous_monitor_{datetime.now().isoformat()}"
            )
            
            # Strategy analizi
            strategy_result = await self.strategy_engine.analyze_pair(self.symbol)
            
            # Prediction analizi
            prediction_result = None
            if strategy_result:
                current_price = data['close'].iloc[-1]
                prediction_result = self.prediction_engine.predict(
                    temporal_result, strategy_result, current_price, data
                )
            
            return {
                'temporal': temporal_result,
                'strategy': strategy_result,
                'prediction': prediction_result,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Analiz hatası: {e}")
            return {
                'temporal': None,
                'strategy': None,
                'prediction': None,
                'timestamp': datetime.now()
            }
    
    def generate_trading_decision(
        self,
        market_state: MarketState,
        analysis_results: Dict[str, Any]
    ) -> TradingDecision:
        """Analiz sonuçlarına göre alım-satım kararı ver."""
        
        try:
            current_price = market_state.price
            temporal_result = analysis_results.get('temporal')
            strategy_result = analysis_results.get('strategy')
            prediction_result = analysis_results.get('prediction')
            
            # Karar verme faktörleri
            decision_factors = {
                'direction_score': 0.0,
                'confidence_score': 0.0,
                'temporal_confidence': 0.0,
                'risk_score': 0.0,
                'momentum_score': 0.0
            }
            
            reasoning = []
            
            # Strategy sonuçlarını değerlendir
            if strategy_result:
                decision_factors['direction_score'] = strategy_result.overall_direction
                decision_factors['confidence_score'] = strategy_result.overall_confidence
                
                if abs(strategy_result.overall_direction) > 0.3:
                    direction_text = "YUKARI" if strategy_result.overall_direction > 0 else "AŞAĞI"
                    reasoning.append(f"Güçlü {direction_text} yönlü strateji sinyali ({strategy_result.overall_direction:.3f})")
                
                if strategy_result.strategy_consensus > 0.7:
                    reasoning.append(f"Yüksek strateji mutabakatı ({strategy_result.strategy_consensus:.3f})")
            
            # Temporal sonuçları değerlendir
            if temporal_result:
                decision_factors['temporal_confidence'] = temporal_result.confidence_score
                
                if temporal_result.confidence_score > 0.7:
                    reasoning.append(f"Yüksek temporal güven ({temporal_result.confidence_score:.3f})")
                
                # Emergent signals kontrolü
                for signal in temporal_result.emergent_signals:
                    if signal.get('strength', 0) > 0.7:
                        reasoning.append(f"Güçlü emergent sinyal: {signal.get('type', 'unknown')}")
                
                # Risk değerlendirmesi
                risk_assessment = temporal_result.risk_assessment
                avg_risk = np.mean(list(risk_assessment.values())) if risk_assessment else 0.5
                decision_factors['risk_score'] = avg_risk
            
            # Momentum değerlendirmesi
            decision_factors['momentum_score'] = market_state.momentum
            if abs(market_state.momentum) > 0.02:
                momentum_text = "pozitif" if market_state.momentum > 0 else "negatif"
                reasoning.append(f"Güçlü {momentum_text} momentum ({market_state.momentum:.3f})")
            
            # Prediction sonuçları
            if prediction_result:
                if prediction_result.overall_outlook != "neutral":
                    reasoning.append(f"Prediction outlook: {prediction_result.overall_outlook}")
            
            # Karar verme algoritması
            overall_direction = decision_factors['direction_score']
            overall_confidence = np.mean([
                decision_factors['confidence_score'],
                decision_factors['temporal_confidence'],
                1.0 - decision_factors['risk_score']  # Düşük risk = yüksek güven
            ])
            
            # Risk seviyesi belirleme
            risk_score = decision_factors['risk_score']
            if risk_score > 0.7:
                risk_level = "HIGH"
            elif risk_score > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Karar verme
            action = "HOLD"
            position_size = 0.0
            
            if overall_confidence >= self.decision_threshold:
                if overall_direction > 0.2:
                    action = "BUY"
                    position_size = min(0.1, overall_confidence * 0.15)  # Max %15 pozisyon
                    reasoning.append(f"ALIŞ kararı - Güven: {overall_confidence:.3f}")
                elif overall_direction < -0.2:
                    action = "SELL"
                    position_size = min(0.1, overall_confidence * 0.15)
                    reasoning.append(f"SATIŞ kararı - Güven: {overall_confidence:.3f}")
                else:
                    reasoning.append("Yeterli yön sinyali yok - BEKLE")
            else:
                reasoning.append(f"Düşük güven seviyesi ({overall_confidence:.3f}) - BEKLE")
            
            # Stop loss ve take profit hesaplama
            stop_loss = None
            take_profit = None
            
            if action in ["BUY", "SELL"]:
                volatility_buffer = market_state.volatility * 0.5
                
                if action == "BUY":
                    stop_loss = current_price * (1 - volatility_buffer)
                    take_profit = current_price * (1 + volatility_buffer * 2)
                else:  # SELL
                    stop_loss = current_price * (1 + volatility_buffer)
                    take_profit = current_price * (1 - volatility_buffer * 2)
            
            # Strategy breakdown
            strategy_breakdown = {}
            if strategy_result:
                for signal in strategy_result.contributing_strategies:
                    strategy_breakdown[signal.strategy_type.value] = signal.confidence
            
            return TradingDecision(
                timestamp=datetime.now(),
                symbol=self.symbol,
                action=action,
                confidence=overall_confidence,
                price=current_price,
                reasoning=reasoning,
                risk_level=risk_level,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                temporal_signals={
                    'confidence': decision_factors['temporal_confidence'],
                    'emergent_count': len(temporal_result.emergent_signals) if temporal_result else 0,
                    'risk_factors': list(temporal_result.risk_assessment.keys()) if temporal_result and temporal_result.risk_assessment else []
                },
                strategy_breakdown=strategy_breakdown
            )
            
        except Exception as e:
            logger.error(f"❌ Karar verme hatası: {e}")
            return TradingDecision(
                timestamp=datetime.now(),
                symbol=self.symbol,
                action="HOLD",
                confidence=0.0,
                price=current_price if 'current_price' in locals() else 0.0,
                reasoning=[f"Hata nedeniyle karar verilemedi: {str(e)}"],
                risk_level="HIGH",
                position_size=0.0
            )
    
    def execute_virtual_trade(self, decision: TradingDecision):
        """Sanal ticaret gerçekleştir (backtesting için)."""
        try:
            if decision.action == "HOLD":
                return
            
            # Position güncelleme
            if decision.action == "BUY":
                if self.current_position <= 0:  # Yeni pozisyon veya short'tan çıkış
                    self.current_position = decision.position_size
                    logger.info(f"📈 ALIŞ POZİSYONU: {decision.position_size:.3f} @ {decision.price:.5f}")
                
            elif decision.action == "SELL":
                if self.current_position >= 0:  # Yeni pozisyon veya long'dan çıkış
                    self.current_position = -decision.position_size
                    logger.info(f"📉 SATIŞ POZİSYONU: {decision.position_size:.3f} @ {decision.price:.5f}")
            
            # P&L hesaplama (basitleştirilmiş)
            if self.last_price and self.current_position != 0:
                price_change = (decision.price - self.last_price) / self.last_price
                position_pnl = self.current_position * price_change * self.portfolio_value
                self.portfolio_value += position_pnl
                
                if abs(position_pnl) > 0:
                    pnl_text = f"+${position_pnl:.2f}" if position_pnl > 0 else f"-${abs(position_pnl):.2f}"
                    logger.info(f"💰 P&L: {pnl_text}, Portfolio: ${self.portfolio_value:.2f}")
            
            self.last_price = decision.price
            
        except Exception as e:
            logger.error(f"❌ Sanal ticaret hatası: {e}")
    
    def print_decision_report(self, decision: TradingDecision, market_state: MarketState):
        """Karar raporunu yazdır."""
        
        print("\n" + "="*80)
        print(f"📊 {self.symbol} ALİM-SATIŞ KARAR RAPORU")
        print("="*80)
        
        # Temel bilgiler
        print(f"🕐 Zaman: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💲 Fiyat: {decision.price:.5f}")
        print(f"📊 Trend: {market_state.trend}")
        print(f"⚡ Momentum: {market_state.momentum:.3f}")
        print(f"📈 Volatilite: {market_state.volatility:.3f}")
        
        # Karar bilgileri
        action_emoji = "🟢" if decision.action == "BUY" else "🔴" if decision.action == "SELL" else "🟡"
        print(f"\n{action_emoji} KARAR: {decision.action}")
        print(f"🎯 Güven Seviyesi: {decision.confidence:.3f}")
        print(f"⚠️ Risk Seviyesi: {decision.risk_level}")
        
        if decision.position_size > 0:
            print(f"📏 Pozisyon Büyüklüğü: {decision.position_size:.3f} ({decision.position_size*100:.1f}%)")
        
        if decision.stop_loss:
            print(f"🛑 Stop Loss: {decision.stop_loss:.5f}")
        if decision.take_profit:
            print(f"🎯 Take Profit: {decision.take_profit:.5f}")
        
        # Temporal intelligence bilgileri
        if decision.temporal_signals:
            print(f"\n🧠 TEMPORAL INTELLIGENCE:")
            print(f"   Temporal Güven: {decision.temporal_signals.get('confidence', 0):.3f}")
            print(f"   Emergent Sinyal Sayısı: {decision.temporal_signals.get('emergent_count', 0)}")
            
            risk_factors = decision.temporal_signals.get('risk_factors', [])
            if risk_factors:
                print(f"   Risk Faktörleri: {', '.join(risk_factors)}")
        
        # Strategy breakdown
        if decision.strategy_breakdown:
            print(f"\n📈 STRATEJİ DAĞILIMI:")
            for strategy, confidence in decision.strategy_breakdown.items():
                print(f"   {strategy.capitalize()}: {confidence:.3f}")
        
        # Reasoning
        print(f"\n💭 KARAR GEREKÇESİ:")
        for i, reason in enumerate(decision.reasoning, 1):
            print(f"   {i}. {reason}")
        
        # Performans özeti
        print(f"\n💼 PERFORMANS ÖZETİ:")
        print(f"   Toplam Karar: {self.total_decisions}")
        print(f"   Mevcut Pozisyon: {self.current_position:.3f}")
        print(f"   Portfolio Değeri: ${self.portfolio_value:.2f}")
        
        if self.total_decisions > 0:
            success_rate = (self.correct_decisions / self.total_decisions) * 100
            print(f"   Başarı Oranı: {success_rate:.1f}%")
        
        print("="*80)
    
    def save_analysis_to_file(self, decision: TradingDecision, market_state: MarketState):
        """Analiz sonuçlarını dosyaya kaydet."""
        try:
            # Veri hazırlama
            analysis_data = {
                'timestamp': decision.timestamp.isoformat(),
                'symbol': decision.symbol,
                'market_state': asdict(market_state),
                'trading_decision': asdict(decision),
                'portfolio_value': self.portfolio_value,
                'current_position': self.current_position
            }
            
            # JSON dosyasına kaydet
            filename = f"trading_analysis_{self.symbol.replace('/', '_').replace('=', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
            filepath = Path(__file__).parent.parent / 'data' / filename
            
            # Dizin oluştur
            filepath.parent.mkdir(exist_ok=True)
            
            # Mevcut verileri yükle
            existing_data = []
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except:
                    existing_data = []
            
            # Yeni veriyi ekle
            existing_data.append(analysis_data)
            
            # Dosyaya kaydet
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📁 Analiz sonuçları kaydedildi: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Dosya kaydetme hatası: {e}")
    
    def create_performance_chart(self):
        """Performans grafiği oluştur."""
        try:
            if len(self.trading_decisions) < 2:
                return
            
            # Veri hazırlama
            timestamps = [d.timestamp for d in self.trading_decisions]
            prices = [d.price for d in self.trading_decisions]
            actions = [d.action for d in self.trading_decisions]
            confidences = [d.confidence for d in self.trading_decisions]
            
            # Grafik oluşturma
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
            
            # Fiyat grafiği
            ax1.plot(timestamps, prices, 'b-', linewidth=2, label='Fiyat')
            
            # Alım-satım noktalarını işaretle
            for i, (ts, price, action, conf) in enumerate(zip(timestamps, prices, actions, confidences)):
                if action == "BUY":
                    ax1.scatter(ts, price, color='green', s=100, marker='^', alpha=0.8)
                    ax1.annotate(f'BUY\n{conf:.2f}', (ts, price), xytext=(5, 10), 
                               textcoords='offset points', fontsize=8, color='green')
                elif action == "SELL":
                    ax1.scatter(ts, price, color='red', s=100, marker='v', alpha=0.8)
                    ax1.annotate(f'SELL\n{conf:.2f}', (ts, price), xytext=(5, -15), 
                               textcoords='offset points', fontsize=8, color='red')
            
            ax1.set_title(f'{self.symbol} - Fiyat ve Alım-Satım Kararları')
            ax1.set_ylabel('Fiyat')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Güven seviyesi grafiği
            colors = ['green' if a == 'BUY' else 'red' if a == 'SELL' else 'gray' for a in actions]
            ax2.bar(range(len(confidences)), confidences, color=colors, alpha=0.7)
            ax2.axhline(y=self.decision_threshold, color='orange', linestyle='--', label=f'Karar Eşiği ({self.decision_threshold})')
            ax2.set_title('Karar Güven Seviyeleri')
            ax2.set_ylabel('Güven')
            ax2.set_xlabel('Karar Numarası')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Portfolio değeri grafiği (eğer tracking varsa)
            portfolio_values = [10000]  # Başlangıç değeri
            for i in range(1, len(self.trading_decisions)):
                # Basit P&L hesabı
                prev_price = prices[i-1]
                curr_price = prices[i]
                price_change = (curr_price - prev_price) / prev_price
                
                # Pozisyon etkisi (basitleştirilmiş)
                if i > 0 and actions[i-1] == "BUY":
                    portfolio_values.append(portfolio_values[-1] * (1 + price_change * 0.1))
                elif i > 0 and actions[i-1] == "SELL":
                    portfolio_values.append(portfolio_values[-1] * (1 - price_change * 0.1))
                else:
                    portfolio_values.append(portfolio_values[-1])
            
            ax3.plot(range(len(portfolio_values)), portfolio_values, 'purple', linewidth=2)
            ax3.set_title('Portfolio Değeri Simülasyonu')
            ax3.set_ylabel('Portfolio Değeri ($)')
            ax3.set_xlabel('Karar Numarası')
            ax3.grid(True, alpha=0.3)
            
            # Grafiği kaydet
            plt.tight_layout()
            chart_path = Path(__file__).parent.parent / 'data' / f'performance_chart_{self.symbol.replace("/", "_").replace("=", "_")}.png'
            chart_path.parent.mkdir(exist_ok=True)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Performans grafiği kaydedildi: {chart_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Grafik oluşturma hatası: {e}")
    
    async def monitoring_cycle(self):
        """Ana izleme döngüsü."""
        
        while self.is_monitoring:
            try:
                logger.info(f"\n🔄 Analiz döngüsü başlıyor - {datetime.now().strftime('%H:%M:%S')}")
                
                # 1. Piyasa verilerini topla
                market_data = await self.collect_market_data()
                if market_data is None:
                    logger.warning("⚠️ Veri alınamadı, döngü atlanıyor")
                    await asyncio.sleep(self.update_interval)
                    continue
                
                # 2. Piyasa durumunu hesapla
                market_state = self.calculate_market_state(market_data)
                self.market_states.append(market_state)
                
                # 3. Kapsamlı analiz yap
                analysis_results = await self.perform_comprehensive_analysis(market_data)
                self.analysis_history.append(analysis_results)
                
                # Temporal confidence'ı güncelle
                if analysis_results.get('temporal'):
                    market_state.temporal_confidence = analysis_results['temporal'].confidence_score
                    market_state.emergent_patterns = analysis_results['temporal'].emergent_signals
                
                # 4. Trading kararı ver
                trading_decision = self.generate_trading_decision(market_state, analysis_results)
                self.trading_decisions.append(trading_decision)
                self.total_decisions += 1
                
                # 5. Sanal ticaret gerçekleştir
                self.execute_virtual_trade(trading_decision)
                
                # 6. Rapor yazdır
                self.print_decision_report(trading_decision, market_state)
                
                # 7. Verileri kaydet
                self.save_analysis_to_file(trading_decision, market_state)
                
                # 8. Her 10 karardan sonra grafik güncelle
                if self.total_decisions % 10 == 0:
                    self.create_performance_chart()
                
                # Sonraki döngü için bekle
                logger.info(f"⏰ {self.update_interval} saniye bekleniyor...")
                await asyncio.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Kullanıcı tarafından durduruldu")
                break
            except Exception as e:
                logger.error(f"❌ Monitoring döngüsü hatası: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def start_monitoring(self, duration_minutes: Optional[int] = None):
        """Sürekli izlemeyi başlat."""
        
        try:
            # Sistemi başlat
            await self.initialize_system()
            
            self.is_monitoring = True
            self.start_time = datetime.now()
            
            logger.info(f"🚀 {self.symbol} için sürekli izleme başladı!")
            if duration_minutes:
                logger.info(f"⏱️ Süre: {duration_minutes} dakika")
                end_time = self.start_time + timedelta(minutes=duration_minutes)
                logger.info(f"🏁 Bitiş zamanı: {end_time.strftime('%H:%M:%S')}")
            
            # Duration kontrolü için task oluştur
            monitoring_task = asyncio.create_task(self.monitoring_cycle())
            
            if duration_minutes:
                # Belirli bir süre sonra durdur
                await asyncio.sleep(duration_minutes * 60)
                self.is_monitoring = False
                await monitoring_task
            else:
                # Süresiz çalıştır
                await monitoring_task
            
        except Exception as e:
            logger.error(f"❌ Monitoring başlatma hatası: {e}")
        finally:
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """İzlemeyi durdur ve özet rapor oluştur."""
        
        self.is_monitoring = False
        
        logger.info("\n🏁 Sürekli izleme durduruldu")
        
        if self.start_time:
            duration = datetime.now() - self.start_time
            logger.info(f"⏱️ Toplam çalışma süresi: {duration}")
        
        # Özet rapor
        if self.total_decisions > 0:
            logger.info(f"\n📊 ÖZET RAPOR:")
            logger.info(f"   Toplam Karar: {self.total_decisions}")
            logger.info(f"   Buy Kararları: {sum(1 for d in self.trading_decisions if d.action == 'BUY')}")
            logger.info(f"   Sell Kararları: {sum(1 for d in self.trading_decisions if d.action == 'SELL')}")
            logger.info(f"   Hold Kararları: {sum(1 for d in self.trading_decisions if d.action == 'HOLD')}")
            logger.info(f"   Ortalama Güven: {np.mean([d.confidence for d in self.trading_decisions]):.3f}")
            logger.info(f"   Son Portfolio Değeri: ${self.portfolio_value:.2f}")
            
            # Final performans grafiği
            self.create_performance_chart()
        
        logger.info("✅ Monitoring tamamlandı!")


async def main():
    """Ana fonksiyon."""
    
    print("🚀 TL Algorithmic Trading System - Sürekli Pair Monitörü")
    print("=" * 60)
    
    # Parametreler
    SYMBOL = "EURUSD=X"  # İzlenecek pair
    UPDATE_INTERVAL = 180  # 3 dakika (test için kısa)
    DURATION_MINUTES = 30  # 30 dakika izle (test için)
    DECISION_THRESHOLD = 0.5  # Düşük eşik (daha fazla karar için)
    
    print(f"📊 İzlenecek Pair: {SYMBOL}")
    print(f"⏱️ Güncelleme Aralığı: {UPDATE_INTERVAL} saniye")
    print(f"🎯 Karar Eşiği: {DECISION_THRESHOLD}")
    print(f"⏰ Toplam Süre: {DURATION_MINUTES} dakika")
    print()
    
    # Monitörü oluştur ve başlat
    monitor = ContinuousPairMonitor(
        symbol=SYMBOL,
        update_interval=UPDATE_INTERVAL,
        decision_threshold=DECISION_THRESHOLD
    )
    
    try:
        await monitor.start_monitoring(duration_minutes=DURATION_MINUTES)
    except KeyboardInterrupt:
        print("\n🛑 Program kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Program hatası: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Windows uyumluluğu için event loop policy ayarla
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Programı çalıştır
    asyncio.run(main())