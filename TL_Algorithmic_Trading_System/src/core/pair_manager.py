"""
Pair Manager for Trading System
=============================

Manages trading pairs, data collection, and pair-specific configurations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import aiohttp
import yfinance as yf
from pathlib import Path
import json
import logging


class PairType(Enum):
    FOREX = "forex"
    CRYPTO = "crypto"
    STOCKS = "stocks"
    COMMODITIES = "commodities"


@dataclass
class PairInfo:
    """Information about a trading pair."""
    symbol: str
    pair_type: PairType
    base_asset: str
    quote_asset: str
    description: str
    min_tick: float = 0.0001
    lot_size: float = 1.0
    margin_requirement: float = 0.02
    trading_hours: Optional[Dict[str, str]] = None
    

@dataclass
class PairData:
    """Container for pair market data."""
    symbol: str
    timeframe: str
    data: pd.DataFrame
    last_update: datetime
    data_quality: float = 1.0
    

class PairDataCollector:
    """Collects market data from various sources."""
    
    def __init__(self):
        self.data_sources = {
            'yfinance': self._collect_yfinance_data,
            'alpha_vantage': self._collect_alpha_vantage_data,
            'binance': self._collect_binance_data
        }
        self.cache = {}
        
    async def collect_pair_data(
        self,
        symbol: str,
        timeframe: str = '1d',
        period: str = '1y',
        source: str = 'yfinance'
    ) -> PairData:
        """Collect data for a trading pair."""
        
        cache_key = f"{symbol}_{timeframe}_{period}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if (datetime.now() - cached_data.last_update).seconds < 3600:  # 1 hour cache
                return cached_data
        
        # Collect fresh data
        if source in self.data_sources:
            data_df = await self.data_sources[source](symbol, timeframe, period)
        else:
            data_df = await self._collect_yfinance_data(symbol, timeframe, period)
        
        # Create PairData object
        pair_data = PairData(
            symbol=symbol,
            timeframe=timeframe,
            data=data_df,
            last_update=datetime.now(),
            data_quality=self._assess_data_quality(data_df)
        )
        
        # Cache the data
        self.cache[cache_key] = pair_data
        
        return pair_data
    
    async def _collect_yfinance_data(
        self,
        symbol: str,
        timeframe: str,
        period: str
    ) -> pd.DataFrame:
        """Collect data using yfinance."""
        try:
            # Convert timeframe to yfinance format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk', '1M': '1mo'
            }
            
            yf_interval = interval_map.get(timeframe, '1d')
            
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=yf_interval)
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    if col == 'volume':
                        data[col] = 0
                    else:
                        data[col] = data['close'] if 'close' in data.columns else 0
            
            return data[required_cols]
            
        except Exception as e:
            logging.error(f"Error collecting yfinance data for {symbol}: {e}")
            return self._create_dummy_data()
    
    async def _collect_alpha_vantage_data(
        self,
        symbol: str,
        timeframe: str,
        period: str
    ) -> pd.DataFrame:
        """Collect data using Alpha Vantage API."""
        # Placeholder for Alpha Vantage implementation
        return await self._collect_yfinance_data(symbol, timeframe, period)
    
    async def _collect_binance_data(
        self,
        symbol: str,
        timeframe: str,
        period: str
    ) -> pd.DataFrame:
        """Collect crypto data using Binance API."""
        # Placeholder for Binance implementation
        return await self._collect_yfinance_data(symbol, timeframe, period)
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess the quality of collected data."""
        if data.empty:
            return 0.0
        
        quality_score = 1.0
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        quality_score -= missing_ratio * 0.5
        
        # Check for zero volume (if applicable)
        if 'volume' in data.columns:
            zero_volume_ratio = (data['volume'] == 0).sum() / len(data)
            quality_score -= zero_volume_ratio * 0.2
        
        # Check for unusual price gaps
        if 'close' in data.columns:
            price_changes = data['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.2).sum()  # >20% change
            if len(data) > 0:
                quality_score -= (extreme_changes / len(data)) * 0.3
        
        return max(0.0, min(1.0, quality_score))
    
    def _create_dummy_data(self) -> pd.DataFrame:
        """Create dummy data for testing purposes."""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        # Generate synthetic price data
        np.random.seed(42)
        price = 100.0
        prices = []
        
        for _ in dates:
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            price *= (1 + change)
            prices.append(price)
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(10, 1, len(dates))
        }, index=dates)
        
        return data


class PairManager:
    """Main pair management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.pairs_registry: Dict[str, PairInfo] = {}
        self.data_collector = PairDataCollector()
        self.active_pairs: List[str] = []
        
        # Load configuration
        if config_path:
            self.load_pairs_config(config_path)
        else:
            self._initialize_default_pairs()
    
    def load_pairs_config(self, config_path: str):
        """Load pairs configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            for pair_data in config.get('pairs', []):
                pair_info = PairInfo(
                    symbol=pair_data['symbol'],
                    pair_type=PairType(pair_data['type']),
                    base_asset=pair_data['base_asset'],
                    quote_asset=pair_data['quote_asset'],
                    description=pair_data.get('description', ''),
                    min_tick=pair_data.get('min_tick', 0.0001),
                    lot_size=pair_data.get('lot_size', 1.0),
                    margin_requirement=pair_data.get('margin_requirement', 0.02)
                )
                self.register_pair(pair_info)
                
        except Exception as e:
            logging.error(f"Error loading pairs config: {e}")
            self._initialize_default_pairs()
    
    def _initialize_default_pairs(self):
        """Initialize with default trading pairs."""
        default_pairs = [
            # Forex majors
            PairInfo('EURUSD=X', PairType.FOREX, 'EUR', 'USD', 'Euro/US Dollar'),
            PairInfo('GBPUSD=X', PairType.FOREX, 'GBP', 'USD', 'British Pound/US Dollar'),
            PairInfo('USDJPY=X', PairType.FOREX, 'USD', 'JPY', 'US Dollar/Japanese Yen'),
            
            # Crypto majors
            PairInfo('BTC-USD', PairType.CRYPTO, 'BTC', 'USD', 'Bitcoin/US Dollar'),
            PairInfo('ETH-USD', PairType.CRYPTO, 'ETH', 'USD', 'Ethereum/US Dollar'),
            
            # Stock pairs
            PairInfo('SPY', PairType.STOCKS, 'SPY', 'USD', 'S&P 500 ETF'),
            PairInfo('QQQ', PairType.STOCKS, 'QQQ', 'USD', 'NASDAQ ETF'),
            
            # Commodities
            PairInfo('GC=F', PairType.COMMODITIES, 'GOLD', 'USD', 'Gold Futures'),
            PairInfo('CL=F', PairType.COMMODITIES, 'OIL', 'USD', 'Crude Oil Futures')
        ]
        
        for pair in default_pairs:
            self.register_pair(pair)
    
    def register_pair(self, pair_info: PairInfo):
        """Register a new trading pair."""
        self.pairs_registry[pair_info.symbol] = pair_info
        logging.info(f"Registered pair: {pair_info.symbol} ({pair_info.description})")
    
    def get_pair_info(self, symbol: str) -> Optional[PairInfo]:
        """Get pair information."""
        return self.pairs_registry.get(symbol)
    
    def list_pairs(self, pair_type: Optional[PairType] = None) -> List[PairInfo]:
        """List available pairs, optionally filtered by type."""
        if pair_type:
            return [pair for pair in self.pairs_registry.values() if pair.pair_type == pair_type]
        return list(self.pairs_registry.values())
    
    async def get_pair_data(
        self,
        symbol: str,
        timeframe: str = '1d',
        period: str = '1y'
    ) -> Optional[PairData]:
        """Get market data for a trading pair."""
        if symbol not in self.pairs_registry:
            logging.error(f"Pair {symbol} not registered")
            return None
        
        try:
            pair_data = await self.data_collector.collect_pair_data(
                symbol, timeframe, period
            )
            return pair_data
        except Exception as e:
            logging.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def activate_pair(self, symbol: str) -> bool:
        """Activate a pair for monitoring."""
        if symbol in self.pairs_registry and symbol not in self.active_pairs:
            self.active_pairs.append(symbol)
            logging.info(f"Activated pair: {symbol}")
            return True
        return False
    
    def deactivate_pair(self, symbol: str) -> bool:
        """Deactivate a pair."""
        if symbol in self.active_pairs:
            self.active_pairs.remove(symbol)
            logging.info(f"Deactivated pair: {symbol}")
            return True
        return False
    
    def get_active_pairs(self) -> List[str]:
        """Get list of active pairs."""
        return self.active_pairs.copy()
    
    async def update_all_active_pairs(
        self,
        timeframe: str = '1d',
        period: str = '1y'
    ) -> Dict[str, PairData]:
        """Update data for all active pairs."""
        updated_data = {}
        
        for symbol in self.active_pairs:
            pair_data = await self.get_pair_data(symbol, timeframe, period)
            if pair_data:
                updated_data[symbol] = pair_data
        
        logging.info(f"Updated data for {len(updated_data)} active pairs")
        return updated_data
    
    def validate_pair_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Validate a pair symbol format."""
        if not symbol:
            return False, "Empty symbol"
        
        if len(symbol) < 3:
            return False, "Symbol too short"
        
        # Basic validation - could be expanded
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-=.')
        if not all(c in valid_chars for c in symbol.upper()):
            return False, "Invalid characters in symbol"
        
        return True, "Valid symbol"
    
    def get_pair_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed pairs."""
        stats = {
            'total_pairs': len(self.pairs_registry),
            'active_pairs': len(self.active_pairs),
            'pair_types': {},
            'data_quality': {}
        }
        
        # Count by type
        for pair in self.pairs_registry.values():
            pair_type = pair.pair_type.value
            stats['pair_types'][pair_type] = stats['pair_types'].get(pair_type, 0) + 1
        
        # Data quality from cache
        for symbol, cached_data in self.data_collector.cache.items():
            stats['data_quality'][symbol] = cached_data.data_quality
        
        return stats
    
    def save_pairs_config(self, config_path: str):
        """Save current pairs configuration to file."""
        config = {
            'pairs': [
                {
                    'symbol': pair.symbol,
                    'type': pair.pair_type.value,
                    'base_asset': pair.base_asset,
                    'quote_asset': pair.quote_asset,
                    'description': pair.description,
                    'min_tick': pair.min_tick,
                    'lot_size': pair.lot_size,
                    'margin_requirement': pair.margin_requirement
                }
                for pair in self.pairs_registry.values()
            ],
            'active_pairs': self.active_pairs
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"Saved pairs configuration to {config_path}")


class PairCorrelationAnalyzer:
    """Analyzes correlations between trading pairs."""
    
    def __init__(self, pair_manager: PairManager):
        self.pair_manager = pair_manager
        self.correlation_cache = {}
    
    async def calculate_correlation_matrix(
        self,
        symbols: List[str],
        timeframe: str = '1d',
        period: str = '6m'
    ) -> pd.DataFrame:
        """Calculate correlation matrix for given pairs."""
        
        # Collect data for all pairs
        pair_data = {}
        for symbol in symbols:
            data = await self.pair_manager.get_pair_data(symbol, timeframe, period)
            if data and not data.data.empty:
                pair_data[symbol] = data.data['close']
        
        if len(pair_data) < 2:
            return pd.DataFrame()
        
        # Create DataFrame with all close prices
        price_df = pd.DataFrame(pair_data)
        
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Cache the result
        cache_key = f"{'_'.join(sorted(symbols))}_{timeframe}_{period}"
        self.correlation_cache[cache_key] = {
            'matrix': correlation_matrix,
            'timestamp': datetime.now()
        }
        
        return correlation_matrix
    
    def get_highly_correlated_pairs(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """Find pairs with high correlation."""
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                symbol1 = correlation_matrix.columns[i]
                symbol2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append((symbol1, symbol2, corr_value))
        
        return sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    
    def find_diversification_opportunities(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """Find pairs with low correlation for diversification."""
        low_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                symbol1 = correlation_matrix.columns[i]
                symbol2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) <= threshold:
                    low_corr_pairs.append((symbol1, symbol2, corr_value))
        
        return sorted(low_corr_pairs, key=lambda x: abs(x[2]))