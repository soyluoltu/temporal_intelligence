"""
Temporal Intelligence Framework
==============================

Bu paket, zamansal farkındalık ile donatılmış yapay zekâ sistemleri için
Hebbian öğrenme ve dikkat mekanizmalarını birleştiren bir çerçeve sunar.

Makaleden ilham alınarak: "Sinirsel Sistemlerde Zamansal Zekâya Doğru"
"""

__version__ = "0.1.0"
__author__ = "Temporal Intelligence Research"

from .core.temporal_system import TemporalIntelligenceSystem
from .hebbian.hebbian_learning import HebbianLearner
from .attention.temporal_attention import TemporalAttention
from .memory.memory_hierarchy import MemoryHierarchy
from .validation.model_validator import ModelValidator

__all__ = [
    'TemporalIntelligenceSystem',
    'HebbianLearner', 
    'TemporalAttention',
    'MemoryHierarchy',
    'ModelValidator'
]