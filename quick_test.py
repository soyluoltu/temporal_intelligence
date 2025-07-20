#!/usr/bin/env python3
"""
Quick Test Script
================

Import problemlerini test etmek için hızlı test scripti.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = str(Path(__file__).parent)
sys.path.insert(0, current_dir)

print("🧪 Import Test Başlatılıyor...")
print("=" * 40)

try:
    print("1. HebbianLearner import ediliyor...")
    from hebbian.hebbian_learning import HebbianLearner
    print("   ✅ HebbianLearner başarılı")
    
    print("2. TemporalAttention import ediliyor...")
    from attention.temporal_attention import TemporalAttention
    print("   ✅ TemporalAttention başarılı")
    
    print("3. MemoryHierarchy import ediliyor...")
    from memory.memory_hierarchy import MemoryHierarchy
    print("   ✅ MemoryHierarchy başarılı")
    
    print("4. ModelValidator import ediliyor...")
    from validation.model_validator import ModelValidator, ValidationResult
    print("   ✅ ModelValidator başarılı")
    
    print("5. EmergentBehaviorManager import ediliyor...")
    from core.emergent_behavior import EmergentBehaviorManager
    print("   ✅ EmergentBehaviorManager başarılı")
    
    print("6. TemporalIntelligenceSystem import ediliyor...")
    from core.temporal_system import TemporalIntelligenceSystem
    print("   ✅ TemporalIntelligenceSystem başarılı")
    
    print("\n🎉 Tüm import'lar başarılı!")
    
    # Basit test
    print("\n🔬 Basit fonksiyonalite testi...")
    import torch
    
    system = TemporalIntelligenceSystem(d_model=32, n_heads=2, hebbian_hidden=16)
    x = torch.randn(2, 4, 32)
    
    print("   Sistem oluşturuldu...")
    
    results = system(x)
    print("   Forward pass başarılı...")
    print(f"   Çıkış boyutu: {results['output'].shape}")
    print(f"   Validation sonucu: {results['validation']['result']}")
    
    print("\n✅ Tüm testler geçti! Sistem çalışır durumda.")
    
except ImportError as e:
    print(f"\n❌ Import hatası: {e}")
    print("\nMevcut Python path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
except Exception as e:
    print(f"\n❌ Genel hata: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 40)
print("Test tamamlandı!")