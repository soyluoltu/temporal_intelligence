#!/usr/bin/env python3
"""
Quick Fix Test
==============

Memory tensor boyut problemlerini test eden script.
"""

import sys
from pathlib import Path
import torch

# Add current directory to path
current_dir = str(Path(__file__).parent)
sys.path.insert(0, current_dir)

print("🔧 Memory Tensor Boyut Fix Testi")
print("=" * 40)

try:
    from memory.memory_hierarchy import MemoryHierarchy
    
    print("1. MemoryHierarchy oluşturuluyor...")
    memory = MemoryHierarchy(d_model=32)
    
    print("2. Test query oluşturuluyor...")
    # Batch boyutlu query (problematik)
    query_batch = torch.randn(4, 32)  # [batch_size, d_model]
    
    print("3. Memory retrieve testi...")
    results = memory.retrieve(query_batch, memory_types=["semantic"])
    print(f"   ✅ Semantic memory retrieve başarılı: {len(results['semantic'])} sonuç")
    
    print("4. Kısa süreli bellek testi...")
    # Önce bir şey ekle
    test_item = torch.randn(32)
    memory.store(test_item, "short_term", importance=0.8)
    
    results = memory.retrieve(query_batch, memory_types=["short_term"])
    print(f"   ✅ Short-term memory retrieve başarılı: {len(results['short_term'])} sonuç")
    
    print("5. Epizodik bellek testi...")
    memory.store(test_item, "episodic", context="test", importance=0.9)
    
    results = memory.retrieve(query_batch, memory_types=["episodic"])
    print(f"   ✅ Episodic memory retrieve başarılı: {len(results['episodic'])} sonuç")
    
    print("6. Tam sistem testi...")
    from core.temporal_system import TemporalIntelligenceSystem
    
    system = TemporalIntelligenceSystem(d_model=32, n_heads=2, hebbian_hidden=16)
    x = torch.randn(2, 4, 32)
    
    print("   Sistem forward pass test...")
    results = system(x)
    print(f"   ✅ Sistem başarılı! Çıkış: {results['output'].shape}")
    
    print("\n🎉 Tüm memory tensor boyut sorunları çözüldü!")
    
except Exception as e:
    print(f"\n❌ Hata: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 40)