#!/usr/bin/env python3
"""
İyileştirme Doğrulama Testi
==========================

Threshold düzeltmelerinin etkisini test eder.
"""

import sys
from pathlib import Path
import torch

# Add current directory to path
current_dir = str(Path(__file__).parent)
sys.path.insert(0, current_dir)

print("🔧 Threshold İyileştirmeleri Test Raporu")
print("=" * 50)

try:
    from core.temporal_system import TemporalIntelligenceSystem
    from core.emergent_behavior import ConstraintMode
    
    # Test sistemi oluştur
    system = TemporalIntelligenceSystem(
        d_model=64, n_heads=4, hebbian_hidden=32
    )
    
    print("✅ Sistem oluşturuldu")
    print(f"📊 Varsayılan validation threshold: {system.model_validator.validation_threshold}")
    print(f"📊 Varsayılan novelty threshold: {system.emergent_behavior.novelty_threshold}")
    
    # Test verisi
    test_data = torch.randn(4, 6, 64)
    
    print("\n🔄 5 Adımlık Test...")
    accept_count = 0
    quarantine_count = 0
    reject_count = 0
    novelty_scores = []
    
    for i in range(5):
        results = system(test_data)
        validation_result = results['validation']['result']
        novelty_score = results['behavior_analysis']['novelty_score']
        
        novelty_scores.append(novelty_score)
        
        if validation_result.value == 'accept':
            accept_count += 1
        elif validation_result.value == 'quarantine':
            quarantine_count += 1
        else:
            reject_count += 1
            
        print(f"  Adım {i+1}: {validation_result.value}, novelty={novelty_score:.3f}")
    
    print(f"\n📊 Sonuçlar:")
    print(f"  ✅ Accept: {accept_count}/5 ({accept_count*20}%)")
    print(f"  ⚠️  Quarantine: {quarantine_count}/5 ({quarantine_count*20}%)")
    print(f"  ❌ Reject: {reject_count}/5 ({reject_count*20}%)")
    print(f"  📈 Ortalama novelty: {sum(novelty_scores)/len(novelty_scores):.3f}")
    
    # Bellek konsolidasyonu testi
    print(f"\n🧠 Bellek konsolidasyonu öncesi:")
    stats_before = system.memory_hierarchy.get_memory_stats()
    print(f"  ST: {stats_before['short_term_size']}, EP: {stats_before['episodic_size']}")
    
    system.consolidate_memory()
    
    stats_after = system.memory_hierarchy.get_memory_stats()
    print(f"🧠 Bellek konsolidasyonu sonrası:")
    print(f"  ST: {stats_after['short_term_size']}, EP: {stats_after['episodic_size']}")
    
    # Constraint mode testi
    print(f"\n🎛️  Constraint Mode Testi:")
    
    for mode in [ConstraintMode.CONSERVATIVE, ConstraintMode.EXPLORATORY, ConstraintMode.ADAPTIVE]:
        system.set_constraint_mode(mode)
        results = system(test_data)
        validation_result = results['validation']['result']
        novelty_score = results['behavior_analysis']['novelty_score']
        patterns = results['behavior_analysis'].get('pattern_info', None)
        
        print(f"  {mode.value}: {validation_result.value}, novelty={novelty_score:.3f}")
        if patterns:
            print(f"    Pattern ID: {patterns.pattern_id}")
    
    print(f"\n🎉 İyileştirme testleri tamamlandı!")
    
    # İyileştirme özeti
    improvement_score = accept_count + quarantine_count * 0.5
    print(f"\n📈 İyileştirme Skoru: {improvement_score:.1f}/5.0")
    
    if improvement_score >= 2.0:
        print("✅ Sistem başarıyla iyileştirildi!")
    elif improvement_score >= 1.0:
        print("⚠️  Kısmi iyileştirme - daha fazla ayar gerekebilir")
    else:
        print("❌ İyileştirme yetersiz - parametreleri gözden geçirin")
        
except Exception as e:
    print(f"\n❌ Test hatası: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)