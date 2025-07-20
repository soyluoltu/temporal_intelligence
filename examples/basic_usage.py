"""
Temel Kullanım Örneği
====================

Zamansal Zekâ Sisteminin nasıl kullanılacağını gösteren basit örnekler.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Add the grandparent directory as well (for temporal_intelligence module)
grandparent_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, grandparent_dir)

# Direct imports without package qualification
from core.temporal_system import TemporalIntelligenceSystem
from core.emergent_behavior import ConstraintMode
from validation.model_validator import ValidationResult


def create_sample_data(batch_size: int = 32, seq_len: int = 10, d_model: int = 128):
    """
    Örnek veri oluşturur.
    """
    # Zamansal korelasyonu olan veri
    base_signal = torch.randn(batch_size, d_model)
    
    # Sekans oluştur - zamansal bağımlılık ekle
    sequences = []
    for t in range(seq_len):
        # Önceki adımın etkisini ekle
        temporal_weight = 0.8 ** t
        noise = torch.randn(batch_size, d_model) * 0.3
        
        if t == 0:
            step_data = base_signal + noise
        else:
            step_data = temporal_weight * sequences[0] + (1 - temporal_weight) * base_signal + noise
        
        sequences.append(step_data)
    
    # [batch_size, seq_len, d_model]
    data = torch.stack(sequences, dim=1)
    
    # Zamansal bilgi
    time_deltas = torch.arange(seq_len, dtype=torch.float32)
    
    return data, time_deltas


def basic_system_demo():
    """
    Temel sistem gösterimi.
    """
    print("🧠 Zamansal Zekâ Sistemi - Temel Demo")
    print("=" * 50)
    
    # Sistem parametreleri
    d_model = 128
    batch_size = 16
    seq_len = 8
    
    # Sistem oluştur
    system = TemporalIntelligenceSystem(
        d_model=d_model,
        n_heads=4,
        hebbian_hidden=64,
        learning_rate=0.01,
        validation_threshold=0.7
    )
    
    print(f"✅ Sistem oluşturuldu: d_model={d_model}")
    
    # Örnek veri
    data, time_deltas = create_sample_data(batch_size, seq_len, d_model)
    print(f"📊 Veri oluşturuldu: {data.shape}")
    
    # Sistem işleme
    print("\n🔄 Sistem işleme başladı...")
    
    results_history = []
    
    for step in range(5):
        print(f"\n--- Adım {step + 1} ---")
        
        # İşlem
        results = system(data, context=f"demo_step_{step}", time_deltas=time_deltas)
        results_history.append(results)
        
        # Sonuçları göster
        validation_result = results['validation']['result']
        validation_score = results['validation']['metrics']['overall_score']
        
        print(f"🔍 Doğrulama: {validation_result.value} (skor: {validation_score:.3f})")
        
        behavior_stats = results['behavior_analysis']
        novelty = behavior_stats.get('novelty_score', 0)
        print(f"✨ Novelty skoru: {novelty:.3f}")
        
        attention_stats = results['attention_stats']
        print(f"👁️  Ortalama entropi: {attention_stats['mean_entropy']:.3f}")
        
        memory_stats = results['memory_stats']
        print(f"💾 Bellek: ST={memory_stats['short_term_size']}, "
              f"EP={memory_stats['episodic_size']}, "
              f"SEM={memory_stats['semantic_concepts']}")
    
    # Final istatistikler
    print("\n📈 Final Sistem İstatistikleri:")
    final_stats = system.get_system_statistics()
    
    for key, value in final_stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    return system, results_history


def constraint_mode_demo():
    """
    Farklı kısıtlama modlarını test eder.
    """
    print("\n🎛️  Kısıtlama Modları Demo")
    print("=" * 50)
    
    system = TemporalIntelligenceSystem(d_model=64, hebbian_hidden=32)
    data, time_deltas = create_sample_data(8, 5, 64)
    
    modes = [ConstraintMode.CONSERVATIVE, ConstraintMode.EXPLORATORY, ConstraintMode.ADAPTIVE]
    
    for mode in modes:
        print(f"\n🔧 {mode.value.upper()} modu test ediliyor...")
        
        system.set_constraint_mode(mode)
        system.reset_system_state()
        
        accept_count = 0
        quarantine_count = 0
        reject_count = 0
        
        for i in range(10):
            results = system(data, context=f"{mode.value}_{i}")
            validation_result = results['validation']['result']
            
            if validation_result == ValidationResult.ACCEPT:
                accept_count += 1
            elif validation_result == ValidationResult.QUARANTINE:
                quarantine_count += 1
            else:
                reject_count += 1
        
        print(f"  ✅ Kabul: {accept_count}/10")
        print(f"  ⚠️  Karantina: {quarantine_count}/10") 
        print(f"  ❌ Red: {reject_count}/10")
        
        behavior_stats = system.emergent_behavior.get_behavior_statistics()
        print(f"  📊 Toplam desen: {behavior_stats['total_patterns']}")


def memory_consolidation_demo():
    """
    Bellek konsolidasyonu demonstrasyonu.
    """
    print("\n🧠 Bellek Konsolidasyonu Demo")
    print("=" * 50)
    
    system = TemporalIntelligenceSystem(d_model=64)
    
    # Çok sayıda veri işle
    for batch_idx in range(20):
        data, time_deltas = create_sample_data(4, 3, 64)
        results = system(data, context=f"batch_{batch_idx}")
        
        if batch_idx % 5 == 0:
            stats = system.memory_hierarchy.get_memory_stats()
            print(f"Batch {batch_idx}: ST={stats['short_term_size']}, "
                  f"EP={stats['episodic_size']}, SEM={stats['semantic_concepts']}")
    
    print("\n🔄 Bellek konsolidasyonu yapılıyor...")
    system.consolidate_memory()
    
    final_stats = system.memory_hierarchy.get_memory_stats()
    print(f"Final: ST={final_stats['short_term_size']}, "
          f"EP={final_stats['episodic_size']}, SEM={final_stats['semantic_concepts']}")


def temporal_pattern_learning_demo():
    """
    Zamansal desen öğrenme demonstrasyonu.
    """
    print("\n⏰ Zamansal Desen Öğrenme Demo")
    print("=" * 50)
    
    system = TemporalIntelligenceSystem(d_model=64, learning_rate=0.05)
    
    # Periyodik desen oluştur
    def create_periodic_pattern(phase: float, amplitude: float = 1.0):
        t = torch.linspace(0, 4 * np.pi, 64)
        pattern = amplitude * torch.sin(t + phase)
        return pattern.unsqueeze(0).unsqueeze(0)  # [1, 1, 64]
    
    print("📊 Periyodik desenler öğretiliyor...")
    
    phases = [0, np.pi/2, np.pi, 3*np.pi/2]
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        
        for i, phase in enumerate(phases):
            pattern = create_periodic_pattern(phase)
            time_deltas = torch.tensor([i * 0.5])
            
            results = system(pattern, context=f"pattern_{i}", time_deltas=time_deltas)
            
            novelty = results['behavior_analysis'].get('novelty_score', 0)
            validation = results['validation']['result']
            
            print(f"  Desen {i}: novelty={novelty:.3f}, validation={validation.value}")
        
        # Hebbian bağlantı güçlerini kontrol et
        connection_norm = torch.norm(system.hebbian_learner.hebbian_weights).item()
        print(f"  Bağlantı gücü normu: {connection_norm:.3f}")


def visualization_demo():
    """
    Basit görselleştirme demonstrasyonu.
    """
    print("\n📊 Görselleştirme Demo")
    print("=" * 50)
    
    system = TemporalIntelligenceSystem(d_model=32)
    
    # Veri topla
    novelty_scores = []
    validation_scores = []
    attention_entropies = []
    
    for i in range(50):
        data, time_deltas = create_sample_data(4, 3, 32)
        results = system(data, context=f"viz_{i}")
        
        novelty_scores.append(results['behavior_analysis'].get('novelty_score', 0))
        validation_scores.append(results['validation']['metrics']['overall_score'])
        attention_entropies.append(results['attention_stats']['mean_entropy'])
    
    # Basit plot
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(novelty_scores)
        plt.title('Novelty Scores')
        plt.xlabel('Step')
        plt.ylabel('Score')
        
        plt.subplot(1, 3, 2)
        plt.plot(validation_scores)
        plt.title('Validation Scores')
        plt.xlabel('Step')
        plt.ylabel('Score')
        
        plt.subplot(1, 3, 3)
        plt.plot(attention_entropies)
        plt.title('Attention Entropy')
        plt.xlabel('Step')
        plt.ylabel('Entropy')
        
        plt.tight_layout()
        plt.savefig('temporal_intelligence_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📈 Grafik 'temporal_intelligence_demo.png' olarak kaydedildi")
        
    except Exception as e:
        print(f"⚠️  Görselleştirme hatası: {e}")
        print("Matplotlib yüklü değil olabilir")


if __name__ == "__main__":
    print("🚀 Zamansal Zekâ Sistemi Demonstrasyonu")
    print("=" * 60)
    
    try:
        # Ana demo
        system, history = basic_system_demo()
        
        # Kısıtlama modları
        constraint_mode_demo()
        
        # Bellek konsolidasyonu  
        memory_consolidation_demo()
        
        # Zamansal desen öğrenme
        temporal_pattern_learning_demo()
        
        # Görselleştirme
        visualization_demo()
        
        print("\n✅ Tüm demonstrasyonlar tamamlandı!")
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()