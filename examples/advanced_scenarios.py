"""
Gelişmiş Senaryolar
==================

Zamansal Zekâ Sisteminin karmaşık görevlerdeki performansını test eden örnekler.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time
from typing import List, Dict, Any

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Add the grandparent directory as well
grandparent_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, grandparent_dir)

from core.temporal_system import TemporalIntelligenceSystem
from core.emergent_behavior import ConstraintMode
from validation.model_validator import ValidationResult


class SequenceClassificationTask:
    """
    Zamansal sekans sınıflandırma görevi.
    """
    
    def __init__(self, d_model: int = 128, num_classes: int = 3):
        self.d_model = d_model
        self.num_classes = num_classes
        
    def generate_sequence_data(self, batch_size: int = 32, seq_len: int = 15):
        """
        Farklı zamansal desenlere sahip sekanslar üretir.
        """
        sequences = []
        labels = []
        
        for _ in range(batch_size):
            # Rastgele sınıf seç
            class_id = np.random.randint(0, self.num_classes)
            
            if class_id == 0:
                # Artan trend
                base_seq = torch.linspace(0, 1, seq_len).unsqueeze(-1)
                noise = torch.randn(seq_len, self.d_model - 1) * 0.1
                seq = torch.cat([base_seq, noise], dim=-1)
                
            elif class_id == 1:
                # Periyodik desen
                t = torch.linspace(0, 4 * np.pi, seq_len)
                base_seq = torch.sin(t).unsqueeze(-1)
                noise = torch.randn(seq_len, self.d_model - 1) * 0.1
                seq = torch.cat([base_seq, noise], dim=-1)
                
            else:
                # Rastgele (kontrol grubu)
                seq = torch.randn(seq_len, self.d_model) * 0.5
            
            sequences.append(seq)
            labels.append(class_id)
        
        return torch.stack(sequences), torch.tensor(labels)


class ContinualLearningScenario:
    """
    Sürekli öğrenme senaryosu - felaket unutmayı test eder.
    """
    
    def __init__(self, system: TemporalIntelligenceSystem):
        self.system = system
        self.task_history = []
        
    def create_task_data(self, task_id: int, d_model: int):
        """
        Görev spesifik veri oluşturur.
        """
        # Her görev farklı özellik alanı kullanır
        task_signature = torch.zeros(d_model)
        start_idx = (task_id * d_model // 4) % d_model
        end_idx = min(start_idx + d_model // 4, d_model)
        task_signature[start_idx:end_idx] = 1.0
        
        # Bu imzayı taşıyan sekanslar
        batch_size = 16
        seq_len = 8
        
        sequences = []
        for _ in range(batch_size):
            seq = torch.randn(seq_len, d_model) * 0.3
            # Görev imzasını ekle
            seq = seq + task_signature.unsqueeze(0) * 0.7
            sequences.append(seq)
        
        return torch.stack(sequences)
    
    def run_continual_learning(self, num_tasks: int = 5, steps_per_task: int = 10):
        """
        Sürekli öğrenme protokolünü çalıştırır.
        """
        print(f"🔄 Sürekli öğrenme: {num_tasks} görev, görev başına {steps_per_task} adım")
        
        results = {
            'task_performances': [],
            'memory_stats_over_time': [],
            'validation_trends': []
        }
        
        for task_id in range(num_tasks):
            print(f"\n📚 Görev {task_id + 1} öğreniliyor...")
            
            # Görev verisi oluştur
            task_data = self.create_task_data(task_id, self.system.d_model)
            
            task_validations = []
            
            # Görev üzerinde eğitim
            for step in range(steps_per_task):
                system_results = self.system(
                    task_data, 
                    context=f"task_{task_id}_step_{step}"
                )
                
                validation_score = system_results['validation']['metrics']['overall_score']
                task_validations.append(validation_score)
            
            # Bu görevin ortalama performansı
            avg_performance = np.mean(task_validations)
            results['task_performances'].append(avg_performance)
            results['validation_trends'].append(task_validations)
            
            print(f"  Ortalama doğrulama skoru: {avg_performance:.3f}")
            
            # Bellek durumu
            memory_stats = self.system.get_system_statistics()['memory_stats']
            results['memory_stats_over_time'].append(memory_stats)
            
            # Bellek konsolidasyonu
            if task_id % 2 == 1:
                print("  🧠 Bellek konsolidasyonu yapılıyor...")
                self.system.consolidate_memory()
        
        # Geçmiş görevleri hatırlama testi
        print("\n🔍 Geçmiş görev hatırlama testi...")
        self._test_task_recall(num_tasks)
        
        return results
    
    def _test_task_recall(self, num_tasks: int):
        """
        Geçmiş görevlerin ne kadar hatırlandığını test eder.
        """
        recall_scores = []
        
        for task_id in range(num_tasks):
            # Eski görev verisini yeniden oluştur
            test_data = self.create_task_data(task_id, self.system.d_model)
            
            # Sistem tepkisini test et
            results = self.system(test_data, context=f"recall_test_task_{task_id}")
            
            # Bellek sisteminden ne kadar benzer veri bulabiliyor?
            memory_results = self.system.memory_hierarchy.retrieve(
                query=test_data.mean(dim=(0, 1)),
                memory_types=["episodic", "semantic"]
            )
            
            episodic_matches = len(memory_results.get('episodic', []))
            semantic_matches = len(memory_results.get('semantic', []))
            
            recall_score = (episodic_matches + semantic_matches) / 10  # Normalize
            recall_scores.append(recall_score)
            
            print(f"  Görev {task_id + 1}: hatırlama skoru = {recall_score:.3f}")
        
        avg_recall = np.mean(recall_scores)
        print(f"\n📊 Ortalama hatırlama skoru: {avg_recall:.3f}")
        
        return recall_scores


class EmergentBehaviorExperiment:
    """
    Ortaya çıkan davranış deneyi.
    """
    
    def __init__(self, system: TemporalIntelligenceSystem):
        self.system = system
        
    def create_novelty_inducing_data(self, phase: int, d_model: int):
        """
        Novelty oluşturacak veri yaratır.
        """
        if phase == 0:
            # Normal veri
            return torch.randn(8, 5, d_model) * 0.5
            
        elif phase == 1:
            # Desen değişimi
            data = torch.randn(8, 5, d_model) * 0.5
            # Belirli boyutlarda güçlü sinyal
            data[:, :, :d_model//4] *= 3.0
            return data
            
        elif phase == 2:
            # Tamamen yeni desen
            data = torch.zeros(8, 5, d_model)
            # Sinüzoidal desen
            for i in range(5):
                t = torch.linspace(0, 2*np.pi, d_model)
                data[:, i, :] = torch.sin(t + i * np.pi/4).unsqueeze(0)
            return data
            
        else:
            # Hibrit desen
            normal_data = torch.randn(8, 5, d_model) * 0.3
            novel_data = torch.sin(torch.linspace(0, 4*np.pi, d_model)).unsqueeze(0).unsqueeze(0)
            return normal_data + novel_data
    
    def run_emergence_experiment(self, phases: int = 4, steps_per_phase: int = 15):
        """
        Ortaya çıkan davranış deneyini çalıştırır.
        """
        print(f"✨ Ortaya çıkan davranış deneyi: {phases} faz, faz başına {steps_per_phase} adım")
        
        results = {
            'novelty_evolution': [],
            'pattern_discovery': [],
            'quarantine_events': [],
            'validation_changes': []
        }
        
        for phase in range(phases):
            print(f"\n🔬 Faz {phase + 1} başladı...")
            
            phase_novelty = []
            phase_validations = []
            
            for step in range(steps_per_phase):
                # Faza özgü veri
                data = self.create_novelty_inducing_data(phase, self.system.d_model)
                
                # Sistem işleme
                system_results = self.system(
                    data, 
                    context=f"emergence_phase_{phase}_step_{step}"
                )
                
                # Novelty ve validation takibi
                novelty = system_results['behavior_analysis'].get('novelty_score', 0)
                validation_score = system_results['validation']['metrics']['overall_score']
                
                phase_novelty.append(novelty)
                phase_validations.append(validation_score)
                
                # Karantina olayları
                quarantine_decision = system_results['behavior_analysis']['quarantine_decision']
                if quarantine_decision['action'] in ['quarantine', 'validate', 'reject']:
                    results['quarantine_events'].append({
                        'phase': phase,
                        'step': step,
                        'action': quarantine_decision['action'],
                        'reason': quarantine_decision['reason']
                    })
                    print(f"  📋 Karantina olayı: {quarantine_decision['action']}")
            
            # Faz istatistikleri
            avg_novelty = np.mean(phase_novelty)
            avg_validation = np.mean(phase_validations)
            
            results['novelty_evolution'].append(phase_novelty)
            results['validation_changes'].append(phase_validations)
            
            print(f"  Ortalama novelty: {avg_novelty:.3f}")
            print(f"  Ortalama validation: {avg_validation:.3f}")
            
            # Ortaya çıkan desen sayısı
            behavior_stats = self.system.emergent_behavior.get_behavior_statistics()
            pattern_count = behavior_stats['total_patterns']
            results['pattern_discovery'].append(pattern_count)
            
            print(f"  Toplam keşfedilen desen: {pattern_count}")
        
        # Final analiz
        self._analyze_emergence_results(results)
        
        return results
    
    def _analyze_emergence_results(self, results: Dict[str, Any]):
        """
        Ortaya çıkan davranış sonuçlarını analiz eder.
        """
        print("\n📊 Ortaya Çıkan Davranış Analizi:")
        
        # Novelty trend
        all_novelty = [score for phase in results['novelty_evolution'] for score in phase]
        novelty_trend = np.polyfit(range(len(all_novelty)), all_novelty, 1)[0]
        print(f"  Novelty trendi: {novelty_trend:.4f} (pozitif = artıyor)")
        
        # Desen keşif hızı
        pattern_counts = results['pattern_discovery']
        if len(pattern_counts) > 1:
            discovery_rate = pattern_counts[-1] - pattern_counts[0]
            print(f"  Desen keşif hızı: {discovery_rate} desen/faz")
        
        # Karantina aktivitesi
        quarantine_events = results['quarantine_events']
        action_counts = {}
        for event in quarantine_events:
            action = event['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"  Karantina aktivitesi: {action_counts}")
        
        # Validasyon kararlılığı
        all_validations = [score for phase in results['validation_changes'] for score in phase]
        validation_std = np.std(all_validations)
        print(f"  Validation kararlılığı (düşük std = kararlı): {validation_std:.3f}")


class PerformanceBenchmark:
    """
    Performans karşılaştırma testi.
    """
    
    def __init__(self):
        self.results = {}
    
    def benchmark_system_sizes(self, d_models: List[int] = [64, 128, 256]):
        """
        Farklı sistem boyutlarını benchmark eder.
        """
        print("⚡ Sistem boyutu performans testi")
        
        for d_model in d_models:
            print(f"\n🔧 d_model = {d_model} test ediliyor...")
            
            system = TemporalIntelligenceSystem(
                d_model=d_model,
                hebbian_hidden=d_model // 2
            )
            
            # Veri oluştur
            data = torch.randn(16, 8, d_model)
            
            # Zaman ölçümü
            start_time = time.time()
            
            # Birden fazla işlem
            for i in range(10):
                results = system(data, context=f"benchmark_{i}")
            
            end_time = time.time()
            
            # İstatistikler
            total_time = end_time - start_time
            avg_time_per_step = total_time / 10
            
            final_stats = system.get_system_statistics()
            
            self.results[d_model] = {
                'total_time': total_time,
                'avg_time_per_step': avg_time_per_step,
                'memory_usage': final_stats['memory_stats'],
                'validation_distribution': final_stats.get('recent_validation_distribution', {}),
                'hebbian_norm': final_stats['hebbian_stats']['connection_matrix_norm']
            }
            
            print(f"  ⏱️  Ortalama adım süresi: {avg_time_per_step:.4f}s")
            print(f"  💾 Bellek kullanımı: {final_stats['memory_stats']}")
    
    def compare_constraint_modes(self):
        """
        Kısıtlama modlarının performansını karşılaştırır.
        """
        print("\n🎛️  Kısıtlama modu karşılaştırması")
        
        modes = [ConstraintMode.CONSERVATIVE, ConstraintMode.EXPLORATORY, ConstraintMode.ADAPTIVE]
        mode_results = {}
        
        for mode in modes:
            print(f"\n🔧 {mode.value} modu test ediliyor...")
            
            system = TemporalIntelligenceSystem(d_model=128)
            system.set_constraint_mode(mode)
            
            # Test verisi
            novel_data = torch.randn(8, 5, 128) * 2.0  # Yüksek novelty
            
            accept_count = 0
            total_novelty = 0
            
            for i in range(20):
                results = system(novel_data, context=f"{mode.value}_{i}")
                
                if results['validation']['result'] == ValidationResult.ACCEPT:
                    accept_count += 1
                
                total_novelty += results['behavior_analysis'].get('novelty_score', 0)
            
            mode_results[mode.value] = {
                'acceptance_rate': accept_count / 20,
                'avg_novelty': total_novelty / 20,
                'final_patterns': system.emergent_behavior.get_behavior_statistics()['total_patterns']
            }
            
            print(f"  ✅ Kabul oranı: {accept_count}/20")
            print(f"  ✨ Ortalama novelty: {total_novelty/20:.3f}")
        
        # Sonuçları karşılaştır
        print("\n📊 Mod Karşılaştırması:")
        for mode, stats in mode_results.items():
            print(f"  {mode.upper()}:")
            print(f"    Kabul oranı: {stats['acceptance_rate']:.2f}")
            print(f"    Ortalama novelty: {stats['avg_novelty']:.3f}")
            print(f"    Keşfedilen desen: {stats['final_patterns']}")
        
        return mode_results
    
    def print_benchmark_summary(self):
        """
        Benchmark özetini yazdırır.
        """
        print("\n📈 PERFORMANS ÖZETİ")
        print("=" * 50)
        
        if self.results:
            print("Sistem Boyutu Performansı:")
            for d_model, stats in self.results.items():
                print(f"  d_model={d_model}: {stats['avg_time_per_step']:.4f}s/adım")


def main():
    """
    Tüm gelişmiş senaryoları çalıştırır.
    """
    print("🚀 Zamansal Zekâ Sistemi - Gelişmiş Senaryolar")
    print("=" * 60)
    
    # 1. Sürekli öğrenme testi
    print("\n1️⃣  Sürekli Öğrenme Senaryosu")
    system1 = TemporalIntelligenceSystem(d_model=128, learning_rate=0.02)
    continual_scenario = ContinualLearningScenario(system1)
    continual_results = continual_scenario.run_continual_learning(num_tasks=4, steps_per_task=8)
    
    # 2. Ortaya çıkan davranış deneyi
    print("\n2️⃣  Ortaya Çıkan Davranış Deneyi")
    system2 = TemporalIntelligenceSystem(d_model=96, validation_threshold=0.6)
    emergence_experiment = EmergentBehaviorExperiment(system2)
    emergence_results = emergence_experiment.run_emergence_experiment(phases=3, steps_per_phase=12)
    
    # 3. Performans benchmark
    print("\n3️⃣  Performans Benchmark")
    benchmark = PerformanceBenchmark()
    benchmark.benchmark_system_sizes([64, 128, 192])
    mode_comparison = benchmark.compare_constraint_modes()
    benchmark.print_benchmark_summary()
    
    print("\n✅ Tüm gelişmiş senaryolar tamamlandı!")
    return {
        'continual_learning': continual_results,
        'emergence_experiment': emergence_results,
        'performance_benchmark': benchmark.results,
        'mode_comparison': mode_comparison
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\n🎯 Tüm testler başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()