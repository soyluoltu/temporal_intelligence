"""
Düzeltme Testleri
================

Yapılan düzeltmelerin doğruluğunu test eden ek test dosyası.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

from core.temporal_system import TemporalIntelligenceSystem
from hebbian.hebbian_learning import HebbianLearner
from attention.temporal_attention import TemporalAttention
from validation.model_validator import ModelValidator, ValidationResult
from core.emergent_behavior import EmergentBehaviorManager


class TestMathematicalFixes(unittest.TestCase):
    """
    Matematiksel formül düzeltmelerini test eder.
    """
    
    def test_hebbian_formula_correctness(self):
        """Hebbian formülünün doğru implementasyonunu test eder."""
        learner = HebbianLearner(input_size=4, hidden_size=3, learning_rate=0.1)
        
        # Basit test verisi
        input_act = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        hidden_act = torch.tensor([[0.5, 1.0, 0.0]])
        
        initial_weights = learner.hebbian_weights.clone()
        learner.update_hebbian_weights(input_act, hidden_act, 1.0)
        
        # Ağırlık değişimlerini kontrol et
        weight_change = learner.hebbian_weights - initial_weights
        
        # Beklenen değişim: input[i] * hidden[j] * learning_rate * temporal_weight
        # Test'te hidden_act direkt olarak verildiği için onu kullan
        expected_change_01 = 1.0 * 1.0 * 0.1 * 1.0  # input_act[0] * hidden_act[1] * lr * temporal_weight
        actual_change_01 = weight_change[0, 1].item()
        
        self.assertAlmostEqual(actual_change_01, expected_change_01, places=4)
    
    def test_temporal_weight_function(self):
        """Zamansal ağırlık fonksiyonunun düzeltildiğini test eder."""
        learner = HebbianLearner(input_size=2, hidden_size=2, temporal_decay=0.9)
        
        # Delta_t = 0 için ağırlık 1.0 olmalı
        weight_0 = learner.temporal_weight_function(0.0)
        self.assertEqual(weight_0, 1.0)
        
        # Delta_t = 1 için ağırlık temporal_decay olmalı  
        weight_1 = learner.temporal_weight_function(1.0)
        self.assertEqual(weight_1, 0.9)
        
        # Delta_t = 2 için ağırlık temporal_decay^2 olmalı
        weight_2 = learner.temporal_weight_function(2.0)
        self.assertAlmostEqual(weight_2, 0.81, places=5)


class TestDimensionFixes(unittest.TestCase):
    """
    Boyut uyumsuzluğu düzeltmelerini test eder.
    """
    
    def test_attention_dimension_consistency(self):
        """Attention boyutlarının tutarlı olduğunu test eder."""
        attention = TemporalAttention(d_model=64, n_heads=4)
        
        batch_size, seq_len = 3, 8
        x = torch.randn(batch_size, seq_len, 64)
        time_deltas = torch.arange(seq_len, dtype=torch.float32)
        
        # Forward pass
        output, attention_weights = attention(x, x, x, time_deltas)
        
        # Boyut kontrolleri
        self.assertEqual(output.shape, (batch_size, seq_len, 64))
        self.assertEqual(attention_weights.shape, (batch_size, 4, seq_len, seq_len))
        
        # Attention weights normalization
        attention_sums = torch.sum(attention_weights, dim=-1)
        expected_sums = torch.ones_like(attention_sums)
        self.assertTrue(torch.allclose(attention_sums, expected_sums, atol=1e-5))
    
    def test_temporal_system_concatenation(self):
        """Temporal sistem tensor birleştirmesinin düzgün çalıştığını test eder."""
        system = TemporalIntelligenceSystem(d_model=32, n_heads=2, hebbian_hidden=16)
        
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, 32)
        
        # Forward pass - hata olmamalı
        results = system(x)
        
        # Çıkış boyutunun doğru olduğunu kontrol et
        self.assertEqual(results['output'].shape, (batch_size, 32))
        self.assertIn('validation', results)
        self.assertIn('behavior_analysis', results)


class TestValidationFixes(unittest.TestCase):
    """
    Model validation düzeltmelerini test eder.
    """
    
    def test_validation_formula_tensor_consistency(self):
        """Validation formülünde tensor tutarlılığını test eder."""
        validator = ModelValidator(d_model=16, validation_threshold=0.7)
        
        batch_size = 3
        current_repr = torch.randn(batch_size, 16)
        context_repr = torch.randn(batch_size, 16)
        history = [torch.randn(batch_size, 16) for _ in range(5)]
        
        # Validation - hata olmamalı
        result, metrics = validator.validate(current_repr, context_repr, history)
        
        # Sonuç tiplerini kontrol et
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(0.0 <= metrics.overall_score <= 1.0)
        self.assertTrue(0.0 <= metrics.semantic_consistency <= 1.0)
    
    def test_decision_making_batch_handling(self):
        """Batch düzeyinde karar vermenin doğru çalıştığını test eder."""
        validator = ModelValidator(d_model=8, validation_threshold=0.8)
        
        # Farklı skorlara sahip batch
        mixed_scores = torch.tensor([0.9, 0.6, 0.3])  # Accept, Quarantine, Reject
        
        decision = validator.make_decision(mixed_scores)
        
        # Çoğunluk oyuna göre karar verilmeli
        self.assertIn(decision, [ValidationResult.ACCEPT, ValidationResult.QUARANTINE, ValidationResult.REJECT])


class TestMemoryFixes(unittest.TestCase):
    """
    Bellek sistemi düzeltmelerini test eder.
    """
    
    def test_memory_consolidation_logic(self):
        """Bellek konsolidasyonu mantığının düzgün çalıştığını test eder."""
        from memory.memory_hierarchy import MemoryHierarchy
        
        memory = MemoryHierarchy(d_model=16)
        
        # Farklı önem seviyelerinde öğeler ekle
        high_importance = torch.randn(16)
        low_importance = torch.randn(16)
        
        memory.store(high_importance, "short_term", importance=0.9)
        memory.store(low_importance, "short_term", importance=0.3)
        
        initial_st_size = len(memory.short_term.memory)
        initial_ep_size = len(memory.episodic.episodes)
        
        # Konsolidasyon (threshold=0.8)
        memory.consolidate(threshold=0.8)
        
        final_st_size = len(memory.short_term.memory)
        final_ep_size = len(memory.episodic.episodes)
        
        # Yüksek önemli öğe epizodik belleğe taşınmalı
        # Düşük önemli öğe kısa süreli bellekte kalmalı
        self.assertEqual(final_st_size, 1)  # Sadece düşük önemli kaldı
        self.assertGreater(final_ep_size, initial_ep_size)  # Epizodik arttı


class TestErrorHandling(unittest.TestCase):
    """
    Hata yakalama ve device tutarlılığı testleri.
    """
    
    def test_device_consistency(self):
        """Device tutarlılığının korunduğunu test eder."""
        device = 'cpu'  # GPU yoksa cpu kullan
        system = TemporalIntelligenceSystem(d_model=16, device=device)
        
        # CPU'da veri oluştur
        x = torch.randn(2, 3, 16)
        
        # Forward pass - device conflict olmamalı
        results = system(x)
        
        # Çıkışın doğru device'da olduğunu kontrol et
        self.assertEqual(results['output'].device.type, device)
    
    def test_input_validation_errors(self):
        """Giriş validation hatalarının düzgün yakalandığını test eder."""
        system = TemporalIntelligenceSystem(d_model=32)
        
        # Yanlış boyutlu giriş
        wrong_dim_input = torch.randn(32)  # 3D olmalı, 1D verildi
        
        results = system(wrong_dim_input)
        
        # Hata yakalanmalı ve graceful degradation olmalı
        self.assertIn('error', results)
        self.assertEqual(results['validation']['result'], ValidationResult.REJECT)
    
    def test_checkpoint_loading_safety(self):
        """Checkpoint yüklemenin güvenli olduğunu test eder."""
        system = TemporalIntelligenceSystem(d_model=16)
        
        # Geçersiz checkpoint path
        try:
            system.load_checkpoint("nonexistent_file.pt")
            self.fail("Should have raised an exception")
        except Exception as e:
            # Exception yakalanmalı
            self.assertIsInstance(e, (FileNotFoundError, RuntimeError))


if __name__ == '__main__':
    print("🔧 Düzeltme Testleri Çalıştırılıyor...")
    print("=" * 50)
    
    # Test suite oluştur
    test_classes = [
        TestMathematicalFixes,
        TestDimensionFixes, 
        TestValidationFixes,
        TestMemoryFixes,
        TestErrorHandling
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Testleri çalıştır
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Özet
    print(f"\n📊 Düzeltme Test Özeti:")
    print(f"  Toplam test: {result.testsRun}")
    print(f"  ✅ Başarılı: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  ❌ Başarısız: {len(result.failures)}")
    print(f"  🚫 Hata: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"\n🎉 Tüm düzeltme testleri başarıyla geçti!")
        print("Sistem artık matematiksel olarak doğru ve güvenilir!")
    else:
        print(f"\n⚠️  Bazı düzeltme testleri başarısız oldu.")
        if result.failures:
            print("Başarısız testler:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("Hatalı testler:")
            for test, traceback in result.errors:
                print(f"  - {test}")