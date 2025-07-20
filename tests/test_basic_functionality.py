"""
Temel Fonksiyonalite Testleri
============================

Zamansal Zekâ Sisteminin temel bileşenlerini test eden birim testler.
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
from memory.memory_hierarchy import MemoryHierarchy
from validation.model_validator import ModelValidator, ValidationResult
from core.emergent_behavior import EmergentBehaviorManager


class TestHebbianLearning(unittest.TestCase):
    """
    Hebbian öğrenme mekanizması testleri.
    """
    
    def setUp(self):
        self.input_size = 64
        self.hidden_size = 32
        self.learner = HebbianLearner(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            learning_rate=0.01
        )
    
    def test_initialization(self):
        """İlklendirme testi."""
        self.assertEqual(self.learner.input_size, self.input_size)
        self.assertEqual(self.learner.hidden_size, self.hidden_size)
        self.assertEqual(self.learner.hebbian_weights.shape, (self.input_size, self.hidden_size))
    
    def test_forward_pass(self):
        """İleri besleme testi."""
        x = torch.randn(1, self.input_size)
        hidden, weights = self.learner(x)
        
        self.assertEqual(hidden.shape, (1, self.hidden_size))
        self.assertEqual(weights.shape, (self.input_size, self.hidden_size))
        
        # Çıkış tanh aktivasyonunda olmalı
        self.assertTrue(torch.all(hidden >= -1) and torch.all(hidden <= 1))
    
    def test_temporal_weight_function(self):
        """Zamansal ağırlık fonksiyonu testi."""
        weight_0 = self.learner.temporal_weight_function(0.0)
        weight_1 = self.learner.temporal_weight_function(1.0)
        weight_2 = self.learner.temporal_weight_function(2.0)
        
        # Zaman arttıkça ağırlık azalmalı
        self.assertGreater(weight_0, weight_1)
        self.assertGreater(weight_1, weight_2)
        self.assertEqual(weight_0, 1.0)  # t=0'da ağırlık 1 olmalı
    
    def test_weight_update(self):
        """Ağırlık güncelleme testi."""
        x = torch.randn(1, self.input_size)
        
        # İlk ağırlıkları sakla
        initial_weights = self.learner.hebbian_weights.clone()
        
        # İleri besleme yap (ağırlıkları günceller)
        hidden, weights = self.learner(x)
        
        # Ağırlıkların değiştiğini kontrol et (tolerance artırıldı)
        self.assertFalse(torch.allclose(initial_weights, self.learner.hebbian_weights, atol=1e-6))
    
    def test_activation_correlation(self):
        """Aktivasyon korelasyonu testi."""
        # Birkaç örnek işle
        for _ in range(5):
            x = torch.randn(1, self.input_size)
            self.learner(x)
        
        correlation = self.learner.get_activation_correlation()
        self.assertEqual(correlation.shape, (self.hidden_size, self.hidden_size))
        
        # Diagonal elementler 1'e yakın olmalı
        diagonal = torch.diag(correlation)
        self.assertTrue(torch.all(diagonal >= 0.9))


class TestTemporalAttention(unittest.TestCase):
    """
    Zamansal dikkat mekanizması testleri.
    """
    
    def setUp(self):
        self.d_model = 128
        self.n_heads = 4
        self.attention = TemporalAttention(
            d_model=self.d_model,
            n_heads=self.n_heads
        )
    
    def test_initialization(self):
        """İlklendirme testi."""
        self.assertEqual(self.attention.d_model, self.d_model)
        self.assertEqual(self.attention.n_heads, self.n_heads)
        self.assertEqual(self.attention.d_k, self.d_model // self.n_heads)
    
    def test_forward_pass(self):
        """İleri besleme testi."""
        batch_size, seq_len = 4, 8
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        output, attention_weights = self.attention(x, x, x)
        
        # Çıkış boyutları
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))
        self.assertEqual(attention_weights.shape, (batch_size, self.n_heads, seq_len, seq_len))
        
        # Attention weights normalizasyonu
        attention_sums = torch.sum(attention_weights, dim=-1)
        self.assertTrue(torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-6))
    
    def test_temporal_bias(self):
        """Zamansal önyargı testi."""
        seq_len = 6
        batch_size = 2
        temporal_bias = self.attention.compute_temporal_bias(seq_len, batch_size)
        
        # Güncellenen şekil: [batch_size, n_heads, seq_len, seq_len]
        self.assertEqual(temporal_bias.shape, (batch_size, self.n_heads, seq_len, seq_len))
        
        # Zamansal yakınlık: diagonal elementler en yüksek olmalı (ilk batch, ilk head)
        bias_sample = temporal_bias[0, 0]  # [seq_len, seq_len]
        diagonal = torch.diag(bias_sample)
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    self.assertGreaterEqual(diagonal[i].item(), bias_sample[i, j].item())
    
    def test_attention_statistics(self):
        """Dikkat istatistikleri testi."""
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        _, attention_weights = self.attention(x, x, x)
        stats = self.attention.get_attention_statistics(attention_weights)
        
        self.assertIn('mean_entropy', stats)
        self.assertIn('mean_max_attention', stats)
        self.assertIn('attention_sparsity', stats)
        
        # Entropy pozitif olmalı
        self.assertGreater(stats['mean_entropy'], 0)


class TestMemoryHierarchy(unittest.TestCase):
    """
    Bellek hiyerarşisi testleri.
    """
    
    def setUp(self):
        self.d_model = 64
        self.memory = MemoryHierarchy(d_model=self.d_model)
    
    def test_short_term_memory(self):
        """Kısa süreli bellek testi."""
        item = torch.randn(self.d_model)
        
        # Belleğe kaydet
        self.memory.store(item, memory_type="short_term", importance=0.8)
        
        # Geri getir
        query = item + torch.randn(self.d_model) * 0.1  # Biraz gürültülü sorgu
        results = self.memory.retrieve(query, memory_types=["short_term"])
        
        self.assertIn("short_term", results)
        self.assertGreater(len(results["short_term"]), 0)
    
    def test_episodic_memory(self):
        """Epizodik bellek testi."""
        item = torch.randn(self.d_model)
        context = "test_context"
        
        # Epizot kaydet
        self.memory.store(item, memory_type="episodic", context=context, importance=0.9)
        
        # Bağlam ile geri getir
        results = self.memory.retrieve(item, memory_types=["episodic"], context=context)
        
        self.assertIn("episodic", results)
        self.assertGreater(len(results["episodic"]), 0)
        self.assertEqual(results["episodic"][0].context_id, context)
    
    def test_semantic_memory(self):
        """Anlamsal bellek testi."""
        concept_embedding = torch.randn(self.d_model)
        concept_name = "test_concept"
        
        # Kavram kaydet
        concept_id = self.memory.semantic.store_concept(concept_embedding, concept_name)
        
        # Kavram geri getir
        results = self.memory.semantic.retrieve_concept(concept_embedding, top_k=1)
        
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], concept_id)  # Concept ID eşleşmeli
        self.assertEqual(results[0][1], concept_name)  # Concept name eşleşmeli
    
    def test_memory_consolidation(self):
        """Bellek konsolidasyonu testi."""
        # Kısa süreli belleğe yüksek önem ile kaydet
        high_importance_item = torch.randn(self.d_model)
        self.memory.store(high_importance_item, memory_type="short_term", importance=0.9)
        
        initial_st_size = len(self.memory.short_term.memory)
        initial_ep_size = len(self.memory.episodic.episodes)
        
        # Konsolidasyon
        self.memory.consolidate(threshold=0.8)
        
        final_st_size = len(self.memory.short_term.memory)
        final_ep_size = len(self.memory.episodic.episodes)
        
        # Kısa süreli bellek temizlenmeli
        self.assertEqual(final_st_size, 0)
        
        # Epizodik bellek artmalı (yüksek önem nedeniyle)
        self.assertGreater(final_ep_size, initial_ep_size)


class TestModelValidator(unittest.TestCase):
    """
    Model doğrulayıcı testleri.
    """
    
    def setUp(self):
        self.d_model = 32
        self.validator = ModelValidator(d_model=self.d_model, validation_threshold=0.7)
    
    def test_semantic_consistency(self):
        """Anlamsal tutarlılık testi."""
        # Benzer vektörler
        current = torch.randn(1, self.d_model)
        context = current + torch.randn(1, self.d_model) * 0.1
        
        consistency = self.validator.compute_semantic_consistency(current, context)
        
        self.assertEqual(consistency.shape, (1,))
        self.assertTrue(0.0 <= consistency.item() <= 1.0)
        
        # Tamamen aynı vektörler için yüksek tutarlılık
        same_consistency = self.validator.compute_semantic_consistency(current, current)
        self.assertGreater(same_consistency.item(), consistency.item())
    
    def test_logical_coherence(self):
        """Mantıksal tutarlılık testi."""
        representation = torch.randn(1, self.d_model)
        coherence = self.validator.compute_logical_coherence(representation)
        
        self.assertEqual(coherence.shape, (1,))
        self.assertTrue(0.0 <= coherence.item() <= 1.0)
    
    def test_validation_decision(self):
        """Doğrulama kararı testi."""
        current = torch.randn(1, self.d_model)
        context = torch.randn(1, self.d_model)
        history = [torch.randn(1, self.d_model) for _ in range(3)]
        
        result, metrics = self.validator.validate(current, context, history)
        
        # Sonuç enum değerlerinden biri olmalı
        self.assertIn(result, [ValidationResult.ACCEPT, ValidationResult.QUARANTINE, ValidationResult.REJECT])
        
        # Metrikler doğru aralıklarda olmalı
        self.assertTrue(0.0 <= metrics.semantic_consistency <= 1.0)
        self.assertTrue(0.0 <= metrics.logical_coherence <= 1.0)
        self.assertTrue(0.0 <= metrics.temporal_stability <= 1.0)


class TestEmergentBehavior(unittest.TestCase):
    """
    Ortaya çıkan davranış yöneticisi testleri.
    """
    
    def setUp(self):
        self.d_model = 32
        self.manager = EmergentBehaviorManager(d_model=self.d_model)
    
    def test_novelty_detection(self):
        """Novelty tespiti testi."""
        representation = torch.randn(self.d_model)
        novelty = self.manager._compute_novelty(representation)
        
        self.assertTrue(0.0 <= novelty <= 1.0)
    
    def test_pattern_detection(self):
        """Desen tespiti testi."""
        representation = torch.randn(self.d_model)
        
        # İlk desen
        pattern1 = self.manager._check_for_new_pattern(representation, 0.8, 1.0)
        self.assertIsNotNone(pattern1)
        self.assertEqual(pattern1.occurrence_count, 1)
        
        # Aynı desen tekrar
        pattern2 = self.manager._check_for_new_pattern(representation, 0.8, 2.0)
        self.assertEqual(pattern2.pattern_id, pattern1.pattern_id)
        self.assertEqual(pattern2.occurrence_count, 2)
    
    def test_quarantine_protocol(self):
        """Karantina protokolü testi."""
        representation = torch.randn(self.d_model)
        
        # Yeni desen oluştur
        pattern = self.manager._check_for_new_pattern(representation, 0.9, 1.0)
        
        # Karantina protokolünü uygula
        decision = self.manager._apply_quarantine_protocol(pattern)
        
        self.assertEqual(decision['action'], 'quarantine')
        self.assertEqual(pattern.validation_status, 'quarantine')


class TestTemporalIntelligenceSystem(unittest.TestCase):
    """
    Ana sistem testleri.
    """
    
    def setUp(self):
        self.d_model = 64
        self.system = TemporalIntelligenceSystem(
            d_model=self.d_model,
            n_heads=2,
            hebbian_hidden=32
        )
    
    def test_system_initialization(self):
        """Sistem ilklendirme testi."""
        self.assertEqual(self.system.d_model, self.d_model)
        self.assertIsNotNone(self.system.hebbian_learner)
        self.assertIsNotNone(self.system.temporal_attention)
        self.assertIsNotNone(self.system.memory_hierarchy)
        self.assertIsNotNone(self.system.model_validator)
        self.assertIsNotNone(self.system.emergent_behavior)
    
    def test_forward_pass(self):
        """Sistem ileri beslemesi testi."""
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        results = self.system(x)
        
        # Temel çıkışlar mevcut olmalı
        self.assertIn('output', results)
        self.assertIn('attention_weights', results)
        self.assertIn('validation', results)
        self.assertIn('behavior_analysis', results)
        
        # Çıkış boyutu doğru olmalı
        self.assertEqual(results['output'].shape, (batch_size, self.d_model))
    
    def test_memory_integration(self):
        """Bellek entegrasyonu testi."""
        x = torch.randn(1, 3, self.d_model)
        
        # Birkaç adım işle
        for i in range(5):
            results = self.system(x, context=f"test_{i}")
        
        # Bellek istatistikleri
        memory_stats = self.system.memory_hierarchy.get_memory_stats()
        
        # Kısa süreli bellekte veri olmalı
        self.assertGreater(memory_stats['short_term_size'], 0)
    
    def test_system_statistics(self):
        """Sistem istatistikleri testi."""
        x = torch.randn(1, 2, self.d_model)
        
        # Birkaç adım
        for _ in range(3):
            self.system(x)
        
        stats = self.system.get_system_statistics()
        
        # İstatistik kategorileri mevcut olmalı
        self.assertIn('processing_steps', stats)
        self.assertIn('memory_stats', stats)
        self.assertIn('validation_stats', stats)
        self.assertIn('behavior_stats', stats)
        
        # İşlem adımı sayısı doğru olmalı
        self.assertEqual(stats['processing_steps'], 3)


if __name__ == '__main__':
    print("🧪 Zamansal Zekâ Sistemi - Birim Testleri")
    print("=" * 50)
    
    # Test suite oluştur
    test_classes = [
        TestHebbianLearning,
        TestTemporalAttention,
        TestMemoryHierarchy,
        TestModelValidator,
        TestEmergentBehavior,
        TestTemporalIntelligenceSystem
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
    print(f"\n📊 Test Özeti:")
    print(f"  Toplam test: {result.testsRun}")
    print(f"  ✅ Başarılı: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  ❌ Başarısız: {len(result.failures)}")
    print(f"  🚫 Hata: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Başarısız Testler:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\n🚫 Hatalı Testler:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Başarı durumu
    if result.wasSuccessful():
        print(f"\n🎉 Tüm testler başarıyla geçti!")
    else:
        print(f"\n⚠️  Bazı testler başarısız oldu.")