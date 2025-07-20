"""
Ana Zamansal Zekâ Sistemi
=========================

Tüm bileşenleri birleştiren ana sistem sınıfı.
Makaledeki Hebbian-Dikkat-Doğrulama çerçevesini uygular.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import time

# Flexible import system
try:
    # Try relative imports first (when used as package)
    from ..hebbian.hebbian_learning import HebbianLearner
    from ..attention.temporal_attention import TemporalAttention
    from ..memory.memory_hierarchy import MemoryHierarchy
    from ..validation.model_validator import ModelValidator, ValidationResult, ValidationMetrics
    from .emergent_behavior import EmergentBehaviorManager, ConstraintMode
except ImportError:
    try:
        # Try temporal_intelligence package imports
        from temporal_intelligence.hebbian.hebbian_learning import HebbianLearner
        from temporal_intelligence.attention.temporal_attention import TemporalAttention
        from temporal_intelligence.memory.memory_hierarchy import MemoryHierarchy
        from temporal_intelligence.validation.model_validator import ModelValidator, ValidationResult, ValidationMetrics
        from temporal_intelligence.core.emergent_behavior import EmergentBehaviorManager, ConstraintMode
    except ImportError:
        # Direct imports (when run from temporal_intelligence directory)
        from hebbian.hebbian_learning import HebbianLearner
        from attention.temporal_attention import TemporalAttention
        from memory.memory_hierarchy import MemoryHierarchy
        from validation.model_validator import ModelValidator, ValidationResult, ValidationMetrics
        from core.emergent_behavior import EmergentBehaviorManager, ConstraintMode


class TemporalIntelligenceSystem(nn.Module):
    """
    Zamansal farkındalıklı yapay zekâ sistemi.
    
    Bu sistem şu bileşenleri entegre eder:
    - Hebbian öğrenme mekanizması
    - Zamansal dikkat
    - Bellek hiyerarşisi (kısa-orta-uzun vadeli)
    - İçsel model doğrulama
    - Ortaya çıkan davranış yönetimi
    """
    
    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 hebbian_hidden: int = 256,
                 learning_rate: float = 0.01,
                 validation_threshold: float = 0.4,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.d_model = d_model
        self.device = device
        
        # Ana bileşenler
        self.hebbian_learner = HebbianLearner(
            input_size=d_model,
            hidden_size=hebbian_hidden,
            learning_rate=learning_rate,
            device=device
        )
        
        self.temporal_attention = TemporalAttention(
            d_model=d_model,
            n_heads=n_heads,
            device=device
        )
        
        self.memory_hierarchy = MemoryHierarchy(d_model=d_model)
        
        self.model_validator = ModelValidator(
            d_model=d_model,
            validation_threshold=validation_threshold,
            device=device
        )
        
        self.emergent_behavior = EmergentBehaviorManager(d_model=d_model)
        
        # Giriş ve çıkış projeksiyon katmanları
        self.input_projection = nn.Linear(d_model, d_model, device=device)
        self.output_projection = nn.Linear(hebbian_hidden + d_model, d_model, device=device)
        
        # Zamansal kodlama
        self.temporal_encoding = nn.Parameter(torch.randn(1000, d_model, device=device) * 0.1)
        
        # Sistem durumu
        self.representation_history = []
        self.validation_history = []
        self.step_count = 0
        
    def forward(self, 
                x: torch.Tensor,
                context: Optional[str] = None,
                time_deltas: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Ana ileri besleme fonksiyonu.
        
        Args:
            x: [batch_size, seq_len, d_model] giriş tensörü
            context: İsteğe bağlı bağlam bilgisi
            time_deltas: Zamansal bilgi
            
        Returns:
            Sistem çıkışları ve iç durumlar
        """
        
        # Device ve boyut kontrolleri
        try:
            # Tensor'i doğru device'a taşı
            x = x.to(self.device)
            if time_deltas is not None:
                time_deltas = time_deltas.to(self.device)
            
            # Boyut kontrolleri
            if x.dim() != 3:
                raise ValueError(f"Input tensor must be 3D [batch, seq_len, d_model], got {x.dim()}D")
            
            batch_size, seq_len, input_dim = x.size()
            
            if input_dim != self.d_model:
                raise ValueError(f"Input dimension {input_dim} doesn't match model dimension {self.d_model}")
            
            context = context or "default"
            current_time = time.time()
            
            # 1. Giriş projeksionu ve zamansal kodlama
            x_proj = self.input_projection(x)
            
        except Exception as e:
            # Safe batch_size extraction for error handling
            try:
                batch_size = x.size(0) if x.dim() > 0 else 1
            except:
                batch_size = 1
                
            return {
                'error': f"Input processing failed: {str(e)}",
                'output': torch.zeros(batch_size, self.d_model, device=self.device),
                'validation': {'result': ValidationResult.REJECT, 'metrics': {}},
                'system_state': {'step': self.step_count, 'error': True}
            }
        
        # Zamansal pozisyon kodlama ekle - sekans pozisyonuna göre
        for seq_pos in range(seq_len):
            if seq_pos < len(self.temporal_encoding):
                temporal_code = self.temporal_encoding[seq_pos].unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
                x_proj[:, seq_pos:seq_pos+1, :] = x_proj[:, seq_pos:seq_pos+1, :] + temporal_code
        
        # 2. Bellek sistemi - ilgili bilgileri getir
        memory_context = self._retrieve_memory_context(x_proj.mean(dim=1))  # [batch_size, d_model]
        
        # 3. Zamansal dikkat mekanizması
        attended_output, attention_weights = self.temporal_attention(
            query=x_proj,
            key=x_proj, 
            value=x_proj,
            time_deltas=time_deltas
        )
        
        # 4. Hebbian öğrenme - sekans bilgisini koruyarak
        attended_mean = attended_output.mean(dim=1)  # [batch_size, d_model]
        hebbian_output, hebbian_weights = self.hebbian_learner(attended_mean)
        
        # 5. Çıkış kombinasyonu - boyut kontrollü
        # attended_mean: [batch_size, d_model]
        # hebbian_output: [batch_size, hebbian_hidden]
        
        # Boyut kontrolü
        assert attended_mean.size(-1) == self.d_model, f"Expected d_model {self.d_model}, got {attended_mean.size(-1)}"
        assert hebbian_output.size(-1) == self.hebbian_learner.hidden_size, f"Expected hebbian_hidden {self.hebbian_learner.hidden_size}, got {hebbian_output.size(-1)}"
        
        combined_features = torch.cat([
            attended_mean,  # [batch_size, d_model]
            hebbian_output  # [batch_size, hebbian_hidden]
        ], dim=-1)  # [batch_size, d_model + hebbian_hidden]
        
        final_output = self.output_projection(combined_features)
        
        try:
            # 6. İçsel model doğrulama
            validation_result, validation_metrics = self._validate_output(
                final_output, attended_mean
            )
            
            # 7. Ortaya çıkan davranış analizi
            behavior_analysis = self.emergent_behavior.detect_emergent_behavior(
                representation=final_output,
                attention_weights=attention_weights,
                context_id=context
            )
            
            # 8. Bellek güncelleme
            self._update_memory(final_output, context, validation_result)
            
            # 9. Sistem durumunu güncelle
            self.representation_history.append(final_output.detach().clone())
            self.validation_history.append(validation_result)
            self.step_count += 1
            
        except Exception as e:
            print(f"Warning: Post-processing error: {e}")
            # Graceful degradation
            validation_result = ValidationResult.QUARANTINE
            validation_metrics = ValidationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
            behavior_analysis = {'error': str(e), 'novelty_score': 0.0}
        
        # 10. Sonuçları topla
        results = {
            'output': final_output,
            'attention_weights': attention_weights,
            'hebbian_weights': hebbian_weights,
            'validation': {
                'result': validation_result,
                'metrics': validation_metrics.__dict__
            },
            'behavior_analysis': behavior_analysis,
            'memory_stats': self.memory_hierarchy.get_memory_stats(),
            'attention_stats': self.temporal_attention.get_attention_statistics(attention_weights),
            'system_state': {
                'step': self.step_count,
                'constraint_mode': self.emergent_behavior.constraint_mode.value,
                'representation_history_size': len(self.representation_history)
            }
        }
        
        return results
    
    def _retrieve_memory_context(self, query: torch.Tensor) -> Dict[str, Any]:
        """Bellek hiyerarşisinden ilgili bilgileri getirir."""
        memory_results = self.memory_hierarchy.retrieve(
            query=query,
            memory_types=["short_term", "episodic", "semantic"],
            top_k=5
        )
        
        return memory_results
    
    def _validate_output(self, 
                        current_output: torch.Tensor,
                        context_output: torch.Tensor) -> Tuple[ValidationResult, Any]:
        """Çıkışı doğrular."""
        validation_result, metrics = self.model_validator.validate(
            current_repr=current_output,
            context_repr=context_output,
            representation_history=self.representation_history[-10:]  # Son 10 adım
        )
        
        return validation_result, metrics
    
    def _update_memory(self, 
                      representation: torch.Tensor,
                      context: str,
                      validation_result: ValidationResult):
        """Bellek sistemini günceller."""
        
        # Doğrulama sonucuna göre önem skoru
        importance_scores = {
            ValidationResult.ACCEPT: 1.0,
            ValidationResult.QUARANTINE: 0.5,
            ValidationResult.REJECT: 0.1
        }
        
        importance = importance_scores.get(validation_result, 0.5)
        
        # Batch tensor'ı tek örneklere böl
        batch_size = representation.size(0)
        
        for i in range(batch_size):
            # Her batch elemanını ayrı ayrı kaydet
            item_representation = representation[i:i+1, :].squeeze(0)  # [d_model]
            
            # Kısa süreli belleğe ekle
            self.memory_hierarchy.store(
                content=item_representation,
                memory_type="short_term",
                context=f"{context}_batch_{i}",
                importance=importance
            )
            
            # Yüksek önemli olanları epizodik belleğe de ekle
            if importance >= 0.8:
                self.memory_hierarchy.store(
                    content=item_representation,
                    memory_type="episodic", 
                    context=f"{context}_batch_{i}",
                    importance=importance
                )
    
    def set_constraint_mode(self, mode: ConstraintMode):
        """Kısıtlama modunu ayarlar."""
        self.emergent_behavior.set_constraint_mode(mode)
        
        # Doğrulama eşiğini de ayarla
        if mode == ConstraintMode.CONSERVATIVE:
            self.model_validator.validation_threshold = 0.6
        elif mode == ConstraintMode.EXPLORATORY:
            self.model_validator.validation_threshold = 0.2
        elif mode == ConstraintMode.ADAPTIVE:
            # Varsayılan değeri koru
            pass
    
    def consolidate_memory(self):
        """Bellek konsolidasyonu yapar."""
        self.memory_hierarchy.consolidate(threshold=0.3)
    
    def reset_system_state(self):
        """Sistem durumunu sıfırlar."""
        self.representation_history = []
        self.validation_history = []
        self.step_count = 0
        self.hebbian_learner.reset_history()
        self.memory_hierarchy.short_term.clear()
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Kapsamlı sistem istatistikleri."""
        stats = {
            'processing_steps': self.step_count,
            'memory_stats': self.memory_hierarchy.get_memory_stats(),
            'validation_stats': self.model_validator.get_validation_statistics(),
            'behavior_stats': self.emergent_behavior.get_behavior_statistics(),
            'hebbian_stats': {
                'connection_matrix_norm': torch.norm(self.hebbian_learner.hebbian_weights).item(),
                'activation_history_size': len(self.hebbian_learner.activation_history)
            }
        }
        
        # Son doğrulama skorları
        if self.validation_history:
            recent_validations = self.validation_history[-10:]
            validation_counts = {result.value: 0 for result in ValidationResult}
            for result in recent_validations:
                validation_counts[result.value] += 1
            stats['recent_validation_distribution'] = validation_counts
        
        return stats
    
    def save_checkpoint(self, path: str):
        """Sistem durumunu kaydeder."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'step_count': self.step_count,
            'representation_history': self.representation_history[-50:],  # Son 50 adım
            'validation_history': [r.value for r in self.validation_history[-50:]],
            'memory_stats': self.memory_hierarchy.get_memory_stats(),
            'system_config': {
                'd_model': self.d_model,
                'device': self.device
            }
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Sistem durumunu yükler."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Güvenli model yükleme
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise ValueError("Checkpoint doesn't contain model_state_dict")
            
            # Opsiyonel alanları güvenli yükleme
            self.step_count = checkpoint.get('step_count', 0)
            self.representation_history = checkpoint.get('representation_history', [])
            
            # Validation history'yi geri dönüştür
            if 'validation_history' in checkpoint:
                validation_mapping = {v.value: v for v in ValidationResult}
                self.validation_history = [
                    validation_mapping.get(v, ValidationResult.QUARANTINE) 
                    for v in checkpoint['validation_history']
                ]
            else:
                self.validation_history = []
                
            print(f"Checkpoint loaded successfully from {path}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise