"""
İçsel Model Doğrulama Sistemi
=============================

Makaledeki çok katmanlı doğrulama mimarisi:
1. Anlamsal tutarlılık kontrolü
2. Karar deseni analizi: C_d = f(A_h, C_i, S_t)
3. Zamansal olasılık değerlendirmesi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationResult(Enum):
    ACCEPT = "accept"
    QUARANTINE = "quarantine" 
    REJECT = "reject"


@dataclass
class ValidationMetrics:
    semantic_consistency: float
    logical_coherence: float
    confidence_calibration: float
    temporal_stability: float
    overall_score: float


class ModelValidator(nn.Module):
    """
    İçsel model doğrulama sistemi.
    Anlamsal tutarlılık, mantıksal coherence ve zamansal kararlılığı değerlendirir.
    """
    
    def __init__(self,
                 d_model: int,
                 validation_threshold: float = 0.4,
                 quarantine_threshold: float = 0.4,
                 temporal_window: int = 10,
                 device: str = 'cpu'):
        super().__init__()
        
        self.d_model = d_model
        self.validation_threshold = validation_threshold
        self.quarantine_threshold = quarantine_threshold
        self.temporal_window = temporal_window
        self.device = device
        
        # Anlamsal tutarlılık ağı
        self.semantic_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Mantıksal coherence ağı
        self.logic_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Zamansal kararlılık değerlendirici
        self.temporal_network = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            device=device
        )
        
        self.temporal_classifier = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Karar geçmişi
        self.decision_history = []
        self.accuracy_history = []
        
    def compute_semantic_consistency(self, 
                                   current_repr: torch.Tensor,
                                   context_repr: torch.Tensor) -> torch.Tensor:
        """
        Anlamsal tutarlılık skorunu hesaplar.
        Vektör uyumu ve semantik bölge kontrolü.
        """
        # İki temsili birleştir
        combined = torch.cat([current_repr, context_repr], dim=-1)
        consistency_score = self.semantic_network(combined)
        
        # Cosine benzerlik ekle
        cosine_sim = F.cosine_similarity(current_repr, context_repr, dim=-1, eps=1e-8)
        cosine_sim = (cosine_sim + 1) / 2  # [0,1] aralığına normalize et
        
        # Ağırlıklı kombinasyon
        final_score = 0.7 * consistency_score.squeeze(-1) + 0.3 * cosine_sim
        
        return final_score
    
    def compute_logical_coherence(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Mantıksal tutarlılık kontrolü.
        İçsel çelişki tespiti.
        """
        coherence_score = self.logic_network(representation)
        
        # Entropi bazlı belirsizlik ölçümü
        prob_dist = F.softmax(representation, dim=-1)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8), dim=-1)
        
        # Düşük entropi = yüksek tutarlılık
        max_entropy = torch.log(torch.tensor(representation.size(-1), dtype=torch.float))
        normalized_entropy = 1.0 - (entropy / max_entropy)
        
        # Kombinasyon
        final_score = 0.6 * coherence_score.squeeze(-1) + 0.4 * normalized_entropy
        
        return final_score
    
    def compute_temporal_stability(self, 
                                 representations: List[torch.Tensor]) -> torch.Tensor:
        """
        Zamansal kararlılık değerlendirmesi.
        C_d = f(A_h, C_i, S_t) formülüne göre.
        """
        if len(representations) < 2:
            return torch.ones(representations[0].size(0), device=self.device)
        
        batch_size = representations[0].size(0)
        
        # Son temporal_window kadar temsili al
        recent_reprs = representations[-self.temporal_window:]
        
        # LSTM ile zamansal desen analizi
        sequence = torch.stack(recent_reprs, dim=1)  # [batch, seq_len, d_model]
        lstm_out, _ = self.temporal_network(sequence)
        
        # Son çıkışı kullan
        temporal_score = self.temporal_classifier(lstm_out[:, -1, :]).squeeze(-1)  # [batch_size]
        
        # Tarihsel doğruluk A_h - tensor haline getir
        if len(self.accuracy_history) > 0:
            historical_accuracy = np.mean(self.accuracy_history[-10:])
        else:
            historical_accuracy = 0.5
        
        # Tensor'e dönüştür ve batch boyutuna genişlet
        historical_accuracy_tensor = torch.full((batch_size,), historical_accuracy, 
                                               device=self.device, dtype=torch.float32)
        
        # İçsel tutarlılık C_i (varyans bazlı) - doğru tensor işlem
        if len(recent_reprs) > 1:
            repr_stack = torch.stack(recent_reprs, dim=0)  # [seq_len, batch_size, d_model]
            # Her batch elemanı için varyans hesapla
            variance = torch.var(repr_stack, dim=0)  # [batch_size, d_model]
            variance_mean = variance.mean(dim=-1)  # [batch_size]
            internal_consistency = torch.exp(-variance_mean)  # [batch_size]
        else:
            internal_consistency = torch.ones(batch_size, device=self.device)
        
        # Formül: C_d = f(A_h, C_i, S_t) - tüm bileşenler tensor
        final_score = (
            0.4 * temporal_score +              # [batch_size]
            0.3 * historical_accuracy_tensor +  # [batch_size]  
            0.3 * internal_consistency          # [batch_size]
        )
        
        return final_score
    
    def validate(self, 
                current_repr: torch.Tensor,
                context_repr: torch.Tensor,
                representation_history: List[torch.Tensor]) -> Tuple[ValidationResult, ValidationMetrics]:
        """
        Kapsamlı model doğrulaması.
        """
        batch_size = current_repr.size(0)
        
        # 1. Anlamsal tutarlılık
        semantic_score = self.compute_semantic_consistency(current_repr, context_repr)
        
        # 2. Mantıksal coherence
        logic_score = self.compute_logical_coherence(current_repr)
        
        # 3. Zamansal kararlılık  
        temporal_score = self.compute_temporal_stability(representation_history + [current_repr])
        
        # 4. Güven kalibrasyonu (belirsizlik ölçümü)
        uncertainty = self.compute_uncertainty(current_repr)
        confidence_score = 1.0 - uncertainty
        
        # Toplam skor
        overall_score = (
            0.3 * semantic_score +
            0.25 * logic_score +
            0.25 * temporal_score +
            0.2 * confidence_score
        )
        
        # Karar verme
        decision = self.make_decision(overall_score)
        
        # Metrikleri oluştur
        metrics = ValidationMetrics(
            semantic_consistency=semantic_score.mean().item(),
            logical_coherence=logic_score.mean().item(),
            confidence_calibration=confidence_score.mean().item(),
            temporal_stability=temporal_score.mean().item(),
            overall_score=overall_score.mean().item()
        )
        
        # Geçmişi güncelle
        self.decision_history.append(decision)
        
        return decision, metrics
    
    def compute_uncertainty(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Belirsizlik ölçümü (entropi bazlı).
        """
        prob_dist = F.softmax(representation, dim=-1)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8), dim=-1)
        max_entropy = torch.log(torch.tensor(representation.size(-1), dtype=torch.float))
        normalized_uncertainty = entropy / max_entropy
        
        return normalized_uncertainty
    
    def make_decision(self, overall_score: torch.Tensor) -> ValidationResult:
        """
        Doğrulama skoruna göre karar verir.
        Batch'teki çoğunluk oyu kullanır.
        """
        # Batch içindeki her örnek için karar
        decisions = []
        for score in overall_score:
            score_val = score.item()
            if score_val >= self.validation_threshold:
                decisions.append(ValidationResult.ACCEPT)
            elif score_val >= self.quarantine_threshold:
                decisions.append(ValidationResult.QUARANTINE)
            else:
                decisions.append(ValidationResult.REJECT)
        
        # Çoğunluk oyu
        accept_count = decisions.count(ValidationResult.ACCEPT)
        quarantine_count = decisions.count(ValidationResult.QUARANTINE)
        reject_count = decisions.count(ValidationResult.REJECT)
        
        if accept_count >= quarantine_count and accept_count >= reject_count:
            return ValidationResult.ACCEPT
        elif quarantine_count >= reject_count:
            return ValidationResult.QUARANTINE
        else:
            return ValidationResult.REJECT
    
    def update_accuracy(self, predicted_correct: bool):
        """
        Doğruluk geçmişini günceller.
        """
        self.accuracy_history.append(1.0 if predicted_correct else 0.0)
        
        # Geçmişi sınırla
        if len(self.accuracy_history) > 100:
            self.accuracy_history = self.accuracy_history[-100:]
    
    def get_validation_statistics(self) -> Dict[str, float]:
        """
        Doğrulama istatistiklerini döndürür.
        """
        if not self.decision_history:
            return {}
        
        recent_decisions = self.decision_history[-50:]
        
        stats = {
            'accept_rate': sum(1 for d in recent_decisions if d == ValidationResult.ACCEPT) / len(recent_decisions),
            'quarantine_rate': sum(1 for d in recent_decisions if d == ValidationResult.QUARANTINE) / len(recent_decisions),
            'reject_rate': sum(1 for d in recent_decisions if d == ValidationResult.REJECT) / len(recent_decisions),
        }
        
        if self.accuracy_history:
            stats['avg_accuracy'] = np.mean(self.accuracy_history[-50:])
        
        return stats