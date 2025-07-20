"""
Ortaya Çıkan Davranış Tespiti ve Yönetimi
=========================================

Makaledeki sistem:
1. Gerçek zamanlı desen takibi
2. Karantina protokolü
3. Kısıtlama modları (tutucu, keşifsel, uyarlayıcı)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time

# NetworkX için güvenli import
try:
    import networkx as nx
except ImportError:
    print("Warning: NetworkX not found. Graph evolution tracking will be disabled.")
    nx = None


class ConstraintMode(Enum):
    CONSERVATIVE = "conservative"  # θ_v yüksek
    EXPLORATORY = "exploratory"   # θ_v düşük  
    ADAPTIVE = "adaptive"         # θ_v = f(C_d)


@dataclass
class EmergentPattern:
    pattern_id: str
    representation: torch.Tensor
    confidence: float
    novelty_score: float
    stability: float
    first_seen: float
    last_seen: float
    occurrence_count: int
    validation_status: str  # "new", "quarantine", "validated", "rejected"


class GraphEvolutionTracker:
    """
    Grafik evrimi G_t takipçisi.
    """
    
    def __init__(self, max_nodes: int = 1000):
        self.max_nodes = max_nodes
        
        if nx is not None:
            self.graph = nx.DiGraph()
            self.graph_enabled = True
        else:
            self.graph = None
            self.graph_enabled = False
            print("Graph evolution tracking disabled - NetworkX not available")
            
        self.node_embeddings = {}
        self.evolution_history = deque(maxlen=100)
        
    def add_node(self, node_id: str, embedding: torch.Tensor):
        """Yeni düğüm ekler."""
        if not self.graph_enabled:
            return
            
        self.graph.add_node(node_id)
        self.node_embeddings[node_id] = embedding.detach().clone()
        
        # Kapasite kontrolü
        if len(self.graph.nodes) > self.max_nodes:
            self._prune_graph()
    
    def add_edge(self, source: str, target: str, weight: float = 1.0):
        """Yeni kenar ekler."""
        if not self.graph_enabled:
            return
            
        self.graph.add_edge(source, target, weight=weight)
    
    def detect_evolution(self) -> Dict[str, Any]:
        """Grafik evrimini tespit eder."""
        if not self.graph_enabled:
            return {'graph_disabled': True}
            
        current_state = {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'avg_degree': np.mean([self.graph.degree(n) for n in self.graph.nodes]) if self.graph.nodes else 0,
            'clustering': nx.average_clustering(self.graph) if len(self.graph.nodes) > 2 else 0
        }
        
        self.evolution_history.append(current_state)
        
        # Değişim tespiti
        if len(self.evolution_history) >= 2:
            prev_state = self.evolution_history[-2]
            evolution_metrics = {
                'node_change': current_state['num_nodes'] - prev_state['num_nodes'],
                'edge_change': current_state['num_edges'] - prev_state['num_edges'],
                'degree_change': current_state['avg_degree'] - prev_state['avg_degree'],
                'clustering_change': current_state['clustering'] - prev_state['clustering']
            }
            return evolution_metrics
        
        return {}
    
    def _prune_graph(self):
        """
        Grafik boyutunu sınırlar.
        En az bağlantılı ve en eski düğümleri kaldırır.
        """
        current_size = len(self.graph.nodes)
        target_size = self.max_nodes - 50  # Güvenli marj
        
        if current_size <= target_size:
            return
        
        nodes_to_remove_count = current_size - target_size
        
        # Düğümleri skorla (derece + yaş faktörü)
        current_time = time.time()
        node_scores = []
        
        for node in self.graph.nodes:
            degree = self.graph.degree(node)
            
            # Yaş faktörü (node ID'den zaman çıkarsamaya çalış)
            try:
                node_time = float(node.split('_')[-1]) if '_' in node else current_time
                age_hours = (current_time - node_time) / 3600  # saat cinsinden
                age_penalty = min(age_hours / 24, 1.0)  # maksimum 1 gün
            except:
                age_penalty = 0.5  # varsayılan
            
            # Düşük skor = kaldırılacak
            score = degree - age_penalty
            node_scores.append((score, node))
        
        # En düşük skorlu düğümleri seç
        node_scores.sort(key=lambda x: x[0])
        nodes_to_remove = [node for _, node in node_scores[:nodes_to_remove_count]]
        
        # Düğümleri kaldır
        for node in nodes_to_remove:
            if node in self.graph.nodes:
                self.graph.remove_node(node)
            if node in self.node_embeddings:
                del self.node_embeddings[node]
        
        print(f"Grafik budandı: {current_size} -> {len(self.graph.nodes)} düğüm")


class AttentionShiftDetector:
    """
    Dikkat odağında kayma ∇A(t) tespitçisi.
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.attention_history = deque(maxlen=window_size)
        
    def track_attention(self, attention_weights: torch.Tensor):
        """Dikkat ağırlıklarını takip eder."""
        # Attention weights: [batch, heads, seq_len, seq_len]
        # Ortalama dikkat dağılımını hesapla
        avg_attention = attention_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
        
        self.attention_history.append({
            'attention': avg_attention.detach().clone(),
            'timestamp': time.time(),
            'entropy': self._compute_entropy(avg_attention)
        })
    
    def detect_shift(self) -> Dict[str, float]:
        """Dikkat kaymasını tespit eder."""
        if len(self.attention_history) < 2:
            return {}
        
        current = self.attention_history[-1]
        previous = self.attention_history[-2]
        
        # Cosine distance
        current_flat = current['attention'].flatten()
        previous_flat = previous['attention'].flatten()
        
        cosine_sim = torch.cosine_similarity(current_flat, previous_flat, dim=0)
        shift_magnitude = 1.0 - cosine_sim.item()
        
        # Entropi değişimi
        entropy_change = current['entropy'] - previous['entropy']
        
        # Gradyan büyüklüğü
        if len(self.attention_history) >= 3:
            gradient = self._compute_gradient()
        else:
            gradient = 0.0
        
        return {
            'shift_magnitude': shift_magnitude,
            'entropy_change': entropy_change,
            'gradient_magnitude': gradient
        }
    
    def _compute_entropy(self, attention: torch.Tensor) -> float:
        """Dikkat dağılımının entropisi."""
        # Satır bazında entropi hesapla
        entropies = []
        for row in attention:
            prob_dist = torch.softmax(row, dim=0)
            entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8))
            entropies.append(entropy.item())
        
        return np.mean(entropies)
    
    def _compute_gradient(self) -> float:
        """Dikkat değişiminin gradyanı."""
        if len(self.attention_history) < 3:
            return 0.0
        
        recent_entropies = [h['entropy'] for h in list(self.attention_history)[-3:]]
        gradient = np.gradient(recent_entropies)
        
        return np.abs(gradient[-1])


class EmergentBehaviorManager:
    """
    Ortaya çıkan davranışları yöneten ana sistem.
    """
    
    def __init__(self,
                 d_model: int,
                 novelty_threshold: float = 0.4,
                 stability_threshold: float = 0.6,
                 validation_window: int = 10):
        
        self.d_model = d_model
        self.novelty_threshold = novelty_threshold
        self.stability_threshold = stability_threshold
        self.validation_window = validation_window
        
        # Alt sistemler
        self.graph_tracker = GraphEvolutionTracker()
        self.attention_detector = AttentionShiftDetector()
        
        # Desen takibi
        self.emergent_patterns = {}
        self.quarantine_patterns = {}
        
        # Kısıtlama modu
        self.constraint_mode = ConstraintMode.ADAPTIVE
        self.validation_threshold = 0.5  # θ_v
        
        # Novelty detector ağı
        self.novelty_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def detect_emergent_behavior(self, 
                               representation: torch.Tensor,
                               attention_weights: torch.Tensor,
                               context_id: str = "default") -> Dict[str, Any]:
        """
        Ortaya çıkan davranışları tespit eder.
        """
        current_time = time.time()
        
        # 1. Novelty skorunu hesapla
        novelty_score = self._compute_novelty(representation)
        
        # 2. Grafik evrimini takip et
        node_id = f"{context_id}_{current_time}"
        self.graph_tracker.add_node(node_id, representation)
        graph_evolution = self.graph_tracker.detect_evolution()
        
        # 3. Dikkat kaymasını tespit et
        self.attention_detector.track_attention(attention_weights)
        attention_shift = self.attention_detector.detect_shift()
        
        # 4. Yeni desen kontrolü
        pattern_info = self._check_for_new_pattern(representation, novelty_score, current_time)
        
        # 5. Karantina protokolü
        quarantine_decision = self._apply_quarantine_protocol(pattern_info)
        
        return {
            'novelty_score': novelty_score,
            'graph_evolution': graph_evolution,
            'attention_shift': attention_shift,
            'pattern_info': pattern_info,
            'quarantine_decision': quarantine_decision,
            'constraint_mode': self.constraint_mode.value,
            'validation_threshold': self.validation_threshold
        }
    
    def _compute_novelty(self, representation: torch.Tensor) -> float:
        """Novelty skorunu hesaplar."""
        # Batch boyutunu kontrol et
        if representation.dim() > 1 and representation.size(0) > 1:
            # Batch varsa, ortalama al
            representation = representation.mean(dim=0, keepdim=True)
        elif representation.dim() > 1:
            # Tek batch elemanı varsa, squeeze
            representation = representation.squeeze(0)
        
        # Novelty hesapla
        novelty_tensor = self.novelty_network(representation.unsqueeze(0))
        novelty_score = novelty_tensor.mean().item()  # Güvenli scalar conversion
        
        # Mevcut desenlerle karşılaştır
        if self.emergent_patterns:
            max_similarity = 0.0
            representation_flat = representation.flatten()
            
            for pattern in self.emergent_patterns.values():
                try:
                    similarity = torch.cosine_similarity(
                        representation_flat.unsqueeze(0), 
                        pattern.representation.flatten().unsqueeze(0), 
                        dim=1
                    ).item()
                    max_similarity = max(max_similarity, similarity)
                except Exception as e:
                    print(f"Similarity calculation error: {e}")
                    continue
            
            # Novelty = 1 - max_similarity
            adjusted_novelty = max(novelty_score, 1.0 - max_similarity)
            return adjusted_novelty
        
        return novelty_score
    
    def _check_for_new_pattern(self, 
                              representation: torch.Tensor,
                              novelty_score: float,
                              current_time: float) -> Optional[EmergentPattern]:
        """Yeni desen kontrolü yapar."""
        
        if novelty_score < self.novelty_threshold:
            return None
        
        # Representation boyutunu normalize et
        if representation.dim() > 1:
            if representation.size(0) > 1:
                # Batch varsa ortalama al
                representation = representation.mean(dim=0)
            else:
                # Tek element varsa squeeze
                representation = representation.squeeze(0)
        
        # Benzer desen var mı kontrol et - uyarlanabilir eşik
        similarity_threshold = 0.75 if self.constraint_mode == ConstraintMode.CONSERVATIVE else 0.65
        
        for pattern_id, pattern in self.emergent_patterns.items():
            try:
                similarity = torch.cosine_similarity(
                    representation.flatten().unsqueeze(0),
                    pattern.representation.flatten().unsqueeze(0),
                    dim=1
                ).item()
                
                if similarity > similarity_threshold:  # Benzer desen bulundu
                    # Mevcut deseni güncelle
                    pattern.last_seen = current_time
                    pattern.occurrence_count += 1
                    pattern.stability = self._compute_stability(pattern)
                    
                    # Representasyonu güncelle (ağırlıklı ortalama)
                    alpha = 0.1  # öğrenme katsayısı
                    pattern.representation = (1 - alpha) * pattern.representation + alpha * representation.detach()
                    
                    return pattern
                    
            except Exception as e:
                print(f"Pattern similarity calculation error: {e}")
                continue
        
        # Yeni desen oluştur
        pattern_id = f"pattern_{len(self.emergent_patterns)}_{int(current_time)}"
        new_pattern = EmergentPattern(
            pattern_id=pattern_id,
            representation=representation.detach().clone(),
            confidence=novelty_score,
            novelty_score=novelty_score,
            stability=0.0,
            first_seen=current_time,
            last_seen=current_time,
            occurrence_count=1,
            validation_status="new"
        )
        
        self.emergent_patterns[pattern_id] = new_pattern
        return new_pattern
    
    def _compute_stability(self, pattern: EmergentPattern) -> float:
        """
        Desen kararlılığını hesaplar.
        Frekans, süreklilik ve yaş faktörlerini birleştirir.
        """
        time_span = pattern.last_seen - pattern.first_seen
        if time_span == 0:
            return 1.0
        
        # Occurrence frequency (normalized by time)
        time_span_hours = max(time_span / 3600, 0.1)  # minimum 0.1 saat
        frequency = pattern.occurrence_count / time_span_hours
        
        # Süreklilik faktörü - eşit aralıklarla görülmüş mü?
        continuity_factor = 1.0
        if pattern.occurrence_count > 2:
            expected_interval = time_span / (pattern.occurrence_count - 1)
            # Bu basit bir yaklaşım - gerçek implementasyon için interval tracking gerekir
            continuity_factor = min(1.0, 2.0 / max(expected_interval / 3600, 1.0))  # saat normalize
        
        # Yaş faktörü - çok eski desenler biraz cezalandırılır
        current_time = time.time()
        age_hours = (current_time - pattern.first_seen) / 3600
        age_factor = max(0.5, 1.0 - (age_hours / (7 * 24)))  # 1 hafta sonra %50'ye düş
        
        # Toplam kararlılık skoru
        stability = min(1.0, frequency * continuity_factor * age_factor * 0.1)
        
        return stability
    
    def _apply_quarantine_protocol(self, pattern: Optional[EmergentPattern]) -> Dict[str, Any]:
        """Karantina protokolünü uygular."""
        if not pattern:
            return {"action": "none", "reason": "no_pattern"}
        
        if pattern.validation_status == "new":
            # Yeni desen -> karantinaya al
            pattern.validation_status = "quarantine"
            self.quarantine_patterns[pattern.pattern_id] = pattern
            return {
                "action": "quarantine",
                "reason": "new_pattern",
                "pattern_id": pattern.pattern_id
            }
        
        elif pattern.validation_status == "quarantine":
            # Karantinada -> kararlılık kontrolü
            if pattern.stability >= self.stability_threshold:
                pattern.validation_status = "validated"
                if pattern.pattern_id in self.quarantine_patterns:
                    del self.quarantine_patterns[pattern.pattern_id]
                return {
                    "action": "validate",
                    "reason": "stability_achieved",
                    "pattern_id": pattern.pattern_id
                }
            elif pattern.occurrence_count > self.validation_window:
                pattern.validation_status = "rejected"
                if pattern.pattern_id in self.quarantine_patterns:
                    del self.quarantine_patterns[pattern.pattern_id]
                if pattern.pattern_id in self.emergent_patterns:
                    del self.emergent_patterns[pattern.pattern_id]
                return {
                    "action": "reject",
                    "reason": "failed_validation",
                    "pattern_id": pattern.pattern_id
                }
        
        return {"action": "monitor", "reason": "ongoing_validation"}
    
    def set_constraint_mode(self, mode: ConstraintMode, decision_confidence: Optional[float] = None):
        """Kısıtlama modunu ayarlar."""
        self.constraint_mode = mode
        
        if mode == ConstraintMode.CONSERVATIVE:
            self.validation_threshold = 0.6
        elif mode == ConstraintMode.EXPLORATORY:
            self.validation_threshold = 0.2
        elif mode == ConstraintMode.ADAPTIVE and decision_confidence is not None:
            # θ_v = f(C_d)
            self.validation_threshold = 0.2 + 0.6 * decision_confidence
    
    def get_behavior_statistics(self) -> Dict[str, Any]:
        """Davranış istatistiklerini döndürür."""
        total_patterns = len(self.emergent_patterns)
        quarantine_count = len(self.quarantine_patterns)
        validated_count = sum(
            1 for p in self.emergent_patterns.values() 
            if p.validation_status == "validated"
        )
        
        return {
            'total_patterns': total_patterns,
            'quarantine_patterns': quarantine_count,
            'validated_patterns': validated_count,
            'graph_nodes': len(self.graph_tracker.graph.nodes),
            'graph_edges': len(self.graph_tracker.graph.edges),
            'constraint_mode': self.constraint_mode.value,
            'validation_threshold': self.validation_threshold
        }