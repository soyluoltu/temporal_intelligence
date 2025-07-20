"""
Bellek Hiyerarşisi Yönetimi
===========================

Makaledeki üç katmanlı bellek sistemi:
- M_s: Kısa süreli bellek (anlık bağlam)
- M_e: Epizodik bellek (oturum bazlı öğrenim) 
- M_l: Anlamsal bellek (kalıcı bilgi)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class MemoryItem:
    content: torch.Tensor
    timestamp: float
    importance: float
    access_count: int
    context_id: str


class ShortTermMemory:
    """
    Kısa süreli bellek - anlık bağlam için.
    """
    
    def __init__(self, capacity: int = 50, decay_rate: float = 0.95):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.memory = deque(maxlen=capacity)
        
    def store(self, item: torch.Tensor, importance: float = 1.0):
        """Kısa süreli belleğe öğe ekler."""
        memory_item = MemoryItem(
            content=item.detach().clone(),
            timestamp=time.time(),
            importance=importance,
            access_count=0,
            context_id="short_term"
        )
        self.memory.append(memory_item)
    
    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> List[MemoryItem]:
        """Sorguya en benzer öğeleri getirir."""
        if not self.memory:
            return []
        
        similarities = []
        current_time = time.time()
        
        # Query boyutunu normalize et
        if query.dim() > 1:
            query_flat = query.flatten()
        else:
            query_flat = query
        
        for item in self.memory:
            try:
                # Temporal decay uygula
                time_decay = self.decay_rate ** (current_time - item.timestamp)
                
                # Item content boyutunu normalize et
                if item.content.dim() > 1:
                    item_flat = item.content.flatten()
                else:
                    item_flat = item.content
                
                # Cosine similarity - boyut kontrolü ile
                if query_flat.size(0) == item_flat.size(0):
                    similarity = torch.cosine_similarity(
                        query_flat.unsqueeze(0), item_flat.unsqueeze(0), dim=1
                    ).item()
                else:
                    # Boyut uyumsuzluğu durumunda düşük similarity
                    similarity = 0.0
                
                # Önem ve zaman içeren final skor
                final_score = similarity * item.importance * time_decay
                similarities.append((final_score, item))
                
            except Exception as e:
                print(f"Short-term memory similarity calculation error: {e}")
                # Hatalı öğe için 0 similarity
                similarities.append((0.0, item))
        
        # En iyi k öğeyi seç
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in similarities[:top_k]]
    
    def clear(self):
        """Belleği temizler."""
        self.memory.clear()


class EpisodicMemory:
    """
    Epizodik bellek - oturum bazlı öğrenim için.
    """
    
    def __init__(self, capacity: int = 500, consolidation_threshold: float = 0.8):
        self.capacity = capacity
        self.consolidation_threshold = consolidation_threshold
        self.episodes = []
        
    def store_episode(self, 
                     content: torch.Tensor,
                     context: str,
                     importance: float = 1.0):
        """Epizot olarak saklar."""
        episode = MemoryItem(
            content=content.detach().clone(),
            timestamp=time.time(),
            importance=importance,
            access_count=0,
            context_id=context
        )
        
        self.episodes.append(episode)
        
        # Kapasite kontrolü
        if len(self.episodes) > self.capacity:
            self._consolidate_memories()
    
    def retrieve_episodes(self, 
                         query: torch.Tensor,
                         context: Optional[str] = None,
                         top_k: int = 10) -> List[MemoryItem]:
        """Benzer epizotları getirir."""
        if not self.episodes:
            return []
        
        candidates = self.episodes
        
        # Bağlam filtresi
        if context:
            candidates = [ep for ep in candidates if ep.context_id == context]
        
        # Query boyutunu normalize et
        if query.dim() > 1:
            query_flat = query.flatten()
        else:
            query_flat = query
        
        similarities = []
        for episode in candidates:
            try:
                # Episode content boyutunu normalize et
                if episode.content.dim() > 1:
                    episode_flat = episode.content.flatten()
                else:
                    episode_flat = episode.content
                
                # Cosine similarity - boyut kontrolü ile
                if query_flat.size(0) == episode_flat.size(0):
                    similarity = torch.cosine_similarity(
                        query_flat.unsqueeze(0), episode_flat.unsqueeze(0), dim=1
                    ).item()
                else:
                    # Boyut uyumsuzluğu durumunda düşük similarity
                    similarity = 0.0
                
                # Erişim sıklığı bonusu
                access_bonus = min(episode.access_count * 0.1, 0.5)
                final_score = similarity * episode.importance + access_bonus
                
                similarities.append((final_score, episode))
                
            except Exception as e:
                print(f"Episodic memory similarity calculation error: {e}")
                # Hatalı öğe için düşük skor
                similarities.append((0.0, episode))
        
        # Sırala ve döndür
        similarities.sort(key=lambda x: x[0], reverse=True)
        selected = [item for _, item in similarities[:top_k]]
        
        # Erişim sayısını güncelle
        for item in selected:
            item.access_count += 1
        
        return selected
    
    def _consolidate_memories(self):
        """
        Bellek konsolidasyonu - önemli olanları tutar.
        Önem skoru, erişim sıklığı ve yakınlık faktörlerini birleştirir.
        """
        current_time = time.time()
        
        # Gelişmiş konsolidasyon skoru hesaplama
        scored_episodes = []
        for episode in self.episodes:
            # Önem skoru
            importance_score = episode.importance
            
            # Erişim sıklığı bonusu
            access_bonus = min(episode.access_count * 0.1, 0.5)
            
            # Yakınlık faktörü (yeni olanlar biraz daha değerli)
            time_factor = min((current_time - episode.timestamp) / (24 * 3600), 1.0)  # 24 saat normalize
            recency_bonus = (1.0 - time_factor) * 0.2
            
            # Toplam skor
            total_score = importance_score + access_bonus + recency_bonus
            scored_episodes.append((total_score, episode))
        
        # Skor bazında sırala
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        
        # En iyi skorlu olanları tut
        self.episodes = [episode for _, episode in scored_episodes[:self.capacity]]
        
        print(f"Epizodik bellek konsolidasyonu: {len(scored_episodes)} -> {len(self.episodes)} epizot")


class SemanticMemory:
    """
    Anlamsal bellek - kalıcı bilgi için.
    """
    
    def __init__(self, d_model: int, num_concepts: int = 1000):
        self.d_model = d_model
        self.num_concepts = num_concepts
        
        # Kavram vektörleri
        self.concept_embeddings = nn.Embedding(num_concepts, d_model)
        self.concept_names = {}
        self.concept_count = 0
        
        # Kavramlar arası ilişkiler
        self.relation_matrix = torch.zeros(num_concepts, num_concepts)
        
    def store_concept(self, 
                     embedding: torch.Tensor,
                     name: str,
                     strengthen_relations: bool = True) -> int:
        """Yeni kavram ekler veya mevcut olanı günceller."""
        concept_id = self._get_or_create_concept_id(name)
        
        # Embedding'i güncelle
        with torch.no_grad():
            self.concept_embeddings.weight[concept_id] = embedding.detach()
        
        # İlişkileri güçlendir
        if strengthen_relations:
            self._update_relations(concept_id, embedding)
        
        return concept_id
    
    def retrieve_concept(self, query: torch.Tensor, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """Sorguya en benzer kavramları getirir."""
        # Eğer hiç kavram yoksa boş liste döndür
        if self.concept_count == 0:
            return []
        
        # Query boyutunu kontrol et
        if query.dim() == 2:
            query = query.squeeze(0)  # [batch_size, d_model] -> [d_model]
        elif query.dim() > 2:
            query = query.view(-1)  # Flatten to 1D
        
        # Tüm kavramlarla benzerlik hesapla
        all_embeddings = self.concept_embeddings.weight[:self.concept_count]
        
        # Boyut kontrolü
        if all_embeddings.size(0) == 0:
            return []
        
        # Query'yi doğru boyuta getir
        query_expanded = query.unsqueeze(0)  # [1, d_model]
        
        # Cosine similarity hesapla
        similarities = torch.cosine_similarity(
            query_expanded, all_embeddings, dim=1
        )
        
        # En iyi k tanesini seç
        actual_k = min(top_k, self.concept_count)
        if actual_k == 0:
            return []
            
        top_indices = torch.topk(similarities, actual_k)[1]
        
        results = []
        for idx in top_indices:
            concept_id = idx.item()
            concept_name = self._get_concept_name(concept_id)
            similarity = similarities[idx].item()
            results.append((concept_id, concept_name, similarity))
        
        return results
    
    def get_related_concepts(self, concept_id: int, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """İlişkili kavramları getirir."""
        if concept_id >= self.concept_count:
            return []
        
        relations = self.relation_matrix[concept_id, :self.concept_count]
        top_indices = torch.topk(relations, min(top_k, self.concept_count))[1]
        
        results = []
        for idx in top_indices:
            related_id = idx.item()
            if related_id != concept_id:
                relation_strength = relations[idx].item()
                concept_name = self._get_concept_name(related_id)
                results.append((related_id, concept_name, relation_strength))
        
        return results
    
    def _get_or_create_concept_id(self, name: str) -> int:
        """Kavram ID'si alır veya oluşturur."""
        for concept_id, concept_name in self.concept_names.items():
            if concept_name == name:
                return concept_id
        
        # Yeni kavram oluştur
        if self.concept_count < self.num_concepts:
            concept_id = self.concept_count
            self.concept_names[concept_id] = name
            self.concept_count += 1
            return concept_id
        else:
            # Kapasitede yer yok, en az kullanılanı değiştir
            return self._replace_least_used_concept(name)
    
    def _get_concept_name(self, concept_id: int) -> str:
        """Kavram adını getirir."""
        return self.concept_names.get(concept_id, f"concept_{concept_id}")
    
    def _update_relations(self, concept_id: int, embedding: torch.Tensor):
        """Kavramlar arası ilişkileri günceller."""
        if self.concept_count <= 1:
            return
        
        with torch.no_grad():
            # Diğer kavramlarla benzerlik hesapla
            other_embeddings = self.concept_embeddings.weight[:self.concept_count]
            similarities = torch.cosine_similarity(
                embedding.unsqueeze(0), other_embeddings, dim=1
            )
            
            # İlişki matrisini güncelle (öğrenme katsayısı ile)
            learning_rate = 0.1
            self.relation_matrix[concept_id, :self.concept_count] = (
                (1 - learning_rate) * self.relation_matrix[concept_id, :self.concept_count] +
                learning_rate * similarities
            )
            
            # Simetrik güncelleme
            self.relation_matrix[:self.concept_count, concept_id] = (
                self.relation_matrix[concept_id, :self.concept_count]
            )
    
    def _replace_least_used_concept(self, name: str) -> int:
        """En az kullanılan kavramı değiştirir."""
        # Basit strateji: ilişki matrisindeki toplam değeri en düşük olanı seç
        relation_sums = torch.sum(self.relation_matrix[:self.concept_count], dim=1)
        least_used_id = torch.argmin(relation_sums).item()
        
        # Yeni kavramla değiştir
        self.concept_names[least_used_id] = name
        return least_used_id


class MemoryHierarchy:
    """
    Üç katmanlı bellek hiyerarşisini yöneten ana sınıf.
    """
    
    def __init__(self, 
                 d_model: int,
                 short_term_capacity: int = 50,
                 episodic_capacity: int = 500,
                 semantic_concepts: int = 1000):
        
        self.d_model = d_model
        
        # Bellek katmanları
        self.short_term = ShortTermMemory(short_term_capacity)
        self.episodic = EpisodicMemory(episodic_capacity)
        self.semantic = SemanticMemory(d_model, semantic_concepts)
        
    def store(self, 
              content: torch.Tensor,
              memory_type: str = "short_term",
              context: str = "default",
              importance: float = 1.0,
              concept_name: Optional[str] = None):
        """Uygun bellek katmanına saklar."""
        
        if memory_type == "short_term":
            self.short_term.store(content, importance)
            
        elif memory_type == "episodic":
            self.episodic.store_episode(content, context, importance)
            
        elif memory_type == "semantic" and concept_name:
            self.semantic.store_concept(content, concept_name)
            
    def retrieve(self, 
                query: torch.Tensor,
                memory_types: List[str] = ["short_term", "episodic", "semantic"],
                context: Optional[str] = None,
                top_k: int = 5) -> Dict[str, List[Any]]:
        """Bellek katmanlarından ilgili bilgileri getirir."""
        
        results = {}
        
        # Query boyutunu normalize et (batch boyutunu kaldır)
        if query.dim() > 1:
            query_normalized = query.view(-1)  # Flatten to 1D
        else:
            query_normalized = query
        
        try:
            if "short_term" in memory_types:
                results["short_term"] = self.short_term.retrieve(query_normalized, top_k)
        except Exception as e:
            print(f"Short-term memory retrieval error: {e}")
            results["short_term"] = []
            
        try:
            if "episodic" in memory_types:
                results["episodic"] = self.episodic.retrieve_episodes(query_normalized, context, top_k)
        except Exception as e:
            print(f"Episodic memory retrieval error: {e}")
            results["episodic"] = []
            
        try:
            if "semantic" in memory_types:
                results["semantic"] = self.semantic.retrieve_concept(query_normalized, top_k)
        except Exception as e:
            print(f"Semantic memory retrieval error: {e}")
            results["semantic"] = []
            
        return results
    
    def consolidate(self, threshold: float = 0.3):
        """
        Bellek konsolidasyonu yapar.
        Önemli kısa süreli bellekleri epizodik belleğe taşır,
        çok önemli olanları anlamsal belleğe de ekler.
        """
        # Kısa süreli bellekten konsolidasyon adaylarını topla
        items_to_consolidate = []
        items_to_keep = []
        
        for item in list(self.short_term.memory):
            if item.importance >= threshold:
                items_to_consolidate.append(item)
            else:
                # Düşük önemli olanları kısa süreli bellekte tut
                items_to_keep.append(item)
        
        # Konsolidasyon işlemi
        for item in items_to_consolidate:
            # Epizodik belleğe taşı
            self.episodic.store_episode(
                item.content, 
                item.context_id, 
                item.importance
            )
            
            # Çok yüksek önemli olanları anlamsal belleğe de ekle
            if item.importance >= 0.9:
                concept_name = f"consolidated_{item.context_id}_{int(item.timestamp)}"
                self.semantic.store_concept(
                    item.content, 
                    concept_name, 
                    strengthen_relations=True
                )
        
        # Kısa süreli belleği güncelle - tüm öğeleri silmek yerine seçici temizlik
        self.short_term.memory.clear()
        for item in items_to_keep:
            self.short_term.memory.append(item)
            
        print(f"Konsolidasyon tamamlandı: {len(items_to_consolidate)} öğe taşındı, "
              f"{len(items_to_keep)} öğe kısa süreli bellekte kaldı")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Bellek istatistiklerini döndürür."""
        return {
            "short_term_size": len(self.short_term.memory),
            "episodic_size": len(self.episodic.episodes),
            "semantic_concepts": self.semantic.concept_count,
            "semantic_capacity": self.semantic.num_concepts
        }