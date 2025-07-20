"""
Hebbian Öğrenme Mekanizması
===========================

Makaledeki formüle göre:
Δw_ij(t) = α · a_i(t) · a_j(t) · ω(t)

burada:
- α: öğrenme katsayısı
- a_i, a_j: i, j nöronlarının aktivasyon seviyeleri  
- ω(t): zamansal ağırlık fonksiyonu
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
import time


class HebbianLearner(nn.Module):
    """
    Zamansal ağırlıklı Hebbian öğrenme mekanizması.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 learning_rate: float = 0.01,
                 temporal_decay: float = 0.95,
                 device: str = 'cpu'):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.temporal_decay = temporal_decay
        self.device = device
        
        # Hebbian bağlantı matrisi
        self.hebbian_weights = nn.Parameter(
            torch.randn(input_size, hidden_size, device=device) * 0.1
        )
        
        # Aktivasyon geçmişi
        self.activation_history = []
        self.time_stamps = []
        
        # Zamansal ağırlık fonksiyonu
        self.temporal_weights = torch.ones(1, device=device)
        
    def temporal_weight_function(self, delta_t: float) -> float:
        """
        Zamansal ağırlık fonksiyonu ω(t)
        Yakın geçmişteki aktivasyonlara daha fazla ağırlık verir.
        """
        return self.temporal_decay ** delta_t
    
    def update_hebbian_weights(self, 
                             input_activation: torch.Tensor,
                             hidden_activation: torch.Tensor,
                             current_time: float) -> None:
        """
        Hebbian öğrenme kuralına göre ağırlıkları günceller.
        Doğru formül: Δw_ij = α · a_i · a_j · ω(t)
        Her bağlantı için ayrı ayrı hesaplanır.
        """
        # Zamansal ağırlık hesaplama
        if len(self.time_stamps) > 0:
            delta_t = current_time - self.time_stamps[-1]
            temporal_weight = self.temporal_weight_function(delta_t)
        else:
            temporal_weight = 1.0
            
        # Batch boyutunu kontrol et
        if input_activation.dim() == 1:
            input_activation = input_activation.unsqueeze(0)
        if hidden_activation.dim() == 1:
            hidden_activation = hidden_activation.unsqueeze(0)
            
        batch_size = input_activation.size(0)
        
        # Doğru Hebbian güncellemesi: her örnek için ayrı hesaplama
        delta_w = torch.zeros_like(self.hebbian_weights)
        
        for b in range(batch_size):
            # Her bağlantı için: w_ij += α * a_i * a_j * ω(t)
            input_vec = input_activation[b].unsqueeze(1)  # [input_size, 1]
            hidden_vec = hidden_activation[b].unsqueeze(0)  # [1, hidden_size]
            
            # Element-wise Hebbian update
            delta_w += self.learning_rate * (input_vec * hidden_vec) * temporal_weight
        
        # Batch ortalaması al
        delta_w = delta_w / batch_size
        
        with torch.no_grad():
            self.hebbian_weights += delta_w
            # Ağırlık sınırlandırması
            self.hebbian_weights.clamp_(-1.0, 1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        İleri besleme ve Hebbian güncelleme.
        """
        current_time = time.time()
        
        # Doğrusal dönüşüm
        hidden = torch.tanh(torch.matmul(x, self.hebbian_weights))
        
        # Aktivasyon geçmişini kaydet
        self.activation_history.append({
            'input': x.detach().clone(),
            'hidden': hidden.detach().clone(),
            'time': current_time
        })
        
        # Geçmişi sınırla (bellek yönetimi) - senkronizasyon korunur
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]
        if len(self.time_stamps) > 100:
            self.time_stamps = self.time_stamps[-100:]
        
        # Hebbian ağırlıkları güncelle
        self.update_hebbian_weights(x, hidden, current_time)
        self.time_stamps.append(current_time)
        
        return hidden, self.hebbian_weights
    
    def get_connection_strength(self, i: int, j: int) -> float:
        """
        Belirli bir bağlantının gücünü döndürür.
        """
        return self.hebbian_weights[i, j].item()
    
    def get_activation_correlation(self, window_size: int = 10) -> torch.Tensor:
        """
        Son aktivasyonlar arasındaki korelasyonu hesaplar.
        """
        if len(self.activation_history) < 2:
            return torch.zeros(self.hidden_size, self.hidden_size)
            
        recent_activations = [
            h['hidden'] for h in self.activation_history[-window_size:]
        ]
        
        if len(recent_activations) < 2:
            return torch.zeros(self.hidden_size, self.hidden_size)
            
        activation_matrix = torch.stack(recent_activations)
        correlation_matrix = torch.corrcoef(activation_matrix.T)
        
        # NaN değerleri temizle
        correlation_matrix = torch.nan_to_num(correlation_matrix, 0.0)
        
        return correlation_matrix
    
    def reset_history(self):
        """
        Aktivasyon geçmişini temizler.
        """
        self.activation_history = []
        self.time_stamps = []