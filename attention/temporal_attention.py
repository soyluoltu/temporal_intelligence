"""
Zamansal Dikkat Mekanizması  
============================

Makaledeki formüle göre:
Attention(Q, K, V) = softmax(QK^T/√d_k + τ_t)V

burada τ_t zamansal yakınlık ve doğrulama güvenini kodlar.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class TemporalAttention(nn.Module):
    """
    Zamansal bilgileri içeren dikkat mekanizması.
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 temporal_decay: float = 0.9,
                 validation_weight: float = 0.1,
                 device: str = 'cpu'):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k or d_model // n_heads
        self.d_v = d_v or d_model // n_heads
        self.temporal_decay = temporal_decay
        self.validation_weight = validation_weight
        self.device = device
        
        # Doğrusal dönüşümler
        self.w_q = nn.Linear(d_model, n_heads * self.d_k, device=device)
        self.w_k = nn.Linear(d_model, n_heads * self.d_k, device=device)
        self.w_v = nn.Linear(d_model, n_heads * self.d_v, device=device)
        self.w_o = nn.Linear(n_heads * self.d_v, d_model, device=device)
        
        # Zamansal kodlama
        self.temporal_encoder = nn.Linear(1, self.d_k, device=device)
        
        # Doğrulama güveni hesaplama
        self.confidence_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        ).to(device)
        
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def compute_temporal_bias(self, 
                            seq_len: int, 
                            batch_size: int,
                            time_deltas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Zamansal önyargı τ_t hesaplama.
        Returns: [batch_size, n_heads, seq_len, seq_len]
        """
        if time_deltas is None:
            # Varsayılan: ardışık zaman adımları
            time_deltas = torch.arange(seq_len, device=self.device).float()
        
        # Time deltas boyutunu kontrol et
        if time_deltas.dim() == 1:
            time_deltas = time_deltas.unsqueeze(0).expand(batch_size, -1)
        
        # Her batch için zamansal uzaklık matrisi
        temporal_bias_batch = []
        for b in range(batch_size):
            batch_time_deltas = time_deltas[b] if time_deltas.size(0) > 1 else time_deltas[0]
            time_diff_matrix = batch_time_deltas.unsqueeze(0) - batch_time_deltas.unsqueeze(1)
            
            # Zamansal ağırlık: exp(-|Δt|/decay)
            temporal_weights = torch.exp(-torch.abs(time_diff_matrix) / self.temporal_decay)
            
            # Log-space'e dönüştür (softmax için)
            temporal_bias = torch.log(temporal_weights + 1e-8)
            temporal_bias_batch.append(temporal_bias)
        
        # Stack ve broadcast to all heads
        temporal_bias = torch.stack(temporal_bias_batch, dim=0)  # [batch_size, seq_len, seq_len]
        temporal_bias = temporal_bias.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [batch_size, n_heads, seq_len, seq_len]
        
        return temporal_bias
    
    def compute_validation_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Doğrulama güveni C_v hesaplama.
        """
        return self.confidence_network(x)
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor,
                time_deltas: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zamansal dikkat hesaplama.
        
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]  
            value: [batch_size, seq_len, d_model]
            time_deltas: [seq_len] zamansal bilgi
            mask: [batch_size, seq_len, seq_len] dikkat maskesi
        """
        batch_size, seq_len, _ = query.size()
        
        # Q, K, V dönüşümleri
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        
        # Standart dikkat skorları
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Zamansal önyargı hesaplama - doğru boyutlarla
        temporal_bias = self.compute_temporal_bias(seq_len, batch_size, time_deltas)
        
        # Doğrulama güveni - boyut düzeltmesi
        confidence = self.compute_validation_confidence(query)  # [batch_size, seq_len, 1]
        # Confidence'ı attention score boyutuna genişlet
        confidence_expanded = confidence.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [batch_size, n_heads, seq_len, 1]
        confidence_bias = confidence_expanded * self.validation_weight
        
        # Toplam dikkat skoru: QK^T/√d_k + τ_t + C_v
        # temporal_bias: [batch_size, n_heads, seq_len, seq_len]
        # confidence_bias: [batch_size, n_heads, seq_len, 1] - broadcast olacak
        attention_scores = attention_scores + temporal_bias + confidence_bias
        
        # Maske uygulama
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Değer ağırlıklandırması
        context = torch.matmul(attention_weights, V)
        
        # Çıkış projeksionu
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.d_v
        )
        output = self.w_o(context)
        
        return output, attention_weights
    
    def get_attention_statistics(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """
        Dikkat ağırlıklarından istatistikler çıkarır.
        """
        # Entropi hesaplama (ne kadar odaklanmış)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        
        # Maksimum dikkat değeri
        max_attention = torch.max(attention_weights, dim=-1)[0]
        
        return {
            'mean_entropy': entropy.mean().item(),
            'mean_max_attention': max_attention.mean().item(),
            'attention_sparsity': (attention_weights < 0.01).float().mean().item()
        }