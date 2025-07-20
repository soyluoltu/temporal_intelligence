# 🧠 Temporal Intelligence Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red)](https://pytorch.org)
[![🤗 Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/SoyluOltu/dynamic-temporal-learning-intelligence-model)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b)](temporal_intelligence_advanced_paper.md)

Bu proje, **zamansal farkındalığa sahip yapay zekâ sistemleri** için yenilikçi bir framework sunar. Hebbian öğrenme, zamansal dikkat mekanizmaları ve içsel model doğrulama sistemlerini birleştirerek **catastrophic forgetting** problemine çözüm getirir.

## 🌟 **Hugging Face Model**
Modelimiz Hugging Face Hub'da yayınlandı: **[SoyluOltu/dynamic-temporal-learning-intelligence-model](https://huggingface.co/SoyluOltu/dynamic-temporal-learning-intelligence-model)**

## 🧠 Temel Özellikler

### 1. Hebbian Öğrenme Mekanizması
- **Formül**: `Δw_ij(t) = α · a_i(t) · a_j(t) · ω(t)`
- Zamansal ağırlıklı sinaptik plastisite
- Aktivasyon geçmişi takibi
- Dinamik bağlantı güçlendirme

### 2. Zamansal Dikkat Sistemi
- **Formül**: `Attention(Q, K, V) = softmax(QK^T/√d_k + τ_t)V`
- Zamansal önyargı entegrasyonu
- Doğrulama güveni hesaplama
- Çok başlı dikkat mekanizması

### 3. Bellek Hiyerarşisi
- **Kısa Süreli Bellek (M_s)**: Anlık bağlam
- **Epizodik Bellek (M_e)**: Oturum bazlı öğrenim
- **Anlamsal Bellek (M_l)**: Kalıcı bilgi
- Otomatik konsolidasyon

### 4. İçsel Model Doğrulama
- **Karar Formülü**: `C_d = f(A_h, C_i, S_t)`
- Anlamsal tutarlılık kontrolü
- Mantıksal coherence değerlendirmesi
- Zamansal kararlılık analizi

### 5. Ortaya Çıkan Davranış Yönetimi
- Gerçek zamanlı desen takibi
- Karantina protokolü
- Üç kısıtlama modu (tutucu, keşifsel, uyarlayıcı)
- Grafik evrimi izleme

## 📦 Kurulum

### Gereksinimler
```bash
pip install -r requirements.txt
```

### Paket İçeriği
```
temporal_intelligence/
├── core/                    # Ana sistem bileşenleri
│   ├── temporal_system.py   # Ana zamansal zekâ sistemi
│   └── emergent_behavior.py # Ortaya çıkan davranış yönetimi
├── hebbian/                 # Hebbian öğrenme
│   └── hebbian_learning.py
├── attention/               # Dikkat mekanizmaları
│   └── temporal_attention.py
├── memory/                  # Bellek hiyerarşisi
│   └── memory_hierarchy.py
├── validation/              # Model doğrulama
│   └── model_validator.py
├── examples/                # Kullanım örnekleri
│   ├── basic_usage.py
│   └── advanced_scenarios.py
└── tests/                   # Birim testler
    └── test_basic_functionality.py
```

## 🚀 Hızlı Başlangıç

### Temel Kullanım
```python
from temporal_intelligence import TemporalIntelligenceSystem
import torch

# Sistem oluştur
system = TemporalIntelligenceSystem(
    d_model=128,
    n_heads=8,
    hebbian_hidden=64,
    learning_rate=0.01
)

# Veri hazırla
data = torch.randn(16, 10, 128)  # [batch, seq_len, d_model]
time_deltas = torch.arange(10, dtype=torch.float32)

# Sistem işleme
results = system(data, context="örnek_bağlam", time_deltas=time_deltas)

# Sonuçları incele
print(f"Doğrulama: {results['validation']['result']}")
print(f"Novelty: {results['behavior_analysis']['novelty_score']}")
print(f"Bellek: {results['memory_stats']}")
```

### Gelişmiş Özellikler
```python
from temporal_intelligence.core.emergent_behavior import ConstraintMode

# Kısıtlama modunu ayarla
system.set_constraint_mode(ConstraintMode.ADAPTIVE)

# Bellek konsolidasyonu
system.consolidate_memory()

# Sistem istatistikleri
stats = system.get_system_statistics()
print(stats)

# Checkpoint kaydetme/yükleme
system.save_checkpoint("model_checkpoint.pt")
system.load_checkpoint("model_checkpoint.pt")
```

## 📊 Örnekler

### 0. Hızlı Test (Import Kontrolü)
```bash
cd temporal_intelligence/
python3 quick_test.py
```

### 1. Temel Demo  
```bash
cd temporal_intelligence/examples/
python3 basic_usage.py
```

Bu örnek şunları gösterir:
- Temel sistem kullanımı
- Farklı kısıtlama modları
- Bellek konsolidasyonu
- Zamansal desen öğrenme
- Basit görselleştirme

### 2. Gelişmiş Senaryolar
```bash
cd temporal_intelligence/examples/
python3 advanced_scenarios.py
```

Bu örnek şunları içerir:
- Sürekli öğrenme (catastrophic forgetting testi)
- Ortaya çıkan davranış deneyleri
- Performans benchmark'ları
- Kısıtlama modu karşılaştırmaları

## 🧪 Testler

```bash
cd temporal_intelligence/tests/
python3 test_basic_functionality.py
python3 test_fixes.py
```

Test kapsamı:
- Hebbian öğrenme mekanizması
- Zamansal dikkat sistemi
- Bellek hiyerarşisi
- Model doğrulayıcı
- Ortaya çıkan davranış yöneticisi
- Ana sistem entegrasyonu

## 📈 Performans Karakteristikleri

### Kuramsal Projeksiyonlar (Makaleden)
| Metrik | Geleneksel Sistemler | Önerilen Yapı |
|--------|----------------------|-----------------|
| Bilgi Tutma | ~%67 | ~%89 |
| Entegrasyon Hatası | Yüksek | ~%50 azalma |
| Zamansal Akıl Yürütme | Statik | Dinamik |

### Hesaplama Maliyeti
- Bellek: ~1.3× baz modeller
- Yeni girdi gecikmesi: ~1.75×
- Tekrarlayan bağlam: ~1.1×
- Eğitim süresi: 2.1×

## 🎯 Uygulama Alanları

- **Otonom Sistemler**: Zamansal karar verme
- **Eğitim Teknolojileri**: Uyarlanabilir öğretim
- **Yaratıcı Üretim**: Bağlamsal içerik oluşturma
- **Bilimsel Simülasyonlar**: Temporal pattern analysis

## 🔧 Yapılandırma

### Sistem Parametreleri
```python
system = TemporalIntelligenceSystem(
    d_model=512,              # Model boyutu
    n_heads=8,                # Dikkat başı sayısı
    hebbian_hidden=256,       # Hebbian gizli boyutu
    learning_rate=0.01,       # Öğrenme hızı
    validation_threshold=0.7, # Doğrulama eşiği
    device='cuda'             # Hesaplama cihazı
)
```

### Bellek Ayarları
```python
from temporal_intelligence.memory.memory_hierarchy import MemoryHierarchy

memory = MemoryHierarchy(
    d_model=512,
    short_term_capacity=50,    # Kısa süreli bellek kapasitesi
    episodic_capacity=500,     # Epizodik bellek kapasitesi
    semantic_concepts=1000     # Anlamsal kavram sayısı
)
```

### Kısıtlama Modları
- **CONSERVATIVE**: Yüksek doğrulama eşiği (θ_v = 0.8)
- **EXPLORATORY**: Düşük doğrulama eşiği (θ_v = 0.3)
- **ADAPTIVE**: Dinamik eşik (θ_v = f(C_d))

## 📚 Referanslar

Bu uygulama aşağıdaki makaleden ilham almıştır:
- **"Sinirsel Sistemlerde Zamansal Zekâya Doğru"**
- Hebbian öğrenme + Dikkat mekanizmaları + İçsel model doğrulama
- Zamansal farkındalık ve ortaya çıkan davranış yönetimi

### Temel Kaynaklar
1. Hebb, D.O. (1949). *The Organization of Behavior*
2. Vaswani, A. et al. (2017). *Attention Is All You Need*
3. Kirkpatrick, J. et al. (2017). *Overcoming Catastrophic Forgetting*
4. Parisi, G.I. et al. (2019). *Continual Lifelong Learning*

## 🤝 Katkıda Bulunma

Bu proje araştırma amaçlı geliştirilmiştir. Katkılarınızı bekliyoruz:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'i push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir. Ticari kullanım için izin gereklidir.

## ⚠️ Dikkat

Bu uygulama kuramsal bir çerçevenin proof-of-concept implementasyonudur. Üretim ortamında kullanmadan önce kapsamlı testler yapılmalıdır.

---

*Bu kuramsal çerçeve, zamansal öğrenme ve ortaya çıkan yapay zekâ davranışları üzerine süregelen araştırmaları teşvik etmek amacıyla sunulmuştur.*