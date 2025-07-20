# Hızlı Düzeltme Önerileri - 5 Dakikada Uygulayın

## 🚀 Acil Parametre Düzeltmeleri

Sistem şu anda çok muhafazakar ayarlanmış. Bu 5 basit değişiklik ile dramatik iyileştirme sağlayabilirsiniz:

## 1. Validation Threshold Düzelt

**Dosya**: `validation/model_validator.py`
```python
# Satır ~25 civarında
def __init__(self, 
             d_model: int, 
             validation_threshold: float = 0.4,  # 0.7 → 0.4 değiştir
             device: str = 'cpu'):
```

## 2. Novelty Detection Hassasiyeti Artır

**Dosya**: `core/emergent_behavior.py`
```python
# Satır ~237 civarında  
def __init__(self,
             d_model: int,
             novelty_threshold: float = 0.4,      # 0.7 → 0.4 değiştir
             stability_threshold: float = 0.6,    # 0.8 → 0.6 değiştir
             validation_window: int = 10):
```

## 3. Pattern Similarity Gevşet

**Dosya**: `core/emergent_behavior.py`
```python
# Satır ~362 civarında
# Benzer desen var mı kontrol et - uyarlanabilir eşik
similarity_threshold = 0.75 if self.constraint_mode == ConstraintMode.CONSERVATIVE else 0.65  # 0.85 → 0.75, 0.75 → 0.65
```

## 4. Bellek Konsolidasyonu Aktifleştir

**Dosya**: `memory/memory_hierarchy.py`
```python
# Satır ~447 civarında
def consolidate(self, threshold: float = 0.3):  # 0.8 → 0.3 değiştir
```

## 5. Kısıtlama Modları Yumuşat

**Dosya**: `core/emergent_behavior.py`
```python
# Satır ~479 civarında
if mode == ConstraintMode.CONSERVATIVE:
    self.validation_threshold = 0.6        # 0.8 → 0.6
elif mode == ConstraintMode.EXPLORATORY:
    self.validation_threshold = 0.2        # 0.3 → 0.2
```

## 🔄 Test Etmek İçin

Değişiklikleri yaptıktan sonra:

```bash
cd temporal_intelligence/examples/
python3 basic_usage.py
```

**Beklenen İyileştirmeler**:
- Validation skorları: "reject" → "quarantine" veya "accept"
- Novelty skorları: Daha geniş aralık
- Bellek kullanımı: Episodic memory'de aktivite
- Desen keşfi: İlk desenler görünmeye başlar

## 📊 Hızlı Kontrol

Bu komutla sonuçları hızlıca test edin:
```bash
cd temporal_intelligence/examples/
python3 advanced_scenarios.py | grep -E "(Kabul oranı|Ortalama novelty|Toplam desen)"
```

**Hedef Sonuçlar**:
- Kabul oranı: 0/20 → 3-7/20
- Ortalama novelty: 0.45-0.50 → 0.40-0.60 (daha geniş aralık)
- Toplam desen: 0 → 1-5

## ⚡ Ekstra İpucu

Eğer hala çok muhafazakar ise, bu ek değişikliği de yapın:

**Dosya**: `core/temporal_system.py`
```python
# Satır ~56 civarında
validation_threshold: float = 0.3,  # 0.7 → 0.3
```

Bu değişiklikler sistemi araştırma için daha kullanışlı hale getirecek!

---
*Not: Bu değişiklikler sistemi daha "keşifsel" yapar. Güvenlik-kritik uygulamalar için orijinal değerleri kullanın.*