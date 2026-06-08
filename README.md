# 🍷 Kırmızı Şeker Kalitesi Sınıflandırması
### Decision Tree | SMOTE | GridSearchCV

> UCI Machine Learning Repository - Wine Quality (Red) veri seti üzerinde  
> Karar Ağacı yöntemiyle çok sınıflı kalite tahmini

---

## 📋 Proje Özeti

Bu proje, kırmızı şekerin 11 fizikokimyasal özelliğini kullanarak kalite sınıfını (**Low / Medium / High**) tahmin eden bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır. Sınıf dengesizliği SMOTE ile giderilmiş, model hiperparametreleri GridSearchCV ile optimize edilmiş ve sonuçlar Cortez et al. (2009) akademik çalışmasıyla karşılaştırılmıştır.

---

## 📁 Proje Yapısı

```
wine-quality-classification/
│
├── main.py                    # Ana kaynak kodu
├── README.md                  # Bu dosya
│
├── outputs/
│   ├── class_distribution.png     # Sınıf dağılımı grafiği
│   ├── correlation_matrix.png     # Korelasyon matrisi
│   ├── smote_comparison.png       # SMOTE öncesi/sonrası
│   ├── cm_baseline.png            # Baseline confusion matrix
│   ├── cm_optimized.png           # Optimized confusion matrix
│   ├── model_comparison.png       # Model karşılaştırma grafikleri
│   ├── feature_importance.png     # Özellik önem grafiği
│   └── decision_tree.png          # Karar ağacı görseli
│
└── report/
    └── wine_quality_DETAYLI_RAPOR.docx   # Proje raporu
```

---

## 🗂️ Veri Seti

| Özellik | Değer |
|---------|-------|
| **Kaynak** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) |
| **Dosya** | `winequality-red.csv` |
| **Örnek Sayısı** | 1.599 |
| **Özellik Sayısı** | 11 bağımsız değişken |
| **Hedef Değişken** | `quality` → `quality_class` (Low / Medium / High) |
| **Eksik Veri** | Yok |

### Sınıf Dağılımı

| Sınıf | Kalite Puanları | Örnek | Oran |
|-------|----------------|-------|------|
| Low | 3, 4, 5 | 744 | %46.5 |
| Medium | 6 | 638 | %39.9 |
| High | 7, 8 | 217 | %13.6 |

---

## ⚙️ Kurulum

### Gereksinimler

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Çalıştırma

```bash
python main.py
```

---

## 🔬 Yöntem

### İşlem Adımları

1. **Veri Yükleme & EDA** — İstatistiksel özet, dağılım ve korelasyon analizi
2. **Eksik Veri Kontrolü** — Tüm sütunlar tam
3. **Sınıf Etiketi Oluşturma** — `quality` → Low / Medium / High
4. **Eğitim/Test Bölünmesi** — %80 / %20, `stratify=y`
5. **SMOTE** — Sınıf dengesizliği giderildi (her sınıf: 595 örnek)
6. **Baseline Model** — Varsayılan hiperparametrelerle Decision Tree
7. **GridSearchCV** — 5-fold Stratified CV, `scoring='f1_weighted'`
8. **Optimized Model** — En iyi parametrelerle Decision Tree
9. **Değerlendirme** — 7 farklı metrik + overfitting analizi

### GridSearch Parametre Uzayı

```python
param_grid = {
    "criterion":          ["gini", "entropy"],
    "max_depth":          [3, 4, 5, 6, 7, 8, 10],
    "min_samples_split":  [2, 5, 10],
    "min_samples_leaf":   [1, 2, 4]
}
# Toplam: 2 × 7 × 3 × 3 = 126 kombinasyon × 5-fold = 630 model
```

---

## 📊 Sonuçlar

### Model Performans Karşılaştırması

| Metrik | Baseline DT | Optimized DT | Cortez et al. (2009) SVM |
|--------|-------------|--------------|--------------------------|
| **Accuracy** | **0.6969** | 0.6281 | ~0.6240 |
| Balanced Accuracy | **0.7047** | 0.6335 | — |
| Precision | **0.7055** | 0.6370 | — |
| Recall / Sensitivity | **0.6969** | 0.6281 | — |
| Specificity | **0.8392** | 0.8026 | — |
| F1 Score | **0.6985** | 0.6282 | — |
| 5-Fold CV F1 | 0.7052 ± 0.038 | 0.6955 ± 0.023 | — |

### En İyi Hiperparametreler

```python
{
    "criterion":         "entropy",
    "max_depth":         10,
    "min_samples_leaf":  2,
    "min_samples_split": 2
}
```

### ⚠️ Overfitting Tespiti

```
Train Accuracy : 0.9070
Test  Accuracy : 0.6281
Fark           : 0.2789  ← Kritik eşik (0.10) aşılıyor!
```

Optimized modelde `max_depth=10` nedeniyle belirgin overfitting tespit edilmiştir.  
Baseline model test performansı açısından daha başarılı olmuştur.

---

## 🔍 Özellik Önemi

| Sıra | Özellik | Önem Skoru |
|------|---------|------------|
| 1 | alcohol | 0.2711 |
| 2 | sulphates | 0.1290 |
| 3 | total sulfur dioxide | 0.1069 |
| 4 | volatile acidity | 0.0997 |
| 5 | free sulfur dioxide | 0.0732 |
| … | … | … |

> **Alkol içeriği** (%27.1), şeker kalitesini tahmin etmede en belirleyici kimyasal parametre olarak öne çıkmaktadır.

---

## 📚 Akademik Karşılaştırma

> **Referans:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).  
> *Modeling wine preferences by data mining from physicochemical properties.*  
> Decision Support Systems, 47(4), 547–553.  
> DOI: [10.1016/j.dss.2009.05.016](https://doi.org/10.1016/j.dss.2009.05.016)

Cortez et al. aynı veri seti üzerinde SVM ile **~%62.4** doğruluk elde etmiştir.  
Bu çalışmanın Baseline modeli **%69.7** ile bu değeri geçmektedir.  
*(Not: Cortez et al. 10 sınıflı orijinal problemi; bu çalışma 3 sınıflı basitleştirilmiş problemi ele almaktadır.)*

---

## 💡 İleriye Yönelik Öneriler

- [ ] `max_depth` parametresini 3–6 aralığına kısıtlamak veya `ccp_alpha` ile budama uygulamak
- [ ] Random Forest veya Gradient Boosting (XGBoost) ile karşılaştırma
- [ ] ADASYN veya Borderline-SMOTE alternatifleri denemek
- [ ] ROC-AUC analizi eklemek
- [ ] Özellik mühendisliği (alkol/asit oranı gibi türetilmiş özellikler)

---

## 🛠️ Kullanılan Teknolojiler

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)
![matplotlib](https://img.shields.io/badge/matplotlib-3.x-11557c)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green)

---

## 📄 Lisans

Bu proje eğitim amaçlı hazırlanmıştır.  
Veri seti: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) — Cortez et al. (2009)
