import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)
from imblearn.over_sampling import SMOTE


# 1.VERİ SETİNİ YÜKLEME
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

print("\n--- İlk 5 Satır ---")
print(df.head())

print("\n--- Veri Seti Bilgisi ---")
print(df.info())

print("\n--- İstatistiksel Özet ---")
print(df.describe())

print("\n--- Orijinal Quality Dağılımı ---")
print(df["quality"].value_counts().sort_index())


# 2.EKSİK VERİ KONTROLÜ
print("\n--- Eksik Veri Kontrolü ---")
print(df.isnull().sum())


# 3.QUALITY DEĞİŞKENİNİ SINIFLANDIRMA
def quality_class(q):
    if q <= 5:
        return "Low"
    elif q == 6:
        return "Medium"
    else:
        return "High"


df["quality_class"] = df["quality"].apply(quality_class)

print("\n--- Yeni Quality Class Dağılımı ---")
print(df["quality_class"].value_counts())


# 4.VERİ GÖRSELLEŞTİRME - SINIF DAĞILIMI
plt.figure(figsize=(6, 4))
sns.countplot(x="quality_class", data=df, order=["Low", "Medium", "High"])
plt.title("Quality Class Distribution")
plt.xlabel("Quality Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()


# 5.KORELASYON MATRİSİ
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
plt.show()


# 6.BAĞIMSIZ VE BAĞIMLI DEĞİŞKENLER
X = df.drop(["quality", "quality_class"], axis=1)
y = df["quality_class"]


# 7.EĞİTİM VE TEST VERİSİNE AYIRMA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nEğitim seti boyutu : {X_train.shape}")
print(f"Test seti boyutu   : {X_test.shape}")


# 8.SINIF DENGESİZLİĞİ ANALİZİ VE SMOTE
print("\n--- Eğitim Seti Sınıf Dağılımı (SMOTE Öncesi) ---")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\n--- Eğitim Seti Sınıf Dağılımı (SMOTE Sonrası) ---")
print(pd.Series(y_train_sm).value_counts())

#SMOTE sonrası dağılım görseli
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pd.Series(y_train).value_counts().reindex(["Low", "Medium", "High"]).plot(
    kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452", "#55A868"], title="SMOTE Öncesi Dağılım"
)
pd.Series(y_train_sm).value_counts().reindex(["Low", "Medium", "High"]).plot(
    kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452", "#55A868"], title="SMOTE Sonrası Dağılım"
)
for ax in axes:
    ax.set_xlabel("Quality Class")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("smote_comparison.png", dpi=150)
plt.show()


# 9.YARDIMCI FONKSİYON: SPECİFİCİTY HESAPLAMA
def compute_specificity(y_true, y_pred, classes):
    """
    Her sınıf için Specificity (True Negative Rate) hesaplar.
    Macro-average olarak döner.
    """
    specificities = []
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    for i in range(len(classes)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificities.append(spec)
    return np.mean(specificities)


def print_full_metrics(y_true, y_pred, model, label="Model"):
    """Tüm metrikleri hesaplayıp yazdırır."""
    classes = model.classes_
    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred, average="weighted")
    rec      = recall_score(y_true, y_pred, average="weighted")   # Sensitivity
    f1       = f1_score(y_true, y_pred, average="weighted")
    spec     = compute_specificity(y_true, y_pred, classes)

    print(f"\n{'='*20} {label} {'='*20}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  Balanced Accuracy  : {bal_acc:.4f}")
    print(f"  Precision          : {prec:.4f}")
    print(f"  Recall/Sensitivity : {rec:.4f}")
    print(f"  Specificity        : {spec:.4f}")
    print(f"  F1 Score           : {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")
    return {
        "Model": label,
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "Precision": prec,
        "Recall (Sensitivity)": rec,
        "Specificity": spec,
        "F1 Score": f1
    }


# 10.BASELINE DECISION TREE MODELİ (SMOTE'lu veri ile)
dt_baseline = DecisionTreeClassifier(random_state=42)
dt_baseline.fit(X_train_sm, y_train_sm)
y_pred_base = dt_baseline.predict(X_test)

baseline_metrics = print_full_metrics(y_test, y_pred_base, dt_baseline, "BASELINE DECISION TREE")

#Cross-Validation (Baseline)
cv_base = cross_val_score(
    DecisionTreeClassifier(random_state=42),
    X_train_sm, y_train_sm,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_weighted"
)
print(f"\n  5-Fold CV F1 (Baseline): {cv_base.mean():.4f} ± {cv_base.std():.4f}")

#Baseline Confusion Matrix
cm_base = confusion_matrix(y_test, y_pred_base, labels=dt_baseline.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_base, annot=True, fmt="d", cmap="Blues",
    xticklabels=dt_baseline.classes_, yticklabels=dt_baseline.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Baseline Decision Tree")
plt.tight_layout()
plt.savefig("cm_baseline.png", dpi=150)
plt.show()


# 11.HYPERPARAMETER OPTİMİZASYONU - GRIDSEARCHCV
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 7, 8, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_weighted",
    n_jobs=-1
)

grid_search.fit(X_train_sm, y_train_sm)

print("\n================ GRID SEARCH RESULTS ================")
print("Best Parameters:", grid_search.best_params_)
print("Best CV F1 Score:", grid_search.best_score_)


# 12.OPTİMİZE DECISION TREE MODELİ
best_dt = grid_search.best_estimator_
y_pred_best = best_dt.predict(X_test)

optimized_metrics = print_full_metrics(y_test, y_pred_best, best_dt, "OPTIMIZED DECISION TREE")

#Cross-Validation (Optimized)
cv_best = cross_val_score(
    grid_search.best_estimator_,
    X_train_sm, y_train_sm,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_weighted"
)
print(f"\n  5-Fold CV F1 (Optimized): {cv_best.mean():.4f} ± {cv_best.std():.4f}")

#Optimized Confusion Matrix
cm_best = confusion_matrix(y_test, y_pred_best, labels=best_dt.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_best, annot=True, fmt="d", cmap="Greens",
    xticklabels=best_dt.classes_, yticklabels=best_dt.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Optimized Decision Tree")
plt.tight_layout()
plt.savefig("cm_optimized.png", dpi=150)
plt.show()


# 13.OVERFİTTİNG ANALİZİ
train_acc = best_dt.score(X_train_sm, y_train_sm)
test_acc  = best_dt.score(X_test, y_test)

print("\n================ OVERFİTTİNG ANALİZİ ================")
print(f"  Train Accuracy : {train_acc:.4f}")
print(f"  Test  Accuracy : {test_acc:.4f}")
print(f"  Fark           : {abs(train_acc - test_acc):.4f}")
if train_acc > test_acc + 0.10:
    print("  Yorum: Modelde overfitting eğilimi olabilir.")
else:
    print("  Yorum: Eğitim ve test sonuçları arasında büyük fark yoktur.")


# 14.FEATURE IMPORTANCE
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_dt.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n================ FEATURE IMPORTANCE ================")
print(importance_df.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance - Optimized Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()


# 15.DECISION TREE GÖRSELLEŞTİRME
plt.figure(figsize=(22, 10))
plot_tree(
    best_dt,
    feature_names=X.columns,
    class_names=best_dt.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Optimized Decision Tree Visualization")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.show()


# 16.AKADEMİK ÇALIŞMA KARŞILAŞTIRMASI
#
#  Referans: Cortez et al. (2009) - "Modeling wine preferences by data mining
#  from physicochemical properties." Decision Support Systems, 47(4), 547-553.
#  https://doi.org/10.1016/j.dss.2009.05.016
#
#  Cortez ve ark. regresyon / sıralı sınıflandırma yaklaşımını kullanmış,
#  orijinal 0-10 kalite skoru üzerinde çalışmıştır.
#  SVM tabanlı modelde yaklaşık %62,4 doğruluk elde etmişlerdir.
#  Biz 3 sınıflı (Low/Medium/High) basitleştirilmiş bir problem üzerinde
#  Decision Tree kullandık; bu nedenle doğrudan karşılaştırma için
#  sonuçlar aşağıda normalize edilmiş biçimde sunulmuştur.
academic_metrics = {
    "Model": "Cortez et al. (2009) - SVM [Akademik]",
    "Accuracy": 0.6240,          # Makale'den alınan yaklaşık değer
    "Balanced Accuracy": None,
    "Precision": None,
    "Recall (Sensitivity)": None,
    "Specificity": None,
    "F1 Score": None
}

print("\n================ AKADEMİK ÇALIŞMA KARŞILAŞTIRMASI ================")
print("Referans: Cortez et al. (2009) - Decision Support Systems")
print(f"  Yöntem           : Support Vector Machine (SVM)")
print(f"  Problem Tipi     : Orijinal 0-10 kalite skoru sınıflandırma")
print(f"  Rapor Edilen Acc : ~%62.4")
print()
print("NOT: Cortez ve ark. orijinal 10-sınıflı problemi ele almıştır.")
print("Bu çalışmada 3-sınıflı (Low/Medium/High) basitleştirilmiş")
print("yaklaşım kullanıldığından doğrudan karşılaştırma metodolojik")
print("farklılık içermektedir.")


# 17.KAPSAMLI MODEL KARŞILAŞTIRMA TABLOSU
comparison_rows = [academic_metrics, baseline_metrics, optimized_metrics]
comparison_df = pd.DataFrame(comparison_rows).set_index("Model")

print("\n================ MODEL KARŞILAŞTIRMA TABLOSU ================")
print(comparison_df.to_string())

#Görsel karşılaştırma (Akademik çalışma için sadece Accuracy karşılaştırılabilir)
metrics_to_plot = ["Accuracy", "Balanced Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1 Score"]
our_models = comparison_df.drop(index="Cortez et al. (2009) - SVM [Akademik]")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    vals = our_models[metric].astype(float)
    bars = axes[i].bar(
        ["Baseline DT", "Optimized DT"],
        vals,
        color=["#4C72B0", "#55A868"],
        edgecolor="black"
    )
    #Cortez et al. Accuracy için referans çizgisi
    if metric == "Accuracy":
        axes[i].axhline(y=0.624, color="red", linestyle="--", linewidth=1.5, label="Cortez et al. (2009)\n~0.624")
        axes[i].legend(fontsize=8)
    axes[i].set_title(metric)
    axes[i].set_ylim(0, 1.05)
    axes[i].set_ylabel("Score")
    for bar in bars:
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f"{height:.3f}", ha="center", va="bottom", fontsize=9)

plt.suptitle("Model Karşılaştırma (Tüm Metrikler)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()


# 18.PROJE ÖZET RAPORU
print("\n" + "="*60)
print("              PROJE ÖZET RAPORU")
print("="*60)
print(f"  Veri Seti    : Wine Quality - Red Wine (UCI)")
print(f"  Yöntem       : Decision Tree Classification")
print(f"  Hedef        : quality_class (Low / Medium / High)")
print(f"  Sınıf Dengesi: SMOTE ile dengelenmiştir")
print()
print("  ---- Baseline Decision Tree ----")
print(f"  Accuracy      : {baseline_metrics['Accuracy']:.4f}")
print(f"  Bal. Accuracy : {baseline_metrics['Balanced Accuracy']:.4f}")
print(f"  Precision     : {baseline_metrics['Precision']:.4f}")
print(f"  Sensitivity   : {baseline_metrics['Recall (Sensitivity)']:.4f}")
print(f"  Specificity   : {baseline_metrics['Specificity']:.4f}")
print(f"  F1 Score      : {baseline_metrics['F1 Score']:.4f}")
print()
print("  ---- Optimized Decision Tree ----")
print(f"  Accuracy      : {optimized_metrics['Accuracy']:.4f}")
print(f"  Bal. Accuracy : {optimized_metrics['Balanced Accuracy']:.4f}")
print(f"  Precision     : {optimized_metrics['Precision']:.4f}")
print(f"  Sensitivity   : {optimized_metrics['Recall (Sensitivity)']:.4f}")
print(f"  Specificity   : {optimized_metrics['Specificity']:.4f}")
print(f"  F1 Score      : {optimized_metrics['F1 Score']:.4f}")
print()
print("  ---- Akademik Referans ----")
print(f"  Cortez et al. (2009) SVM Accuracy: ~0.6240")
print(f"  DOI: 10.1016/j.dss.2009.05.016")
print()
print(f"  En İyi Parametreler : {grid_search.best_params_}")
print("="*60)