import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
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

# =====================================================
# 1. VERİ SETİNİ YÜKLEME
# =====================================================

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


# =====================================================
# 2. EKSİK VERİ KONTROLÜ
# =====================================================

print("\n--- Eksik Veri Kontrolü ---")
print(df.isnull().sum())


# =====================================================
# 3. QUALITY DEĞİŞKENİNİ SINIFLANDIRMA
# =====================================================

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


# =====================================================
# 4. VERİ GÖRSELLEŞTİRME - SINIF DAĞILIMI
# =====================================================

plt.figure(figsize=(6, 4))
sns.countplot(x="quality_class", data=df, order=["Low", "Medium", "High"])
plt.title("Quality Class Distribution")
plt.xlabel("Quality Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# =====================================================
# 5. KORELASYON MATRİSİ
# =====================================================

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# =====================================================
# 6. BAĞIMSIZ VE BAĞIMLI DEĞİŞKENLER
# =====================================================

X = df.drop(["quality", "quality_class"], axis=1)
y = df["quality_class"]


# =====================================================
# 7. EĞİTİM VE TEST VERİSİNE AYIRMA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# 8. BASELINE DECISION TREE MODELİ
# =====================================================

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

print("\n================ BASELINE DECISION TREE RESULTS ================")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall / Sensitivity:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =====================================================
# 9. BASELINE CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, y_pred, labels=dt_model.classes_)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=dt_model.classes_,
    yticklabels=dt_model.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Baseline Decision Tree")
plt.tight_layout()
plt.show()


# =====================================================
# 10. HYPERPARAMETER OPTIMIZATION - GRIDSEARCHCV
# =====================================================

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 7, 8, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n================ GRID SEARCH RESULTS ================")
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)


# =====================================================
# 11. OPTIMIZED DECISION TREE MODELİ
# =====================================================

best_dt = grid_search.best_estimator_

y_pred_best = best_dt.predict(X_test)

print("\n================ OPTIMIZED DECISION TREE RESULTS ================")

print("Optimized Accuracy:", accuracy_score(y_test, y_pred_best))
print("Optimized Precision:", precision_score(y_test, y_pred_best, average="weighted"))
print("Optimized Recall / Sensitivity:", recall_score(y_test, y_pred_best, average="weighted"))
print("Optimized F1 Score:", f1_score(y_test, y_pred_best, average="weighted"))
print("Balanced Accuracy:",balanced_accuracy_score(y_test, y_pred_best))

print("\nOptimized Classification Report:\n")
print(classification_report(y_test, y_pred_best))


# =====================================================
# 12. OPTIMIZED CONFUSION MATRIX
# =====================================================

cm_best = confusion_matrix(y_test, y_pred_best, labels=best_dt.classes_)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_best,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=best_dt.classes_,
    yticklabels=best_dt.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Optimized Decision Tree")
plt.tight_layout()
plt.show()


# =====================================================
# 13. OVERFITTING ANALİZİ
# =====================================================

train_accuracy = best_dt.score(X_train, y_train)
test_accuracy = best_dt.score(X_test, y_test)

print("\n================ OVERFITTING ANALYSIS ================")
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

if train_accuracy > test_accuracy + 0.10:
    print("Yorum: Modelde overfitting eğilimi olabilir.")
else:
    print("Yorum: Eğitim ve test sonuçları arasında büyük fark yoktur.")


# =====================================================
# 14. FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_dt.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n================ FEATURE IMPORTANCE ================")
print(importance_df)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)
plt.title("Feature Importance - Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# =====================================================
# 15. DECISION TREE GÖRSELLEŞTİRME
# =====================================================

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
plt.show()


# =====================================================
# 16. RAPOR İÇİN KISA ÖZET
# =====================================================

print("\n================ PROJECT SUMMARY ================")
print("Dataset: Wine Quality - Red Wine")
print("Method: Decision Tree Classification")
print("Target Variable: quality_class")
print("Classes: Low, Medium, High")
print("Baseline Accuracy:", accuracy_score(y_test, y_pred))
print("Optimized Accuracy:", accuracy_score(y_test, y_pred_best))
print("Baseline F1 Score:", f1_score(y_test, y_pred, average="weighted"))
print("Optimized F1 Score:", f1_score(y_test, y_pred_best, average="weighted"))
print("Best Parameters:", grid_search.best_params_)

results_df = pd.DataFrame({
    "Model": ["Baseline Decision Tree", "Optimized Decision Tree"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_best)
    ],
    "Precision": [
        precision_score(y_test, y_pred, average="weighted"),
        precision_score(y_test, y_pred_best, average="weighted")
    ],
    "Recall": [
        recall_score(y_test, y_pred, average="weighted"),
        recall_score(y_test, y_pred_best, average="weighted")
    ],
    "F1 Score": [
        f1_score(y_test, y_pred, average="weighted"),
        f1_score(y_test, y_pred_best, average="weighted")
    ]
})

print("\n================ MODEL COMPARISON ================")
print(results_df)