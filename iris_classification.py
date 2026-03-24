"""
Task 4: Iris Flower Classification
====================================
Dataset   : Iris (sklearn built-in)
Goal      : Classify flowers into Setosa / Versicolor / Virginica
Models    : Logistic Regression (baseline), KNN, SVM, Random Forest (main)
Evaluation: Accuracy, Confusion Matrix, Classification Report, ROC-AUC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.linear_model   import LogisticRegression
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.svm            import SVC
from sklearn.ensemble       import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc
)

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  IRIS FLOWER CLASSIFICATION")
print("=" * 55)

raw = load_iris(as_frame=True)
df  = raw.frame.copy()
df.columns = [*raw.feature_names, "species_id"]
df["species"] = df["species_id"].map(dict(enumerate(raw.target_names)))

print(f"\n[1] Dataset loaded: {df.shape[0]} samples × {df.shape[1]} columns")
print(f"   Classes : {raw.target_names.tolist()}")
print(f"   Samples per class:\n{df['species'].value_counts().to_string()}")

# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
print("\n[2] Basic statistics:")
print(df.drop(columns=["species_id","species"]).describe().round(3).to_string())

missing = df.isnull().sum()
print(f"\nMissing values: {missing.sum()} total  ✓" if missing.sum() == 0 else f"\nMissing values:\n{missing[missing > 0]}")

# ─────────────────────────────────────────────
# 3. FEATURES & SPLIT
# ─────────────────────────────────────────────
FEATURES = raw.feature_names          # all 4 features
X = df[FEATURES]
y = df["species_id"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[3] Split: {len(X_train)} train  |  {len(X_test)} test  (80/20, stratified)")

# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[4] Training models ...")

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=500, random_state=42))
    ]),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  KNeighborsClassifier(n_neighbors=5))
    ]),
    "SVM (RBF kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None,
        random_state=42, n_jobs=-1
    ),
}

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc     = accuracy_score(y_test, y_pred)
    cv_acc  = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    # One-vs-Rest AUC
    y_bin   = label_binarize(y_test, classes=[0, 1, 2])
    auc_ovr = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    results[name] = {
        "Accuracy": acc,
        "CV Accuracy": cv_acc.mean(),
        "CV Std": cv_acc.std(),
        "AUC (macro)": auc_ovr,
        "preds": y_pred,
        "proba": y_proba,
    }
    print(f"   {name:<24}  Acc={acc:.4f}  CV={cv_acc.mean():.4f}±{cv_acc.std():.4f}  AUC={auc_ovr:.4f}")

# Best model
best_name  = max(results, key=lambda k: results[k]["CV Accuracy"])
best_model = models[best_name]
best_preds = results[best_name]["preds"]
best_proba = results[best_name]["proba"]
print(f"\n   Best model → {best_name}  (CV Acc={results[best_name]['CV Accuracy']:.4f})")

# ─────────────────────────────────────────────
# 5. DETAILED REPORT
# ─────────────────────────────────────────────
print(f"\n[5] Classification report — {best_name}:")
print(classification_report(y_test, best_preds, target_names=raw.target_names))

# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
rf = models["Random Forest"]
fi_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)
print("[6] Feature importances (Random Forest):")
print(fi_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Iris Flower Classification — Results", fontsize=15, fontweight="bold")

CLASS_NAMES = raw.target_names
PALETTE     = ["#4C72B0", "#DD8452", "#55A868"]

# (a) Confusion matrix
ax = axes[0, 0]
cm = confusion_matrix(y_test, best_preds)
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}")

# (b) Feature importance
ax = axes[0, 1]
colors = ["#4C72B0","#DD8452","#55A868","#C44E52"]
ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color=colors[::-1])
ax.set_xlabel("Importance")
ax.set_title("Feature Importances (Random Forest)")

# (c) Model accuracy comparison
ax = axes[0, 2]
names = list(results.keys())
accs  = [results[n]["CV Accuracy"] for n in names]
stds  = [results[n]["CV Std"]      for n in names]
bars  = ax.barh(names, accs, xerr=stds, color="#4C72B0", ecolor="gray", capsize=4)
ax.set_xlim(0.85, 1.02)
ax.set_xlabel("CV Accuracy")
ax.set_title("Model Comparison (5-fold CV)")
for bar, acc in zip(bars, accs):
    ax.text(acc + 0.002, bar.get_y() + bar.get_height()/2,
            f"{acc:.3f}", va="center", fontsize=9)

# (d) Pairplot-style scatter (Petal Length vs Petal Width)
ax = axes[1, 0]
for i, (cls, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
    mask = y == i
    ax.scatter(X.loc[mask, "petal length (cm)"],
               X.loc[mask, "petal width (cm)"],
               label=cls, color=color, alpha=0.7, s=40, edgecolors="white", linewidth=0.3)
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.set_title("Petal Length vs Width")
ax.legend()

# (e) Sepal scatter
ax = axes[1, 1]
for i, (cls, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
    mask = y == i
    ax.scatter(X.loc[mask, "sepal length (cm)"],
               X.loc[mask, "sepal width (cm)"],
               label=cls, color=color, alpha=0.7, s=40, edgecolors="white", linewidth=0.3)
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Sepal Width (cm)")
ax.set_title("Sepal Length vs Width")
ax.legend()

# (f) ROC curves (best model)
ax = axes[1, 2]
y_bin = label_binarize(y_test, classes=[0, 1, 2])
for i, (cls, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], best_proba[:, i])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={roc_auc:.2f})")
ax.plot([0,1],[0,1],"k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curves — {best_name}")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/iris_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[7] Plot saved → outputs/iris_results.png")

# ─────────────────────────────────────────────
# 8. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FINAL RESULTS SUMMARY")
print("=" * 55)
print(f"  Best Model     : {best_name}")
print(f"  Test Accuracy  : {results[best_name]['Accuracy']:.4f}")
print(f"  CV  Accuracy   : {results[best_name]['CV Accuracy']:.4f} ± {results[best_name]['CV Std']:.4f}")
print(f"  Macro AUC-ROC  : {results[best_name]['AUC (macro)']:.4f}")
print("=" * 55)
