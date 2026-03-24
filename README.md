# Iris Flower Classification

**ARCH Technologies — Machine Learning Internship (Month 2) | Task 4**

---

## Overview

A machine learning pipeline that classifies iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — using petal and sepal measurements. The project covers preprocessing, multi-model training, cross-validation, and visualisation including ROC curves.

---

## Project Structure

```
task4_iris_classification/
│
├── iris_classification.py   # Full ML pipeline
├── outputs/
│   └── iris_results.png     # Auto-generated result plots
└── README.md
```

---

## Requirements

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

The dataset is loaded directly from `sklearn.datasets.load_iris` — no downloads needed.

---

## Dataset

| Property | Value |
|---|---|
| Source | `sklearn.datasets.load_iris` |
| Samples | 150 (perfectly balanced — 50 per class) |
| Features | 4 |
| Target | Species: Setosa · Versicolor · Virginica |

| Feature | Description |
|---|---|
| `sepal length (cm)` | Length of the outer leaf-like structure |
| `sepal width (cm)` | Width of the outer leaf-like structure |
| `petal length (cm)` | Length of the inner petal |
| `petal width (cm)` | Width of the inner petal |

---

## Methodology

### 1. Preprocessing
- Zero missing values — no imputation needed
- `StandardScaler` applied inside `Pipeline` objects for scale-sensitive models (Logistic Regression, KNN, SVM)
- Stratified 80/20 train/test split to preserve class balance in both sets

### 2. Models

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=500`, scaled features |
| K-Nearest Neighbors | `k=5`, scaled features |
| SVM (RBF kernel) | `C=1.0`, probability outputs enabled for ROC |
| Random Forest | 200 trees — also used for feature importance analysis |

### 3. Evaluation
- **Accuracy** — overall fraction of correct predictions
- **Precision / Recall / F1** — full per-class classification report
- **Stratified 5-fold CV** — robust accuracy estimate across folds
- **Macro AUC-ROC** (One-vs-Rest) — discrimination ability across all three classes

---

## Results

| Model | Test Accuracy | CV Accuracy | Macro AUC |
|---|---|---|---|
| Logistic Regression | 0.9333 | 0.9533 ± 0.045 | 0.9967 |
| **K-Nearest Neighbors ✓** | **0.9333** | **0.9733 ± 0.025** | **0.9933** |
| SVM (RBF) | 0.9667 | 0.9600 ± 0.039 | 0.9967 |
| Random Forest | 0.9000 | 0.9600 ± 0.039 | 0.9867 |

**Best Model (by CV stability): K-Nearest Neighbors**

| Metric | Value |
|---|---|
| Test Accuracy | 0.9333 |
| CV Accuracy (5-fold) | **0.9733 ± 0.0249** |
| Macro AUC-ROC | 0.9933 |

### Classification Report (KNN)

```
              precision  recall  f1-score  support
setosa           1.00    1.00      1.00       10
versicolor       0.83    1.00      0.91       10
virginica        1.00    0.80      0.89       10
accuracy                           0.93       30
```

### Feature Importances (Random Forest)

| Feature | Importance |
|---|---|
| petal length (cm) | 0.454 |
| petal width (cm) | 0.412 |
| sepal length (cm) | 0.116 |
| sepal width (cm) | 0.018 |

### Key Findings
- **Petal dimensions** account for ~87% of predictive importance — far more discriminative than sepal features
- **Setosa** is perfectly separable from the other two classes (a well-known property of this dataset)
- **Versicolor and Virginica** show minor overlap in sepal space, causing the 2 misclassifications
- KNN was chosen as the best model due to its lowest CV variance (±2.5%), indicating the most consistent generalisation

---

## Output Plots

Running the script generates `outputs/iris_results.png` with 6 panels:

1. **Confusion Matrix** — highlights the 2 misclassified Virginica samples
2. **Feature Importances** — Random Forest importance ranking
3. **Model Comparison** — CV accuracy with error bars across all 4 models
4. **Petal Scatter Plot** — clear 3-cluster separation in petal space
5. **Sepal Scatter Plot** — overlapping Versicolor/Virginica clusters in sepal space
6. **ROC Curves** — per-class AUC near 1.0 for all three species

---

## How to Run

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python iris_classification.py
```

Plots are saved to `outputs/iris_results.png` automatically.

---

*ARCH Technologies ML Internship — Month 2, Task 4*
