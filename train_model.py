"""
Student Performance Prediction — Full ML Training Pipeline
==========================================================
1. Load & inspect data
2. Exploratory Data Analysis (plots saved to reports/)
3. Preprocessing  (label encoding + standard scaling)
4. Train 4 classifiers and compare
5. Detailed evaluation of the best model
6. Save model artefacts for the web app
7. Write a plain-text evaluation report

Usage:
    python train_model.py          # expects student_performance.csv in cwd
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")                                      # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)

warnings.filterwarnings("ignore")

# ── configuration ────────────────────────────────────────────
DATA_PATH = "student_performance.csv"
REPORTS_DIR = "reports"
MODEL_PATH = "model_artifacts.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(REPORTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

PASS_COLOR = "#27ae60"
FAIL_COLOR = "#e74c3c"


# ═══════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c != "performance"]

print(f"Shape : {df.shape}")
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"\nDtypes:\n{df.dtypes}\n")
print(f"Head:\n{df.head()}\n")
print(f"Describe:\n{df.describe().round(2)}\n")
print(f"Missing values:\n{df.isnull().sum()}\n")
print(f"Class distribution:\n{df['performance'].value_counts()}")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

# 2a  Class distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["performance"].value_counts()
colours = [PASS_COLOR if k == "Pass" else FAIL_COLOR for k in counts.index]
bars = ax.bar(counts.index, counts.values, color=colours, edgecolor="black", lw=0.5)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
            f"{v}\n({v / len(df) * 100:.1f}%)", ha="center", va="bottom", fontweight="bold")
ax.set_title("Class Distribution: Pass vs Fail", fontweight="bold", fontsize=13)
ax.set_ylabel("Number of Students")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/class_distribution.png", bbox_inches="tight")
plt.close()
print("  Saved class_distribution.png")

# 2b  Feature distributions (split by class)
n_cols = 3
n_rows = (len(feature_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    if df[col].nunique() <= 5:
        sns.countplot(data=df, x=col, hue="performance", ax=ax,
                      palette={"Pass": PASS_COLOR, "Fail": FAIL_COLOR})
    else:
        for perf, clr in [("Pass", PASS_COLOR), ("Fail", FAIL_COLOR)]:
            ax.hist(df[df["performance"] == perf][col], bins=20,
                    alpha=0.6, color=clr, label=perf, edgecolor="black", lw=0.3)
        ax.legend(fontsize=8)
    ax.set_title(col.replace("_", " ").title(), fontweight="bold", fontsize=10)
    ax.tick_params(labelsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Distributions by Performance", fontweight="bold", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/feature_distributions.png", bbox_inches="tight")
plt.close()
print("  Saved feature_distributions.png")

# 2c  Correlation heatmap
df_enc = df.copy()
df_enc["_target"] = (df_enc["performance"] == "Pass").astype(int)
# Only include numeric columns for correlation
numeric_df = df_enc.select_dtypes(include=[np.number])
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Data Preprocessing")
print("=" * 60)

X = df[feature_cols].copy()
y = df["performance"].copy()

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
print("StandardScaler applied.")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — MODEL TRAINING & COMPARISON
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Model Training & Comparison")
print("=" * 60)

MODELS = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
    "SVM (RBF)": SVC(random_state=RANDOM_STATE, kernel="rbf", probability=True),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

SCALE_REQUIRED = {"Logistic Regression", "SVM (RBF)", "KNN (k=5)"}

results = {}

for name, mdl in MODELS.items():
    use_scaled = name in SCALE_REQUIRED
    X_tr = X_train_sc if use_scaled else X_train
    X_te = X_test_sc if use_scaled else X_test

    mdl.fit(X_tr, y_train)
    y_pred = mdl.predict(X_te)
    y_prob = mdl.predict_proba(X_te)[:, 1] if hasattr(mdl, "predict_proba") else None
    cv = cross_val_score(mdl, X_tr, y_train, cv=5, scoring="accuracy")

    met = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1 Score":  f1_score(y_test, y_pred, zero_division=0),
        "CV Mean":   cv.mean(),
        "CV Std":    cv.std(),
    }
    results[name] = dict(model=mdl, metrics=met, y_pred=y_pred,
                         y_prob=y_prob, needs_scaling=use_scaled)
    print(f"\n  {name}")
    for k, v in met.items():
        print(f"    {k:12s}: {v:.4f}")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — COMPARISON VISUALISATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Model Comparison Visualisation")
print("=" * 60)

comp_df = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T.round(4)
print(comp_df.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metric_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "CV Mean"]
bar_colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
x = np.arange(len(MODELS))
w = 0.15

for i, (mk, bc) in enumerate(zip(metric_keys, bar_colors)):
    vals = [results[n]["metrics"][mk] for n in MODELS]
    axes[0].bar(x + i * w, vals, w, label=mk, color=bc, edgecolor="black", lw=0.3)

axes[0].set_xticks(x + w * 2)
axes[0].set_xticklabels([n.replace(" ", "\n") for n in MODELS], fontsize=8)
axes[0].set_ylabel("Score")
axes[0].set_title("Metric Comparison", fontweight="bold")
axes[0].legend(fontsize=7, loc="lower right")
axes[0].set_ylim(0.4, 1.05)
axes[0].grid(axis="y", alpha=0.3)

cv_means = [results[n]["metrics"]["CV Mean"] for n in MODELS]
cv_stds = [results[n]["metrics"]["CV Std"] for n in MODELS]
bars = axes[1].barh(list(MODELS.keys()), cv_means, xerr=cv_stds,
                    color="#3498db", edgecolor="black", lw=0.5, capsize=5)
axes[1].set_xlim(0.4, 1.05)
axes[1].set_xlabel("CV Accuracy (Mean ± Std)")
axes[1].set_title("5-Fold Cross-Validation", fontweight="bold")
axes[1].grid(axis="x", alpha=0.3)
for b, m in zip(bars, cv_means):
    axes[1].text(m + 0.01, b.get_y() + b.get_height() / 2, f"{m:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/model_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved model_comparison.png")


# ═══════════════════════════════════════════════════════════════
# STEP 6 — BEST MODEL DETAILED EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Best Model — Detailed Evaluation")
print("=" * 60)

best_name = max(results, key=lambda n: results[n]["metrics"]["F1 Score"])
best = results[best_name]
best_model = best["model"]
best_met = best["metrics"]
X_te_final = X_test_sc if best["needs_scaling"] else X_test

print(f"Best model: {best_name}  (F1 = {best_met['F1 Score']:.4f})")

# Confusion matrix
cm = confusion_matrix(y_test, best["y_pred"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=1, linecolor="black", ax=ax)
ax.set_xlabel("Predicted", fontweight="bold")
ax.set_ylabel("Actual", fontweight="bold")
ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/confusion_matrix.png", bbox_inches="tight")
plt.close()
print("  Saved confusion_matrix.png")

# Classification report string
cls_report = classification_report(y_test, best["y_pred"], target_names=le.classes_)
print(f"\n{cls_report}")

# ROC curve
if best["y_prob"] is not None:
    fpr, tpr, _ = roc_curve(y_test, best["y_prob"])
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=FAIL_COLOR, lw=2,
            label=f"ROC (AUC = {roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "gray", lw=1, ls="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color=FAIL_COLOR)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title(f"ROC Curve — {best_name}", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/roc_curve.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved roc_curve.png   AUC = {roc_auc_val:.4f}")

# Feature importance
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
else:
    from sklearn.inspection import permutation_importance
    _perm = permutation_importance(best_model, X_te_final, y_test,
                                   n_repeats=10, random_state=RANDOM_STATE)
    importances = _perm.importances_mean

sorted_idx = np.argsort(importances)
fig, ax = plt.subplots(figsize=(8, 5))
normed = importances[sorted_idx] / importances.max() if importances.max() > 0 else importances[sorted_idx]
ax.barh(range(len(feature_cols)), importances[sorted_idx],
        color=plt.cm.RdYlGn(normed), edgecolor="black", lw=0.3)
ax.set_yticks(range(len(feature_cols)))
ax.set_yticklabels([feature_cols[i].replace("_", " ").title() for i in sorted_idx])
ax.set_xlabel("Importance", fontweight="bold")
ax.set_title(f"Feature Importance — {best_name}", fontweight="bold", fontsize=13)
ax.grid(axis="x", alpha=0.3)
for bar_obj, val in zip(ax.patches, importances[sorted_idx]):
    if val > 0.005:
        ax.text(val + importances.max() * 0.01,
                bar_obj.get_y() + bar_obj.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/feature_importance.png", bbox_inches="tight")
plt.close()
print("  Saved feature_importance.png")


# ═══════════════════════════════════════════════════════════════
# STEP 7 — SAVE ARTEFACTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Saving Model Artefacts")
print("=" * 60)

artefacts = {
    "model": best_model,
    "scaler": scaler,
    "feature_names": feature_cols,
    "feature_importances": importances,
    "class_mapping": {int(i): lbl for i, lbl in zip(le.transform(le.classes_), le.classes_)},
    "model_name": best_name,
    "metrics": {k: float(v) for k, v in best_met.items()},
    "needs_scaling": best["needs_scaling"],
}

joblib.dump(artefacts, MODEL_PATH)
print(f"  → {MODEL_PATH}  ({os.path.getsize(MODEL_PATH) / 1024:.1f} KB)")
print(f"    Model     : {best_name}")
print(f"    Features  : {len(feature_cols)}")
print(f"    Classes   : {artefacts['class_mapping']}")


# ═══════════════════════════════════════════════════════════════
# STEP 8 — TEXT REPORT
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Generating Evaluation Report")
print("=" * 60)

lines = [
    "STUDENT PERFORMANCE PREDICTION — EVALUATION REPORT",
    "=" * 55,
    "",
    f"Dataset       : {DATA_PATH}",
    f"Total samples : {len(df)}",
    f"Train / Test  : {len(X_train)} / {len(X_test)}",
    f"Features      : {len(feature_cols)}",
    f"Feature list  : {', '.join(feature_cols)}",
    "",
    "CLASS DISTRIBUTION",
    f"  Pass : {(df['performance']=='Pass').sum()}  ({(df['performance']=='Pass').mean()*100:.1f}%)",
    f"  Fail : {(df['performance']=='Fail').sum()}  ({(df['performance']=='Fail').mean()*100:.1f}%)",
    "",
    "MODEL COMPARISON",
    comp_df.to_string(),
    "",
    f"SELECTED MODEL : {best_name}",
    f"  Accuracy  : {best_met['Accuracy']:.4f}",
    f"  Precision : {best_met['Precision']:.4f}",
    f"  Recall    : {best_met['Recall']:.4f}",
    f"  F1 Score  : {best_met['F1 Score']:.4f}",
    f"  CV Mean   : {best_met['CV Mean']:.4f}  (+/- {best_met['CV Std']:.4f})",
    "",
    "CLASSIFICATION REPORT",
    cls_report,
    "",
    "TOP-5 FEATURE IMPORTANCE",
]
for rank, idx in enumerate(np.argsort(importances)[::-1][:5], 1):
    lines.append(f"  {rank}. {feature_cols[idx].replace('_',' ').title():40s} {importances[idx]:.4f}")

lines += [
    "",
    "SAVED FILES",
    f"  Model artefacts : {MODEL_PATH}",
    f"  Report plots    : {REPORTS_DIR}/",
    "",
    "=" * 55,
    "Pipeline completed successfully.",
]

report_txt = "\n".join(lines)
report_path = f"{REPORTS_DIR}/evaluation_report.txt"
with open(report_path, "w") as f:
    f.write(report_txt)
print(f"  → {report_path}")

print("\n" + report_txt)
print("\n" + "=" * 60)
print("ALL DONE.")
print("=" * 60)
