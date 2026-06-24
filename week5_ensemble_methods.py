import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR_sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

print("="*60)
print("WEEK 5 - ENSEMBLE METHODS: RANDOM FOREST")
print("="*60)

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X = (X - X.mean(axis=0)) / X.std(axis=0)

print(f"\n1️⃣ DATASET:")
print(f"  Shape X: {X.shape} (30 features)")
print(f"  Malignant: {sum(y==0)}, Benign: {sum(y==1)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n2️⃣ TRAINING MODELLI...")

print(f"  Training Random Forest (100 alberi)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)

print(f"  Training Logistic Regression...")
lr = LR_sklearn(max_iter=10000, random_state=42)
lr.fit(X_train, y_train)

print(f"\n3️⃣ VALUTAZIONE:")

rf_pred = rf.predict(X_test)
rf_pred_proba = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred_proba)
rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

lr_pred = lr.predict(X_test)
lr_pred_proba = lr.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_pred_proba)
lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred)
lr_rec = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print(f"\n  RANDOM FOREST:")
print(f"    ROC-AUC: {rf_auc:.4f}")
print(f"    Accuracy: {rf_acc:.4f}")
print(f"    Precision: {rf_prec:.4f}")
print(f"    Recall: {rf_rec:.4f}")
print(f"    F1-Score: {rf_f1:.4f}")

print(f"\n  LOGISTIC REGRESSION:")
print(f"    ROC-AUC: {lr_auc:.4f}")
print(f"    Accuracy: {lr_acc:.4f}")
print(f"    Precision: {lr_prec:.4f}")
print(f"    Recall: {lr_rec:.4f}")
print(f"    F1-Score: {lr_f1:.4f}")

print(f"\n  VINCITORE: {'Random Forest' if rf_auc > lr_auc else 'Logistic Regression'} (+{abs(rf_auc-lr_auc):.4f} AUC)")

feature_importance = rf.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:]
top_features_names = [cancer.feature_names[i] for i in top_features_idx]
top_features_values = feature_importance[top_features_idx]

print(f"\n4️⃣ TOP 10 FEATURE IMPORTANCE:")
for name, value in zip(top_features_names, top_features_values):
    print(f"  {name}: {value:.4f}")

print(f"\n📊 CREANDO GRAFICI...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].barh(range(len(top_features_names)), top_features_values, color='steelblue')
axes[0, 0].set_yticks(range(len(top_features_names)))
axes[0, 0].set_yticklabels(top_features_names)
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Top 10 Feature Importance (Random Forest)')
axes[0, 0].invert_yaxis()

models = ['Random Forest', 'Logistic Reg']
scores = [rf_auc, lr_auc]
colors = ['green' if rf_auc > lr_auc else 'gray', 'orange' if lr_auc > rf_auc else 'gray']
axes[0, 1].bar(models, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 1].set_ylabel('ROC-AUC Score')
axes[0, 1].set_title('Model Comparison (ROC-AUC)')
axes[0, 1].set_ylim([0.9, 1.0])
for i, v in enumerate(scores):
    axes[0, 1].text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rf_metrics = [rf_acc, rf_prec, rf_rec, rf_f1]
lr_metrics = [lr_acc, lr_prec, lr_rec, lr_f1]

x = np.arange(len(metrics))
width = 0.35

axes[1, 0].bar(x - width/2, rf_metrics, width, label='Random Forest', color='steelblue', alpha=0.8)
axes[1, 0].bar(x + width/2, lr_metrics, width, label='Logistic Reg', color='orange', alpha=0.8)
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Classification Metrics Comparison')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

cm_rf = confusion_matrix(y_test, rf_pred)
im = axes[1, 1].imshow(cm_rf, cmap='Blues', aspect='auto')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title('Confusion Matrix (Random Forest)')
for i in range(2):
    for j in range(2):
        axes[1, 1].text(j, i, str(cm_rf[i, j]), ha='center', va='center', color='white' if cm_rf[i, j] > cm_rf.max()/2 else 'black', fontsize=12)
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])

plt.tight_layout()
plt.savefig('week5_ensemble_methods.png', dpi=100)
print("✅ Grafici salvati come 'week5_ensemble_methods.png'")
plt.show()

print("\n" + "="*60)
print("ENSEMBLE METHODS COMPLETATA!")
print("MILESTONE: ADVANCED ML MODELS ✅")
print("="*60)
