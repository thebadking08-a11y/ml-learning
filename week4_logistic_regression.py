import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

print("="*60)
print("WEEK 4 - LOGISTIC REGRESSION & CLASSIFICATION")
print("="*60)

cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target

X = (X - X.mean(axis=0)) / X.std(axis=0)

print(f"\n1️⃣ DATASET:")
print(f"  Shape X: {X.shape}")
print(f"  Classe 0 (maligno): {sum(y==0)}")
print(f"  Classe 1 (benigno): {sum(y==1)}")

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = np.zeros(X.shape[1])
        self.b = 0
        self.losses = []
    
    def sigmoid(self, z):
        '''Sigmoid activation function'''
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        '''Addestra il modello'''
        n = len(X)
        
        for i in range(self.iterations):
            z = np.dot(X, self.m) + self.b
            predictions = self.sigmoid(z)
            
            loss = -np.mean(y * np.log(predictions + 1e-8) + (1-y) * np.log(1-predictions + 1e-8))
            self.losses.append(loss)
            
            dz = predictions - y
            dm = (1/n) * np.dot(X.T, dz)
            db = (1/n) * np.sum(dz)
            
            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db
            
            if (i + 1) % 200 == 0:
                print(f"  Iteration {i+1}: Loss = {loss:.4f}")
    
    def predict(self, X):
        '''Fa previsioni (probabilità)'''
        z = np.dot(X, self.m) + self.b
        return self.sigmoid(z)
    
    def predict_class(self, X, threshold=0.5):
        '''Previsioni come classe (0 o 1)'''
        return (self.predict(X) >= threshold).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n2️⃣ TRAINING (Binary Cross-Entropy Loss):")
model = LogisticRegression(learning_rate=0.1, iterations=1000)
model.fit(X_train, y_train)

print(f"\n3️⃣ RISULTATI:")
y_pred_proba = model.predict(X_test)
y_pred_class = model.predict_class(X_test)
cm = confusion_matrix(y_test, y_pred_class)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"  ROC-AUC Score: {roc_auc:.4f}")
print(f"  Confusion Matrix:")
print(f"    True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"    False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
precision = cm[1,1] / (cm[1,1] + cm[0,1])
recall = cm[1,1] / (cm[1,1] + cm[1,0])
f1 = 2 * (precision * recall) / (precision + recall)

print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")

print(f"\n📊 CREANDO GRAFICI...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(model.losses, color='blue', linewidth=2)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Binary Cross-Entropy Loss')
axes[0, 0].set_title('Training Loss Curve')
axes[0, 0].grid(True, alpha=0.3)

xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

axes[0, 1].contourf(xx, yy, Z, alpha=0.3, levels=[0, 0.5, 1], colors=['red', 'blue'])
axes[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdBu', s=50, edgecolors='black')
axes[0, 1].set_xlabel('Feature 1')
axes[0, 1].set_ylabel('Feature 2')
axes[0, 1].set_title('Decision Boundary')

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

im = axes[1, 1].imshow(cm, cmap='Blues', aspect='auto')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=12)
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])

plt.tight_layout()
plt.savefig('week4_logistic_regression.png', dpi=100)
print("✅ Grafici salvati come 'week4_logistic_regression.png'")
plt.show()

print("\n" + "="*60)
print("LOGISTIC REGRESSION COMPLETATA!")
print("MILESTONE: CLASSIFICATION FUNDAMENTALS ✅")
print("="*60)
