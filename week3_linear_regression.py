import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("WEEK 3 - LINEAR REGRESSION DA ZERO")
print("="*60)

np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 7, 8, 8, 9, 10])

print(f"\n1️⃣ DATASET:")
print(f"  X (Size): {X.flatten()}")
print(f"  y (Price): {y}")

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0
        self.b = 0
        self.losses = []
    
    def calculate_loss(self, X, y, predictions):
        '''MSE - Mean Squared Error'''
        return np.mean((y - predictions) ** 2)
    
    def fit(self, X, y):
        '''Addestra il modello'''
        n = len(X)
        for i in range(self.iterations):
            predictions = self.m * X.flatten() + self.b
            errors = y - predictions
            loss = self.calculate_loss(X, y, predictions)
            self.losses.append(loss)
            dm = (-2/n) * np.sum(X.flatten() * errors)
            db = (-2/n) * np.sum(errors)
            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i+1}: Loss = {loss:.4f}")
    
    def predict(self, X):
        '''Fa previsioni'''
        return self.m * X.flatten() + self.b

print(f"\n2️⃣ TRAINING (Gradient Descent):")
model = LinearRegression(learning_rate=0.01, iterations=500)
model.fit(X, y)

print(f"\n3️⃣ RISULTATI FINALI:")
print(f"  m (pendenza): {model.m:.4f}")
print(f"  b (intercetta): {model.b:.4f}")
print(f"  Loss finale: {model.losses[-1]:.4f}")

X_test = np.array([5.5, 7.5, 11]).reshape(-1, 1)
predictions = model.predict(X_test)
print(f"\n4️⃣ PREVISIONI SU NUOVI DATI:")
for x_val, pred in zip(X_test.flatten(), predictions):
    print(f"  Size {x_val} → Price €{pred*100:.0f}k")

print(f"\n📊 CREANDO GRAFICI...")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].scatter(X, y, color='blue', s=100, label='Dati reali', alpha=0.7)
X_line = np.linspace(0, 11, 100).reshape(-1, 1)
y_line = model.predict(X_line)
axes[0].plot(X_line, y_line, color='red', linewidth=2, label='Linea regressione')
axes[0].set_xlabel('Size (migliaia mq)')
axes[0].set_ylabel('Price (€100k)')
axes[0].set_title('Linear Regression Fit')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(model.losses, color='green', linewidth=2)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Loss (MSE)')
axes[1].set_title('Loss Curve - Gradient Descent Convergence')
axes[1].grid(True, alpha=0.3)

predictions_train = model.predict(X)
residuals = y - predictions_train
axes[2].scatter(X, residuals, color='purple', s=100, alpha=0.7)
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[2].set_xlabel('Size')
axes[2].set_ylabel('Residuale (errore)')
axes[2].set_title('Residuals Plot')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('week3_linear_regression.png', dpi=100)
print("✅ Grafici salvati come 'week3_linear_regression.png'")
plt.show()

print("\n" + "="*60)
print("LINEAR REGRESSION COMPLETATA!")
print("MILESTONE: ML FUNDAMENTALS RAGGIUNTI ✅")
print("="*60)
