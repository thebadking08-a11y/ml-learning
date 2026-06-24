import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

print("="*60)
print("WEEK 2 - NUMPY FUNDAMENTALS + SEABORN VISUALIZATION")
print("="*60)

# DATASET: Diabetes (più interessante di Iris)
diabetes = load_diabetes()
X = diabetes.data  # 442 samples, 10 features
y = diabetes.target  # target: disease progression

print(f"\n1️⃣ DATASET INFO:")
print(f"  Shape X: {X.shape}")
print(f"  Shape y: {y.shape}")
print(f"  Features: {diabetes.feature_names}")

# NUMPY OPERATIONS
print(f"\n2️⃣ NUMPY OPERATIONS:")
print(f"  Media X: {np.mean(X, axis=0)}")
print(f"  Std X: {np.std(X, axis=0)}")
print(f"  Min y: {np.min(y)}, Max y: {np.max(y)}")
print(f"  Correlazione X[0] e y: {np.corrcoef(X[:, 0], y)[0, 1]:.3f}")

# Crea DataFrame
df = pd.DataFrame(X, columns=diabetes.feature_names)
df['target'] = y

print(f"\n3️⃣ DATAFRAME:")
print(df.head())

# SEABORN VISUALIZATIONS
print(f"\n📊 CREANDO GRAFICI SEABORN...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution plot
sns.histplot(data=df, x='target', kde=True, ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Distribuzione Target (Disease Progression)')

# 2. Correlation heatmap (prime 5 features)
corr_matrix = df.iloc[:, :5].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 1], cbar_kws={'label': 'Correlazione'})
axes[0, 1].set_title('Correlazione Features (prime 5)')

# 3. Scatter con regression line
sns.regplot(data=df, x='age', y='target', ax=axes[1, 0], scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
axes[1, 0].set_title('Relazione Age vs Target (con trend line)')

# 4. Box plot per feature selection
df_melted = df[['age', 'sex', 'bmi', 'target']].copy()
sns.boxplot(data=df, x='sex', y='target', ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Target Distribution by Sex')

plt.tight_layout()
plt.savefig('week2_diabetes_analysis.png', dpi=100)
print("✅ Grafici salvati come 'week2_diabetes_analysis.png'")
plt.show()

print("\n" + "="*60)
print("WEEK 2 ANALYSIS COMPLETE!")
print("="*60)
