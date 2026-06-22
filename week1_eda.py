import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Carica dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target_names[iris.target]

print("="*60)
print("EXPLORATORY DATA ANALYSIS - IRIS DATASET")
print("="*60)

# 1. Info generali
print("\n1️⃣ SHAPE & INFO:")
print(f"  Righe: {df.shape[0]}, Colonne: {df.shape[1]}")
print(f"\n{df.info()}")

# 2. Statistiche descrittive
print("\n2️⃣ STATISTICHE DESCRITTIVE:")
print(df.describe())

# 3. Prime righe
print("\n3️⃣ PRIME 5 RIGHE:")
print(df.head())

# 4. Conteggio per specie
print("\n4️⃣ DISTRIBUZIONE SPECIE:")
print(df['Species'].value_counts())

# 5. Correlazione
print("\n5️⃣ CORRELAZIONE TRA FEATURES:")
print(df.iloc[:, :-1].corr())

# 6. Visualizzazioni
print("\n📊 CREANDO GRAFICI...")

# Histogram - distribuzione di una feature
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(df['sepal length (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribuzione Sepal Length')
plt.xlabel('cm')

# Scatter plot - relazione tra due features
plt.subplot(2, 2, 2)
for species in iris.target_names:
    mask = df['Species'] == species
    plt.scatter(df[mask]['sepal length (cm)'], 
               df[mask]['sepal width (cm)'], 
               label=species, alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Width (per specie)')
plt.legend()

# Box plot - confronto tra specie
plt.subplot(2, 2, 3)
df.boxplot(column='petal length (cm)', by='Species')
plt.title('Petal Length per Specie')
plt.suptitle('')

# Scatter plot - Petal
plt.subplot(2, 2, 4)
for species in iris.target_names:
    mask = df['Species'] == species
    plt.scatter(df[mask]['petal length (cm)'], 
               df[mask]['petal width (cm)'], 
               label=species, alpha=0.7)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Petal Length vs Width (per specie)')
plt.legend()

plt.tight_layout()
plt.savefig('iris_eda.png', dpi=100)
print("✅ Grafico salvato come 'iris_eda.png'")
plt.show()

print("\n" + "="*60)
print("EDA COMPLETATA!")
print("="*60)
