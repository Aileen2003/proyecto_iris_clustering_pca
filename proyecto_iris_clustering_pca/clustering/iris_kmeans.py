# clustering/iris_kmeans.py
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Cargar datos
iris = load_iris()
X = iris.data  # 4 características

# 2. Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Probar distintos valores de k y evaluar con silhouette
valores_k = [2, 3, 4, 5]
puntajes_silhouette = []

for k in valores_k:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    etiquetas = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, etiquetas)
    puntajes_silhouette.append(score)
    print(f"k = {k}, silhouette = {score:.4f}")

# 4. Entrenar el modelo final con k=3
k_optimo = 3
kmeans_final = KMeans(n_clusters=k_optimo, n_init=10, random_state=42)
labels_final = kmeans_final.fit_predict(X_scaled)

print("\nCentroides (espacio estandarizado):")
print(kmeans_final.cluster_centers_)

# 5. Gráfica sencilla usando dos primeras características (solo para ilustrar)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_final)
plt.title("Agrupación K-Means sobre Iris (2 características)")
plt.xlabel("Característica 1 (escalada)")
plt.ylabel("Característica 2 (escalada)")
plt.show()
