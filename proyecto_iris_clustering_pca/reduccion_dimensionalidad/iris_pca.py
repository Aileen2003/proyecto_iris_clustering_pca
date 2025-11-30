# reduccion_dimensionalidad/iris_pca.py

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# 1. Cargar datos
iris = load_iris()
X = iris.data
y = iris.target  # solo para colorear la gráfica

# 2. Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA con todos los componentes para ver varianza explicada
pca_full = PCA(n_components=4)
X_pca_full = pca_full.fit_transform(X_scaled)

var_explicada = pca_full.explained_variance_ratio_
var_acumulada = var_explicada.cumsum()

print("Varianza explicada por componente:", var_explicada)
print("Varianza explicada acumulada:", var_acumulada)

# 4. Tabla de evaluación
df_resultados = pd.DataFrame({
    "n_componentes": [1, 2, 3, 4],
    "varianza_explicada_acumulada": var_acumulada
})
print("\nTabla de varianza acumulada:")
print(df_resultados)

# 5. PCA con 2 componentes (modelo final)
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

# 6. Gráfica de los datos en 2D
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y)
plt.title("PCA con 2 componentes - Iris")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.show()

# 7. Guardar resultados en CSV (opcional)
df_pca = pd.DataFrame(X_pca_2, columns=["PC1", "PC2"])
df_pca["clase_real"] = y
df_pca.to_csv("resultados_pca_iris.csv", index=False)
