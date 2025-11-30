# Modelo de agrupación y reducción de dimensionalidad - Iris

Este repositorio contiene:

- Un modelo de **agrupación (clustering)** usando K-Means sobre el conjunto de datos Iris.
- Un modelo de **reducción de dimensionalidad** usando PCA sobre el mismo conjunto de datos.

## Estructura

- `clustering/iris_kmeans.py`: código para entrenar y evaluar K-Means con diferentes valores de k.
- `reduccion_dimensionalidad/iris_pca.py`: código para aplicar PCA, analizar la varianza explicada y visualizar los datos en 2D.
- `resultados_pca_iris.csv`: (opcional) archivo generado con las componentes principales.

## Requisitos

```bash
pip install scikit-learn matplotlib pandas
