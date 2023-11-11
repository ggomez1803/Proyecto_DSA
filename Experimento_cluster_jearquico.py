import pandas as pd
from DLTV import lfv
import gower
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def calcular_outliers(df: pd.DataFrame, col:str):
    """ Calcula los outliers de una variable
    Args:
        df (DataFrame): Dataset a utilizar
        col (str): Nombre de la columna
    Returns:
        DataFrame: Dataset sin outliers
        DataFrame: Dataset con outliers"""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    no_outliers = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f'Se encontraron {len(outliers)} outliers en la variable {col} con un límite inferior de {lower} y un límite superior de {upper}')
    print(f'Se analizará una base con {len(no_outliers)} registros')
    return no_outliers, outliers

# Calcular el rango IQR para la variable DLTV
donaciones_sin_outlier, outliers = calcular_outliers(lfv, 'DLTV')

# Definir variables a utilizar
model_cols = ['DLTV', 'FK_ID_Genero', 'Canal_Principal', 'VL_Edad', 'Prom_Cuotas_Pagadas', 'Prom_Cuotas_No_Pagadas', 'FK_ID_Estado_Civil']
gower_dist = gower.gower_matrix(donaciones_sin_outlier[model_cols])
gower_dist = pd.DataFrame(gower_dist)

# Definir el dendograma
Z = linkage(gower_dist, method='average', metric='euclidean')
clusters = fcluster(Z, t=4, criterion='maxclust')

# Graficar
fig, ax = plt.subplots(figsize=(12, 8))
d = dendrogram(Z, show_leaf_counts=True, leaf_font_size=14, ax=ax)
ax.set_xlabel('Observaciones', fontsize=14)
ax.set_yticks(np.arange(0,50,10))
ax.set_ylabel('Distancia', fontsize=14)
plt.show()

# Numero de clusters
n_clusters = len(np.unique(clusters))
# Tamaño de clusters
cluster_size = pd.Series(clusters).value_counts().sort_index()
# Porcentaje de observaciones en cada cluster
cluster_pct = cluster_size / len(clusters)
# DLTV promedio por cluster
cluster_dltv = donaciones_sin_outlier.groupby(clusters)['DLTV'].mean()
# Promedio cuotas pagadas por cluster
cluster_cuotas_pagadas = donaciones_sin_outlier.groupby(clusters)['Prom_Cuotas_Pagadas'].mean()
# Promedio cuotas no pagadas por cluster
cluster_cuotas_no_pagadas = donaciones_sin_outlier.groupby(clusters)['Prom_Cuotas_No_Pagadas'].mean()
# Promedio edad por cluster
cluster_edad = donaciones_sin_outlier.groupby(clusters)['VL_Edad'].mean()
# Cuenta de generos por cluster
cluster_genero = donaciones_sin_outlier.groupby(clusters)['FK_ID_Genero'].value_counts().unstack().fillna(0)
# Cuenta de canales por cluster
cluster_canal = donaciones_sin_outlier.groupby(clusters)['Canal_Principal'].value_counts().unstack().fillna(0)
# Cuenta de estado civil por cluster
cluster_estado_civil = donaciones_sin_outlier.groupby(clusters)['FK_ID_Estado_Civil'].value_counts().unstack().fillna(0)