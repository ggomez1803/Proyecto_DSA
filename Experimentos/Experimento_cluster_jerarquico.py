import pandas as pd
import gower
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

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

lfv = pd.read_csv('./Archivos_Cliente/Base_lfv.csv', encoding='latin1')
df_donantes = pd.read_csv('./Archivos_Cliente/Base_donantes_preprocesada.csv', encoding='latin1')
df_transacciones = pd.read_csv('./Archivos_Cliente/Transacciones_Individuales_preprocesada.csv', encoding='latin1')

# Hacer una tabla dinámica que muestre por donante en qué canal entró su donación
tipo_transacciones = df_transacciones.pivot_table(index='FK_ID_Donante', columns='FK_CD_Registro', values='VL_Importe', aggfunc='count').fillna(0)
# Calcular el número de canales por donante
tipo_transacciones['Canales_Donacion'] = tipo_transacciones.apply(lambda x: np.count_nonzero(x), axis=1)
# Calcular el canal principal de donación
tipo_transacciones['Canal_Principal'] = tipo_transacciones.apply(lambda x: x.idxmax(), axis=1)
# Agregar variables a la tabla de donantes
lfv = pd.merge(lfv, df_donantes[['ID_Donante', 'FK_ID_Genero', 'FK_ID_Estado_Civil']], on='ID_Donante', how='left')
lfv = pd.merge(lfv, tipo_transacciones[['Canales_Donacion', 'Canal_Principal']], left_on='ID_Donante', right_on= 'FK_ID_Donante', how='left')

# Calcular el rango IQR para la variable DLTV
donaciones_sin_outlier, outliers = calcular_outliers(lfv, 'DLTV')

# Tomar una muestra aleatoria de donantes
donaciones_sin_outlier = donaciones_sin_outlier.sample(n=1000, random_state=1)

experiment = mlflow.set_experiment("cluster_jerarquico")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Definir variables a utilizar
    model_cols = ['DLTV', 'FK_ID_Genero', 'Canal_Principal', 'VL_Edad', 'Prom_Cuotas_Pagadas', 'Prom_Cuotas_No_Pagadas', 'FK_ID_Estado_Civil']
    gower_dist = gower.gower_matrix(donaciones_sin_outlier[model_cols])
    gower_dist = pd.DataFrame(gower_dist)
  
    # Registre los parámetros
    mlflow.log_param("variables", model_cols)

    # Definir el dendograma
    Z = linkage(gower_dist, method='average', metric='euclidean')
    clusters = fcluster(Z, t=4, criterion='maxclust')

    # Graficar
    fig, ax = plt.subplots(figsize=(12, 8))
    d = dendrogram(Z, show_leaf_counts=True, leaf_font_size=10, ax=ax)
    ax.set_xlabel('Observaciones', fontsize=10)
    ax.set_yticks(np.arange(0,50,10))
    ax.set_ylabel('Distancia', fontsize=10)
    plt.show()
  
    # Registre el modelo
    mlflow.sklearn.log_model(clusters, "cluster_jerarquico")

    # Cree y registre la métrica de interés
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
    
    # Registrar métricasw en mlflow
    mlflow.log_metric("n_clusters", n_clusters)
    mlflow.log_metric("cluster_size", cluster_size)
    mlflow.log_metric("cluster_pct", cluster_pct)
    mlflow.log_metric("cluster_dltv", cluster_dltv)
    mlflow.log_metric("cluster_cuotas_pagadas", cluster_cuotas_pagadas)
    mlflow.log_metric("cluster_cuotas_no_pagadas", cluster_cuotas_no_pagadas)
    mlflow.log_metric("cluster_edad", cluster_edad)
    mlflow.log_metric("cluster_genero", cluster_genero)
    mlflow.log_metric("cluster_canal", cluster_canal)
    mlflow.log_metric("cluster_estado_civil", cluster_estado_civil)

    print(f'El número de clusters es {n_clusters}')
    print(f'El tamaño de los clusters es {cluster_size}')
    print(f'El porcentaje de observaciones en cada cluster es {cluster_pct}')
    print(f'El DLTV promedio por cluster es {cluster_dltv}')
    print(f'El promedio de cuotas pagadas por cluster es {cluster_cuotas_pagadas}')
    print(f'El promedio de cuotas no pagadas por cluster es {cluster_cuotas_no_pagadas}')
    print(f'El promedio de edad por cluster es {cluster_edad}')
    print(f'La cuenta de generos por cluster es {cluster_genero}')
    print(f'La cuenta de canales por cluster es {cluster_canal}')
    print(f'La cuenta de estado civil por cluster es {cluster_estado_civil}')