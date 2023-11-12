import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def obtener_clusters(df:pd.DataFrame, cols: list, k: int):
    """ Calcula los clusters de un dataset dado
    Args:
        df (DataFrame): Dataset a utilizar
        k (int): Número de clusters
    Returns:
        DataFrame: Dataset con los clusters
        KMeans: Modelo de KMeans"""
    df_copy = df[cols].copy()
    df_copy = StandardScaler().fit_transform(df_copy)
    kmeans = KMeans(n_clusters = k, n_init=10, random_state=0)
    kmeans.fit(df_copy)
    df['cluster'] = kmeans.labels_
    return df, kmeans

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
print(lfv.columns)
#df_donantes = pd.read_csv('./Archivos_Cliente/Base_donantes_preprocesada.csv', encoding='latin1')
#df_transacciones = pd.read_csv('./Archivos_Cliente/Transacciones_Individuales_preprocesada.csv', encoding='latin1')

# Calcular el rango IQR para la variable DLTV
donaciones_sin_outlier, outliers = calcular_outliers(lfv, 'DLTV')

mlflow.set_tracking_uri('http://3.83.143.68:8080/')
experiment = mlflow.set_experiment("exp_kmeans")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Definir variables a utilizar
    model_cols = ['DLTV', 'VL_Edad', 'VL_NChurn']
    k = 5    

    # Variables a evaluar
    df_segmentado, kmeans_model = obtener_clusters(donaciones_sin_outlier, model_cols, k)

    # Registre los parámetros
    mlflow.log_param("clusters", k)
    mlflow.log_param("variables", model_cols)

    # Registre el modelo
    mlflow.sklearn.log_model(kmeans_model, "k_means")

    # Cree y registre la métrica de interés
    # Numero de clusters
    n_clusters = k
    # Tamaño de los clusters
    cluster_size = df_segmentado.groupby("cluster").size()
    # Porcentaje de observaciones en cada cluster
    cluster_pct = cluster_size / len(df_segmentado)
    # DLTV promedio por cluster
    cluster_dltv = df_segmentado.groupby("cluster").agg({"DLTV":"mean"})
    # Promedio de cuotas pagadas por cluster
    cluster_cuotas_pagadas = df_segmentado.groupby("cluster").agg({"Prom_Cuotas_Pagadas":"mean"})
    # Promedio de cuotas no pagadas por cluster
    cluster_cuotas_no_pagadas = df_segmentado.groupby("cluster").agg({"Prom_Cuotas_No_Pagadas":"mean"})
    # Promedio de edad por cluster
    cluster_edad = df_segmentado.groupby("cluster").agg({"VL_Edad":"mean"})
    # Promedio de churn por cluster
    cluster_churn = df_segmentado.groupby("cluster").agg({"VL_NChurn":"mean"})
  
    # Registrar métricasw en mlflow
    mlflow.log_metric("n_clusters", n_clusters)
    for cluster in range(n_clusters):
        mlflow.log_metric(f"cluster_{cluster}_size", cluster_size[cluster])
        print(f"cluster_{cluster}_size", cluster_size[cluster])
        mlflow.log_metric(f"cluster_{cluster}_pct", cluster_pct[cluster])
        print(f"cluster_{cluster}_pct", cluster_pct[cluster])
        mlflow.log_metric(f"cluster_{cluster}_dltv", cluster_dltv.iloc[cluster])
        print(f"cluster_{cluster}_dltv", cluster_dltv.iloc[cluster])
        mlflow.log_metric(f"cluster_{cluster}_cuotas_pagadas", cluster_cuotas_pagadas.iloc[cluster])
        print(f"cluster_{cluster}_cuotas_pagadas", cluster_cuotas_pagadas.iloc[cluster])
        mlflow.log_metric(f"cluster_{cluster}_cuotas_no_pagadas", cluster_cuotas_no_pagadas.iloc[cluster])
        print(f"cluster_{cluster}_cuotas_no_pagadas", cluster_cuotas_no_pagadas.iloc[cluster])
        mlflow.log_metric(f"cluster_{cluster}_edad", cluster_edad.iloc[cluster])
        print(f"cluster_{cluster}_edad", cluster_edad.iloc[cluster])
        mlflow.log_metric(f"cluster_{cluster}_churn", cluster_churn.iloc[cluster])
        print(f"cluster_{cluster}_churn", cluster_churn.iloc[cluster])
