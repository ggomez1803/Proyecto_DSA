#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import datetime as dt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from DLTV import df_donantes, df_transacciones, df_inflacion, lfv, donaciones_anuales
from xgboost import XGBClassifier, plot_importance
import numpy as np
import matplotlib.pyplot as plt
from pca import pca
from sklearn.preprocessing import StandardScaler
import mlflow

#Funciones de Experimentación
def obtener_mejor_k(df: pd.DataFrame):
    """ Calcula el mejor número de clusters para un dataset dado
    Args:
        df (DataFrame): Dataset a utilizar
    Returns:
        int: Número de clusters óptimo"""
    df = StandardScaler().fit_transform(df)
    sil = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters = k, random_state=123).fit(df)
        labels = kmeans.labels_
        sil.append(silhouette_score(df, labels, metric = 'euclidean'))
    # Graficar silhouette_score
    fig = px.line(x = range(2, 10), y = sil)
    fig.show()
    # Crear dataframe con los valores de silhouette_score
    df_sil = pd.DataFrame({'k': range(2, 10), 'sil': sil})
    # Ordenar de mayor a menor
    df_sil = df_sil.sort_values('sil', ascending = False)
    # Obtener el mejor k
    best_k = df_sil.iloc[0, 0]
    # Si best_k = 2, tomar el segundo mejor
    if best_k == 2:
        best_k = df_sil.iloc[1, 0]
    
    print('El mejor número de clusters es: ', best_k)
    return best_k

def obtener_mejor_k_prototype(df: pd.DataFrame):
    """ Calcula el mejor número de clusters para un dataset dado
    Args:
        df (DataFrame): Dataset a utilizar
    Returns:
        None"""
    # Convertir dataframe en array
    array = df.values    

    # Obtener índices de variables categóricas
    idx = []
    for col in df.columns:
        if df[col].dtype == 'object':
            idx.append(df.columns.get_loc(col))
        else:
            array[:,df.columns.get_loc(col)] = array[:,df.columns.get_loc(col)].astype(float)

    # Calcular silhouette_score
    cost_values = []
    for k in range(2, 10):
        kprototypes = KPrototypes(n_clusters=k, init='Huang', verbose=1, max_iter=20, random_state=123)
        kprototypes.fit_predict(array, categorical=idx)
        cost_values.append(kprototypes.cost_)
    # Graficar elbow
    fig = px.line(x = range(2, 10), y = cost_values, title='Gráfica de codo')
    fig.update_xaxes(title='Número de clusters')
    fig.update_yaxes(title='Costo')
    fig.show()


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

def graficar_clusters(df: pd.DataFrame, x: str, y: str):
    """ Grafica los clusters de un dataset dado
    Args:
        df (DataFrame): Dataset a utilizar
        x (str): Nombre de la columna x
        y (str): Nombre de la columna y
    Returns:
        None"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df[x], df[y], c=df['cluster'], cmap='rainbow')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.show()

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

#Agregar variables
# Pasar columnas de fecha a formato fecha
df_donantes['DT_Captacion'] = pd.to_datetime(df_donantes['DT_Captacion'])
df_donantes['DT_Primer_Cobro'] = pd.to_datetime(df_donantes['DT_Primer_Cobro'])
# Calcular el tiempo de conversión como la diferencia entre la fecha de primer cobro y la fecha de captación
df_donantes['T_Conversion'] = df_donantes['DT_Primer_Cobro'] - df_donantes['DT_Captacion']
# Convertir el tiempo de conversión a número de días
df_donantes['T_Conversion'] = df_donantes['T_Conversion'].dt.days
# Hacer una tabla dinámica que muestre por donante en qué canal entró su donación
tipo_transacciones = df_transacciones.pivot_table(index='FK_ID_Donante', columns='FK_CD_Registro', values='VL_Importe', aggfunc='count').fillna(0)
# Calcular el número de canales por donante
tipo_transacciones['Canales_Donacion'] = tipo_transacciones.apply(lambda x: np.count_nonzero(x), axis=1)
# Calcular el canal principal de donación
tipo_transacciones['Canal_Principal'] = tipo_transacciones.apply(lambda x: x.idxmax(), axis=1)
# Agregar variables a la tabla de donantes
lfv = pd.merge(lfv, df_donantes[['ID_Donante', 'T_Conversion', 'FK_ID_Genero', 'FK_ID_Estado_Civil']], on='ID_Donante', how='left')
lfv = pd.merge(lfv, tipo_transacciones[['Canales_Donacion', 'Canal_Principal']], left_on='ID_Donante', right_on= 'FK_ID_Donante', how='left')

efect_cobro = df_transacciones.pivot_table(index='FK_ID_Donante', columns='FK_CD_Etapa', values='VL_Importe', aggfunc='count').fillna(0)

#Calcular outliers
# Calcular el rango IQR para la variable DLTV
donaciones_sin_outlier, outliers = calcular_outliers(lfv, 'DLTV')

#Definición de variables a utilizar
# Variables a evaluar
model_cols = ['VL_Edad', 'Prom_Cuotas_Pagadas', 'Prom_Cuotas_No_Pagadas', 'Canales_Donacion', 'DLTV', 'VL_NChurn']

# Estandarizar variables
scaler = StandardScaler()
scaler.fit(donaciones_sin_outlier[model_cols])
donaciones_sin_outlier[model_cols] = scaler.transform(donaciones_sin_outlier[model_cols])

#mlflow.set_tracking_uri('http://0.0.0.0:5000')
experiment = mlflow.set_experiment("pca-kmeans")
with mlflow.start_run(experiment_id=experiment.experiment_id):
    n_components=3
    

#PCA
    pca_model = pca(n_components=n_components)
    pca_donantes = pca_model.fit_transform(donaciones_sin_outlier[model_cols])

    df_pca = pca_donantes['PC']
    df_pca = pd.DataFrame(df_pca)
    df_pca = df_pca.rename(columns={'PC1':'DV-Fuga', 'PC2':'Edad-Cuotas', 'PC3':'Canales_Donacion'})

    best_k = obtener_mejor_k(df_pca)
# Obtener clusters
# df_segmentado, kmeans_model = obtener_clusters(donaciones_sin_outlier, model_cols, best_k)
    model_cols = df_pca.columns
    df_segmentado, kmeans_model = obtener_clusters(df_pca, model_cols, best_k)
    tabla = df_segmentado.groupby("cluster").mean()
    tabla = tabla.to_dict(orient="records")
    #model_cols = ["DLTV", "FK_ID_Genero", "Canal_Principal"]

#Clasificar outliers
# Obtener clusters
    #df_segmentado, kmeans_model = obtener_clusters(donaciones_sin_outlier, model_cols, best_k)
# Agregar clasificación de outliers al df de outliers
    #outliers['cluster'] = kmeans_model.predict(outliers[model_cols])
# Agregar outliers al df_segmentado
    #df_segmentado_con_outliers = pd.concat([df_segmentado, outliers])

#Caracterizar segmentos
# suppress scientific notation by setting float_format
    #pd.options.display.float_format = '{:.2f}'.format
# Mostrar estadísticas de los clusters
    #tabla = df_segmentado[model_cols+['cluster']].groupby('cluster').describe().T
    
    mlflow.log_param("n_components", n_components)
    mlflow.log_metric("best_k", best_k)
    mlflow.sklearn.log_model(kmeans_model, "kmeans_model")
    #mlflow.log_table('tabla', tabla)
    




