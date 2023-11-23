import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from Preprocesamiento_datos import df_transacciones

# Función para agregar variables de interés al dataset de donantes
def agregar_variables(df: pd.DataFrame):
    """ Agrega variables como DLTV, efectividad de cobro, etc. al dataset de transacciones preprocesadas por donante
    Args:
        df (DataFrame): Dataset a utilizar
    Returns:
        DataFrame: Dataset con las variables agregadas"""
    # Leer archivo de transacciones individuales completo
    # Hacer tabla dinámica para obtener el total de transacciones por donante
    resumen_trans = pd.pivot_table(df_transacciones, values='VL_Importe', index=['FK_ID_Donante'], columns=['FK_CD_Etapa'], aggfunc=np.sum, fill_value=0)
    resumen_trans['Cobrada'] = resumen_trans['Cobrada'].astype(int)
    resumen_trans['Perdida'] = resumen_trans['Perdida'].astype(int)
    resumen_trans['Total_Importe'] = resumen_trans['Cobrada'] + resumen_trans['Perdida']
    # Calcular la efectividad de cobro
    resumen_trans['Efectividad_cobro'] = resumen_trans['Cobrada'] / resumen_trans['Total_Importe']
    # Reiniciar índice
    resumen_trans.reset_index(inplace=True)
    # Cambiar los nombres de las columnas
    resumen_trans.rename(columns={'FK_ID_Donante':'ID_Donante', 'Cobrada':'Total_Cobrado'}, inplace=True)
    # Unir variables al dataset de donantes
    df = df.merge(resumen_trans[['ID_Donante', 'Total_Cobrado', 'Total_Importe', 'Efectividad_cobro']], on='ID_Donante', how='left')
    return df

# Función para obtener el mejor número de clusters
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

# Función para obtener los clusters de la corrida
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

# Función para obtener los centroides de los clusters
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

# Función para obtener los centroides de los clusters en 3D
def graficar_clusters3D(df: pd.DataFrame, x:str, y:str, z:str):
    """ Grafica los clusters de un dataset dado en 3D
    Args:
        df (DataFrame): Dataset a utilizar
        x (str): Nombre de la columna x
        y (str): Nombre de la columna y
        z (str): Nombre de la columna z
    Returns:
        None"""
    fig = px.scatter_3d(df, x=x, y=y, z=z, color='cluster')
    fig.show()

# Función para graficar la importancia de las variables del modelo de segmentación
def graficar_importancia(df: pd.DataFrame, cols: list):
    """ Grafica la importancia de las variables de un dataset dado
    Args:
        df (DataFrame): Dataset a utilizar
        cols (list): Lista de columnas a utilizar
    Returns:
        None"""
    # Crear conjunto de entrenamiento y prueba
    X = df[cols]
    y = df['cluster']

    # Dividir en datos de prueba y entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model = XGBClassifier(eval_metric='mlogloss')
    # Entrenar modelo
    model.fit(X_train, y_train)
    # Predecir con datos de prueba
    y_pred = model.predict(X_test)
    # Calcular precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'La precisión del modelo es: {accuracy}')
    # Graficar importancia de variables
    plot_importance(model)

# Función para graficar los segmentos del modelo de segmentación
def graficar(df: pd.DataFrame, cols: list):
    if len(cols) == 2:
        graficar_clusters(df, cols[1], cols[0])
    elif len(cols) == 3:
        graficar_clusters3D(df, cols[0], cols[1], cols[2])
    else:
        print('No se puede graficar porque son más de 3 dimensiones')

# Función para encontrar los outliers de una variable
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