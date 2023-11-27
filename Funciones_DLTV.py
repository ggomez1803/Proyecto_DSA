# Importar librerias
import pandas as pd
import datetime as dt
import numpy as np
import Rutas

today = dt.date.today()
anio_actual = today.year

# Función para calcular el lifespan
def lifespan(df: pd.DataFrame):
    """Función para calcular el lifespan de un cliente
    Args:
        df (DataFrame): DataFrame con las columnas VL_Churn y VL_Lapsed
    Returns:
        df (DataFrame): DataFrame con las columnas VL_NChurn y VL_Lifespan"""
    
    # Calcular el mayor entre la probabilidad de churn y lapsed
    df['VL_NChurn'] = df[['VL_Churn_Prob', 'VL_Lapsed_Prob']].max(axis=1)
    df['VL_NChurn'].fillna(0, inplace=True)

    # Eliminar registros con valor 0
    df = df[df['VL_NChurn'] != 0]

    # Calcular el lifespan
    df['VL_Lifespan'] = 100/df['VL_NChurn']

    # Modificar lifespan si la suma de lifespan más la edad es mayor a 90
    df.loc[df['VL_Lifespan'] + df['VL_Edad'] > 90, 'VL_Lifespan'] = 90 - df['VL_Edad']
    return df

#Función que calcula el valor presente acumulado
def acum_value(serie: pd.Series, acum: float, inflacion:pd.DataFrame):
    """Función que calcula el valor presente acumulado
    Args:
        serie (Series): Serie con los valores a calcular
        acum (float): Valor acumulado de años anteriores
        inflacion (DataFrame): DataFrame con la inflación por año
    Returns:
        acumulado (float): Valor presente acumulado"""
    anio = serie.name
    #Si estoy en el año actual solo sumo el total
    if anio == anio_actual:
        return serie + acum
    else:
        #Extraigo el ipc del año en el que estoy
        ipc = inflacion.loc[anio]["Inflación total 1"]
        #Paso a valor presente el acumulado de años anteriores + el del año en el que estoy
        acumulado = (serie + acum)*(1+ipc)
        return acumulado

#Función que calcula el valor presente
def calcular_vp(row, inflacion:pd.DataFrame):
    """Función que calcula el valor presente
    Args:
        row (Series): Serie con los valores a calcular
        inflacion (DataFrame): DataFrame con la inflación por año
    Returns:
        acumulado (float): Valor presente acumulado"""
    #Inflación más reciente
    ipc = inflacion.loc[inflacion.index.max()]["Inflación total 1"]
    if int(row['VL_Lifespan']) == 0:
        lifespan = 2
    else:
        lifespan = int(row['VL_Lifespan']+1)

    montos = [row['Total_Anual']/((1+ipc)**t) for t in range(1, lifespan)]
    return sum(montos)

# Función para cargar el archivo de inflación
def cargar_inflacion():
    """Función para cargar el archivo de inflación
    Returns:
        df_inflacion (DataFrame): DataFrame con la inflación por año"""
    # Leer el archivo de datos
    df_inflacion = pd.read_excel("1.1.INF_Serie historica Meta de inflacion IQY.xlsx",skiprows=range(7),skipfooter=8,dtype={'Año(aaaa)-Mes(mm)': str})
    ## Extraer el año de la fecha
    df_inflacion["anio"] = df_inflacion["Año(aaaa)-Mes(mm)"].apply(lambda x: x[0:4])
    # Sacar la media de inflación por año
    (df_inflacion.groupby("anio").agg({"Inflación total 1":"mean"}) / 100).to_csv(Rutas.ruta_inflacion_prom_anual, encoding='latin-1')
    # Leer el archivo procesado
    df_inflacion = pd.read_csv(Rutas.ruta_inflacion_prom_anual, index_col="anio", encoding='latin-1')
    return df_inflacion

# Función para cargar las transacciones válidas para el análisis
def cargar_transacciones(df: pd.DataFrame):
    """Función para cargar las transacciones válidas para el análisis
    Args:
        df (DataFrame): DataFrame con las transacciones
    Returns:
        df_transacciones (DataFrame): DataFrame con las transacciones válidas para el análisis"""
    # Leer el archivo de datos
    df = pd.read_csv(Rutas.ruta_transacciones_individuales_preprocesada, encoding='latin1')
    # Filtrar transacciones cobradas
    df = df[df['FK_CD_Etapa'] == 'Cobrada']
    # Convertir la fecha a datetime
    df['DT_Fecha'] = pd.to_datetime(df['DT_Fecha'])
    # Extraer el mes y el año de la fecha
    df['Mes_Donacion'] = df['DT_Fecha'].dt.month
    df['Anio_Donacion'] = df['DT_Fecha'].dt.year
    ##Diferencia entre el año actual y en el que se hizo la donación
    df['Ant_Primer_Transaccion'] = anio_actual - df["Anio_Donacion"]
    return df
