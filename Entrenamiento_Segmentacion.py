from DLTV import lfv, df_donantes
import pandas as pd
import Funciones_Segmentacion as fs
import Rutas
import joblib

# Unir variables al dataset de donantes
df = fs.agregar_variables(lfv)

# Agregar churn
df = pd.merge(df, df_donantes[['ID_Donante', 'VL_Churn_Prob']])

# Remover outliers de DLTV
df, df_outliers = fs.calcular_outliers(df, 'DLTV')
df, df_outliers2 = fs.calcular_outliers(df, 'VL_Churn_Prob')

df_outliers = pd.concat([df_outliers, df_outliers2], axis=0)

# Definir variables de segmentación
model_cols = ['DLTV', 'VL_Churn_Prob', 'Efectividad_cobro']

# Obtener el mejor número de clusters
best_k = fs.obtener_mejor_k(df[model_cols])
print(f'El mejor número de segmentos es: {best_k}')
# Obtener los clusters de la corrida
df, kmeans = fs.obtener_clusters(df, model_cols, best_k)

# Graficar clusters
#fs.graficar(df, model_cols)

# Clasificar outliers
df_outliers['cluster'] = kmeans.predict(df_outliers[model_cols])

# Unir outliers con dataset original segmentados
df = pd.concat([df, df_outliers], axis=0)

# Exportar dataset con clusters
df_donantes = df_donantes.merge(df[['ID_Donante', 'cluster', 'DLTV', 'Efectividad_cobro']], on='ID_Donante', how='left')
df_donantes.to_csv(Rutas.ruta_base_donantes_segmentada, index=False, encoding='latin-1')

# Guardar modelo entrenado
joblib.dump(kmeans, 'modelo_segmentacion.pkl')