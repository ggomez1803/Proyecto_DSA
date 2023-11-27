import joblib
import Rutas
import Funciones_Segmentacion as fs
from DLTV import lfv
from Preprocesamiento_datos import df_donantes, df_nuevos
from sklearn.preprocessing import StandardScaler

# Cargar modelo entrenado
kmeans = joblib.load('modelo_segmentacion.pkl')

# Cargar DLTV y Efectividad_cobro a tabla de transacciones resumen por donante
df = fs.agregar_variables(lfv)

# Definir variables de segmentación
model_cols = ['DLTV_std', 'VL_Churn_Prob_std', 'Efectividad_cobro_std']

# Agregar DLTV y Efectividad_cobro a la base de donantes
df_donantes = df_donantes.merge(df[['ID_Donante', 'VL_Lifespan', 'DLTV', 'Efectividad_cobro']], on='ID_Donante', how='left')

# Remover valores nulos
df_segmentacion = df_donantes.dropna(subset=['DLTV', 'VL_Churn_Prob', 'Efectividad_cobro'])

# Transformar los datos antes de predecir
df_segmentacion['DLTV_std'] = StandardScaler().fit_transform(df_segmentacion[['DLTV']])
df_segmentacion['VL_Churn_Prob_std'] = StandardScaler().fit_transform(df_segmentacion[['VL_Churn_Prob']])
df_segmentacion['Efectividad_cobro_std'] = StandardScaler().fit_transform(df_segmentacion[['Efectividad_cobro']])

# Predecir clusters
df_segmentacion['cluster'] = kmeans.predict(df_segmentacion[model_cols])
segmentos = {0:'Hibernando', 1:'Comprometidos', 2:'En riesgo', 3:'Campeones', 4:'En fuga'}
df_segmentacion['cluster'] = df_segmentacion['cluster'].map(segmentos)

# Agregar clusters a base de donantes
df_donantes = df_donantes.merge(df_segmentacion[['ID_Donante', 'cluster']], on='ID_Donante', how='left')
df_donantes['cluster'] = df_donantes['cluster'].fillna('No asignado')

# Si pertenece al df_nuevos, marcar cluster como "nuevo"
for donante in df_nuevos['ID_Donante']:
    if donante in df_donantes['ID_Donante']:
        df_donantes.loc[df_donantes['ID_Donante'] == donante, 'cluster'] = 'Nuevo'

# Exportar resultados
df_donantes.to_csv(Rutas.ruta_base_donantes_segmentada, index=False, encoding='latin-1')