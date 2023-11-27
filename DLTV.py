# Importar librerias
import pandas as pd
import Funciones_DLTV as fdv
import Funciones_Preprocesamiento as fpp
from Preprocesamiento_datos import df_donantes as donantes, df_transacciones as transacciones, df_cancelaciones as cancelaciones

# Leer el archivo de datos
df_donantes = fdv.lifespan(donantes)
df_transacciones = fdv.cargar_transacciones(transacciones)
df_inflacion = fdv.cargar_inflacion()

# Crear objeto de preprocesamiento
pr_donantes = fpp.Preprocesamiento_Donantes(df_donantes)
# Filtrar donantes activos
df_donantes = pr_donantes.filtrar_donantes_activos('CD_Donante_Activo')

# Filtrar únicamente las donaciones regulares
donaciones_regulares = ['DI - Regular', 'DI - Regular Montos Variables (BO)', 'DI - Regular Tiempo determinado', 'PF - Padrinazgo familiar', 'PI - Padrinazgo individual']
df_transacciones = df_transacciones[df_transacciones['FK_CD_Registro'].isin(donaciones_regulares)]

# Tabla resumen de transacciones por donante y año
donaciones_anuales = pd.pivot_table(df_transacciones, values='VL_Importe', index='FK_ID_Donante', columns=['Anio_Donacion'], aggfunc='sum', fill_value=0)
#print(df_donantes[df_donantes['ID_Donante'] == 1020483945][['ID_Donante', 'VL_Lifespan', 'VL_Churn_Prob']])

acumulado = 0
##Calcular el valor presente acumulado
for col in donaciones_anuales.columns:
    acumulado = fdv.acum_value(donaciones_anuales[col], acumulado, df_inflacion)

lfv = pd.concat([acumulado.to_frame("Valor_Total"),df_transacciones.groupby("FK_ID_Donante").agg({"Anio_Donacion":"min"}), df_donantes[['VL_Lifespan',"VL_Edad","VL_NChurn","ID_Donante"]].set_index("ID_Donante")],axis = 1,join='inner')
lfv["Ant_Primer_Transaccion"] = fdv.anio_actual - lfv["Anio_Donacion"]
lfv["Total_Anual"] = lfv["Valor_Total"] /  (lfv["Ant_Primer_Transaccion"]+1)

# Calcular el Donor Lifetime Value
lfv['DLTV'] = lfv.apply(lambda x: fdv.calcular_vp(x, df_inflacion), axis=1)
lfv = lfv.reset_index()
lfv = lfv.rename(columns={'index': 'ID_Donante'})
lfv.to_csv('./Archivos_Cliente/lfv.csv', index=False, encoding='latin-1')
