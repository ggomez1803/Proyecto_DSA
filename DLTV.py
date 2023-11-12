# Importar librerias
import pandas as pd
import Funciones_DLTV as fdv
import Funciones_Preprocesamiento as fpp

# Leer el archivo de datos
df_donantes = pd.read_csv('./Archivos_Cliente/Base_donantes_preprocesada.csv', encoding='latin1')
df_donantes = fdv.lifespan(df_donantes)
df_transacciones = fdv.cargar_transacciones()
df_inflacion = fdv.cargar_inflacion()

# Cambiar formato de columna de fecha
df_transacciones['DT_Primer_Cobro'] = pd.to_datetime(df_transacciones['DT_Primer_Cobro'])
# Agregar Fecha de primera donación al dataframe
df_donantes = pd.merge(df_donantes, df_transacciones.groupby("FK_ID_Donante").agg({"DT_Primer_Cobro":"min"}), how="left", left_on="ID_Donante", right_on="FK_ID_Donante")
# Crear objeto de preprocesamiento
pr_donantes = fpp.Preprocesamiento_Donantes(df_donantes)
# Filtrar donantes activos
df_donantes = pr_donantes.filtrar_donantes_activos('CD_Donante_Activo')
# Filtrar donantes con fecha de primera donación mayor a 3 meses
df_donantes = pr_donantes.remover_donantes_menos_3_meses('DT_Primer_Cobro')
# Calcular el promedio de cuotas pagadas y no pagadas al año
df_donantes['Prom_Cuotas_Pagadas'] = df_donantes['VL_Cuotas_Pagadas'] / ((fdv.today.year+1) - df_donantes['DT_Primer_Cobro'].dt.year)
df_donantes['Prom_Cuotas_No_Pagadas'] = df_donantes['VL_Cuotas_No_Pagadas'] / ((fdv.today.year+1) - df_donantes['DT_Primer_Cobro'].dt.year)


# Tabla resumen de transacciones por donante y año
donaciones_anuales = pd.pivot_table(df_transacciones, values='VL_Importe', index='FK_ID_Donante', columns=['Anio_Donacion'], aggfunc='sum', fill_value=0)

acumulado = 0
# Calcular el valor presente acumulado
for col in donaciones_anuales.columns:
    acumulado = fdv.acum_value(donaciones_anuales[col], acumulado, df_inflacion)

lfv = pd.concat([acumulado.to_frame("Valor_Total"),df_transacciones.groupby("FK_ID_Donante").agg({"Anio_Donacion":"min"}), df_donantes[['VL_Lifespan',"VL_Edad","VL_NChurn","Prom_Cuotas_Pagadas","Prom_Cuotas_No_Pagadas","ID_Donante"]].set_index("ID_Donante")],axis = 1,join='inner')
lfv["Ant_Primer_Transaccion"] = fdv.anio_actual - lfv["Anio_Donacion"]
lfv["Total_Anual"] = lfv["Valor_Total"] /  (lfv["Ant_Primer_Transaccion"]+1)

# Calcular el Donor Lifetime Value
lfv['DLTV'] = lfv.apply(lambda x: fdv.calcular_vp(x, df_inflacion), axis=1)
lfv = lfv.reset_index()
lfv = lfv.rename(columns={'index': 'ID_Donante'})
