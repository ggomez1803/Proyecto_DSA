import Funciones_Preprocesamiento
import Rutas
import pandas as pd

#### Preprocesamiento Donantes ####

# Cargar datos 
df_donantes = pd.read_csv(Rutas.ruta_base_donantes, encoding='latin-1')

# Definir si se quiere retornar o no los datos con errores
return_errors = False

# Crear objeto de la clase
pr_donantes = Funciones_Preprocesamiento.Preprocesamiento_Donantes(df_donantes)

# Corregir columnas de fechas
df_donantes = pr_donantes.corregir_fechas('Fecha de nacimiento', 'DT_Nacimiento', return_errors)
df_donantes = pr_donantes.corregir_fechas('Fecha de Captación', 'DT_Captacion', return_errors)
df_donantes = pr_donantes.corregir_fechas('Fecha Aniversario Pago', 'DT_Aniversario_Pago', return_errors)

# Corregir las edades que no hacen sentido
df_donantes, df_error_edades = pr_donantes.corregir_edades(return_errors=True)

# Corregir el número de hijos
df_donantes = pr_donantes.corregir_cantidad_hijos('Cantidad de Hijos', return_errors)

# Corregir la columna probabilidades
#df_donantes = pr_donantes.corregir_prob('Churn Probability')
#df_donantes = pr_donantes.corregir_prob('Lapsed Probability')

# Corregir las columnas de ciudad y departamento
df_donantes = pr_donantes.corregir_ciudades_o_departamentos('Ciudad de correo', 'Ciudad')
df_donantes = pr_donantes.corregir_ciudades_o_departamentos('Estado o provincia de correo', 'Departamento')

# Corregir nombres y tipos de datos
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('PSN', 'ID_Donante', 'int')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Edad', 'VL_Edad', 'int')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Cantidad de Hijos', 'VL_Num_Hijos', 'int')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Churn Probability', 'VL_Churn_Prob', 'float')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Lapsed Probability', 'VL_Lapsed_Prob', 'float')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Tiene hijos', 'CD_Tiene_Hijos', 'bool')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Donante Activo', 'CD_Donante_Activo', 'bool')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Tipo de registro del contacto', 'CD_Tipo_Registro', 'str')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Género', 'FK_ID_Genero', 'str')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Estado Civil', 'FK_ID_Estado_Civil', 'str')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Ocupación', 'FK_ID_Ocupacion', 'str')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Otra Clasificación RFM Actual', 'FK_ID_Segmento_RFM', 'str')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Cantidad Cuotas Pagadas Global', 'VL_Cuotas_Pagadas', 'int')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Cantidad Cuotas No Pagadas Global', 'VL_Cuotas_No_Pagadas', 'int')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Campaña Inicial: Nombre', 'FK_ID_Campana', 'str')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Ciudad de correo', 'CD_Ciudad', 'str')
df_donantes = pr_donantes.ajustar_nombre_y_tipo_columnas('Estado o provincia de correo', 'CD_Departamento', 'str')

# Filtrar donantes con captación menor a 3 meses
df_donantes, df_nuevos = pr_donantes.remover_donantes_menos_3_meses('DT_Captacion')

# Corregir la columna Tiene Hijos
df_donantes = pr_donantes.corregir_tiene_hijos()

# Guardar datos
df_donantes.to_csv(Rutas.ruta_base_donantes_preprocesada, index=False, encoding='latin-1')
df_nuevos.to_csv(Rutas.ruta_base_donantes_nuevos, index=False, encoding='latin-1')
df_error_edades.to_csv(Rutas.ruta_base_donantes_error_edades, index=False, encoding='latin-1')

# Imprimir mensaje de finalización
print('Proceso finalizado para la base de donantes')

#### Preprocesamiento Transacciones Individuales ####

# Cargar datos
df_transacciones = pd.read_csv(Rutas.ruta_transacciones_individuales, sep=';', encoding='latin-1')

# Definir si se quiere retornar o no los datos con errores
return_errors = False

# Crear objeto de la clase
pr_transacciones = Funciones_Preprocesamiento.Preprocesamiento_Transacciones(df_transacciones)

# Corregir columnas de fechas
df_transacciones = pr_transacciones.corregir_fechas('Fecha de Donación', 'DT_Fecha', return_errors)
df_transacciones = pr_transacciones.corregir_fechas('Fecha de Registro Contable', 'DT_Registro_Contable', return_errors)
df_transacciones = pr_transacciones.corregir_fechas('Fecha efectiva de primer cobro', 'DT_Primer_Cobro', return_errors)
df_transacciones = pr_transacciones.corregir_fechas('Fecha de última donación', 'DT_Ultima_Donacion', return_errors)

# Corregir decimales en columna de importe
df_transacciones = pr_transacciones.corregir_decimales('Importe')

# Corregir nombres y tipos de datos
df_transacciones = pr_transacciones.ajustar_nombre_y_tipo_columnas('PSN', 'FK_ID_Donante', 'int')
df_transacciones = pr_transacciones.ajustar_nombre_y_tipo_columnas('Etapa', 'FK_CD_Etapa', 'str')
df_transacciones = pr_transacciones.ajustar_nombre_y_tipo_columnas('Canal de la Campaña', 'FK_CD_Canal_campana', 'str')
df_transacciones = pr_transacciones.ajustar_nombre_y_tipo_columnas('Importe', 'VL_Importe', 'float')
df_transacciones = pr_transacciones.ajustar_nombre_y_tipo_columnas('Estado/Tipo', 'FK_CD_Estado', 'str')
df_transacciones = pr_transacciones.ajustar_nombre_y_tipo_columnas('Tipo de Compromiso', 'FK_CD_Compromiso', 'str')
df_transacciones = pr_transacciones.ajustar_nombre_y_tipo_columnas('Medio de Pago', 'FK_CD_Medio_Pago', 'str')
df_transacciones = pr_transacciones.ajustar_nombre_y_tipo_columnas('Tipo de registro', 'FK_CD_Registro', 'str')

# Guardar datos
df_transacciones.to_csv(Rutas.ruta_transacciones_individuales_preprocesada, index=False, encoding='latin-1')

# Imprimir mensaje de finalización
print('Proceso finalizado para transacciones individuales')

#### Preprocesamiento Cancelaciones ####
# Cargar datos
df_cancelaciones = pd.read_csv(Rutas.ruta_base_cancelaciones, encoding='latin-1')

# Definir si se quiere retornar o no los datos con errores
return_errors = False

# Crear objeto de la clase
pr_cancelaciones = Funciones_Preprocesamiento.Preprocesamiento_Donantes(df_cancelaciones)

# Corregir columnas de fechas
df_cancelaciones = pr_cancelaciones.corregir_fechas('Fecha de Baja', 'DT_Cancelacion', return_errors)
df_cancelaciones = pr_cancelaciones.corregir_fechas('Fecha efectiva de primer cobro', 'DT_Primer_Cobro', return_errors)
df_cancelaciones = pr_cancelaciones.corregir_fechas('Fecha de última donación', 'DT_Ultima_Donacion', return_errors)

# Corregir nombres y tipos de datos
df_cancelaciones = pr_cancelaciones.ajustar_nombre_y_tipo_columnas('PSN', 'ID_Donante', 'int')
df_cancelaciones = pr_cancelaciones.ajustar_nombre_y_tipo_columnas('Motivo de Baja', 'CD_Motivo_Cancelacion', 'str')
df_cancelaciones = pr_cancelaciones.ajustar_nombre_y_tipo_columnas('Estado/Tipo', 'CD_Estado', 'str')

# Guardar datos
df_cancelaciones.to_csv(Rutas.ruta_base_cancelaciones_preprocesada, index=False, encoding='latin-1')

# Imprimir mensaje de finalización
print('Proceso finalizado para cancelaciones')
