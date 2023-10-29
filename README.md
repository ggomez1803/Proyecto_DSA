# Proyecto_DSA
Proyecto para Despliegue de Soluciones Analíticas
prueba 1 
  # Método para corregir las columnas de fechas
    def corregir_fechas(self, col:str, n_col:str, return_errors: bool=False):
        """Método para corregir las columnas de fechas
        Args:
            df (pd.DataFrame): DataFrame con los datos de los donantes
            col (str): Nombre de la columna a revisar
            n_col (str): Nombre de la nueva columna
            return_errors (bool, optional): Si es True, retorna los valores que no coincidan con una fecha. Defaults to False.
            Returns:
            pd.DataFrame: DataFrame con los valores convertidos en fecha"""
        # Obtener valores que no coincidan con una fecha
        self.df, df_error = self.texto_en_fechas(col)

        # Convertir la columna Fecha de Nacimiento a datetime
        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        # Renombrar columna
        self.df.rename(columns={col:n_col}, inplace=True)

        # Retornar el DataFrame
        if len(df_error) > 0:
            if return_errors:
                return self.df, df_error
        else:
            return self.df