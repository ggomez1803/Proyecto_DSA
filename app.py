import tkinter as tk
from tkinter import ttk
import pickle
import numpy as np

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("App de Segmentación")

        # Variables para almacenar los valores de los campos de entrada
        self.dltv_var = tk.DoubleVar()
        self.fuga_var = tk.DoubleVar()

        # Crear etiquetas y campos de entrada
        ttk.Label(root, text="DLTV:").grid(row=0, column=0, padx=10, pady=10)
        ttk.Entry(root, textvariable=self.dltv_var).grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(root, text="Fuga:").grid(row=1, column=0, padx=10, pady=10)
        ttk.Entry(root, textvariable=self.fuga_var).grid(row=1, column=1, padx=10, pady=10)

        # Botón de predicción
        ttk.Button(root, text="Predecir", command=self.predecir).grid(row=2, column=0, columnspan=2, pady=10)

    def predecir(self):
        # Obtener los valores de los campos de entrada
        dltv = self.dltv_var.get()
        fuga = self.fuga_var.get()

        # Cargar el modelo
        with open("kmeans_model.pkl", "rb") as model_file:
            modelo = pickle.load(model_file)

        # Realizar la predicción
        observacion = np.array([[dltv, fuga]])
        cluster_predicho = modelo.predict(observacion)

        # Mostrar el resultado en una ventana emergente
        resultado_str = f"La observación pertenece al cluster {cluster_predicho[0]}"
        tk.messagebox.showinfo("Resultado", resultado_str)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
