# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class Datos:

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero, print_val=True):       
        # Leer el dataset
        datos = pd.read_csv(nombreFichero, delimiter=',')
        datos[datos.select_dtypes(include=['number']).columns] = datos.select_dtypes(include=['number']).astype(float)

        # Obtener los tipos de datos de cada columna
        tipos = datos.dtypes
        
        # Inicializar la lista para los atributos nominales
        self.nominalAtributos = []
        self.diccionarios = {}

        # Iterar sobre los tipos de cada columna

        #En pandas, el tipo de dato object generalmente 
        #se refiere a columnas que contienen cadenas de texto (strings) o valores categóricos. 
        #Este tipo se usa cuando los datos no son numéricos, como nombres, colores, categorías, o cualquier información que se maneja como texto.
        
        for i, col in enumerate(datos.columns):
            if tipos[col] == 'object' or i == len(tipos) - 1:  # Si es 'object' o es la última columna (clase)
                self.nominalAtributos.append(True)
                
                # Crear diccionario ordenado lexicográficamente para la columna nominal
                valores_unicos = sorted(datos[col].unique())  # Valores únicos ordenados
                self.diccionarios[col] = {valor: idx for idx, valor in enumerate(valores_unicos)}
                
                # Reemplazar los valores nominales por los valores numéricos del diccionario
                datos[col] = datos[col].map(self.diccionarios[col])
            else:
                self.nominalAtributos.append(False)
                self.diccionarios[col] = {}  # Diccionario vacío para atributos numéricos

        # Guardar los datos numéricos transformados como un atributo de la clase
        
        self.datos = datos.to_numpy()
        if print_val:
            # Imprimir resultados para ver la estructura
            print("Diccionarios:\n", self.diccionarios)
            print("Datos transformados:\n", self.datos)
            print("NominalAtributos:\n", self.nominalAtributos)
            print("shape: ", self.datos.shape[0])

    
    # Devuelve el subconjunto de los datos cuyos �ndices se pasan como argumento
    def extraeDatos(self, idx:list):
        if idx is None or len(idx) > self.datos.shape[0]:
            return None

        sub = np.ndarray((len(idx), self.datos.shape[1]))
        
        for i, fila in enumerate(idx):
            for j in range(self.datos.shape[1]):
                sub[i][j] = self.datos[fila][j]

        return sub

    def estandarizarDatos(self, media=True, std=True):
        column = None
        for i, value in enumerate(self.nominalAtributos):
            if value == False:
                if media or std:
                    column = self.datos[:,i]

                u = np.mean(column) if media else 0.0
                s = np.std(column) if std else 1.0 # ddof=1 ?

                for j in range(self.datos.shape[0]): # num. filas
                    # z = (x - u) / s
                    x = self.datos[j][i]
                    self.datos[j][i] = (x - u) / s

        return self.datos
