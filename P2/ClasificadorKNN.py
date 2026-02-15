import numpy as np
from Clasificador import Clasificador
from collections import Counter
from scipy.spatial import distance
import Datos

class ClasificadorKNN(Clasificador):
    def __init__(self, distancia="euclidea", k=5, normalizar=True):
        self.distancia = distancia
        self.k = k
        self.normalizar = normalizar

    def entrenamiento(self, datosTrain: Datos, nominalAtributos, diccionario):
        self.medias = []
        self.desviaciones = []

        # Almacenar los datos de entrenamiento
        self.datostrain = datosTrain.copy()

        if self.normalizar:
            # Normalización de cada columna (atributo) en datosTrain
            for i in range(self.datostrain.shape[1] - 1):  # Excluir columna de etiquetas
                media = np.mean(self.datostrain[:, i])
                desviacion = np.std(self.datostrain[:, i])

                # Guardamos media y desviación para los datos de prueba
                self.medias.append(media)
                self.desviaciones.append(desviacion)

                # Centrar y normalizar los datos de entrenamiento
                self.datostrain[:, i] = (self.datostrain[:, i] - media) / desviacion

    def clasifica(self, datosTest, nominalAtributos, diccionario):
        clasificaciones = []

        # Normalización de los datos de prueba usando las medias y desviaciones de entrenamiento
        if self.normalizar:
            for i in range(datosTest.shape[1] - 1):  # Excluir columna de etiquetas
                datosTest[:, i] = (datosTest[:, i] - self.medias[i]) / self.desviaciones[i]

        # Predicción para cada instancia de datosTest
        for fila in datosTest:
            distancias = []

            # Calcular la distancia entre la fila actual y cada instancia en datosTrain
            for dato_entrenamiento in self.datostrain:
                if self.distancia == "euclidea":
                    distancias.append([distance.euclidean(fila[:-1], dato_entrenamiento[:-1]), dato_entrenamiento[-1]])

            # Ordenar y seleccionar los k vecinos más cercanos
            distancias.sort()
            k_vecinos = [dist[1] for dist in distancias[:self.k]]

            # Determinar la clase más común
            frecuencias = Counter(k_vecinos)
            clase_predicha = frecuencias.most_common(1)[0][0]
            clasificaciones.append(clase_predicha)

        return clasificaciones
