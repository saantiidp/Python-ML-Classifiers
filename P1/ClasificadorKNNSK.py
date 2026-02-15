from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

class ClasificadorKNNSK:
    def __init__(self, distancia="euclidean", k=5, normalizar=True):
        self.distancia = distancia
        self.k = k
        self.normalizar = normalizar
        self.scaler = StandardScaler() if normalizar else None
        self.clf = KNeighborsClassifier(n_neighbors=k, metric=distancia)

    def entrenamiento(self, datosTrain, nominalAtributos=None, diccionario=None):
        if self.normalizar:
            datosTrain = self.scaler.fit_transform(datosTrain)
        self.clf.fit(datosTrain[:, :-1], datosTrain[:, -1].astype(int))

    def clasifica(self, datosTest, nominalAtributos=None, diccionario=None):
        if self.normalizar:
            datosTest = self.scaler.transform(datosTest)
        return self.clf.predict(datosTest[:, :-1])
