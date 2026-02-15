import numpy as np
from Clasificador import Clasificador
import random
import math

class ClaificadorRL(Clasificador):
    def __init__(self, epocas=100, const_aprendizaje=0.1):
        super().__init__()
        self.w = None
        self.const_aprendizaje = const_aprendizaje
        self.epocas = epocas

    def entrenamiento(self,datosTrain,nominalAtributos,diccionario):
        if datosTrain is None or nominalAtributos is None or diccionario is None:
            return None

        # paso 1
        self.w = np.random.uniform(-0.5, 0.5, len(nominalAtributos) - 1)
        
        i = 0
        # paso 2
        for dato in datosTrain:
            if i >= self.epocas:
                break

            # x*t
            x = dato[:len(nominalAtributos) - 1]
            t = dato[-1]

            x_t = np.dot(self.w, x.T) # no seguro

            # calcular probabilidad a posterior de la clase
            prob_posteriori = 1 / (1 + math.exp(- x_t))

            # ajustar w
            self.w = self.w - self.const_aprendizaje * (x * (prob_posteriori - t))

            i += 1
        
        return self.w
                    

    def clasifica(self,datosTest,nominalAtributos,diccionario):
        if datosTest is None or nominalAtributos is None or diccionario is None:
            return None
        
        umbral = 0.5
        lista_probs = []
        for dato in datosTest:
            x = dato[:len(nominalAtributos) - 1]

            x_t = np.dot(self.w, x.T) # no seguro

            prob = 1 / (1 + math.exp(- x_t))

            if prob > umbral:
                lista_probs.append(1)
            else:
                lista_probs.append(0)

        return lista_probs
    
    def obtener_scores(self, datosTest, nominalAtributos, diccionario):
        """
        Método para obtener los scores (probabilidades) de pertenencia a la clase positiva
        para cada instancia de test.
        """
        if datosTest is None or nominalAtributos is None or diccionario is None:
            return None
        
        scores = []
        for dato in datosTest:
            x = dato[:len(nominalAtributos) - 1]
            x_t = np.dot(self.w, x.T)
            prob = 1 / (1 + math.exp(-x_t))
            scores.append(prob)
        
        return scores
