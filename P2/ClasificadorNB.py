import numpy as np
from Clasificador import Clasificador
from collections import Counter


from scipy.stats import norm

class ClaificadorNaiveBayes(Clasificador):
    def __init__(self):
        super().__init__()
        self.aprioris = {}
        self.condicionadas = {}
        self.media_std_atributo = {}

    # TODO: esta funcion debe ser implementada en cada clasificador concreto. Crea el modelo a partir de los datos de entrenamiento
    # datosTrain: matriz numpy o dataframe con los datos de entrenamiento
    # nominalAtributos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self,datosTrain,nominalAtributos,diccionario, laplace=False):
        if datosTrain is None or nominalAtributos is None or diccionario is None:
            return None

        contador_clases = Counter(datosTrain[:, -1])
        total = sum(contador_clases.values())

        # calculo a prioris: num elem clase c / num elems totales
        for nombre_clase in diccionario['Class']:
            valor_numerico_clase = diccionario['Class'][nombre_clase]
            self.aprioris[nombre_clase] = contador_clases[valor_numerico_clase] / total

        indice_atributo_clase = len(nominalAtributos) - 1

        # media y std para continuos
        for indice_atributo, bool_atributo in enumerate(nominalAtributos):
            column = datosTrain[:, indice_atributo]
            if bool_atributo is False: # continuo
                media = np.mean(column)
                desviacion = np.std(column)
                self.media_std_atributo[indice_atributo] = (media, desviacion)

        # calcular probabilidades condicionadas para atributos nominales
        for nombre_clase in diccionario['Class']:
            valor_numerico_clase = diccionario['Class'][nombre_clase]
            num_elem_clase = contador_clases[valor_numerico_clase]
            
            dict_atributo = {}
            for indice_atributo, bool_atributo in enumerate(nominalAtributos):
                if indice_atributo == indice_atributo_clase:
                    break # no hay que hacer claulos para la clase, ya se han calculado lo a prioris

                if bool_atributo is True: # nominal
                    contador_atributos = Counter(datosTrain[:, indice_atributo])
                    dict_valor = {}
                    for valor_atributo in contador_atributos.keys():
                        condicion = (datosTrain[:, indice_atributo] == valor_atributo) & (datosTrain[:, -1] == valor_numerico_clase)
                        num_elem_clase_valor = np.sum(condicion)

                        # correccion de laplace si es necesario
                        if laplace is True:
                            num_elem_clase_valor += 1

                        # calculo prob condicionada
                        prob_condicionada = num_elem_clase_valor / num_elem_clase
                        dict_valor[valor_atributo] = prob_condicionada
                    
                    dict_atributo[indice_atributo] = dict_valor

            self.condicionadas[nombre_clase] = dict_atributo
                    

    # TODO: esta funcion debe ser implementada en cada clasificador concreto. Devuelve un numpy array con las predicciones
    # datosTest: matriz numpy o dataframe con los datos de validaci�n
    # nominalAtributos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    # devuelve un numpy array o vector con las predicciones (clase estimada para cada fila de test)
    def clasifica(self,datosTest,nominalAtributos,diccionario):
        if datosTest is None or nominalAtributos is None or diccionario is None:
            return None
        
        indice_atributo_clase = len(nominalAtributos) - 1
        lista = []
        # calcular con la formula a que clase pertenece (la más probable)
        for dato in datosTest:

            lista_probs = []
            for nombre_clase in diccionario['Class'].keys():

                prob = self.aprioris[nombre_clase]

                for indice_atributo, bool_atributo in enumerate(nominalAtributos):
                    if indice_atributo == indice_atributo_clase:
                        break # no hay que hacer claulos para la clase, ya se han calculado lo a prioris

                    if bool_atributo is True: # nominal
                        # prob cndicionada calculada en self.condicionados
                        valor_atributo = dato[indice_atributo]
                        prob *= self.condicionadas[nombre_clase][indice_atributo][valor_atributo]
                    
                    else: # numerico
                        media, std = self.media_std_atributo[indice_atributo]
                        x = dato[indice_atributo]

                        # calcular con formula
                        numerator = np.exp( (-(x - media) ** 2 ) / (2 * (std ** 2)))
                        denominator = std * np.sqrt(2 * np.pi)
                        prob *= numerator / denominator

                lista_probs.append([prob, diccionario['Class'][nombre_clase]])
            
            max_prob = max(lista_probs, key=lambda x: x[0])
            lista.append(max_prob[1])
        
        return lista


