from abc import ABCMeta,abstractmethod
import statistics

import numpy as np

class Clasificador:

	# Clase abstracta
	__metaclass__ = ABCMeta
	
	# Metodos abstractos que se implementan en casa clasificador concreto
	@abstractmethod
	def entrenamiento(self,datosTrain,nominalAtributos,diccionario):
		pass
	
	
	@abstractmethod
	def clasifica(self,datosTest,nominalAtributos,diccionario):
		pass
	
	# Obtiene el numero de aciertos y errores para calcular la tasa de fallo
	# TODO: implementar
	def error(self,datos,pred):
		# Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
		# devuelve el error
		errores = 0
		fp, fn, tp, tn = 0, 0, 0, 0
		
		# Recorremos los datos para calcular el error y actualizar la matriz de confusión
		for indice, valor_real in enumerate(datos):
			valor_predicho = pred[indice]

			clase1 = list(self.classValues.keys())[0]

			if valor_real[-1] != valor_predicho:
				errores += 1
				if valor_predicho == clase1:
					fp += 1
				else:
					fn += 1
			else:
				if valor_predicho == clase1:
					tp += 1
				else:
					tn += 1

		self.FP.append(fp)
		self.FN.append(fn)
		self.TP.append(tp)
		self.TN.append(tn)

		errores = errores / len(datos)

		return errores


	# Realiza una clasificacion utilizando una estrategia de particionado determinada
	# TODO: implementar esta funcion
	def validacion(self, particionado, dataset, clasificador, seed=None):
		# Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
		particiones = particionado.creaParticiones(dataset, seed)
		
		# Inicializamos el vector de errores y matrices de confusión
		list_errores = []
		self.FP, self.FN, self.TP, self.TN = [], [], [], []

		self.classValues = dataset.diccionarios['Class']

		# Iteramos sobre cada partición creada
		for num_particion in range(len(particiones)):
			# Obtenemos datos de train y test
			datos_train = dataset.extraeDatos(particiones[num_particion].indicesTrain)
			datos_test = dataset.extraeDatos(particiones[num_particion].indicesTest)
			
			# Entrenamos el clasificador sobre los datos de train
			clasificador.entrenamiento(datos_train, dataset.nominalAtributos, dataset.diccionarios)

			# Obtenemos las predicciones para los datos de test
			predicciones = clasificador.clasifica(datos_test, dataset.nominalAtributos, dataset.diccionarios)
			
			# Añadimos el error de la partición al vector de errores
			list_errores.append(self.error(datos_test, predicciones))

		# Calculamos la matriz de confusión promedio
		matrizConfusion = np.array([
			[statistics.mean(self.TP), statistics.mean(self.FP)],
			[statistics.mean(self.FN), statistics.mean(self.TN)]
		])

		return list_errores, matrizConfusion

