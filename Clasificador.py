from abc import ABCMeta,abstractmethod

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
		num_elems = len(datos)
		for i, dato in enumerate(datos):
			if dato[-1] == pred[i]:
				errores += 1
		
		return errores / num_elems

	# Realiza una clasificacion utilizando una estrategia de particionado determinada
	# TODO: implementar esta funcion
	def validacion(self,particionado,dataset,clasificador, laplace=False, seed=None):
		
		# Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
		# - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
		# y obtenemos el error en la particion de test i
		# - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
		# y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
		# devuelve el vector con los errores por cada partici�n
		
		# pasos
		# crear particiones
		particiones = particionado.creaParticiones(dataset)

		# inicializar vector de errores
		list_errores = []
		
		len_particiones = len(particiones)

		for num_particion in range(len_particiones):
			# obtener datos de train
			datos_train = dataset.extraeDatos(particiones[num_particion].indicesTrain)

			# obtener datos de test
			datos_test = dataset.extraeDatos(particiones[num_particion].indicesTest)
			
			# entrenar sobre los datos de train
			clasificador.entrenamiento(datos_train, dataset.nominalAtributos, dataset.diccionarios, laplace)

			# obtener prediciones de los datos de test (llamando a clasifica)
			predicciones = clasificador.clasifica(datos_test, dataset.nominalAtributos, dataset.diccionarios)
			
			# a�adir error de la partici�n al vector de errores
			list_errores.append(self.error(datos_test, predicciones))

		return list_errores

