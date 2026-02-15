from abc import ABCMeta,abstractmethod
import random
import Datos

class Particion():

  # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones  
  def __init__(self, indicesTrain:list, indicesTest:list):
    self.indicesTrain=indicesTrain
    self.indicesTest=indicesTest

#####################################################################################################

class EstrategiaParticionado:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor 
  def __init__(self):
    self.particiones = []
  
  @abstractmethod
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  
  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar

  def __init__(self, proporcionTrain=0.75, numeroEjecuciones=8):
    self.proporcionTrain = proporcionTrain
    self.numeroEjecuciones = numeroEjecuciones
    super().__init__()

  def creaParticiones(self, datos:Datos, seed=None):
    if datos is None:
      return None
    
    num_filas_total = datos.datos.shape[0]

    random.seed(seed)
    indicesTrain = random.sample(range(num_filas_total), round(num_filas_total * self.proporcionTrain))
    indicesTest = [x for x in range(num_filas_total) if x not in indicesTrain]

    self.particiones.append(Particion(indicesTrain, indicesTest))
    return self.particiones

      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  def __init__(self, k=4):
    self.k = k
    super().__init__()
    

  def creaParticiones(self,datos,seed=None):
    if datos is None:
      return None
    
    random.seed(seed)
    random.shuffle(datos.datos)

    num_filas_total = datos.datos.shape[0]

    fold_size = num_filas_total // self.k

    list_partitions = []

    for i in range(self.k):
      if i == 0: # caso incial, empieza en 0
        partition = [0, (i + 1) * fold_size]
      elif i == self.k - 1: # caso ultimo, coge hasta el final por sila division no fue entera
        partition = [i*fold_size + 1, num_filas_total - 1]
      else: # resto de casos "normales"
        partition = [i*fold_size + 1, (i + 1) * fold_size]
      list_partitions.append(partition) # añade a la lista

    # k particiones creadas en rangos
    # hay que hacer las combinaciones de validacion cruzada

    for i in range(self.k):
      # i va a ser la que se use para validacion (test) y el resto para entrenamiento
      # (podria ser de otra forma pero es la más facil)

      # indices train es la suma de todos los rangos que no son el de entrenamiento (k-1)
      indicesTrain = []
      for j in range(self.k):
        if j != i:
          indicesTrain += list(range(list_partitions[j][0], list_partitions[j][1] + 1))
       
      indicesTest = list(range(list_partitions[i][0], list_partitions[i][1] + 1)) # primer y segundo elemento de la particion i respec.

      self.particiones.append(Particion(indicesTrain, indicesTest))
    
    return self.particiones