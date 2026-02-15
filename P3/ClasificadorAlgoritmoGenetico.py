import random
import numpy as np
import Clasificador as clasificador

class ClasificadorAlgoritmoGenetico:
    def __init__(self, tam_poblacion, num_generaciones, prob_cruce, prob_mut, max_reglas, valores_por_atributo, prop_elitismo, logs):
        self.tam_poblacion = tam_poblacion
        self.num_generaciones = num_generaciones
        self.prob_cruce = prob_cruce
        self.prob_mut = prob_mut
        self.max_reglas = max_reglas
        self.valores_por_atributo = valores_por_atributo
        self.prop_elitismo = prop_elitismo
        self.logs = logs
        self.poblacion = []
        self.mejor_individuo = None

    def generar_regla(self):
        """Genera una regla aleatoria con longitudes correctas por atributo"""
        regla = ''.join(
            ''.join(random.choice(['0', '1']) for _ in range(valores))
            for valores in self.valores_por_atributo
        )
    
        return regla + random.choice(['0', '1'])  # Agregar la clase como conclusión

    def inicializar_poblacion(self):
        """Inicializa la población con reglas aleatorias."""
        self.poblacion = [
            [self.generar_regla() for _ in range(random.randint(1, self.max_reglas))]
            for _ in range(self.tam_poblacion)
        ]
        # Validar que todos los individuos tengan al menos una regla
        self.poblacion = [ind for ind in self.poblacion if len(ind) > 0]

    def fitness(self, individuo, X_train, y_train):
        """Calcula el fitness del individuo"""
        aciertos = 0
        for x, y in zip(X_train, y_train):
            for regla in individuo:
                if self.regla_compatible(regla, x):
                    if regla[-1] == str(int(y)):
                        aciertos += 1
                    break
        return aciertos / len(X_train)

    def regla_compatible(self, regla, ejemplo):
        """Comprueba si una regla se activa con un ejemplo"""
        inicio = 0
        for i, valores in enumerate(self.valores_por_atributo):
            subcadena = regla[inicio:inicio + valores]
            
            # Validar índice y evitar accesos fuera de rango
            if int(ejemplo[i]) >= len(subcadena) or int(ejemplo[i]) < 0:
                return False
            
            # Verificar compatibilidad
            if '1' in subcadena and subcadena[int(ejemplo[i])] != '1':
                return False
            
            inicio += valores
        return True


    def evolucionar_poblacion(self, X_train, y_train):
        """Evoluciona la población aplicando cruce y mutación."""
        self.fitness_mejor = []  # Para almacenar el mejor fitness de cada generación
        self.fitness_promedio = []  # Para almacenar el fitness promedio de cada generación

        for generacion in range(self.num_generaciones):
            # Calcular el fitness de cada individuo
            fitness_poblacion = [self.fitness(ind, X_train, y_train) for ind in self.poblacion]
            
            # Verificar que fitness_poblacion no esté vacío o inválido
            if len(fitness_poblacion) == 0 or all(f is None for f in fitness_poblacion):
                raise ValueError("Todos los individuos tienen fitness inválido.")
            
            # Guardar el mejor y promedio fitness de esta generación
            mejor_fitness = max(fitness_poblacion)
            promedio_fitness = sum(fitness_poblacion) / len(fitness_poblacion)

            # Guardar el mejor y promedio fitness de esta generación
            self.fitness_mejor.append(max(fitness_poblacion))
            self.fitness_promedio.append(sum(fitness_poblacion) / len(fitness_poblacion))

            # Mostrar mensajes aclarativos
            if (generacion + 1) % 10 == 0 and self.logs is True:
                print(f"Generación {generacion + 1}/{self.num_generaciones}")
                print(f"Mejor fitness: {mejor_fitness:.4f}")
                print(f"Fitness promedio: {promedio_fitness:.4f}")
                print('-----------------------------------------------------------------------------')

            # Ordenar la población por fitness (descendente)
            self.poblacion.sort(key=lambda ind: -self.fitness(ind, X_train, y_train))
            
            # Generar nuevos individuos
            nueva_poblacion = self.poblacion[:int(len(self.poblacion) // self.prop_elitismo)]  # Elitismo

            while len(nueva_poblacion) < self.tam_poblacion:
                # Selección por ruleta
                #p1, p2 = random.choices(self.poblacion[:len(self.poblacion)//2], k=2)

                fitness_total = sum(fitness_poblacion)
                probabilidades = [f / fitness_total for f in fitness_poblacion]
                p1, p2 = random.choices(self.poblacion, weights=probabilidades, k=2)
                
                # Aplicar cruce
                if random.random() < self.prob_cruce:
                    p1, p2 = self.cruce(p1, p2)
                
                # Aplicar mutación
                nueva_poblacion.append([self.mutar(r) for r in p1])
                nueva_poblacion.append([self.mutar(r) for r in p2])
            
            # Actualizar la población
            self.poblacion = nueva_poblacion

        # Establecer el mejor individuo al final del entrenamiento
        if len(self.poblacion) == 0:
            raise ValueError("La población está vacía al final del entrenamiento.")
        self.mejor_individuo = max(self.poblacion, key=lambda ind: self.fitness(ind, X_train, y_train))


    def cruce(self, ind1, ind2):
        """Realiza un cruce entre dos individuos."""
        # Validar longitud de los individuos
        if len(ind1) <= 1 or len(ind2) <= 1:
            return ind1, ind2  # Si no es posible hacer cruce, devolver originales

        # Elegir punto de cruce al azar
        punto_cruce = random.randrange(1, min(len(ind1), len(ind2)))

        # Realizar el cruce
        nuevo_ind1 = ind1[:punto_cruce] + ind2[punto_cruce:]
        nuevo_ind2 = ind2[:punto_cruce] + ind1[punto_cruce:]

        return nuevo_ind1, nuevo_ind2


    def mutar(self, regla):
        """Mutación estándar"""
        return ''.join(
            bit if random.random() > self.prob_mut else ('0' if bit == '1' else '1')
            for bit in regla
        )

    def entrenamiento(self, X_train, y_train):
        """Entrena el clasificador"""
        self.inicializar_poblacion()
        self.evolucionar_poblacion(X_train, y_train)

    def clasifica(self, X_test):
        """Clasifica usando el mejor individuo"""
        predicciones = []
        for x in X_test:
            for regla in self.mejor_individuo:
                if self.regla_compatible(regla, x):
                    predicciones.append(int(regla[-1]))
                    break
            else:
                predicciones.append(0)  # Clase por defecto
        return predicciones
    
    def error(self, datos_test, predicciones):
            """Calcula la tasa de error comparando las predicciones con las clases reales."""
            clases_reales = datos_test[:, -1]  # Última columna son las etiquetas reales
            num_errores = np.sum(clases_reales != predicciones)  # Conteo de errores
            tasa_error = num_errores / len(clases_reales)
            return tasa_error

