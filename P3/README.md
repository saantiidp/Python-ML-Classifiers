# P3 — Clasificador con Algoritmo Genético

Esta carpeta **`P3/`** contiene la **Práctica 3** del repositorio. En esta práctica se implementa un **clasificador basado en Algoritmos Genéticos (GA)**, donde cada individuo representa un **conjunto de reglas** binarias que se evolucionan para maximizar la precisión de clasificación sobre un dataset.

El proyecto incluye:

- **Clasificador con Algoritmo Genético** basado en población de reglas.
- Operadores genéticos: **selección por ruleta**, **cruce**, **mutación** y **elitismo**.
- **Función de fitness** basada en tasa de aciertos sobre el conjunto de entrenamiento.
- **Carga y preprocesado de datos** (codificación de nominales y estandarización).
- **Estrategias de validación**: validación simple (*hold-out*) y validación cruzada (*k-fold*).
- Notebook **`MainP3.ipynb`** con experimentos y visualización de resultados.
- Exportación de resultados a HTML (`html_MainP3.html`).

> Este proyecto forma parte de un repositorio con varias prácticas:  
> - `P1/`: KNN y Naive Bayes  
> - `P2/`: KNN, Naive Bayes, Regresión Logística y K-Means  
> - `P3/`: Algoritmo Genético para clasificación  

---

## 📁 Estructura de la carpeta `P3/`

```
P3/
├── Datos.py
├── EstrategiaParticionado.py
├── Clasificador.py
├── ClasificadorAlgoritmoGenetico.py
├── MainP3.ipynb
└── html_MainP3.html
```

---

## 🧩 ¿Qué hace cada archivo?

### `Datos.py`
- Carga un dataset desde CSV usando `pandas`.
- Detecta atributos nominales y **trata siempre la última columna como clase**.
- Convierte atributos nominales a valores numéricos mediante diccionarios.
- Mantiene:
  - `datos.datos` → matriz `numpy` con todos los valores numéricos.
  - `datos.nominalAtributos` → lista booleana indicando qué atributos son nominales.
  - `datos.diccionarios` → diccionarios de codificación por columna.
- `extraeDatos(idx)` devuelve subconjuntos por índices.
- `estandarizarDatos()` aplica estandarización z-score a atributos continuos.

### `EstrategiaParticionado.py`
- Define la clase `Particion(indicesTrain, indicesTest)`.
- Implementa dos estrategias:
  - `ValidacionSimple(proporcionTrain, numeroEjecuciones)` (*hold-out*).
  - `ValidacionCruzada(k)` (*k-fold cross-validation*).
- Genera listas de índices para train y test que luego usa el clasificador.

### `Clasificador.py`
- Clase base abstracta para clasificadores.
- Define:
  - `entrenamiento(...)`
  - `clasifica(...)`
- Implementa:
  - `validacion(...)`: ejecuta el ciclo train/test sobre cada partición.
  - `error(...)`: calcula la tasa de error y construye una **matriz de confusión media** (TP, FP, FN, TN).

### `ClasificadorAlgoritmoGenetico.py`
- Implementa un **clasificador basado en Algoritmos Genéticos**.
- Representación:
  - Cada **individuo** es una lista de **reglas binarias**.
  - Cada regla codifica condiciones sobre atributos y una **clase** como conclusión (último bit).
- Componentes principales:
  - `inicializar_poblacion()`: crea individuos con reglas aleatorias.
  - `fitness(individuo, X_train, y_train)`: mide la proporción de aciertos.
  - `cruce(ind1, ind2)`: combina reglas entre dos individuos.
  - `mutar(regla)`: mutación bit a bit con probabilidad `prob_mut`.
  - `evolucionar_poblacion(...)`: aplica selección, cruce, mutación y elitismo durante varias generaciones.
- Entrenamiento:
  - Evoluciona la población durante `num_generaciones`.
  - Guarda el **mejor individuo** según fitness.
- Clasificación:
  - Para cada ejemplo de test, busca la primera regla compatible.
  - Predice la clase indicada por esa regla.
  - Si ninguna regla aplica, usa una **clase por defecto (0)**.

### `MainP3.ipynb`
- Notebook principal de experimentación:
  - Carga de datasets.
  - Configuración del algoritmo genético (tamaño de población, generaciones, probabilidades, etc.).
  - Ejecución del entrenamiento.
  - Evaluación con validación simple y/o cruzada.
  - Gráficas de evolución del fitness (mejor y promedio por generación).

### `html_MainP3.html`
- Versión exportada del notebook con los resultados de los experimentos.

---

## 🧠 Idea del algoritmo genético

1) **Inicialización**: se crea una población de individuos (cada uno con varias reglas aleatorias).  
2) **Evaluación**: se calcula el *fitness* de cada individuo como su precisión en entrenamiento.  
3) **Selección**: se seleccionan individuos (ruleta ponderada por fitness).  
4) **Cruce**: se combinan reglas de dos padres para generar descendientes.  
5) **Mutación**: se invierten bits de las reglas con cierta probabilidad.  
6) **Elitismo**: se conservan los mejores individuos de cada generación.  
7) **Repetición**: el proceso se repite durante `num_generaciones`.  
8) **Resultado**: se elige el mejor individuo final como clasificador.

---

## ▶️ Cómo ejecutar

### Opción recomendada: Notebook

Abrir:

```bash
MainP3.ipynb
```

y ejecutar las celdas (Jupyter / VS Code).

Ahí podrás:
- Cargar datos
- Configurar parámetros del GA
- Entrenar el clasificador
- Ver la evolución del fitness
- Evaluar resultados

---

## 🧯 Notas técnicas

- La representación de reglas es **binaria**, asumiendo atributos discretizados o nominales codificados.
- Si ninguna regla cubre un ejemplo, se predice la clase **0 por defecto**.
- El rendimiento depende fuertemente de:
  - Tamaño de población
  - Número de generaciones
  - Probabilidades de cruce y mutación
  - Número máximo de reglas por individuo
- La selección por ruleta puede verse afectada si todos los fitness son muy similares.

---

## 🛠️ Dependencias

- `numpy`
- `pandas`
- `matplotlib`
- Jupyter Notebook

---

## 👤 Autor

Santiago de Prada Lorenzo
