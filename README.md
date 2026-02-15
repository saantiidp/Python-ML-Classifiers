# Python-ML-Classifiers (KNN + Naive Bayes)

Proyecto en **Python** que implementa un pequeño “framework” académico de **clasificación supervisada** con:

- **KNN propio** (desde cero) con distancia euclídea y normalización opcional.
- **KNN con scikit-learn** (para comparar resultados y validar la implementación).
- **Naive Bayes** (mixto: nominal + continuo con Gaussiana) con Laplace opcional.
- **Carga y codificación de datasets** (atributos nominales → enteros, continuos → float).
- **Estrategias de validación**: *hold-out* (validación simple) y **k-fold cross-validation**.
- Notebook **`MainP1.ipynb`** con experimentos, comparación contra sklearn y resultados.

> El objetivo principal es practicar diseño modular, validación experimental y fundamentos de ML clásicos (KNN y Naive Bayes).

---

## 📁 Estructura del proyecto

```
.
├── Datos.py
├── EstrategiaParticionado.py
├── Clasificador.py
├── ClasificadorKNN.py
├── ClasificadorKNNSK.py
├── ClasificadorNB.py
└── MainP1.ipynb
```

### ¿Qué hace cada archivo?

- **`Datos.py`**
  - Lee un CSV con `pandas`.
  - Detecta atributos nominales (tipo `object`) y **trata siempre la última columna como clase**.
  - Construye:
    - `datos.datos` → matriz `numpy` con todo numérico (incluida la clase).
    - `datos.nominalAtributos` → lista `bool` indicando qué columnas son nominales.
    - `datos.diccionarios` → diccionario por columna para codificar valores nominales.
  - `extraeDatos(idx)` devuelve un subconjunto por índices.
  - `estandarizarDatos()` estandariza **solo atributos continuos** (z-score).

- **`EstrategiaParticionado.py`**
  - Define `Particion(indicesTrain, indicesTest)`.
  - Define interfaz `EstrategiaParticionado.creaParticiones(...)`.
  - Implementa:
    - `ValidacionSimple(proporcionTrain, numeroEjecuciones)` (*hold-out*).
    - `ValidacionCruzada(k)` (k-fold).

- **`Clasificador.py`**
  - Clase base abstracta con:
    - `entrenamiento(...)`
    - `clasifica(...)`
  - Implementa:
    - `validacion(particionado, dataset, clasificador, laplace=False, seed=None)`
    - `error(datos, pred)` (métrica usada en validación; ver notas abajo).

- **`ClasificadorKNN.py`**
  - Implementación propia de **KNN**.
  - Normaliza usando medias y desviaciones del train si `normalizar=True`.
  - Calcula distancia euclídea y vota por mayoría sobre los `k` vecinos.

- **`ClasificadorKNNSK.py`**
  - KNN con **scikit-learn** (`KNeighborsClassifier`) + `StandardScaler` opcional.
  - Útil para comprobar que la implementación propia se comporta razonablemente.

- **`ClasificadorNB.py`**
  - Implementación de **Naive Bayes**:
    - Para atributos nominales: probabilidades condicionadas `P(x_i | clase)`.
    - Para continuos: modelo **Gaussiano** (densidad normal con media y desviación).
  - Calcula a-prioris `P(clase)` y predice la clase con mayor probabilidad.

- **`MainP1.ipynb`**
  - Notebook con experimentos:
    - Estándarización vs `StandardScaler`.
    - Resultados de Naive Bayes y KNN.
    - Validación simple y cruzada.

---

## 🧠 Flujo general de uso

1) Cargar datos desde CSV con `Datos` (codifica nominales a enteros).  
2) Elegir una estrategia de particionado (`ValidacionSimple` o `ValidacionCruzada`).  
3) Elegir clasificador (KNN propio, KNN sklearn o Naive Bayes).  
4) Ejecutar validación con `Clasificador.validacion(...)` para obtener un vector de métricas por partición.

---

## ✅ Estrategias de validación

### 1) Validación simple (hold-out) — `ValidacionSimple`
- Selecciona aleatoriamente un porcentaje de filas para train (`proporcionTrain`).
- El resto se usa como test.
- **Nota:** aunque existe el parámetro `numeroEjecuciones`, en esta versión se construye **una partición** (una ejecución).

### 2) Validación cruzada k-fold — `ValidacionCruzada`
- Baraja el dataset y lo divide en `k` folds.
- Para cada fold `i`:
  - test = fold `i`
  - train = resto de folds
- Devuelve `k` particiones.

---

## 🧩 Clasificadores en detalle

## 1) KNN propio — `ClasificadorKNN`
**Idea:** un punto se clasifica según las etiquetas de sus `k` vecinos más cercanos.

- Entrenamiento:
  - Guarda `datosTrain`.
  - Si `normalizar=True`, estandariza cada atributo continuo del train y guarda `media/std` por columna.
- Predicción:
  - Si `normalizar=True`, estandariza el test usando **la media/std del train**.
  - Calcula distancias euclídeas entre el punto de test y todos los de train.
  - Ordena por distancia y toma los `k` vecinos.
  - Predice por **mayoría** (`Counter.most_common(1)`).

Parámetros:
- `k`: número de vecinos.
- `distancia`: en este código solo se usa `"euclidea"`.
- `normalizar`: recomendado si hay atributos con escalas distintas.

## 2) KNN scikit-learn — `ClasificadorKNNSK`
Permite comparar con una implementación estándar:
- `KNeighborsClassifier(n_neighbors=k, metric=distancia)`
- `StandardScaler` opcional si `normalizar=True`

## 3) Naive Bayes — `ClaificadorNaiveBayes`
**Idea:** asume independencia condicional de atributos dado la clase.

- Calcula a-prioris:
  - `P(clase) = #clase / #total`
- Para atributos nominales:
  - `P(valor | clase)` usando conteo por clase.
  - Laplace opcional (suma 1 al conteo si `laplace=True`).
- Para atributos continuos:
  - Modela `P(x | clase)` con una **Gaussiana**.
  - En esta implementación se guardan medias/std por atributo (nota: ver “Detalles a revisar”).

Predicción:
- Para cada clase, calcula la probabilidad proporcional:
  - `P(clase) * Π_i P(x_i | clase)`
- Devuelve la clase con probabilidad máxima.

---

## ▶️ Cómo ejecutar

### Opción A: Notebook (recomendado)
Abrir `MainP1.ipynb` y ejecutar celdas (Jupyter / VS Code).

### Opción B: Script rápido (ejemplo)
Puedes crear un `main.py` con algo así:

```python
from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from ClasificadorKNN import ClasificadorKNN
from Clasificador import Clasificador

datos = Datos("tu_dataset.csv", print_val=False)
particionado = ValidacionCruzada(k=5)

knn = ClasificadorKNN(k=5, normalizar=True)
base = Clasificador()  # En la práctica, llamarías a validación desde una instancia concreta o moverías validacion() a función util.

# Si prefieres: usa el método validacion desde una instancia de tu clasificador base
errores = base.validacion(particionado, datos, knn)
print(errores)
print("Media:", sum(errores) / len(errores))
```

> Nota: si vas a usarlo “en limpio”, lo ideal es convertir `validacion(...)` en método estático o función de utilidad y no instanciar `Clasificador` directamente (es abstracta).


## 🛠️ Dependencias

- `numpy`
- `pandas`
- `scipy` (para distancia euclídea en KNN propio)
- `scikit-learn` (solo para `ClasificadorKNNSK` y comparativas del notebook)
- Jupyter (si usas el notebook)

---

## 👤 Autor

Santiago de Prada Lorenzo
