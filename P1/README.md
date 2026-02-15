# P1 — Python-ML-Classifiers (KNN + Naive Bayes)

Esta carpeta **`P1/`** contiene la **Práctica 1** del repositorio: un proyecto en **Python** que implementa un pequeño “framework” académico de **clasificación supervisada** con:

- **KNN propio** (desde cero) con distancia euclídea y normalización opcional.  
- **KNN con scikit-learn** (para comparar resultados y validar la implementación).  
- **Naive Bayes** (mixto: nominal + continuo con Gaussiana) con Laplace opcional.  
- **Carga y codificación de datasets** (atributos nominales → enteros, continuos → float).  
- **Estrategias de validación**: *hold-out* (validación simple) y **k-fold cross-validation**.  
- Notebook **`MainP1.ipynb`** con experimentos, comparación contra sklearn y resultados.  

> Este proyecto forma parte de un repositorio con varias prácticas.  
> La **Práctica 2** se encuentra en la carpeta `P2/`.

El objetivo principal es practicar **diseño modular**, **validación experimental** y **fundamentos de ML clásicos** (KNN y Naive Bayes).

---

## 📁 Estructura de la carpeta `P1/`

```
P1/
├── Datos.py
├── EstrategiaParticionado.py
├── Clasificador.py
├── ClasificadorKNN.py
├── ClasificadorKNNSK.py
├── ClasificadorNB.py
└── MainP1.ipynb
```

---

## 🧩 ¿Qué hace cada archivo?

### `Datos.py`
- Lee un CSV con `pandas`.
- Detecta atributos nominales (tipo `object`) y **trata siempre la última columna como clase**.
- Construye:
  - `datos.datos` → matriz `numpy` con todo numérico (incluida la clase).
  - `datos.nominalAtributos` → lista `bool` indicando qué columnas son nominales.
  - `datos.diccionarios` → diccionario por columna para codificar valores nominales.
- `extraeDatos(idx)` devuelve un subconjunto por índices.
- `estandarizarDatos()` estandariza **solo atributos continuos** (z-score).

### `EstrategiaParticionado.py`
- Define `Particion(indicesTrain, indicesTest)`.
- Define interfaz `EstrategiaParticionado.creaParticiones(...)`.
- Implementa:
  - `ValidacionSimple(proporcionTrain, numeroEjecuciones)` (*hold-out*).
  - `ValidacionCruzada(k)` (k-fold).

### `Clasificador.py`
- Clase base abstracta con:
  - `entrenamiento(...)`
  - `clasifica(...)`
- Implementa:
  - `validacion(particionado, dataset, clasificador, laplace=False, seed=None)`
  - `error(datos, pred)` (métrica usada en validación; ver notas abajo).

### `ClasificadorKNN.py`
- Implementación propia de **KNN**.
- Normaliza usando medias y desviaciones del train si `normalizar=True`.
- Calcula distancia euclídea y vota por mayoría sobre los `k` vecinos.

### `ClasificadorKNNSK.py`
- KNN con **scikit-learn** (`KNeighborsClassifier`) + `StandardScaler` opcional.
- Útil para comprobar que la implementación propia se comporta razonablemente.

### `ClasificadorNB.py`
- Implementación de **Naive Bayes**:
  - Para atributos nominales: probabilidades condicionadas `P(x_i | clase)`.
  - Para continuos: modelo **Gaussiano** (densidad normal con media y desviación).
  - Calcula a-prioris `P(clase)` y predice la clase con mayor probabilidad.

### `MainP1.ipynb`
- Notebook con experimentos:
  - Estandarización vs `StandardScaler`.
  - Resultados de Naive Bayes y KNN.
  - Validación simple y cruzada.

---

## 🧠 Flujo general de uso

1) Cargar datos desde CSV con `Datos` (codifica nominales a enteros).  
2) Elegir una estrategia de particionado (`ValidacionSimple` o `ValidacionCruzada`).  
3) Elegir clasificador (KNN propio, KNN sklearn o Naive Bayes).  
4) Ejecutar validación con `Clasificador.validacion(...)` para obtener un vector de métricas por partición.

---

## ▶️ Cómo ejecutar

### Opción A: Notebook (recomendado)
Abrir `MainP1.ipynb` y ejecutar las celdas (Jupyter / VS Code).

---

## 🧯 Detalles a revisar

1) **`error()` realmente calcula precisión (accuracy), no error**.  
2) **`ValidacionSimple.numeroEjecuciones` no se usa** (solo se crea una partición).  
3) **`seed` no se propaga** correctamente a la creación de particiones.  
4) **Naive Bayes continuo**: media/std no están condicionadas por clase.  
5) **Normalización en KNN propio**: posible división por cero si `std == 0`.

---

## 🛠️ Dependencias

- `numpy`  
- `pandas`  
- `scipy`  
- `scikit-learn` (solo para `ClasificadorKNNSK`)  
- Jupyter Notebook  

---

## 👤 Autor

Santiago de Prada Lorenzo
