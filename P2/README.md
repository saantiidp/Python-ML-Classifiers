# P2 — Python-ML-Algorithms (Clasificación + Clustering)

Esta carpeta **`P2/`** contiene la **Práctica 2** del repositorio. Amplía la Práctica 1 añadiendo **nuevos algoritmos de aprendizaje automático** y técnicas de **clustering no supervisado**.

Incluye:

- **Clasificadores supervisados**:
  - KNN (implementación propia)
  - Naive Bayes (nominal + continuo gaussiano)
  - **Regresión Logística** entrenada con descenso de gradiente estocástico
- **Aprendizaje no supervisado**:
  - **K-Means** con inicialización tipo *k-means++*, cálculo de inercia y visualización
- **Carga y preprocesado de datos** (codificación de nominales, estandarización)
- **Estrategias de validación**: *hold-out* y **k-fold cross-validation**
- Notebook **`MainP2.ipynb`** con experimentos y comparativas

> Este proyecto forma parte de un repositorio con varias prácticas.  
> La **Práctica 1** se encuentra en la carpeta `P1/`.

---

## 📁 Estructura de la carpeta `P2/`

```
P2/
├── Datos.py
├── EstrategiaParticionado.py
├── Clasificador.py
├── ClasificadorKNN.py
├── ClasificadorNB.py
├── ClasificadorRL.py
├── KMeans.py
└── MainP2.ipynb
```

---

## 🧩 ¿Qué hace cada archivo?

### `Datos.py`
- Carga un CSV con `pandas`.
- Detecta atributos nominales y **trata siempre la última columna como clase**.
- Construye:
  - `datos.datos` → matriz `numpy` con todo numérico.
  - `datos.nominalAtributos` → lista `bool` indicando qué columnas son nominales.
  - `datos.diccionarios` → diccionarios para codificar valores nominales.
- `extraeDatos(idx)` devuelve subconjuntos por índices.
- `estandarizarDatos()` aplica estandarización z-score a atributos continuos.

### `EstrategiaParticionado.py`
- Define `Particion(indicesTrain, indicesTest)`.
- Implementa:
  - `ValidacionSimple(proporcionTrain, numeroEjecuciones)` (*hold-out*).
  - `ValidacionCruzada(k)` (*k-fold cross-validation*).

### `Clasificador.py`
- Clase base abstracta con:
  - `entrenamiento(...)`
  - `clasifica(...)`
- Implementa:
  - `validacion(...)`: ejecuta train/test sobre cada partición.
  - `error(...)`: calcula la **tasa de error** y acumula **matriz de confusión media**.

### `ClasificadorKNN.py`
- Implementación propia de **KNN**:
  - Distancia euclídea.
  - Normalización opcional usando estadísticas del train.
  - Predicción por mayoría entre los `k` vecinos más cercanos.

### `ClasificadorNB.py`
- Implementación de **Naive Bayes**:
  - A-prioris `P(clase)` por frecuencia.
  - Atributos nominales: conteo de `P(valor | clase)` (Laplace opcional).
  - Atributos continuos: modelo **gaussiano** (media y desviación).
  - Predicción por máxima probabilidad posterior.

### `ClasificadorRL.py`
- Implementación de **Regresión Logística binaria**:
  - Inicializa pesos aleatoriamente.
  - Entrena con **descenso de gradiente estocástico** durante un número de épocas.
  - Usa función sigmoide para obtener probabilidades.
  - Clasifica con umbral 0.5.
  - Incluye `obtener_scores(...)` para devolver probabilidades (útil para curvas ROC, etc.).

### `KMeans.py`
- Implementación de **K-Means**:
  - Inicialización de centroides con **k-means++**.
  - Reasignación iterativa de puntos a centroides.
  - Re-cálculo de centroides hasta convergencia (tolerancia `tol` o `max_iter`).
  - Cálculo de **inercia** (suma de distancias cuadradas intra-clúster).
  - `plot_clusters(...)` para visualización en 2D.
  - `predict(...)` para asignar nuevos puntos a los clústeres aprendidos.

### `MainP2.ipynb`
- Notebook de experimentación:
  - Comparación de clasificadores (KNN, NB, RL).
  - Evaluación con validación simple y cruzada.
  - Pruebas de K-Means y visualización de clústeres.
  - Análisis de resultados.

---

## 🧠 Flujo general de uso (clasificación)

1) Cargar dataset con `Datos`.  
2) Elegir estrategia de particionado (`ValidacionSimple` o `ValidacionCruzada`).  
3) Elegir clasificador (`ClasificadorKNN`, `ClasificadorNB` o `ClasificadorRL`).  
4) Ejecutar `Clasificador.validacion(...)` para entrenar y evaluar en cada partición.  

Se obtienen:
- Vector de tasas de error por partición
- Matriz de confusión media

---

## ▶️ Cómo ejecutar

### Opción recomendada: Notebook

Abrir:

```bash
MainP2.ipynb
```

y ejecutar las celdas (Jupyter / VS Code).

Ahí encontrarás:
- Carga de datos
- Entrenamiento de clasificadores
- Validación y métricas
- Ejemplos de K-Means y gráficas

---

## 🧯 Notas técnicas

- La **Regresión Logística** está pensada para **problemas binarios** (clases 0/1).
- En **Naive Bayes**, las medias/desviaciones de continuos se calculan por atributo (no por clase), lo que simplifica la implementación.
- En **KNN**, si algún atributo tiene desviación 0, habría división por cero (no está controlado).
- En **ValidacionSimple**, el parámetro `numeroEjecuciones` no genera múltiples particiones (solo una).

---

## 🛠️ Dependencias

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- Jupyter Notebook

---

## 👤 Autor

Santiago de Prada Lorenzo
