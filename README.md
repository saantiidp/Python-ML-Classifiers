# Python-ML-Algorithms

Repositorio académico de **Aprendizaje Automático en Python** que agrupa **tres prácticas** centradas en clasificación supervisada y aprendizaje no supervisado.  
Cada práctica está organizada en su propia carpeta:

- 📁 **P1/** → KNN y Naive Bayes  
- 📁 **P2/** → KNN, Naive Bayes, Regresión Logística y K-Means  
- 📁 **P3/** → Clasificador basado en Algoritmos Genéticos  

El objetivo del repositorio es **implementar desde cero** varios algoritmos clásicos de ML, practicar **preprocesado de datos**, **estrategias de validación** y **evaluación experimental**.

---

## 📂 Estructura del repositorio

```
.
├── P1/
│   └── README.md
├── P2/
│   └── README.md
├── P3/
│   └── README.md
└── README.md   <-- (este archivo)
```

Cada carpeta contiene su propio código, notebooks y un `README.md` con la explicación detallada.

---

## 🧠 Contenido de cada práctica

### 🔹 P1 — Clasificación básica (KNN + Naive Bayes)

En **P1/** se implementa un pequeño framework de **clasificación supervisada** con:

- KNN propio (desde cero)
- KNN usando scikit-learn (para comparación)
- Naive Bayes (nominal + continuo gaussiano)
- Carga y codificación de datasets
- Validación simple (*hold-out*) y validación cruzada (*k-fold*)
- Notebook de experimentación (`MainP1.ipynb`)

Objetivo principal: entender **KNN**, **Naive Bayes**, el **preprocesado de datos** y la **evaluación con validación cruzada**.

➡️ Más detalles en: `P1/README.md`

---

### 🔹 P2 — Clasificación avanzada + Clustering

En **P2/** se amplía la práctica anterior añadiendo:

- Clasificadores:
  - KNN
  - Naive Bayes
  - **Regresión Logística** (entrenada con descenso de gradiente estocástico)
- Aprendizaje no supervisado:
  - **K-Means** con inicialización tipo *k-means++*
- Preprocesado y estandarización de datos
- Validación simple y cruzada
- Notebook de experimentación (`MainP2.ipynb`)

Objetivo principal: trabajar con **modelos lineales**, **métodos no supervisados** y **comparar distintos enfoques de clasificación**.

➡️ Más detalles en: `P2/README.md`

---

### 🔹 P3 — Clasificador con Algoritmo Genético

En **P3/** se implementa un **clasificador basado en Algoritmos Genéticos**, donde:

- Cada individuo representa un **conjunto de reglas binarias**
- Se usan operadores genéticos:
  - Selección (ruleta)
  - Cruce
  - Mutación
  - Elitismo
- La función de *fitness* es la **precisión en clasificación**
- Se evalúa con validación simple y cruzada
- Notebook de experimentación (`MainP3.ipynb`) y versión exportada a HTML

Objetivo principal: aplicar **técnicas evolutivas** a un problema de clasificación y analizar su comportamiento frente a métodos clásicos.

➡️ Más detalles en: `P3/README.md`

---

## 🛠️ Tecnologías usadas

- Python 3
- numpy
- pandas
- scipy
- scikit-learn (para comparativas y utilidades)
- matplotlib (visualización)
- Jupyter Notebook

---

## ▶️ Cómo usar el repositorio

1. Entra en la carpeta de la práctica que quieras (`P1/`, `P2/` o `P3/`).
2. Lee el `README.md` de esa carpeta.
3. Abre el notebook correspondiente (`MainP1.ipynb`, `MainP2.ipynb` o `MainP3.ipynb`).
4. Ejecuta las celdas para reproducir los experimentos.

---

## 🎯 Objetivo académico

Este repositorio está pensado como:

- Ejercicio práctico de **Aprendizaje Automático**
- Implementación **desde cero** de algoritmos clásicos
- Práctica de:
  - Preprocesado de datos
  - Diseño modular en Python
  - Validación experimental
  - Comparación de modelos
  - Análisis de resultados

---

## 👤 Autor

Santiago de Prada Lorenzo
