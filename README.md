# Taller 3 – Aprendizaje Automático
## Predicción de Accidentalidad Vial en Medellín 🚦

> **Asignatura:** Aprendizaje Automático  
> **Grupo:** 5  
> **Dataset:** Accidentes de tránsito + variables climáticas — Medellín

---

## 📌 Descripción General

Este proyecto aborda la **predicción de accidentalidad vial en Medellín** a nivel de barrio y franja horaria, combinando datos históricos de accidentes de tránsito con variables meteorológicas. La pregunta central que guía el trabajo es:

> *¿Es posible predecir si ocurrirá al menos un accidente en un barrio específico, durante una hora determinada del día, dadas las condiciones climáticas del momento?*

El resultado es un **clasificador binario** cuya variable objetivo (`TARGET`) vale `1` si ocurrió al menos un accidente en esa combinación (barrio, hora) y `0` en caso contrario.

---

## 🗂️ Estructura del Proyecto

```
Taller-3-Aprendizaje-Automatico-Grupo-5/
│
├── main.ipynb                   # Notebook principal con todo el desarrollo
├── data_accidentes.sqlite3      # Base de datos SQLite (accidentes + clima)
└── README.md                    # Este archivo
```

---

## 🔬 Metodología y Contenido del Notebook

### 1. Carga de Datos
Conexión a la base de datos SQLite y lectura de tres tablas:
- `accidentes` — registros individuales de siniestros viales
- `clima` — variables meteorológicas horarias por barrio
- `raw_accidentes` — tabla cruda original

### 2. Auditoría de Calidad (Punto 4.2)
- Detección de duplicados totales y en llaves de unión `(TW, BARRIO)`
- Análisis de valores faltantes por tabla
- Estandarización de la variable `BARRIO` (strip + upper)
- Verificación de coincidencia de barrios entre tablas

### 3. Construcción de la Variable Objetivo (Punto 3)
- **Unidad de análisis:** bloque horario × barrio
- **JOIN** entre `clima` (universo completo) y `accidentes` (casos positivos)
- Relleno de `NaN` → `0` (sin accidente) y casos cruzados → `1`
- El dataset resultante (`df_maestro`) presenta un **desbalance severo: ~98.5% clase 0 / ~1.5% clase 1**

### 4. Probabilidad Empírica
Cálculo de la probabilidad marginal de accidente mediante frecuencia relativa:

$$P(\text{accidente}) = \frac{\text{casos con accidente}}{\text{total de observaciones}}$$

### 5. Features Cíclicas — Rol: Ingeniero de Datos
Transformación de la hora (0–23) en componentes trigonométricas para preservar la **continuidad cíclica** del tiempo:

$$\text{hora\_sin} = \sin\!\left(\frac{2\pi \cdot h}{24}\right) \qquad \text{hora\_cos} = \cos\!\left(\frac{2\pi \cdot h}{24}\right)$$

> La hora no puede tratarse como número lineal porque el modelo interpretaría que las 23:00 y las 00:00 están a 23 unidades de distancia, cuando en realidad son horas consecutivas.

### 6. Punto 4.4 – Modelo No Sesgado

#### 6.1 Baseline Ingenuo (ZeroR)
Demuestra que una *accuracy* del ~98.5% es **engañosa**: un clasificador que siempre predice "sin accidente" la alcanza sin aprender nada útil. Las métricas relevantes son **F1-score** y **ROC-AUC**.

#### 6.2 Estrategias de Balanceo
| Estrategia | Descripción |
|---|---|
| `class_weight='balanced'` | Ajusta los pesos del modelo para penalizar más los errores en la clase minoritaria |
| **SMOTE** | Genera ejemplos sintéticos de la clase minoritaria mediante interpolación entre vecinos |

#### 6.3 Comparación de Tres Familias de Modelos
Con ajuste de hiperparámetros vía `RandomizedSearchCV` y validación cruzada estratificada (3-fold):

| Modelo | Descripción |
|---|---|
| **Regresión Logística** | Modelo lineal, interpretable, rápido |
| **Random Forest** | Ensamble de árboles, captura relaciones no lineales |
| **Gradient Boosting** (`HistGradientBoosting`) | Boosting eficiente para datasets de gran volumen |

---

## 🛠️ Tecnologías Utilizadas

| Librería | Uso |
|---|---|
| `pandas` | Manipulación y análisis de datos |
| `numpy` | Operaciones numéricas y transformaciones |
| `scikit-learn` | Modelos ML, métricas, validación cruzada |
| `imbalanced-learn` | SMOTE y estrategias de balanceo |
| `matplotlib` / `seaborn` | Visualizaciones |
| `sqlite3` | Conexión a la base de datos |

---

## 👥 Integrantes del Equipo

| Nombre | Rol en el Taller |
|---|---|
| **Integrante 1** | — |
| **Integrante 2** | — |
| **Integrante 3** | — |
| **Integrante 4** | — |
| **Integrante 5** | Ingeniero de Datos – Features Cíclicas |

> ⚠️ *Actualiza la tabla anterior con los nombres reales de los integrantes del Grupo 5.*

---

## ▶️ Cómo Ejecutar

1. Asegúrate de tener el archivo `data_accidentes.sqlite3` en la misma carpeta que `main.ipynb`.
2. Ejecuta el notebook de arriba hacia abajo en orden.
3. La primera celda instala automáticamente todas las dependencias necesarias.

```bash
# Las dependencias se instalan desde la primera celda del notebook:
# pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn
```
