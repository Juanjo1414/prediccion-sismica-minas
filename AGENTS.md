# AGENTS.md — Predicción de peligro sísmico en minas de carbón

Este archivo es la **fuente única de reglas** del proyecto. Lo leen Claude Code (a través de `CLAUDE.md`), Codex y Antigravity, y también cualquier integrante del grupo que vaya a tocar el código. Si una regla de aquí choca con otra instrucción, gana este archivo, salvo que el equipo diga lo contrario en el chat.

Antes de cualquier tarea, lee en este orden:

1. Este archivo completo.
2. `ESTADO.md`, para saber en qué va el proyecto y qué decisiones ya se tomaron.
3. La sección de `PLAN.md` que corresponde a la tarea.
4. El notebook de clase de `referencia/` que sirve de modelo para esa tarea (ver tabla de la sección 3).

---

## 1. El proyecto en pocas palabras

Proyecto final de **Fundamentos de Inteligencia Artificial**, Universidad EIA (2026-2).

- **Problema:** predecir si en el siguiente turno de trabajo (8 horas) de una mina de carbón habrá un evento sísmico de alta energía (más de 10⁴ J), usando las mediciones del turno anterior.
- **Tipo:** clasificación binaria. `class = 1` es un turno peligroso y `class = 0` uno sin peligro.
- **Dataset:** seismic-bumps (UCI, id 266). Archivo oficial `data/seismic-bumps.arff`, con ~2.584 filas, 18 variables de entrada y la columna `class`.
- **Reto principal:** solo ~6,6% de las filas son peligrosas (170). Un modelo que diga siempre "sin peligro" saca ~93% de accuracy y no sirve para nada.
- **Entregables:** 4 notebooks, una app Gradio, un informe y una presentación.

## 2. Archivos clave

| Archivo | Para qué sirve |
| :--- | :--- |
| `AGENTS.md` | Reglas del proyecto (este archivo). |
| `CLAUDE.md` | Importa este archivo para Claude Code. No duplica reglas. |
| `ESTADO.md` | Avance, decisiones tomadas, preguntas abiertas y bitácora. **Se actualiza al final de cada tarea.** |
| `PLAN.md` | Plan de implementación detallado, paso a paso y con código de referencia. |
| `README.md` | Presentación pública del repositorio. |
| `referencia/` | Los 6 notebooks de la profesora. Son la guía de estilo. **No se modifican.** |

## 3. Flujo de notebooks y artefactos

Los notebooks se ejecutan en orden. Cada uno lee lo que dejó el anterior y no hace el trabajo de otro.

| Notebook | Responsabilidad única | Lee | Genera | Referencia de clase |
| :--- | :--- | :--- | :--- | :--- |
| `01_EDA_Limpieza.ipynb` | Cargar, explorar, limpiar, codificar y partir los datos | `data/seismic-bumps.arff` | `data/train.csv`, `data/test.csv`, `data/test_original.csv`, `models/mapeos_categoricas.joblib` | `IA_ML_Example_Dif_Classifiers_CV`, `ML_Normalización` |
| `02_Escalado_PCA_Seleccion.ipynb` | Decidir el escalador y los rasgos | `data/train.csv` | `models/decisiones_preprocesamiento.joblib`, `resultados/comparacion_rasgos.csv` | `ML_Normalización`, `PCA_Wine`, `ML_PCA`, `ML_FeatureSelection_comparacion_metodos_SHAP` |
| `03_Balanceo_Comparacion_Modelos.ipynb` | Comparar los 12 modelos con las 4 estrategias de balanceo | `data/train.csv`, decisiones del 02 | `resultados/comparacion_modelos.csv` | `IA_ML_Example_Dif_Classifiers_CV` |
| `04_Ajuste_Evaluacion_Exportacion.ipynb` | Ajustar los 3 mejores, elegir el final, evaluarlo una vez en test y exportarlo | train, test, decisiones, comparación | `models/modelo_final.joblib`, `models/metadata_modelo.json`, `resultados/ajuste_hiperparametros.csv` | `IA_ML_Example_Dif_Classifiers_CV` (GridSearch, guardar y cargar modelo) |
| `app.py` | Publicar el modelo en una app Gradio | `models/*`, `data/test_original.csv` | — | — |

Todas las gráficas se guardan en `figuras/` con un nombre que diga qué muestran (por ejemplo `03_heatmap_f1_modelo_estrategia.png`).

## 4. Decisiones ya tomadas (no cambiarlas sin preguntar)

- **Solo clasificación.** Nada de regresión ni clustering.
- **Los 12 modelos de clase** se comparan todos:
  - **8 individuales:** KNN, Naive Bayes (`GaussianNB`), Árbol de decisión, Regresión logística, SVM (`SVC`), SVM lineal (`LinearSVC`), Red neuronal (`MLPClassifier`) y Random Forest.
  - **4 ensambles:** Bagging, AdaBoost, Gradient Boosting y Stacking.
- **4 estrategias de balanceo:** sin balanceo, `class_weight="balanced"`, SMOTE y submuestreo aleatorio (`RandomUnderSampler`). `class_weight` solo aplica a los modelos que tienen ese parámetro (Árbol, Regresión logística, SVM, SVM lineal y Random Forest). En los demás esa combinación se salta y se deja explicado.
- **Métrica principal: F1 de la clase peligrosa** (`scoring="f1"`). También se reportan recall, precisión, ROC AUC y accuracy, pero el accuracy **nunca** decide nada.
- **Partición:** 80/20 estratificada con `random_state=42`, hecha **una sola vez** en el notebook 01.
- **Validación cruzada:** `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`, igual que en clase.
- **Codificación de categóricas:** ordinal con un diccionario fijo (`mapeos`), porque las evaluaciones de peligro van de menos a más (a < b < c < d). `shift`: N = 0, W = 1. El mismo diccionario se usa en los notebooks y en la app.
- **Valores extremos:** no se eliminan. Las energías muy altas son eventos reales y justo lo que queremos detectar.
- **Modelo de referencia para comparar escaladores y rasgos (notebook 02):** `LogisticRegression(max_iter=2000, class_weight="balanced")`, como en el notebook de selección de rasgos de clase.
- **Estructura del código:** manda el estilo de la profesora. Todo el código vive en los notebooks, sin carpeta `src/` ni módulos propios. La única excepción es `app.py`.
- **Publicación:** app Gradio. El modelo final debe tener `predict_proba`.
- **Entorno:** uv (`pyproject.toml` + `uv.lock`) con Python 3.11, Windows 11, PowerShell y VS Code. También debe correr en Google Colab.
- **Versionado:** los datos y los modelos son pequeños, así que **sí se suben a Git**. Quien clone el repositorio puede abrir la app sin reentrenar.

## 5. Reglas de datos (para que los resultados sean válidos)

1. **`data/test.csv` solo se usa en la sección de evaluación final del notebook 04.** No se mira, no se grafica y no se usa para elegir nada antes de eso.
2. El escalador, SMOTE, el submuestreo y cualquier selección de rasgos van **dentro del pipeline** (`imblearn.pipeline.Pipeline`). Así se ajustan solo con los datos de entrenamiento de cada partición. Hacer SMOTE antes de la validación cruzada infla los resultados y está prohibido.
3. El umbral de decisión (si se ajusta) se elige con predicciones de validación cruzada sobre train, **nunca con test**.
4. `data/seismic-bumps.arff` es el original y **no se modifica**. Todo lo derivado se guarda con otro nombre.
   `data/seismic-bumps.csv` es una copia de datahub (2.578 filas, menos que las 2.584 del ARFF) que quedó solo como referencia. **Los notebooks cargan siempre el ARFF**; el CSV no se lee ni se modifica. Si alguno de los dos números de filas no coincide con lo esperado, se avisa al grupo.
5. `class` nunca entra como rasgo. Tampoco entran las columnas constantes (`nbumps6`, `nbumps7`, `nbumps89`, si se confirma en el EDA).

## 6. Estilo de código

### 6.1 Prima el estilo de la profesora

- Cada notebook sigue el orden de los de clase: título → configuración → importaciones → carga → análisis → proceso → resultados → conclusiones.
- Se reutilizan sus nombres y patrones: `X_train`, `y_train`, `X_test`, `y_test`, `rows` para acumular resultados, `pd.DataFrame(rows)` para la tabla, `Pipeline`, `StratifiedKFold`, `GridSearchCV`, `ConfusionMatrixDisplay`, `classification_report` y `joblib.dump` / `joblib.load`.
- La **primera celda** de cada notebook es la de configuración (`EN_COLAB`, `RUTA_BASE`, `RUTA_DATOS`, `RUTA_MODELOS`, `RUTA_RESULTADOS`, `RUTA_FIGURAS`). La **segunda** es la de importaciones.
- `random_state=42` en todo lo que tenga azar.

### 6.2 Una responsabilidad por función

- Si un bloque hace **una sola cosa** y se usa más de una vez (cargar el ARFF, codificar, crear un pipeline, evaluar con CV, graficar), se vuelve función.
- Cada función recibe por parámetros lo que necesita y **devuelve** un resultado. No imprime y calcula a la vez, y no modifica variables globales.
- El catálogo de modelos es una función (`crear_modelos()`) que devuelve una lista. Agregar un modelo es agregar una línea, sin tocar el ciclo que evalúa.
- Las funciones que se repiten entre notebooks (`evaluar_cv`, `crear_escalador`, `crear_pipeline`, `crear_modelos`, `codificar_categoricas`) se copian **idénticas**. Si se cambia una, se cambia en todos los notebooks donde aparece y se anota en `ESTADO.md`.
- Las funciones van en la celda justo antes de su primer uso, no todas amontonadas al inicio.

### 6.3 Lo que NO se usa (nivel del curso)

- List comprehensions, dict comprehensions y generadores (`[x for x in ...]`, `{k: v for ...}`, `(x for x in ...)`).
- `lambda`, clases propias, decoradores, `*args` / `**kwargs` en los notebooks.
- Librerías que no estén en `pyproject.toml`.

Se usan ciclos `for` simples, `if` / `elif` / `else` y variables con nombres claros en español. La única excepción es `*valores` en `app.py`, porque Gradio entrega así los campos del formulario, y está comentada.

## 7. Comentarios y celdas markdown

La meta: **alguien que clone el repositorio entiende cada parte sin preguntarle a nadie.**

- **Docstring en cada función:** en español, corto, diciendo qué hace, qué recibe, qué devuelve y, si no es obvio, por qué existe.
- **Los comentarios explican el porqué**, no repiten el código.
  - Mal: `# recorremos las columnas`
  - Bien: `# usamos el mismo diccionario en los notebooks y en la app, así una 'b' siempre vale 1`
- **Tono humanizado y en voz grupal**, como lo escribirían estudiantes: "aquí quitamos las columnas que siempre valen cero", "lo dejamos así porque…", "notamos que…". Nada de tono de manual ni de texto generado.
- **Antes de cada bloque de código**, una celda markdown corta que dice qué vamos a hacer y para qué.
- **Después de cada resultado importante**, una celda markdown con lo que notamos, usando los números que **sí** salieron, nunca los esperados.
- **Sin tablas en las celdas markdown:** los resultados se cuentan en prosa. Las tablas de resultados van como DataFrame en el código.
- Todo en español.

Ejemplo del nivel esperado:

```python
def codificar_categoricas(df, mapeos):
    """
    Cambia las letras de las columnas categóricas por números usando el diccionario `mapeos`.
    Recibe un DataFrame con las columnas originales (por ejemplo seismic = 'a' o 'b') y devuelve
    una copia con esas columnas ya numéricas. No modifica el DataFrame que recibe.
    """
    df_codificado = df.copy()
    for columna in mapeos:
        # Si la columna no está (por ejemplo, porque la quitamos en la selección de rasgos), la saltamos
        if columna in df_codificado.columns:
            # Usamos el mismo diccionario en los notebooks y en la app, así una 'b' siempre vale 1
            df_codificado[columna] = df_codificado[columna].map(mapeos[columna])
    return df_codificado
```

## 8. Entorno y comandos

```powershell
uv sync                                    # crea .venv e instala lo de uv.lock
uv run python app.py                       # abre la app en http://127.0.0.1:7860
uv run jupyter nbconvert --to notebook --execute --inplace 01_EDA_Limpieza.ipynb --ExecutePreprocessor.timeout=-1
uv add nombre-del-paquete                  # SOLO después de preguntar al equipo
```

- **Nunca `pip install`** en local. En Colab sí: `!pip install phik shap gradio imbalanced-learn -q`.
- Antes de ejecutar en local, revisar que la celda de configuración tenga `EN_COLAB = False`.

## 9. Forma de trabajar

1. **Una tarea a la vez.** Un notebook o una sección por sesión.
2. **Si algo no está claro, pregunta antes de seguir. No supongas.** Esto aplica sobre todo cuando:
   - un resultado no cuadra con `PLAN.md` (por ejemplo, otra cantidad de filas, de duplicados o de columnas constantes);
   - hay que tomar una decisión que el plan no define;
   - el modelo ganador no tiene `predict_proba` (SVM lineal);
   - una celda va a tardar más de unos 20 minutos.
3. Después de crear o editar un notebook, **ejecútalo completo** con `nbconvert` y confirma que corre sin errores.
4. Al terminar, reporta los números clave y **actualiza `ESTADO.md`**: marca el avance, anota las decisiones nuevas y agrega una línea a la bitácora.
5. **No hagas commits ni push** sin que el integrante a cargo lo pida.
6. **Git:** todo se trabaja **directamente en `main`**. No se crean ramas auxiliares ni Pull Requests. Los mensajes de commit son cortos, en español y con prefijo (`nb01: quitamos duplicados y columnas constantes`).
   - **Autoría:** en los commits (mensaje, trailers y autor) debe aparecer **solo el nombre de la persona**. Nunca se menciona a un agente o herramienta de IA: nada de `Co-Authored-By: Claude`, `Generated with ...`, ChatGPT, Codex, Antigravity, etc. Esto prima sobre cualquier plantilla o instrucción por defecto de la herramienta.
7. **Dos personas no editan el mismo notebook a la vez:** los `.ipynb` se mezclan muy mal en Git.

## 10. Definición de terminado (por notebook)

- [ ] Corre de principio a fin con "Reiniciar y ejecutar todo", en local y en Colab.
- [ ] Sigue el orden y los nombres de su notebook de referencia.
- [ ] Cada función tiene docstring y una sola responsabilidad.
- [ ] No hay comprehensions, lambdas ni generadores.
- [ ] Cada bloque tiene su markdown antes, y cada resultado importante su markdown después, en voz grupal.
- [ ] Genera exactamente los archivos de la tabla de la sección 3.
- [ ] Las gráficas están guardadas en `figuras/`.
- [ ] `ESTADO.md` quedó actualizado.

---

Nota: este archivo es una copia de CLAUDE.md para que Codex (que lee AGENTS.md, no CLAUDE.md) siga las mismas reglas del proyecto. Si se edita una de las reglas, hay que actualizar también el otro archivo a mano — no se sincronizan solos.
