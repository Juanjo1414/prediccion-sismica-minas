# Plan de implementación — Predicción de peligro sísmico en minas de carbón

5 de octubre de 2026 · Juan José Jaramillo

## 1. Resumen y decisiones tomadas

Vamos a predecir si en el siguiente turno de trabajo (8 horas) de una mina de carbón habrá un evento sísmico de alta energía (más de 10⁴ J), usando las mediciones del turno anterior. Es una clasificación binaria con un desbalance fuerte, y el mejor modelo se publica en una app Gradio.

**Dataset:** seismic-bumps ([UCI, id 266](https://archive.ics.uci.edu/dataset/266/seismic+bumps)), de Marek Sikora y Łukasz Wróbel. Son datos de dos tajos largos de una mina de carbón en Polonia. El archivo oficial es `seismic-bumps.arff`: según UCI tiene 2.584 filas, 18 variables de entrada y la columna `class`. Hay una copia en CSV (datahub) con 2.578 filas que sirve para explorar, pero el proyecto trabaja con el ARFF oficial.

**Variable objetivo:** `class`. Vale 1 si el turno es peligroso (hubo un evento de alta energía en el turno siguiente) y 0 si no. Solo 170 filas son de clase 1, cerca del 6,6%.

**Lo que ya sabemos de los datos** (revisado en la copia CSV; se confirma en el EDA con el ARFF):

- No hay nulos.
- Hay 4 variables categóricas: `seismic` (a, b), `seismoacoustic` (a, b, c), `shift` (W, N) y `ghazard` (a, b, c).
- `nbumps6`, `nbumps7` y `nbumps89` valen 0 en todas las filas.
- Las variables de energía (`genergy`, `energy`, `maxenergy`) son muy asimétricas y tienen valores extremos.
- En una prueba rápida con la copia CSV, el F1 de la clase peligrosa quedó entre 0,14 y 0,26. Es un problema difícil, y eso se analiza en el informe, no se esconde.

**Decisiones tomadas:**

- **Solo clasificación.** Se comparan los **12 modelos vistos en clase** (8 individuales y 4 ensambles) con **4 estrategias de balanceo**: sin balanceo, `class_weight`, SMOTE y submuestreo aleatorio. Son 41 combinaciones, porque `class_weight` solo aplica a 5 de los modelos.
- **Métrica principal: F1 de la clase peligrosa** (`scoring="f1"`, igual que en el notebook de selección de rasgos de clase). El accuracy se reporta, pero nunca decide nada: un modelo que diga siempre "sin peligro" saca cerca del 93%.
- **Partición** única 80/20 estratificada, con `random_state=42`. El 20% de prueba (~34 turnos peligrosos) se usa **una sola vez**, en el notebook 04.
- **Validación cruzada:** `StratifiedKFold` de **10 particiones**, como en clase.
- **Cuatro notebooks**, cada uno con una sola responsabilidad, más `app.py`:
  - 01: carga, EDA, limpieza, codificación y partición.
  - 02: escalado, PCA y selección de rasgos.
  - 03: balanceo y comparación de los 12 modelos.
  - 04: ajuste, evaluación final y exportación.
- **Manda el estilo de la profesora.** Todo el código vive en los notebooks y sigue el orden de clase. Lo que tenga una única responsabilidad se vuelve función con docstring. No hay carpeta `src/`.
- **Publicación:** app Gradio con un formulario para ingresar los datos de un turno.
- **Entregables:** notebooks, app, informe y presentación.
- **Repositorio en GitHub** con `README.md`, `AGENTS.md`, `CLAUDE.md` y `ESTADO.md`, para que todo el equipo (Claude Code, Codex y Antigravity) trabaje con las mismas reglas y sepa en qué va el proyecto.

## 2. Estilo, buenas prácticas y comentarios

Prima el estilo de la profesora. Las buenas prácticas se aplican sin salirse de él. Estas mismas reglas están en `AGENTS.md`, que es la fuente que leen los agentes.

**Orden de cada notebook:** título → configuración → importaciones → carga → análisis → proceso → resultados → conclusiones. Se reutilizan los nombres y patrones de clase: `X_train`, `y_train`, `X_test`, `y_test`, `rows` para acumular resultados, `pd.DataFrame(rows)` para la tabla, `Pipeline`, `StratifiedKFold`, `GridSearchCV`, `ConfusionMatrixDisplay`, `classification_report` y `joblib`.

**Una responsabilidad por función:**

- Si un bloque hace una sola cosa y se usa más de una vez, se vuelve función: cargar el ARFF, codificar las categóricas, crear un escalador, crear un pipeline, evaluar con validación cruzada, graficar.
- Cada función recibe por parámetros lo que necesita y devuelve un resultado. No imprime y calcula a la vez, ni modifica variables globales.
- El catálogo de modelos es una función `crear_modelos()` que devuelve una lista. Agregar un modelo es agregar una línea, sin tocar el ciclo que evalúa.
- La función de evaluación recibe el pipeline y no le importa qué modelo trae adentro.
- Las funciones que se repiten entre notebooks (`evaluar_cv`, `crear_escalador`, `crear_pipeline`, `crear_modelos`, `codificar_categoricas`) se copian **idénticas**. Si una cambia, se cambia en todos lados y se anota en `ESTADO.md`.

**Lo que no se usa (nivel del curso):** list y dict comprehensions, generadores, `lambda`, clases propias, decoradores, `*args` y `**kwargs` en los notebooks. Se usan ciclos `for` simples y nombres claros en español. La única excepción es `*valores` en `app.py`, porque Gradio entrega así los campos del formulario, y queda comentada.

**Otras reglas:**

- `random_state=42` en todo.
- Las rutas se definen solo en la celda de configuración.
- Nada de números mágicos sin explicar.
- Cada gráfica se guarda en `figuras/` con un nombre que diga qué muestra, por ejemplo `03_heatmap_f1.png`.

**Comentarios y markdown.** La meta es que quien clone el repositorio entienda cada parte sin preguntarle a nadie.

- Cada función lleva un docstring en español: qué hace, qué recibe, qué devuelve y, si no es obvio, por qué existe.
- Los comentarios explican el porqué, no repiten el código. Mal: `# recorremos las columnas`. Bien: `# usamos el mismo diccionario en los notebooks y en la app, así una 'b' siempre vale 1`.
- El tono es humanizado y en voz grupal ("aquí quitamos…", "lo dejamos así porque…", "notamos que…"), como lo escribirían estudiantes.
- Antes de cada bloque de código va una celda markdown corta que dice qué vamos a hacer y para qué. Después de cada resultado importante va otra con lo que notamos, usando los números que **sí** salieron.
- Nada de tablas en las celdas markdown: los resultados se cuentan en prosa. Las tablas de resultados van como DataFrame en el código.

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

## 3. Estructura del repositorio y entorno

Se conserva la base que ya teníamos (uv, Colab, GitHub, Claude Code); cambian el dataset, los nombres y la carpeta `resultados/`. Nombre propuesto para el repositorio: `prediccion-sismica-minas`. Si ya lo crearon con el nombre de WiFi, se puede renombrar en GitHub en **Settings → General**. En Drive la carpeta se llama `Proyecto_Sismos`.

```text
prediccion-sismica-minas/
├── data/
│   ├── seismic-bumps.arff                 # original de UCI, no se modifica (el que usan los notebooks)
│   ├── seismic-bumps.csv                  # copia de datahub (2.578 filas), solo de referencia; no se usa
│   ├── train.csv                          # notebook 01 (ya codificado)
│   ├── test.csv                           # notebook 01 (ya codificado)
│   └── test_original.csv                  # notebook 01 (con las letras originales, para la app)
├── models/
│   ├── mapeos_categoricas.joblib          # notebook 01
│   ├── decisiones_preprocesamiento.joblib # notebook 02 (escalador y rasgos)
│   ├── modelo_final.joblib                # notebook 04
│   └── metadata_modelo.json               # notebook 04 (modelo, umbral, métricas, fecha)
├── resultados/
│   ├── comparacion_rasgos.csv             # notebook 02
│   ├── comparacion_modelos.csv            # notebook 03
│   └── ajuste_hiperparametros.csv         # notebook 04
├── figuras/                               # todas las gráficas
├── referencia/                            # los 6 notebooks de clase (no se modifican)
├── 01_EDA_Limpieza.ipynb
├── 02_Escalado_PCA_Seleccion.ipynb
├── 03_Balanceo_Comparacion_Modelos.ipynb
├── 04_Ajuste_Evaluacion_Exportacion.ipynb
├── app.py
├── AGENTS.md          # reglas para personas y agentes (fuente única)
├── CLAUDE.md          # importa AGENTS.md para Claude Code
├── ESTADO.md          # avance, decisiones y bitácora
├── PLAN.md            # este plan
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
└── .gitignore
```

El dataset pesa unos 140 KB y los modelos son pequeños, así que **`data/`, `models/`, `resultados/` y `figuras/` sí se suben a Git**. Así, quien clone el repositorio puede abrir la app sin reentrenar nada. En el `.gitignore` solo quedan `.venv/`, los checkpoints de Jupyter y archivos del sistema.

### Celda de configuración (primera celda de cada notebook)

Es la única parte que cambia entre Colab y local:

```python
# Si corremos en Google Colab ponemos True; en el computador (VS Code), False
EN_COLAB = False

if EN_COLAB:
    # En Colab los archivos viven en Drive, así que primero lo montamos
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/Proyecto_Sismos/'
else:
    RUTA_BASE = './'

# Todas las rutas del notebook salen de aquí, para no tener rutas regadas por el código
RUTA_DATOS = RUTA_BASE + 'data/'
RUTA_MODELOS = RUTA_BASE + 'models/'
RUTA_RESULTADOS = RUTA_BASE + 'resultados/'
RUTA_FIGURAS = RUTA_BASE + 'figuras/'
```

### Librerías

Se agregan con uv y quedan en `pyproject.toml` y `uv.lock`:

- pandas, numpy, scipy (lee el ARFF);
- matplotlib, seaborn, phik;
- scikit-learn, **imbalanced-learn** (SMOTE, submuestreo y el `Pipeline` que los soporta);
- shap, joblib, gradio;
- ipykernel y nbconvert, para que los agentes puedan ejecutar los notebooks desde la terminal.

En Colab hay que instalar al inicio: `!pip install phik shap gradio imbalanced-learn -q`.

### Creación del proyecto (solo una persona, una vez)

```powershell
mkdir prediccion-sismica-minas
cd prediccion-sismica-minas

uv init --bare --python 3.11
uv python pin 3.11
uv add pandas numpy scipy matplotlib seaborn phik scikit-learn imbalanced-learn shap joblib gradio ipykernel nbconvert

mkdir data, models, resultados, figuras, referencia
New-Item models\.gitkeep, resultados\.gitkeep, figuras\.gitkeep -ItemType File
```

Después:

1. Descargar el `.zip` oficial de UCI (botón **Download**) y copiar `seismic-bumps.arff` en `data/`.
2. Copiar los 6 notebooks de clase en `referencia/`.
3. Poner en la raíz `AGENTS.md`, `CLAUDE.md`, `ESTADO.md`, `README.md`, `.gitignore` y este `PLAN.md`.
4. Hacer el primer push.

### Instalación para el resto del grupo

1. Instalar uv con `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`, cerrar y abrir la terminal, y verificar con `uv --version`.
2. Clonar el repositorio, entrar a la carpeta y correr `uv sync`.
3. En VS Code, elegir el kernel `.venv\Scripts\python.exe`.
4. Dejar `EN_COLAB = False`.

Para agregar una librería se usa `uv add nombre-del-paquete`, nunca `pip install` en local, y se hace commit de `pyproject.toml` y `uv.lock`.

## 4. Cronograma de 12 días

El modelo final debe quedar exportado el día 8, para tener cuatro días para la app, el informe y la presentación. Mientras una persona hace el 03, otra puede ir armando la app con un modelo de prueba.

| Día | Qué se hace | Terminado cuando |
| --- | --- | --- |
| 1 | Repositorio, entorno con uv, ARFF oficial en `data/`, notebooks de clase en `referencia/`, `AGENTS.md`, `CLAUDE.md`, `ESTADO.md`. | `uv sync` corre y el ARFF carga en local y en Colab. |
| 2 | Notebook 01: carga, revisión de variables, distribución de la clase, categóricas, numéricas y correlación phik. | Las conclusiones del EDA están escritas. |
| 3 | Notebook 01: duplicados, columnas constantes, partición, codificación y guardado. | Existen `train.csv`, `test.csv`, `test_original.csv` y `mapeos_categoricas.joblib`. |
| 4 | Notebook 02: comparación de escaladores y PCA. | El escalador está elegido. |
| 5 | Notebook 02: los seis métodos de selección de rasgos, tabla comparativa y decisión. | Existe `decisiones_preprocesamiento.joblib`. |
| 6 | Notebook 03: catálogo de modelos, pipelines y las 41 combinaciones. | Existe `comparacion_modelos.csv`. |
| 7 | Notebook 03: heatmaps, box plots y conclusiones. Notebook 04: GridSearch de los 3 mejores. | Los 3 mejores están ajustados. |
| 8 | Notebook 04: umbral, evaluación única en test, exportación y verificación. | `modelo_final.joblib` carga y predice igual que antes de guardarlo. |
| 9 | App Gradio en local y en Colab. | La app pasa las pruebas de la sección 9. |
| 10 | Informe con todas las figuras. | Todas las secciones tienen contenido. |
| 11 | Presentación y revisión cruzada: cada integrante corre los cuatro notebooks de cero. | Todo corre sin errores en un entorno limpio. |
| 12 | Margen, README con resultados y entrega. | Entregado. |

## 5. Notebook 01 — Carga, EDA, limpieza y partición

`01_EDA_Limpieza.ipynb` entrega los datos listos para modelar y el diccionario de codificación. Se basa en `IA_ML_Example_Dif_Classifiers_CV.ipynb` (análisis de datos, codificación, `train_test_split`) y en `ML_Normalización.ipynb` (`describe` y box plots).

Importaciones extra: `from scipy.io import arff`, `from sklearn.model_selection import train_test_split`, `import phik` y `from phik.report import plot_correlation_matrix`.

### 5.1 Encabezado

Celda markdown con el título, el problema en dos o tres líneas, la fuente (UCI, id 266) y la cita de Sikora y Wróbel (2010).

### 5.2 Cargar el ARFF

El ARFF trae las columnas categóricas como bytes (`b'a'`), así que una función de una sola responsabilidad lo lee y lo deja como texto normal:

```python
def cargar_arff(ruta):
    """
    Lee el archivo .arff oficial de UCI y lo devuelve como DataFrame.
    scipy entrega las columnas de texto como bytes (b'a'), así que aquí las pasamos a texto normal ('a')
    para poder trabajarlas igual que si vinieran de un CSV.
    """
    datos_arff, meta = arff.loadarff(ruta)
    df = pd.DataFrame(datos_arff)
    for columna in df.columns:
        if df[columna].dtype == object:
            df[columna] = df[columna].str.decode('utf-8')
    return df

datos = cargar_arff(RUTA_DATOS + 'seismic-bumps.arff')
datos['class'] = datos['class'].astype(int)   # en el ARFF la clase viene como texto '0' / '1'
```

Verificar con `shape`, `head()` e `info()`. Se esperan 2.584 filas y 19 columnas. Si sale otra cantidad, **parar y avisar al grupo**.

### 5.3 Revisión de variables

- Una función `resumen_columnas(df)` arma una tabla con tipo, valores distintos y nulos de cada columna (con un ciclo `for` que llena una lista de diccionarios).
- Después, `datos.describe()` para las numéricas.
- Markdown con lo que notamos: cuáles son categóricas, cuáles constantes y si hay nulos.

### 5.4 Distribución de la clase

- `value_counts()` y gráfica de torta con porcentajes, como en clase.
- Calcular el accuracy de un modelo que siempre dice "sin peligro" (proporción de clase 0) para mostrar la trampa del accuracy.
- Markdown explicando por qué la métrica principal es F1.

### 5.5 Variables categóricas

Una función `graficar_categorica(df, columna, nombre_archivo)` hace dos gráficas lado a lado:

- el conteo de cada categoría;
- el porcentaje de turnos peligrosos en cada categoría (`pd.crosstab(df[columna], df['class'], normalize='index')`).

Se llama para `seismic`, `seismoacoustic`, `shift` y `ghazard`. Markdown: ¿alguna categoría tiene más riesgo?

### 5.6 Variables numéricas

- Histogramas de todas las numéricas y box plots de cada una separados por clase (`sns.boxplot(x='class', y=columna, data=datos)`).
- Markdown sobre la asimetría de las energías y los valores extremos.
- Decisión explicada: **los extremos no se eliminan**, porque son eventos reales y justo lo que queremos detectar.

### 5.7 Limpieza

- **Duplicados:** contar con `datos.duplicated().sum()`, revisar de qué clase son y quitarlos con `drop_duplicates()`. Se esperan alrededor de 6; si salen más o menos, se reporta.
- **Columnas constantes:** una función `columnas_constantes(df)` devuelve la lista de columnas con un solo valor (`nunique() == 1`). Se esperan `nbumps6`, `nbumps7` y `nbumps89`. Se eliminan y se explica por qué no aportan nada.

### 5.8 Correlación con phik

Como en clase, `datos.phik_matrix(interval_cols=columnas_numericas)` y `plot_correlation_matrix`. phik funciona con categóricas y numéricas a la vez. Markdown: qué variables se relacionan más con `class` y cuáles son redundantes entre sí (por ejemplo `energy` y `maxenergy`, o `nbumps` con sus rangos).

### 5.9 Partición 80/20

Se hace **antes** de codificar, para poder guardar una copia de test con las letras originales para la app:

```python
X = datos.drop(columns=['class'])
y = datos['class']

# stratify=y para que train y test queden con el mismo ~6,6% de turnos peligrosos
X_train_original, X_test_original, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

Verificar que la proporción de clase 1 sea casi igual en train y en test. Se esperan unos 34 turnos peligrosos en test.

### 5.10 Codificación ordinal

```python
# Las evaluaciones de peligro van de menos a más (a = sin peligro, b = bajo, c = alto, d = estado de peligro),
# así que números ordenados conservan ese orden. Dejamos la 'd' aunque no aparezca en los datos,
# por si la app recibe un turno con esa evaluación.
mapeos = {
    'seismic': {'a': 0, 'b': 1, 'c': 2, 'd': 3},
    'seismoacoustic': {'a': 0, 'b': 1, 'c': 2, 'd': 3},
    'ghazard': {'a': 0, 'b': 1, 'c': 2, 'd': 3},
    'shift': {'N': 0, 'W': 1},   # N = turno de preparación, W = turno de extracción de carbón
}
```

- Se aplica `codificar_categoricas(df, mapeos)` (la función de la sección 2) a `X_train_original` y `X_test_original`.
- Se verifica que no queden nulos: si aparece uno, hay una letra que no está en `mapeos`.
- Markdown: por qué ordinal y no one-hot (las letras tienen orden), y por qué no `LabelEncoder` (en clase se usó para la etiqueta, y aquí la etiqueta ya es 0/1).

### 5.11 Guardar

- `data/train.csv` y `data/test.csv`: rasgos codificados más la columna `class`.
- `data/test_original.csv`: test con las letras originales más `class`. Lo usa la app.
- `models/mapeos_categoricas.joblib`.

### 5.12 Conclusiones

Markdown en voz grupal con los números reales: filas, duplicados quitados, columnas eliminadas, desbalance, variables más relacionadas con la clase y tamaño de train y test.

## 6. Notebook 02 — Escalado, PCA y selección de rasgos

`02_Escalado_PCA_Seleccion.ipynb` decide **qué escalador y qué rasgos** usan los modelos. Solo lee `train.csv`: **test no se toca**. Se basa en `ML_Normalización.ipynb`, `PCA_Wine.ipynb`, `ML_PCA.ipynb` y `ML_FeatureSelection_comparacion_metodos_SHAP.ipynb`.

### 6.1 Cargar y armar X, y

`X_train` con todas las columnas menos `class`, y `y_train = train['class']`.

### 6.2 Validación cruzada y función de evaluación

Esta función se copia **idéntica** en los notebooks 03 y 04:

```python
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Con tan pocos turnos peligrosos, a veces un modelo no predice ninguno en una partición y la precisión
# queda indefinida (scikit-learn lanza una advertencia). Con zero_division=0 la contamos como 0 y seguimos.
metricas = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
}

def evaluar_cv(pipeline, X, y, nombre):
    """
    Evalúa un pipeline con validación cruzada estratificada de 10 particiones.
    Devuelve un diccionario con el promedio de cada métrica, la desviación del F1, el F1 de cada
    partición (para los box plots) y el tiempo que tardó. Se agrega a la lista rows para armar la tabla.
    """
    inicio = time.time()
    resultados = cross_validate(pipeline, X, y, cv=cv, scoring=metricas, n_jobs=-1)
    return {
        'Nombre': nombre,
        'Accuracy': resultados['test_accuracy'].mean(),
        'Precision': resultados['test_precision'].mean(),
        'Recall': resultados['test_recall'].mean(),
        'F1': resultados['test_f1'].mean(),
        'F1_std': resultados['test_f1'].std(),
        'ROC_AUC': resultados['test_roc_auc'].mean(),
        'F1_por_particion': resultados['test_f1'],
        'Tiempo_s': time.time() - inicio,
    }
```

**Modelo de referencia del notebook:** `LogisticRegression(max_iter=2000, class_weight='balanced')`, igual que en el notebook de selección de rasgos de clase. Sin balanceo casi no predice turnos peligrosos y todas las comparaciones darían F1 cercano a 0. La comparación de balanceos de verdad se hace en el notebook 03.

### 6.3 Escalado

- Una función `graficar_boxplots(df, titulo, nombre_archivo)` muestra los box plots de las numéricas: sin escalar, con MinMax y con Standard.
- Se comparan con `evaluar_cv` tres pipelines (`'passthrough'`, `MinMaxScaler`, `StandardScaler`) con la regresión logística de referencia y con `SVC(class_weight='balanced')`. Son 6 filas en `rows`.
- Markdown: con energías tan extremas, MinMax aplasta casi todos los valores cerca de 0. Se elige el escalador con mejor F1 y se explica.

### 6.4 PCA

1. Con el escalador elegido, `PCA()` completo y la gráfica de varianza acumulada con líneas en 0,90 y 0,95, como en `PCA_Wine`.
2. Reportar cuántas componentes llegan al 90% y al 95%.
3. Gráfica 2D con las dos primeras componentes, coloreada por clase. Se espera que las clases se mezclen mucho, y eso explica por qué el problema es difícil.
4. Comparar con `evaluar_cv`: todos los rasgos contra PCA al 90% y al 95% (escalador → PCA → regresión logística de referencia).
5. Markdown con la decisión. Lo más probable es no usar PCA: son pocos rasgos y se pierde la interpretación. Pero se decide con los números.

### 6.5 Selección de rasgos

Los mismos métodos del notebook de clase, todos con el escalador elegido y la regresión logística de referencia:

- **a) VarianceThreshold:** sobre los datos escalados a [0, 1], con `threshold=0.01`. Detecta columnas que casi nunca cambian; se espera que `nbumps5` caiga.
- **b) SelectKBest:** con `f_classif` y con `mutual_info_classif`, en un ciclo `for k in range(1, n_rasgos + 1)` como en clase. Gráfica de F1 contra k para los dos.
- **c) RFECV:** con la regresión logística balanceada, `step=1`, `StratifiedKFold(5)` y `scoring='f1'`, igual que en clase.
- **d) SequentialFeatureSelector:** hacia adelante, en un ciclo sobre k como en clase. Tarda unos minutos.
- **e) SelectFromModel con L1:** `LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced')`, como en clase.
- **f) SHAP:** `GradientBoostingClassifier` sobre una muestra de train y `shap.TreeExplainer`. Al ser binario, los valores salen de forma `(filas, rasgos)`, así que la importancia es `np.abs(valores).mean(axis=0)`. Gráficas beeswarm y de barras, y selección de los rasgos que acumulan el 90% de la importancia con un ciclo que va sumando.

### 6.6 Tabla comparativa y decisión

- Una función `evaluar_conjunto(rasgos, nombre)` evalúa un conjunto de rasgos con el mismo pipeline y devuelve la fila.
- Se arma la tabla con: todos los rasgos (línea base) y cada método, con número de rasgos, F1, desviación, recall y ROC AUC. Se guarda en `resultados/comparacion_rasgos.csv`.
- **Regla de decisión:** el de mayor F1. Si la diferencia con un conjunto más pequeño es menor que la desviación del F1, gana el más pequeño. Se explica en markdown.

### 6.7 Guardar

```python
decisiones = {'escalador': escalador_elegido,   # 'StandardScaler', 'MinMaxScaler' o 'Sin escalar'
              'rasgos': rasgos_elegidos}         # lista con los nombres de las columnas
joblib.dump(decisiones, RUTA_MODELOS + 'decisiones_preprocesamiento.joblib')
```

Conclusiones en voz grupal: escalador, componentes de PCA, método ganador y rasgos que quedaron.

## 7. Notebook 03 — Balanceo y comparación de los 12 modelos

`03_Balanceo_Comparacion_Modelos.ipynb` evalúa todas las combinaciones de modelo y estrategia de balanceo con validación cruzada, y deja la tabla con el ranking. Se basa en `IA_ML_Example_Dif_Classifiers_CV.ipynb`. **Test no se toca.**

Importaciones extra:

```python
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
```

Además, los 12 modelos de `sklearn`.

### 7.1 Cargar

Cargar `train.csv` y `decisiones_preprocesamiento.joblib`, y armar `X_train = train[decisiones['rasgos']]` e `y_train`. Se copian idénticos `cv`, `metricas` y `evaluar_cv` del notebook 02.

### 7.2 Catálogo de modelos

```python
def crear_modelos():
    """
    Devuelve los 12 modelos que comparamos, como parejas (nombre, modelo sin entrenar).
    Son los mismos que vimos en clase. Si queremos probar otro, basta con agregar una línea aquí;
    el resto del notebook no cambia.
    """
    modelos = [
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('Naive Bayes', GaussianNB()),
        ('Árbol de decisión', DecisionTreeClassifier(random_state=42)),
        ('Regresión logística', LogisticRegression(max_iter=2000, random_state=42)),
        ('SVM', SVC(random_state=42)),
        ('SVM lineal', LinearSVC(max_iter=10000, random_state=42)),
        ('Red neuronal (MLP)', MLPClassifier(max_iter=1000, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('Bagging', BaggingClassifier(random_state=42)),
        ('AdaBoost', AdaBoostClassifier(random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
        ('Stacking', StackingClassifier(
            estimators=[('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                        ('svm', SVC(random_state=42)),
                        ('knn', KNeighborsClassifier(n_neighbors=5))],
            final_estimator=LogisticRegression(max_iter=2000),
            cv=5)),
    ]
    return modelos
```

Los modelos no llevan `n_jobs=-1`, porque la validación cruzada ya corre en paralelo y se estorbarían.

### 7.3 Escalador, estrategias y pipeline

```python
ESTRATEGIAS = ['Sin balanceo', 'class_weight', 'SMOTE', 'Submuestreo']

def crear_escalador(nombre):
    """Devuelve un escalador nuevo según el nombre que guardamos en el notebook 02."""
    if nombre == 'MinMaxScaler':
        return MinMaxScaler()
    elif nombre == 'StandardScaler':
        return StandardScaler()
    return 'passthrough'   # 'Sin escalar': el pipeline deja pasar los datos tal cual

def admite_class_weight(modelo):
    """
    Dice si el modelo tiene el parámetro class_weight. KNN, Naive Bayes, MLP, Bagging, AdaBoost,
    Gradient Boosting y Stacking no lo tienen, así que con ellos esa estrategia no aplica.
    """
    return 'class_weight' in modelo.get_params()

def crear_pipeline(modelo, estrategia, nombre_escalador):
    """
    Arma el pipeline de una combinación modelo + estrategia: escalador -> (SMOTE o submuestreo) -> modelo.
    Usamos el Pipeline de imblearn porque así el balanceo se hace solo con los datos de entrenamiento
    de cada partición y nunca con los de validación; si no, los resultados saldrían inflados.
    """
    modelo = clone(modelo)   # copia limpia, para que una combinación no afecte a la siguiente
    pasos = [('escalador', crear_escalador(nombre_escalador))]
    if estrategia == 'SMOTE':
        pasos.append(('balanceo', SMOTE(random_state=42)))
    elif estrategia == 'Submuestreo':
        pasos.append(('balanceo', RandomUnderSampler(random_state=42)))
    elif estrategia == 'class_weight':
        modelo.set_params(class_weight='balanced')
    pasos.append(('clf', modelo))
    return ImbPipeline(pasos)
```

### 7.4 Comparación de las 41 combinaciones

```python
rows = []
for nombre_modelo, modelo in crear_modelos():
    for estrategia in ESTRATEGIAS:
        if estrategia == 'class_weight' and not admite_class_weight(modelo):
            continue   # esta combinación no existe; lo explicamos en el markdown
        pipeline = crear_pipeline(modelo, estrategia, decisiones['escalador'])
        fila = evaluar_cv(pipeline, X_train, y_train, nombre_modelo + ' | ' + estrategia)
        fila['Modelo'] = nombre_modelo
        fila['Estrategia'] = estrategia
        rows.append(fila)
        print(nombre_modelo, '|', estrategia, '-> F1 =', round(fila['F1'], 3))

tabla_modelos = pd.DataFrame(rows).sort_values('F1', ascending=False)
```

- Son 12 × 3 + 5 = **41 combinaciones**. Stacking es el más lento: puede tardar varios minutos.
- Se guarda `resultados/comparacion_modelos.csv` sin la columna `F1_por_particion`. Esa columna se usa aparte para los box plots.

### 7.5 Análisis

- **Heatmap de F1** con modelos en filas y estrategias en columnas (`tabla_modelos.pivot(index='Modelo', columns='Estrategia', values='F1')` y `sns.heatmap(annot=True, fmt='.2f')`). Las celdas que no aplican quedan vacías. Otro heatmap igual para recall.
- **Box plot del F1 por partición**, como en clase, con la mejor estrategia de cada modelo.
- **Gráfica de accuracy contra F1** para mostrar la trampa: sin balanceo el accuracy es alto y el F1 casi 0.
- Markdown:
  - qué estrategia ayuda más en general;
  - qué modelos aprovechan mejor el balanceo;
  - qué tan grande es la desviación entre particiones;
  - cuáles son las **3 mejores combinaciones** (de modelos distintos), que pasan al notebook 04.

## 8. Notebook 04 — Ajuste, evaluación final y exportación

`04_Ajuste_Evaluacion_Exportacion.ipynb` ajusta las 3 mejores combinaciones, elige el modelo final y el umbral, lo evalúa **una sola vez** en test y lo exporta. Se basa en las secciones de GridSearch y de guardar y cargar modelos de `IA_ML_Example_Dif_Classifiers_CV.ipynb`.

### 8.1 Cargar

- Cargar `train.csv`, `test.csv`, las decisiones del 02 y `comparacion_modelos.csv`.
- Copiar idénticos `cv`, `metricas`, `evaluar_cv`, `crear_modelos`, `crear_escalador`, `admite_class_weight` y `crear_pipeline`.
- Una función `buscar_modelo(nombre)` recorre `crear_modelos()` y devuelve el modelo con ese nombre.

### 8.2 Grillas de hiperparámetros

Una función `obtener_grilla(nombre_modelo, estrategia)` guarda las grillas de los 12 modelos en un diccionario, así funciona sin importar cuáles queden en el top 3. Si la estrategia es SMOTE, agrega `'balanceo__k_neighbors': [3, 5, 7]`. Las grillas son pequeñas a propósito. Todas llevan el prefijo `clf__`.

| Modelo | Parámetros a probar |
| --- | --- |
| KNN | `n_neighbors`: 3, 5, 7, 11, 15 · `weights`: uniform, distance |
| Naive Bayes | `var_smoothing`: 1e-9, 1e-8, 1e-7, 1e-6 |
| Árbol de decisión | `max_depth`: 3, 5, 8, None · `min_samples_leaf`: 1, 5, 10 |
| Regresión logística | `C`: 0.01, 0.1, 1, 10 |
| SVM | `C`: 0.1, 1, 10 · `gamma`: scale, 0.1, 0.01 |
| SVM lineal | `C`: 0.01, 0.1, 1, 10 |
| Red neuronal (MLP) | `hidden_layer_sizes`: (50,), (100,), (100, 50) · `alpha`: 0.0001, 0.001, 0.01 |
| Random Forest | `n_estimators`: 200, 400 · `max_depth`: None, 8, 15 · `min_samples_leaf`: 1, 5 |
| Bagging | `n_estimators`: 10, 50, 100 · `max_samples`: 0.5, 1.0 |
| AdaBoost | `n_estimators`: 50, 100, 200 · `learning_rate`: 0.1, 0.5, 1.0 |
| Gradient Boosting | `n_estimators`: 100, 200 · `learning_rate`: 0.05, 0.1 · `max_depth`: 2, 3 |
| Stacking | `final_estimator__C`: 0.1, 1, 10 (queda `clf__final_estimator__C`) |

### 8.3 GridSearchCV de las 3 mejores combinaciones

- Tomar las 3 primeras filas de `comparacion_modelos.csv` con modelos distintos (`drop_duplicates(subset='Modelo').head(3)`).
- Para cada una: `GridSearchCV(crear_pipeline(...), obtener_grilla(...), cv=cv, scoring='f1', n_jobs=-1, verbose=1)`.
- Después, `evaluar_cv` sobre su `best_estimator_`, para tener todas las métricas.
- Tabla con F1 antes y después del ajuste y los mejores parámetros. Se guarda en `resultados/ajuste_hiperparametros.csv`.

### 8.4 Elegir el modelo final

- Gana el de mayor F1 en validación cruzada, **nunca** el de mejor resultado en test.
- Si gana `SVC`, se reentrena con `clf__probability=True`.
- Si gana `LinearSVC`, que no tiene `predict_proba`, **se pregunta al grupo** si se toma el siguiente o se calibra. La app y el umbral necesitan probabilidades.

### 8.5 Umbral de decisión

Con pocos positivos, el umbral de 0,5 casi nunca es el mejor. Se elige **solo con train**:

```python
probabilidades_cv = cross_val_predict(modelo_final, X_train, y_train, cv=cv, method='predict_proba')[:, 1]

def elegir_umbral(probabilidades, y_real):
    """
    Prueba umbrales de 0,05 a 0,95 y devuelve el que da el mejor F1 para la clase peligrosa,
    junto con la tabla de todos los umbrales para graficarla.
    """
    filas = []
    for umbral in np.arange(0.05, 0.96, 0.05):
        predicciones = (probabilidades >= umbral).astype(int)
        filas.append({'Umbral': round(umbral, 2),
                      'F1': f1_score(y_real, predicciones),
                      'Recall': recall_score(y_real, predicciones),
                      'Precision': precision_score(y_real, predicciones, zero_division=0)})
    tabla_umbrales = pd.DataFrame(filas)
    mejor_fila = tabla_umbrales.loc[tabla_umbrales['F1'].idxmax()]
    return mejor_fila['Umbral'], tabla_umbrales
```

Gráfica de F1, recall y precisión contra el umbral.

### 8.6 Evaluación única en test

1. Entrenar el modelo final con todo train.
2. Calcular `probabilidades_test = modelo_final.predict_proba(X_test)[:, 1]` y `y_pred = (probabilidades_test >= umbral).astype(int)`.
3. Reportar:
   - `classification_report` con `target_names=['Sin peligro', 'Peligroso']`;
   - `ConfusionMatrixDisplay.from_predictions`;
   - `RocCurveDisplay.from_predictions`;
   - `PrecisionRecallDisplay.from_predictions`.
4. Comparar con `DummyClassifier(strategy='most_frequent')` para mostrar que el accuracy solo no dice nada.
5. Markdown:
   - qué tan cerca quedó test de la validación cruzada;
   - cuántos de los ~34 turnos peligrosos detectó y cuántas falsas alarmas dio;
   - que con tan pocos positivos cada acierto mueve mucho las métricas.

### 8.7 Exportación

- `joblib.dump(modelo_final, RUTA_MODELOS + 'modelo_final.joblib')`. El pipeline guardado ya incluye el escalador, y SMOTE y el submuestreo solo actúan al entrenar, no al predecir.
- `models/metadata_modelo.json` con `json.dump(..., indent=2, ensure_ascii=False)`. Lleva:
  - `modelo`, `estrategia_balanceo`, `escalador`, `rasgos`, `umbral`;
  - `hiperparametros` (pasados a texto);
  - `metricas_cv` y `metricas_test` (F1, Recall, Precision, ROC_AUC, Accuracy);
  - `fecha_entrenamiento` y `version_sklearn`.

### 8.8 Cargar y verificar

Como en la sección "Cargar el modelo entrenado" de clase: cargar el `.joblib`, predecir sobre test y comprobar que da exactamente lo mismo que antes de guardarlo.

### 8.9 Conclusiones

Markdown en voz grupal: modelo y estrategia ganadores, umbral, métricas en CV y en test, qué aprendimos del desbalance y qué haríamos con más tiempo. Con estos números se actualizan el README y `ESTADO.md`.

## 9. App Gradio (publicación del modelo)

`app.py` carga lo que exportó el notebook 04 y deja evaluar un turno de tres formas. Como son pocas variables, aquí sí se puede llenar un formulario a mano.

### 9.1 Pestañas

- **Evaluar un turno.** El formulario se arma solo a partir de `rasgos` en la metadata:
  - un menú desplegable para cada categórica, con las opciones de `mapeos`;
  - un campo numérico para cada numérica;
  - cada campo con su descripción en español.

  Debajo hay dos botones. **Evaluar turno** muestra el veredicto ("⚠️ Turno peligroso" o "✅ Sin peligro previsto"), la probabilidad y el umbral usado. **Cargar un turno real al azar** llena el formulario con un turno de `test_original.csv`, lo evalúa (con `.then(...)`) y muestra el valor real.
- **Subir un CSV.** Recibe un archivo con las columnas originales del dataset (con letras). Devuelve una tabla con el número de turno, la probabilidad de peligro y el veredicto.
- **Sobre el modelo.** Muestra el modelo, la estrategia de balanceo, el escalador, los rasgos, el umbral, las métricas de CV y de test y la fecha de entrenamiento, todo sacado de `metadata_modelo.json`.

En el encabezado va un aviso: es un proyecto académico y no reemplaza los sistemas de monitoreo reales de una mina.

### 9.2 Funciones (una responsabilidad cada una, con docstring)

| Función | Responsabilidad |
| --- | --- |
| `cargar_artefactos()` | Leer el modelo, los mapeos, la metadata y `test_original.csv`. |
| `codificar_categoricas(df, mapeos)` | La misma del notebook 01, copiada idéntica. |
| `validar_entrada(df_codificado)` | Si quedó algún vacío (letra desconocida o campo sin llenar), lanzar `gr.Error` con un mensaje claro. |
| `predecir_probabilidades(df_original)` | Codificar, validar, tomar las columnas de `rasgos` en orden y devolver `predict_proba(...)[:, 1]`. |
| `texto_veredicto(probabilidad)` | Armar el texto en Markdown comparando con el umbral. |
| `etiquetas_probabilidad(probabilidad)` | Devolver `{'Peligroso': p, 'Sin peligro': 1 - p}` para `gr.Label`. |
| `evaluar_turno(*valores)` | Armar una fila con los valores del formulario (Gradio los entrega en el orden de `rasgos`) y devolver el veredicto y las probabilidades. |
| `cargar_turno_al_azar()` | Elegir una fila de `test_original.csv` y devolver sus valores y el valor real. |
| `predecir_csv(ruta)` | Revisar que estén las columnas, predecir y devolver la tabla. |
| `resumen_modelo()` | Armar el Markdown de la pestaña "Sobre el modelo". |
| `crear_campo(columna)` | Devolver un `gr.Dropdown` o un `gr.Number` según la columna. |
| `construir_interfaz()` | Armar `gr.Blocks` con las tres pestañas y conectar los botones. |

Al final del archivo va `if __name__ == '__main__': construir_interfaz().launch(share=EN_COLAB)`. La parte de arriba lleva la misma configuración `EN_COLAB` / `RUTA_BASE` de los notebooks y un docstring que explica qué es la app y cómo correrla.

### 9.3 Cómo correrla

- **Local:** `uv run python app.py`. Se abre en `http://127.0.0.1:7860`.
- **Colab:** poner `EN_COLAB = True` en `app.py` y, desde la carpeta del proyecto en Drive, ejecutar `%run app.py`. Gradio crea un enlace público temporal (`*.gradio.live`) que sirve para la presentación.

### 9.4 Pruebas

- [ ] Un turno cargado al azar da la misma probabilidad en la app y en el notebook 04.
- [ ] Un turno con un campo vacío muestra un error claro en vez de romperse.
- [ ] Un CSV con 5 filas de `test_original.csv` (sin `class`) devuelve 5 predicciones.
- [ ] Un CSV al que le falta una columna muestra qué columnas faltan.
- [ ] La pestaña "Sobre el modelo" muestra las mismas métricas del notebook 04.
- [ ] Funciona en local y en Colab.

## 10. Trabajo en equipo con agentes de IA

El equipo usa Claude Code, Codex y Antigravity. Para que todos trabajen igual, las reglas viven en un solo lugar.

- **`AGENTS.md`** es la fuente única de reglas: resumen del proyecto, archivos clave, flujo de notebooks, decisiones tomadas, reglas de datos, estilo de código, comentarios, comandos, forma de trabajar y definición de terminado. Codex lo lee automáticamente. Si Antigravity no lo toma solo, se agrega como regla del workspace o se le pide al iniciar: "lee AGENTS.md y ESTADO.md antes de empezar".
- **`CLAUDE.md`** importa `AGENTS.md` con `@AGENTS.md` y solo agrega tres notas propias de Claude Code. Así nada se duplica: si una regla cambia, se cambia en `AGENTS.md`.
- **`ESTADO.md`** dice en qué va el proyecto: avance por notebook, decisiones con fecha y motivo, resultados clave, preguntas abiertas y bitácora. **Quien termine una tarea, persona o agente, lo actualiza.**
- **`PLAN.md`** es este documento. Si el plan cambia, se actualiza aquí y se hace commit.

### Git

- Todo se trabaja directamente en `main`: sin ramas auxiliares ni Pull Requests.
- Mensajes de commit cortos en español con prefijo, por ejemplo `nb01: quitamos duplicados y columnas constantes`.
- En los commits aparece solo el nombre de la persona: nunca un agente o herramienta de IA (sin `Co-Authored-By`, sin `Generated with ...`).
- **Dos personas no editan el mismo notebook a la vez:** los `.ipynb` se mezclan muy mal.
- Los agentes no hacen commit ni push sin que la persona a cargo lo pida.

### Prompt de arranque para cualquier agente

```text
Lee AGENTS.md, ESTADO.md y la sección [N] de PLAN.md. Revisa el notebook de referencia que indica AGENTS.md para esta tarea.
Dime en pocas líneas qué entendiste y si tienes dudas. Después crea [nombre del notebook] siguiendo el plan,
ejecútalo completo, repórtame los números clave y actualiza ESTADO.md. No hagas commit.
```

## 11. Informe y presentación

Si la profesora dio un formato, extensión o duración, eso manda sobre esta propuesta.

### Informe

1. **Introducción:** el peligro sísmico en minas, por qué es difícil de anticipar y el objetivo del trabajo.
2. **Dataset:** origen, qué mide cada variable, el desbalance y la cita de Sikora y Wróbel (2010).
3. **Análisis exploratorio:** distribución de la clase, categóricas, numéricas y correlación phik (notebook 01).
4. **Preprocesamiento:** duplicados, columnas constantes, por qué no quitamos los extremos, codificación ordinal y partición.
5. **Escalado, PCA y selección de rasgos** (notebook 02).
6. **Balanceo y comparación de modelos:** los heatmaps de las 41 combinaciones y la trampa del accuracy (notebook 03).
7. **Modelo final:** ajuste, umbral y resultados en test (notebook 04).
8. **App:** capturas de las tres pestañas.
9. **Conclusiones, limitaciones y trabajo futuro.**
10. **Referencias.**

Todas las figuras salen de `figuras/`, así no hay que volver a correr nada.

### Presentación (unos 12 a 15 minutos)

1. Título e integrantes.
2. El problema: peligro sísmico en minas y por qué importa.
3. El dataset y qué mide.
4. El reto: solo 6,6% de turnos peligrosos y la trampa del accuracy.
5. Hallazgos del EDA.
6. Preprocesamiento y selección de rasgos.
7. Las 41 combinaciones: heatmap de F1.
8. Modelo final, umbral y matriz de confusión en test.
9. Demo en vivo de la app.
10. Conclusiones y trabajo futuro.

Para la demo: tener abierto el enlace de Gradio antes de empezar y un video corto de respaldo.

## 12. Riesgos y checklist final

El riesgo principal no es técnico: es que el F1 de la clase peligrosa salga modesto y parezca un mal trabajo. Hay que mostrarlo como lo que es, un problema difícil bien analizado.

### 12.1 Trampas conocidas

- **F1 bajo.** En la prueba rápida con la copia CSV salió entre 0,14 y 0,26. Hay que confirmar con la profesora que un resultado así es aceptable si está bien analizado, y compararlo siempre con la línea base que nunca da la alerta.
- **Fuga de información por el balanceo.** SMOTE o el submuestreo fuera del pipeline, o antes de la validación cruzada, inflan los resultados. Siempre van dentro de `ImbPipeline`.
- **Usar test antes de tiempo.** Test solo se usa en la sección 8.6. Elegir modelo, hiperparámetros o umbral mirando test invalida el resultado.
- **Pocos positivos en test.** Son unos 34, y cada acierto mueve el recall cerca de 3 puntos. Los resultados de test se leen junto con la media y la desviación de la CV.
- **Sesgo por elegir entre 41 combinaciones.** El mejor de muchos tiende a verse mejor de lo que es. Por eso el test se reserva para el final.
- **Diferencias entre el ARFF y la copia CSV.** Si la cantidad de filas o de duplicados no es la esperada, se reporta antes de seguir.
- **Modelo sin `predict_proba`.** `LinearSVC` no lo tiene, y si gana se pregunta al grupo.
- **Advertencias.** Las de convergencia del MLP y las de precisión indefinida son esperables; se explican en un markdown y no se ocultan con un `filterwarnings` global.
- **Versiones.** Si imbalanced-learn y scikit-learn chocan, `uv` lo avisa al instalar. No se cambian versiones sin preguntar.
- **Colab.** Hay que instalar imbalanced-learn y los demás paquetes al inicio de cada sesión.

### 12.2 Checklist de entrega

- [ ] Los cuatro notebooks corren de cero con "Reiniciar y ejecutar todo", en local y en Colab.
- [ ] Ningún notebook usa comprehensions, lambdas ni generadores.
- [ ] Cada función tiene docstring y una sola responsabilidad. Las funciones repetidas son idénticas en todos los notebooks.
- [ ] Todos los markdown están en español, en voz grupal y con los números reales.
- [ ] `test.csv` solo se usa en la sección 8.6.
- [ ] Existen `modelo_final.joblib`, `metadata_modelo.json`, `mapeos_categoricas.joblib` y `decisiones_preprocesamiento.joblib`.
- [ ] La app pasa las pruebas de la sección 9.4.
- [ ] El README tiene la tabla de resultados llena y los integrantes.
- [ ] `ESTADO.md` está al día.
- [ ] Informe completo y presentación ensayada con demo y video de respaldo.