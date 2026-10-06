"""
App Gradio: predicción de peligro sísmico en minas de carbón (proyecto final de Fundamentos de IA, Universidad EIA).

Carga el modelo que dejó el notebook 04 (models/modelo_final.joblib, models/metadata_modelo.json y
models/mapeos_categoricas.joblib) y deja evaluar un turno de tres formas:
  1. Llenando a mano el formulario con las mediciones del turno anterior (o cargando un turno real al azar).
  2. Subiendo un archivo CSV con varios turnos.
  3. Mirando el resumen del modelo (rasgos, umbral y métricas).

Cómo correrla:
  - Local:  uv run python app.py   (se abre en http://127.0.0.1:7860)
  - Colab:  poner EN_COLAB = True aquí abajo y ejecutar  %run app.py  desde la carpeta del proyecto en Drive.

Es un proyecto académico: no reemplaza los sistemas de monitoreo reales de una mina.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Si corremos en Google Colab ponemos True; en el computador (VS Code), False
EN_COLAB = False

if EN_COLAB:
    # En Colab los archivos viven en Drive, así que primero lo montamos
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/Proyecto_Sismos/'
else:
    # Así la app funciona sin importar desde qué carpeta se lance
    RUTA_BASE = os.path.dirname(os.path.abspath(__file__)) + '/'

# Todas las rutas de la app salen de aquí, para no tener rutas regadas por el código
RUTA_DATOS = RUTA_BASE + 'data/'
RUTA_MODELOS = RUTA_BASE + 'models/'

# Qué mide cada variable (según la descripción de UCI), para ponerlo en el formulario
DESCRIPCIONES = {
    'seismic': 'Evaluación del peligro del turno anterior con el método sísmico (a = sin peligro, b = bajo, c = alto, d = estado de peligro)',
    'seismoacoustic': 'Evaluación del peligro del turno anterior con el método sismoacústico (a = sin peligro, b = bajo, c = alto, d = estado de peligro)',
    'shift': 'Tipo de turno: W = extracción de carbón, N = preparación',
    'genergy': 'Energía sísmica registrada en el turno anterior por el geófono más activo (en julios)',
    'gpuls': 'Número de pulsos registrados en el turno anterior por el geófono más activo',
    'gdenergy': 'Desviación (en %) de la energía del turno anterior respecto al promedio de los ocho turnos previos',
    'gdpuls': 'Desviación (en %) de los pulsos del turno anterior respecto al promedio de los ocho turnos previos',
    'ghazard': 'Evaluación del peligro con el método sismoacústico usando solo el geófono más activo (a, b, c o d)',
    'nbumps': 'Número total de golpes sísmicos registrados en el turno anterior',
    'nbumps2': 'Número de golpes sísmicos con energía entre 10² y 10³ J en el turno anterior',
    'nbumps3': 'Número de golpes sísmicos con energía entre 10³ y 10⁴ J en el turno anterior',
    'nbumps4': 'Número de golpes sísmicos con energía entre 10⁴ y 10⁵ J en el turno anterior',
    'nbumps5': 'Número de golpes sísmicos con energía entre 10⁵ y 10⁶ J en el turno anterior',
    'energy': 'Energía total de los golpes sísmicos registrados en el turno anterior (en julios)',
    'maxenergy': 'Energía máxima de un golpe sísmico registrado en el turno anterior (en julios)',
}

# Porcentaje de turnos peligrosos en los datos de entrenamiento (lo vimos en el notebook 01); solo para dar contexto
PREVALENCIA = 0.066


def cargar_artefactos():
    """
    Lee lo que dejaron los notebooks: el modelo final, el diccionario de codificación de las categóricas,
    la metadata del modelo (rasgos, umbral y métricas) y los turnos reales de test con las letras originales.
    Devuelve las cuatro cosas, en ese orden.
    """
    modelo = joblib.load(RUTA_MODELOS + 'modelo_final.joblib')
    mapeos = joblib.load(RUTA_MODELOS + 'mapeos_categoricas.joblib')
    with open(RUTA_MODELOS + 'metadata_modelo.json', encoding='utf-8') as archivo:
        metadata = json.load(archivo)
    test_original = pd.read_csv(RUTA_DATOS + 'test_original.csv')
    return modelo, mapeos, metadata, test_original


# Se cargan una sola vez, al abrir la app. Los botones de Gradio no pueden recibir parámetros extra,
# así que las funciones de abajo leen estas variables globales (solo las leen, nunca las modifican)
MODELO, MAPEOS, METADATA, TEST_ORIGINAL = cargar_artefactos()


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


def validar_entrada(df_codificado):
    """
    Revisa que no haya vacíos en las columnas que usa el modelo. Un vacío aparece cuando un campo se dejó
    sin llenar o cuando una letra no está en el diccionario de codificación. Si encuentra alguno, lanza un
    gr.Error con un mensaje claro (nombra las columnas problemáticas) en vez de dejar que la app se rompa.
    """
    columnas_con_vacios = []
    for columna in METADATA['rasgos']:
        if df_codificado[columna].isna().any():
            columnas_con_vacios.append(columna)
    if len(columnas_con_vacios) > 0:
        raise gr.Error('Faltan datos o hay valores no válidos en: ' + ', '.join(columnas_con_vacios) + '.')


def predecir_probabilidades(df_original):
    """
    Devuelve la probabilidad de peligro de cada turno de un DataFrame con las columnas originales (con letras).
    Codifica las categóricas, valida que no falte nada y toma solo las columnas de `rasgos` en el mismo orden
    en que se entrenó el modelo (si no, las predicciones saldrían mal).
    """
    df_codificado = codificar_categoricas(df_original, MAPEOS)
    validar_entrada(df_codificado)
    X = df_codificado[METADATA['rasgos']]
    return MODELO.predict_proba(X)[:, 1]


def texto_veredicto(probabilidad):
    """
    Arma el texto en Markdown con el veredicto, comparando la probabilidad con el umbral del modelo.
    Muestra el riesgo estimado, el umbral y cuánto es el riesgo normal, para que el número tenga contexto.
    """
    umbral = METADATA['umbral']
    if probabilidad >= umbral:
        titulo = '## ⚠️ Turno peligroso'
    else:
        titulo = '## ✅ Sin peligro previsto'
    return (titulo + '\n\n'
            + 'Riesgo estimado de un evento de alta energía en el siguiente turno: **' + str(round(probabilidad * 100, 1)) + '%**.\n\n'
            + 'El modelo da la alerta cuando el riesgo supera el **' + str(round(umbral * 100, 1)) + '%** (umbral de decisión). '
            + 'Como referencia, en los datos de entrenamiento el ' + str(round(PREVALENCIA * 100, 1)) + '% de los turnos fueron peligrosos.')


def etiquetas_probabilidad(probabilidad):
    """Devuelve el diccionario que necesita gr.Label para mostrar las dos probabilidades como barras."""
    return {'Peligroso': float(probabilidad), 'Sin peligro': float(1 - probabilidad)}


def evaluar_turno(*valores):
    """
    Evalúa un turno con los valores del formulario y devuelve el veredicto y las probabilidades.
    Usamos *valores porque Gradio entrega los campos del formulario como argumentos sueltos, en el mismo
    orden de `rasgos`; es la única excepción a la regla de no usar *args.
    """
    fila = {}
    for rasgo, valor in zip(METADATA['rasgos'], valores):
        fila[rasgo] = valor
    # Un campo vacío llega como None y pandas lo cuenta como vacío, así que validar_entrada lo detecta
    df_original = pd.DataFrame([fila])
    probabilidad = predecir_probabilidades(df_original)[0]
    return texto_veredicto(probabilidad), etiquetas_probabilidad(probabilidad)


def cargar_turno_al_azar():
    """
    Elige un turno real al azar de test_original.csv y devuelve sus valores para llenar el formulario
    (en el orden de `rasgos`), más un texto que dice qué pasó de verdad en el turno siguiente.
    """
    turno = TEST_ORIGINAL.sample(n=1).iloc[0]
    salida = []
    for rasgo in METADATA['rasgos']:
        valor = turno[rasgo]
        if isinstance(valor, str):
            salida.append(valor)
        else:
            salida.append(float(valor))
    if int(turno['class']) == 1:
        salida.append('**Valor real:** ⚠️ en el turno siguiente hubo un evento de alta energía.')
    else:
        salida.append('**Valor real:** ✅ en el turno siguiente no hubo evento de alta energía.')
    return salida


def predecir_csv(ruta):
    """
    Predice varios turnos a partir de un archivo CSV con las columnas originales del dataset (con letras).
    Revisa que estén las columnas que usa el modelo (si faltan, dice cuáles) y devuelve una tabla con el
    número de turno, la probabilidad de peligro y el veredicto. Ignora la columna `class` si viene.
    """
    if ruta is None:
        raise gr.Error('Primero sube un archivo CSV.')
    df = pd.read_csv(ruta)

    columnas_faltantes = []
    for rasgo in METADATA['rasgos']:
        if rasgo not in df.columns:
            columnas_faltantes.append(rasgo)
    if len(columnas_faltantes) > 0:
        raise gr.Error('Al archivo le faltan estas columnas: ' + ', '.join(columnas_faltantes)
                       + '. El modelo usa: ' + ', '.join(METADATA['rasgos']) + '.')

    probabilidades = predecir_probabilidades(df)
    tabla = pd.DataFrame({'Turno': range(1, len(df) + 1)})
    tabla['Probabilidad de peligro (%)'] = np.round(probabilidades * 100, 1)
    veredictos = []
    for probabilidad in probabilidades:
        if probabilidad >= METADATA['umbral']:
            veredictos.append('⚠️ Peligroso')
        else:
            veredictos.append('✅ Sin peligro previsto')
    tabla['Veredicto'] = veredictos
    return tabla


def resumen_modelo():
    """
    Arma el Markdown de la pestaña "Sobre el modelo" con todo lo que guardó el notebook 04 en la metadata:
    modelo, estrategia de balanceo, calibración, escalador, rasgos, umbral, métricas y fecha.
    """
    m = METADATA
    texto = '### Modelo\n\n'
    texto += '- **Modelo:** ' + m['modelo'] + '\n'
    texto += '- **Estrategia de balanceo:** ' + m['estrategia_balanceo'] + '\n'
    texto += '- **Calibración de probabilidades:** ' + m['calibracion'] + '\n'
    texto += '- **Escalador:** ' + m['escalador'] + '\n'
    texto += '- **Rasgos que usa:** ' + ', '.join(m['rasgos']) + '\n'
    texto += '- **Umbral de decisión:** ' + str(m['umbral']) + '\n'
    texto += '- **Fecha de entrenamiento:** ' + m['fecha_entrenamiento'] + ' (scikit-learn ' + m['version_sklearn'] + ')\n\n'
    for titulo, clave in [('Métricas en validación cruzada (train)', 'metricas_cv'), ('Métricas en test', 'metricas_test')]:
        texto += '### ' + titulo + '\n\n'
        for nombre_metrica, valor in m[clave].items():
            texto += '- **' + nombre_metrica + ':** ' + str(round(valor, 3)) + '\n'
        texto += '\n'
    texto += ('### Cómo leerlo\n\n'
              + 'Los turnos peligrosos son pocos (alrededor del 6,6%), así que el modelo no es un sistema de certeza: '
              + 'sirve como alerta temprana que ordena los turnos por riesgo. Con el umbral elegido detecta cerca de la mitad '
              + 'de los turnos peligrosos, y la mayoría de sus alertas son falsas alarmas. Es un proyecto académico y no '
              + 'reemplaza los sistemas de monitoreo reales de una mina.')
    return texto


def crear_campo(columna):
    """
    Devuelve el campo del formulario para una columna: un menú desplegable (gr.Dropdown) si es categórica
    (con las opciones del diccionario de codificación) o un campo numérico (gr.Number) si no lo es.
    Todos llevan su descripción en español y empiezan vacíos.
    """
    descripcion = DESCRIPCIONES.get(columna, '')
    if columna in MAPEOS:
        opciones = list(MAPEOS[columna].keys())
        return gr.Dropdown(choices=opciones, value=None, label=columna, info=descripcion)
    return gr.Number(value=None, label=columna, info=descripcion)


def construir_interfaz():
    """
    Arma la interfaz con gr.Blocks y sus tres pestañas (evaluar un turno, subir un CSV y sobre el modelo)
    y conecta los botones con las funciones. Devuelve la app lista para lanzar.
    """
    with gr.Blocks(title='Peligro sísmico en minas de carbón') as interfaz:
        gr.Markdown('# Predicción de peligro sísmico en minas de carbón\n'
                    'Estima si en el siguiente turno de trabajo (8 horas) habrá un evento sísmico de alta energía (más de 10⁴ J), '
                    'a partir de las mediciones del turno anterior.\n\n'
                    '> ⚠️ Proyecto académico de Fundamentos de IA (Universidad EIA). **No reemplaza los sistemas de monitoreo reales de una mina.**')

        with gr.Tab('Evaluar un turno'):
            campos = []
            for rasgo in METADATA['rasgos']:
                campos.append(crear_campo(rasgo))
            with gr.Row():
                boton_evaluar = gr.Button('Evaluar turno', variant='primary')
                boton_azar = gr.Button('Cargar un turno real al azar')
            veredicto = gr.Markdown()
            probabilidades = gr.Label(label='Probabilidades')
            valor_real = gr.Markdown()

            boton_evaluar.click(evaluar_turno, inputs=campos, outputs=[veredicto, probabilidades])
            # Primero llenamos el formulario con el turno real y, cuando termina, lo evaluamos (por eso el .then)
            boton_azar.click(cargar_turno_al_azar, inputs=None, outputs=campos + [valor_real]).then(
                evaluar_turno, inputs=campos, outputs=[veredicto, probabilidades])

        with gr.Tab('Subir un CSV'):
            gr.Markdown('Sube un archivo CSV con un turno por fila y las columnas originales del dataset (con letras en las '
                        'categóricas). El modelo usa estas columnas: **' + ', '.join(METADATA['rasgos']) + '**.')
            archivo = gr.File(label='Archivo CSV', file_types=['.csv'], type='filepath')
            boton_csv = gr.Button('Predecir', variant='primary')
            tabla = gr.Dataframe(label='Resultados')
            boton_csv.click(predecir_csv, inputs=archivo, outputs=tabla)

        with gr.Tab('Sobre el modelo'):
            gr.Markdown(resumen_modelo())
    return interfaz


if __name__ == '__main__':
    construir_interfaz().launch(share=EN_COLAB)
