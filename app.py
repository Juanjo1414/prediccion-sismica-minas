"""
App Gradio: predicción de peligro sísmico en minas de carbón (proyecto final de Fundamentos de IA, Universidad EIA).

Carga el modelo que dejó el notebook 04 (models/modelo_final.joblib, models/metadata_modelo.json y
models/mapeos_categoricas.joblib) y deja evaluar un turno de tres formas:
  1. Llenando a mano el formulario con las mediciones del turno anterior, o cargando un turno real de ejemplo.
  2. Subiendo un archivo CSV con varios turnos.
  3. Mirando el resumen del modelo (rasgos, umbral y métricas).

Cómo correrla:
  - Local:  uv run python app.py   (se abre en http://127.0.0.1:7860)
  - Colab:  poner EN_COLAB = True aquí abajo y ejecutar  %run app.py  desde la carpeta del proyecto en Drive.

Es un proyecto académico: no reemplaza los sistemas de monitoreo reales de una mina.
"""
import os
import json
import math
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

# El medidor llega hasta el 50 %: casi todos los turnos reales quedan por debajo y así las diferencias se ven
RIESGO_MAXIMO_MEDIDOR = 0.5

# Identidad visual (la misma de la presentación): carbón, papel mineral y ámbar de señalización minera
COLOR_CARBON = '#12171E'
COLOR_AMBAR = '#F2A900'
COLOR_AZUL = '#2E7A9B'
COLOR_ROJO = '#C4432B'
COLOR_VERDE = '#2F8F6B'

# Fuentes de Google (Barlow Condensed para títulos y Barlow para el texto). Si no hay internet, el navegador
# usa las fuentes del sistema que dejamos como respaldo en el CSS
CABECERA_HTML = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
                 '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                 '<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&'
                 'family=Barlow+Condensed:wght@500;600;700&display=swap" rel="stylesheet">')

ESTILOS_CSS = """
:root {
  --carbon: #12171E; --papel: #EEF0EE; --papel2: #E1E6E3; --papel3: #CBD3CF;
  --ambar: #F2A900; --azul: #2E7A9B; --rojo: #C4432B; --verde: #2F8F6B;
  --pizarra: #3A4757; --gris: #667282;
  --titulo: 'Barlow Condensed', 'Arial Narrow', 'Roboto Condensed', sans-serif;
}
body, .gradio-container, .dark .gradio-container { background: var(--papel) !important; }
.gradio-container { max-width: 1260px !important; margin: 0 auto !important; font-family: 'Barlow', 'Segoe UI', Arial, sans-serif !important; }
.gradio-container, .gradio-container .prose, .gradio-container label span { color: var(--carbon); }

/* Encabezado: una banda oscura con un sismograma que se dibuja una sola vez al abrir la app */
.hero { position: relative; overflow: hidden; background: var(--carbon); color: #E9EDF0; border-radius: 16px; padding: 34px 40px 30px; margin-bottom: 14px; }
.hero h1 { font-family: var(--titulo); font-weight: 700; font-size: 54px; line-height: .98; letter-spacing: -.01em; margin: 0 0 12px; color: #E9EDF0; max-width: 18ch; }
.hero p { margin: 0 0 16px; color: #B8C2CC; font-size: 19px; line-height: 1.4; max-width: 60ch; }
.hero .sismo { position: absolute; right: -10px; top: 18px; width: 58%; height: 150px; opacity: .95; pointer-events: none; }
.hero .sismo path { fill: none; stroke: var(--ambar); stroke-width: 2.4; stroke-linejoin: round; stroke-dasharray: 2400; stroke-dashoffset: 2400; animation: trazar 3.2s ease-out .2s forwards; }
.hero .sismo line { stroke: #3A4757; stroke-width: 1.4; stroke-dasharray: 6 7; }
@keyframes trazar { to { stroke-dashoffset: 0; } }
.aviso { display: inline-block; background: rgba(242,169,0,.14); border: 1px solid rgba(242,169,0,.55); color: #F6D27A; border-radius: 8px; padding: 7px 14px; font-size: 15px; }

/* Pestañas */
.tab-wrapper, .tab-container { border-color: var(--papel3) !important; }
button[role="tab"] { font-family: var(--titulo) !important; font-size: 21px !important; font-weight: 600 !important; letter-spacing: .005em; color: var(--gris) !important; }
button[role="tab"][aria-selected="true"] { color: var(--carbon) !important; border-bottom: 3px solid var(--ambar) !important; }

/* Tarjetas */
.tarjeta { background: #fff !important; border: 1px solid var(--papel2) !important; border-radius: 16px !important; padding: 22px 24px !important; box-shadow: 0 1px 0 rgba(18,23,30,.04); }
.titulo-tarjeta { font-family: var(--titulo); font-size: 26px; font-weight: 600; line-height: 1; margin: 0 0 4px; color: var(--carbon); }
.sub-tarjeta { color: var(--gris); font-size: 15px; margin: 0 0 8px; }
.tarjeta label span, .tarjeta .block-info { font-size: 14px; }
.tarjeta .form, .tarjeta .row { border: none !important; box-shadow: none !important; background: transparent !important; }
.tarjeta input[type="number"], .tarjeta input[type="text"] { font-size: 18px !important; font-variant-numeric: tabular-nums; }

/* Botones */
.gradio-container button.primary, .gradio-container .primary { background: var(--ambar) !important; border: 2px solid var(--ambar) !important; color: var(--carbon) !important; font-weight: 600 !important; font-size: 17px !important; transition: transform .15s, background .2s; }
.gradio-container button.primary:hover { background: #ffc233 !important; border-color: #ffc233 !important; }
.gradio-container button.secondary { background: transparent !important; border: 2px solid var(--carbon) !important; color: var(--carbon) !important; font-weight: 600 !important; font-size: 16px !important; }
.gradio-container button.secondary:hover { background: var(--carbon) !important; color: var(--papel) !important; }
.gradio-container button:active { transform: scale(.98); }
.gradio-container button:focus-visible { outline: 3px solid var(--azul) !important; outline-offset: 2px; }

/* Resultado */
.resultado { text-align: center; padding: 6px 6px 2px; }
.resultado .veredicto { font-family: var(--titulo); font-size: 44px; font-weight: 700; line-height: 1; margin: 0 0 4px; }
.resultado .veredicto.alerta { color: #B27C00; }
.resultado .veredicto.calma { color: var(--verde); }
.resultado .detalle { color: var(--pizarra); font-size: 16px; margin: 6px auto 0; max-width: 40ch; line-height: 1.4; }
.resultado .fichas { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 12px; }
.resultado .ficha { background: var(--papel); border-radius: 999px; padding: 4px 14px; font-size: 14px; color: var(--pizarra); }
.resultado svg { width: 100%; max-width: 360px; height: auto; }
.resultado .arco-progreso { animation: llenar .9s cubic-bezier(.2,.8,.2,1); }
@keyframes llenar { from { stroke-dasharray: 0 400; } }
.vacio { text-align: center; color: var(--gris); padding: 38px 18px 30px; }
.vacio .titulo-tarjeta { color: var(--pizarra); margin-bottom: 8px; }
.vacio .punto { width: 74px; height: 74px; border-radius: 50%; border: 3px dashed var(--papel3); margin: 0 auto 16px; }
.real { margin-top: 10px; padding: 10px 14px; border-radius: 10px; font-size: 16px; text-align: left; }
.real.si { background: #FBEFC9; border-left: 4px solid var(--ambar); }
.real.no { background: #DDEFE7; border-left: 4px solid var(--verde); }

/* Resumen del modelo */
.fichas-modelo { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 18px; }
.fichas-modelo span { background: var(--carbon); color: #E9EDF0; border-radius: 8px; padding: 6px 14px; font-size: 16px; }
.fichas-modelo span.rasgo { background: var(--ambar); color: var(--carbon); font-weight: 600; }
.metricas { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 10px 0 8px; }
.metrica { background: var(--papel); border-radius: 12px; padding: 14px 16px; }
.metrica .nombre { color: var(--gris); font-size: 14px; }
.metrica .valor { font-family: var(--titulo); font-size: 38px; font-weight: 700; line-height: 1.05; }
.metrica .cv { color: var(--gris); font-size: 14px; }
.nota-modelo { border-left: 4px solid var(--ambar); background: #F7F3E6; border-radius: 0 10px 10px 0; padding: 12px 16px; margin-top: 18px; font-size: 16px; line-height: 1.45; }

/* Tabla de resultados del CSV */
.gradio-container table { font-variant-numeric: tabular-nums; }
.pie { text-align: center; color: var(--gris); font-size: 14px; padding: 14px 0 4px; }
footer { display: none !important; }

@media (max-width: 760px) {
  .hero { padding: 24px 22px; } .hero h1 { font-size: 38px; } .hero .sismo { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  .hero .sismo path { animation: none; stroke-dashoffset: 0; } .resultado .arco-progreso { animation: none; }
}
"""


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


# Probabilidad de cada turno real de test, calculada una vez; sirve para elegir turnos de ejemplo según su riesgo
PROBABILIDADES_TEST = predecir_probabilidades(TEST_ORIGINAL)


def formato_numero(valor, decimales):
    """Devuelve un número como texto con coma decimal (6,6) y no con punto, como escribimos en todo el proyecto."""
    return ('{:.' + str(decimales) + 'f}').format(valor).replace('.', ',')


def svg_medidor(probabilidad):
    """
    Dibuja el medidor semicircular del riesgo como SVG. Recibe la probabilidad de peligro y devuelve el texto
    del SVG: una pista gris, un arco de progreso (azul bajo el umbral, ámbar sobre el umbral, rojo desde el 25 %),
    las marcas del umbral y del riesgo normal, y el porcentaje grande en el centro. Hacemos el dibujo a mano
    para no depender de ninguna librería gráfica y para poder animarlo con CSS.
    """
    centro_x, centro_y, radio = 180, 175, 135
    largo = math.pi * radio
    fraccion = min(probabilidad, RIESGO_MAXIMO_MEDIDOR) / RIESGO_MAXIMO_MEDIDOR
    umbral = METADATA['umbral']
    if probabilidad >= 0.25:
        color = COLOR_ROJO
    elif probabilidad >= umbral:
        color = COLOR_AMBAR
    else:
        color = COLOR_AZUL
    arco = 'M {} {} A {} {} 0 0 1 {} {}'.format(centro_x - radio, centro_y, radio, radio, centro_x + radio, centro_y)

    marcas = ''
    for riesgo, texto in [(umbral, formato_numero(umbral * 100, 0) + '%'), (0.25, '25%')]:
        angulo = math.pi - (riesgo / RIESGO_MAXIMO_MEDIDOR) * math.pi
        x1 = centro_x + (radio - 30) * math.cos(angulo)
        y1 = centro_y - (radio - 30) * math.sin(angulo)
        x2 = centro_x + (radio + 26) * math.cos(angulo)
        y2 = centro_y - (radio + 26) * math.sin(angulo)
        xt = centro_x + (radio + 44) * math.cos(angulo)
        yt = centro_y - (radio + 44) * math.sin(angulo)
        marcas += '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="#12171E" stroke-width="2.5"/>'.format(x1, y1, x2, y2)
        marcas += '<text x="{:.1f}" y="{:.1f}" text-anchor="middle" font-size="15" fill="#667282">{}</text>'.format(xt, yt + 5, texto)
    # Marca punteada del riesgo normal (la proporción de turnos peligrosos en los datos)
    angulo_base = math.pi - (PREVALENCIA / RIESGO_MAXIMO_MEDIDOR) * math.pi
    bx1 = centro_x + (radio - 30) * math.cos(angulo_base)
    by1 = centro_y - (radio - 30) * math.sin(angulo_base)
    bx2 = centro_x + (radio + 18) * math.cos(angulo_base)
    by2 = centro_y - (radio + 18) * math.sin(angulo_base)
    marcas += '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="#667282" stroke-width="2.5" stroke-dasharray="3 4"/>'.format(bx1, by1, bx2, by2)

    return ('<svg viewBox="0 -22 360 237" role="img" aria-label="Riesgo estimado: {texto}%">'
            '<path d="{arco}" fill="none" stroke="#E1E6E3" stroke-width="30"/>'
            '<path class="arco-progreso" d="{arco}" fill="none" stroke="{color}" stroke-width="30" '
            'stroke-dasharray="{lleno:.1f} {largo:.1f}"/>'
            '{marcas}'
            '<text x="{cx}" y="{ty}" text-anchor="middle" font-size="62" font-weight="700" fill="#12171E" '
            'style="font-family:var(--titulo)">{texto}%</text>'
            '<text x="{cx}" y="{ty2}" text-anchor="middle" font-size="15" fill="#667282">riesgo estimado</text>'
            '</svg>').format(arco=arco, color=color, lleno=largo * fraccion, largo=largo, marcas=marcas, cx=centro_x,
                             ty=centro_y - 14, ty2=centro_y + 12, texto=formato_numero(probabilidad * 100, 1))


def texto_veredicto(probabilidad):
    """
    Arma la tarjeta de resultado en HTML: el veredicto, el medidor del riesgo y unas fichas con el umbral y el
    riesgo normal para que el número tenga contexto. Compara la probabilidad con el umbral del modelo.
    """
    umbral = METADATA['umbral']
    if probabilidad >= umbral:
        titulo = '<div class="veredicto alerta">⚠ Turno peligroso</div>'
        frase = 'El riesgo estimado supera el umbral de decisión: el modelo da la alerta.'
    else:
        titulo = '<div class="veredicto calma">✓ Sin peligro previsto</div>'
        frase = 'El riesgo estimado queda por debajo del umbral de decisión: el modelo no da la alerta.'
    fichas = ('<div class="fichas"><span class="ficha">Umbral de alerta: {}%</span>'
              '<span class="ficha">Riesgo normal en los datos: {}%</span></div>').format(
                  formato_numero(umbral * 100, 0), formato_numero(PREVALENCIA * 100, 1))
    return ('<div class="resultado">' + titulo + svg_medidor(probabilidad)
            + '<div class="detalle">' + frase + '</div>' + fichas + '</div>')


def resultado_vacio():
    """Devuelve la tarjeta que se muestra antes de evaluar nada: invita a llenar el formulario o a cargar un turno."""
    return ('<div class="vacio"><div class="punto"></div><div class="titulo-tarjeta">Aún no hay un turno evaluado</div>'
            'Llena las mediciones y pulsa <b>Evaluar turno</b>, o carga un turno real de ejemplo para ver cómo responde el modelo.</div>')


def texto_valor_real(clase):
    """Devuelve el aviso en HTML con lo que pasó de verdad en el turno siguiente de un turno real de test."""
    if int(clase) == 1:
        return '<div class="real si"><b>Valor real:</b> ⚠ en el turno siguiente sí hubo un evento de alta energía.</div>'
    return '<div class="real no"><b>Valor real:</b> ✓ en el turno siguiente no hubo evento de alta energía.</div>'


def evaluar_turno(*valores):
    """
    Evalúa un turno con los valores del formulario y devuelve la tarjeta de resultado en HTML.
    Usamos *valores porque Gradio entrega los campos del formulario como argumentos sueltos, en el mismo
    orden de `rasgos`; es la única excepción a la regla de no usar *args.
    """
    fila = {}
    for rasgo, valor in zip(METADATA['rasgos'], valores):
        fila[rasgo] = valor
    # Un campo vacío llega como None y pandas lo cuenta como vacío, así que validar_entrada lo detecta
    df_original = pd.DataFrame([fila])
    probabilidad = predecir_probabilidades(df_original)[0]
    return texto_veredicto(probabilidad)


def valores_turno(posicion):
    """
    Devuelve los valores de un turno real de test_original.csv (el de la posición dada), en el orden de
    `rasgos` y listos para llenar el formulario, más el aviso con lo que pasó de verdad en el turno siguiente.
    """
    turno = TEST_ORIGINAL.iloc[posicion]
    salida = []
    for rasgo in METADATA['rasgos']:
        valor = turno[rasgo]
        if isinstance(valor, str):
            salida.append(valor)
        else:
            salida.append(float(valor))
    salida.append(texto_valor_real(turno['class']))
    return salida


def cargar_turno_al_azar():
    """Elige un turno real al azar de test_original.csv y devuelve sus valores para llenar el formulario."""
    posicion = int(np.random.randint(0, len(TEST_ORIGINAL)))
    return valores_turno(posicion)


def cargar_turno_tranquilo():
    """Carga un turno real de riesgo bajo: el que está en el percentil 25 del riesgo de los turnos de test."""
    posicion = int(np.argsort(PROBABILIDADES_TEST)[int(0.25 * len(PROBABILIDADES_TEST))])
    return valores_turno(posicion)


def cargar_turno_limite():
    """Carga un turno real cuyo riesgo queda justo en el umbral de decisión, el caso más difícil de decidir."""
    posicion = int(np.argmin(np.abs(PROBABILIDADES_TEST - METADATA['umbral'])))
    return valores_turno(posicion)


def cargar_turno_intenso():
    """Carga un turno real de riesgo muy alto: el que está en el percentil 99 del riesgo de los turnos de test."""
    posicion = int(np.argsort(PROBABILIDADES_TEST)[int(0.99 * len(PROBABILIDADES_TEST))])
    return valores_turno(posicion)


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
    Arma el HTML de la pestaña "Sobre el modelo" con todo lo que guardó el notebook 04 en la metadata:
    modelo, estrategia de balanceo, calibración, escalador, rasgos, umbral, métricas y fecha.
    """
    m = METADATA
    fichas = ''
    for texto in [m['modelo'], 'Balanceo: ' + m['estrategia_balanceo'], 'Calibración sigmoide', 'Escalador: ' + m['escalador'],
                  'Umbral: ' + formato_numero(m['umbral'], 2)]:
        fichas += '<span>' + texto + '</span>'
    rasgos = ''
    for rasgo in m['rasgos']:
        rasgos += '<span class="rasgo">' + rasgo + '</span>'

    nombres = [('F1', 'F1'), ('Recall', 'Recall'), ('Precision', 'Precisión'), ('ROC_AUC', 'ROC AUC'),
               ('Precision_promedio', 'Precisión promedio'), ('Brier', 'Brier'), ('Accuracy', 'Accuracy')]
    tarjetas = ''
    for clave, nombre in nombres:
        tarjetas += ('<div class="metrica"><div class="nombre">{}</div><div class="valor">{}</div>'
                     '<div class="cv">validación cruzada: {}</div></div>').format(
                         nombre, formato_numero(m['metricas_test'][clave], 3), formato_numero(m['metricas_cv'][clave], 3))

    return ('<div class="titulo-tarjeta">Modelo</div><div class="fichas-modelo">' + fichas + '</div>'
            '<div class="titulo-tarjeta">Rasgos que usa</div><div class="fichas-modelo">' + rasgos + '</div>'
            '<div class="titulo-tarjeta">Métricas en la muestra de prueba</div>'
            '<p class="sub-tarjeta">Clase peligrosa. En gris, el valor en validación cruzada sobre el entrenamiento.</p>'
            '<div class="metricas">' + tarjetas + '</div>'
            '<div class="nota-modelo"><b>Cómo leerlo.</b> Los turnos peligrosos son pocos (alrededor del 6,6%), así que el modelo '
            'no es un sistema de certeza: sirve como alerta temprana que ordena los turnos por riesgo. Con el umbral elegido detecta '
            'cerca de la mitad de los turnos peligrosos, y la mayoría de sus alertas son falsas alarmas. El Brier de referencia, '
            'prediciendo siempre el 6,6%, es 0,0616.<br><br><b>Entrenado el</b> ' + m['fecha_entrenamiento']
            + ' con scikit-learn ' + m['version_sklearn'] + '. Es un proyecto académico y no reemplaza los sistemas de '
            'monitoreo reales de una mina.</div>')


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


def trazo_sismograma():
    """
    Genera los puntos del sismograma decorativo del encabezado: un trazo con ruido de fondo y un evento fuerte
    al final. Usamos una semilla fija para que se vea igual cada vez que se abre la app.
    """
    semilla = 11
    puntos = ''
    for x in range(0, 801, 5):
        semilla = (semilla * 16807) % 2147483647
        ruido = (semilla / 2147483647 - 0.5) * 12
        y = 90 + math.sin(x * 0.06) * 6 + ruido
        distancia = x - 560
        if 0 < distancia < 200:
            y += math.sin(distancia * 0.33) * 78 * math.exp(-distancia / 55)
        puntos += ('M' if x == 0 else 'L') + '{} {:.1f} '.format(x, y)
    return puntos


def encabezado_html():
    """Arma el encabezado de la app: título, descripción, aviso académico y el sismograma animado."""
    return ('<div class="hero"><svg class="sismo" viewBox="0 0 800 180" preserveAspectRatio="none" aria-hidden="true">'
            '<line x1="0" y1="42" x2="800" y2="42"/><path d="' + trazo_sismograma() + '"/></svg>'
            '<h1>Peligro sísmico en minas de carbón</h1>'
            '<p>Estima si en el siguiente turno de trabajo (8 horas) habrá un evento sísmico de alta energía '
            '(más de 10⁴ julios), a partir de las mediciones del turno anterior.</p>'
            '<span class="aviso">Proyecto académico de Fundamentos de IA (Universidad EIA). No reemplaza los sistemas de '
            'monitoreo reales de una mina.</span></div>')


def crear_tema():
    """
    Crea el tema de Gradio con los colores del proyecto. Definimos también las variables del modo oscuro con los
    mismos valores para que la app se vea igual aunque el navegador esté en modo oscuro.
    """
    tema = gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont('Barlow'), 'Segoe UI', 'Arial', 'sans-serif'],
        radius_size=gr.themes.sizes.radius_lg,
    )
    return tema.set(
        body_background_fill='#EEF0EE', body_background_fill_dark='#EEF0EE',
        body_text_color='#12171E', body_text_color_dark='#12171E',
        block_background_fill='#FFFFFF', block_background_fill_dark='#FFFFFF',
        block_border_color='#E1E6E3', block_border_color_dark='#E1E6E3',
        block_label_text_color='#3A4757', block_label_text_color_dark='#3A4757',
        input_background_fill='#F7F8F7', input_background_fill_dark='#F7F8F7',
        button_primary_background_fill=COLOR_AMBAR, button_primary_background_fill_dark=COLOR_AMBAR,
        button_primary_text_color=COLOR_CARBON, button_primary_text_color_dark=COLOR_CARBON,
    )


def construir_interfaz():
    """
    Arma la interfaz con gr.Blocks y sus tres pestañas (evaluar un turno, subir un CSV y sobre el modelo)
    y conecta los botones con las funciones. Devuelve la app lista para lanzar con el tema y los estilos.
    """
    with gr.Blocks(title='Peligro sísmico en minas de carbón') as interfaz:
        gr.HTML(encabezado_html())

        with gr.Tab('Evaluar un turno'):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, elem_classes='tarjeta'):
                    gr.HTML('<div class="titulo-tarjeta">Mediciones del turno anterior</div>'
                            '<p class="sub-tarjeta">Escribe los valores a mano o carga un turno real de la muestra de prueba.</p>')
                    campos = []
                    rasgos = METADATA['rasgos']
                    # Los campos van de a dos por fila para que el formulario no quede largo y estrecho
                    for i in range(0, len(rasgos), 2):
                        with gr.Row():
                            for rasgo in rasgos[i:i + 2]:
                                campos.append(crear_campo(rasgo))
                    boton_evaluar = gr.Button('Evaluar turno', variant='primary', size='lg')
                    gr.HTML('<p class="sub-tarjeta" style="margin-top:14px">Turnos reales de ejemplo (de la muestra de prueba)</p>')
                    with gr.Row():
                        boton_tranquilo = gr.Button('Turno tranquilo', variant='secondary')
                        boton_limite = gr.Button('Turno en el límite', variant='secondary')
                        boton_intenso = gr.Button('Turno intenso', variant='secondary')
                    boton_azar = gr.Button('Cargar un turno real al azar', variant='secondary')
                with gr.Column(scale=5, elem_classes='tarjeta'):
                    resultado = gr.HTML(resultado_vacio())
                    valor_real = gr.HTML()

            boton_evaluar.click(evaluar_turno, inputs=campos, outputs=resultado)
            # Primero llenamos el formulario con el turno real y, cuando termina, lo evaluamos (por eso el .then)
            for boton, funcion in [(boton_tranquilo, cargar_turno_tranquilo), (boton_limite, cargar_turno_limite),
                                   (boton_intenso, cargar_turno_intenso), (boton_azar, cargar_turno_al_azar)]:
                boton.click(funcion, inputs=None, outputs=campos + [valor_real]).then(
                    evaluar_turno, inputs=campos, outputs=resultado)

        with gr.Tab('Subir un CSV'):
            with gr.Column(elem_classes='tarjeta'):
                gr.HTML('<div class="titulo-tarjeta">Evaluar varios turnos</div>'
                        '<p class="sub-tarjeta">Sube un archivo CSV con un turno por fila y las columnas originales del dataset '
                        '(con letras en las categóricas). El modelo usa estas columnas: <b>' + ', '.join(METADATA['rasgos'])
                        + '</b>. Las demás se ignoran.</p>')
                archivo = gr.File(label='Archivo CSV', file_types=['.csv'], type='filepath')
                boton_csv = gr.Button('Predecir', variant='primary', size='lg')
                tabla = gr.Dataframe(label='Resultados', interactive=False)
            boton_csv.click(predecir_csv, inputs=archivo, outputs=tabla)

        with gr.Tab('Sobre el modelo'):
            with gr.Column(elem_classes='tarjeta'):
                gr.HTML(resumen_modelo())

        gr.HTML('<div class="pie">Fundamentos de Inteligencia Artificial · Universidad EIA · 2026-2 · '
                'Datos: seismic-bumps (UCI, Sikora y Wróbel, 2010)</div>')
    return interfaz


if __name__ == '__main__':
    # En Gradio 6 el tema, los estilos y la cabecera se pasan al lanzar la app
    construir_interfaz().launch(share=EN_COLAB, theme=crear_tema(), css=ESTILOS_CSS, head=CABECERA_HTML)
