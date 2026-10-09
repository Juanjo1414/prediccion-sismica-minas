# ESTADO.md — En qué va el proyecto

> Este archivo lo actualiza quien termine una tarea (persona o agente). Es la forma de que todos sepamos en qué va el proyecto sin tener que preguntar.
> Regla: al terminar, marca el avance, anota cualquier decisión nueva y agrega una línea a la bitácora (la más reciente arriba).

**Última actualización:** 2026-10-08

## Avance

### Preparación
- [ ] Repositorio creado en GitHub y primer push
- [x] Entorno con uv funcionando en local (`uv sync`)
- [x] `data/seismic-bumps.arff` oficial descargado de UCI
- [x] Notebooks de clase copiados en `referencia/`

### Notebooks
- [x] 01 — Carga, EDA, limpieza, codificación y partición (ejecutado completo sin errores, 2026-10-06)
- [x] 02 — Escalado, PCA y selección de rasgos (ejecutado completo sin errores, 2026-10-06)
- [x] 03 — Balanceo y comparación de los 12 modelos (ejecutado completo sin errores, 2026-10-06)
- [x] 04 — Ajuste, evaluación final y exportación (ejecutado completo sin errores, 2026-10-06)

### Publicación y entregables
- [x] App Gradio funcionando en local (probada el 2026-10-06, ver bitácora)
- [ ] App Gradio funcionando en Colab (enlace público)
- [x] README con resultados finales (2026-10-06; falta llenar los integrantes si cambian)
- [x] Informe (`informe/Informe_Final.pdf`, 22 páginas; fuente en `informe/informe.html`)
- [ ] Presentación ensayada con demo (la presentación interactiva ya está en `presentacion/presentacion.html`; falta ensayar y probar la app en Colab)

## Decisiones tomadas

| Fecha | Decisión | Motivo |
| :--- | :--- | :--- |
| 2026-10-05 | Problema: peligro sísmico en minas de carbón (seismic-bumps, UCI 266) | Clasificación binaria con desbalance real (~6,6%), poco común en cursos |
| 2026-10-05 | Comparar 12 modelos × 4 estrategias de balanceo (sin balanceo, class_weight, SMOTE, submuestreo) | Requisito del trabajo: comparar todos los modelos vistos en clase |
| 2026-10-05 | Métrica principal: F1 de la clase peligrosa | Con 6,6% de positivos el accuracy engaña |
| 2026-10-05 | Todo el código en notebooks, estilo de la profesora; funciones de una sola responsabilidad; sin `src/` | Pedido del grupo: que prime el estilo de clase |
| 2026-10-05 | Partición 80/20 estratificada y CV de 10 particiones | Igual que en clase; con ~2.000 filas no hay problema de tiempo |
| 2026-10-05 | Datos y modelos sí se versionan en Git | Son pequeños; la app funciona apenas se clona |
| 2026-10-06 | Se quitan los 6 duplicados exactos antes de partir (todos de clase 0) | No cuestan ningún turno peligroso y evitan que una misma fila caiga en train y en test |
| 2026-10-06 | Se eliminan `nbumps6`, `nbumps7` y `nbumps89` (constantes en 0) | No aportan información; se confirmó en el EDA |
| 2026-10-06 | Valores extremos no se eliminan | Son eventos sísmicos reales, justo lo que se quiere detectar |
| 2026-10-06 | Escalador: `MinMaxScaler` | Mejor F1 promedio entre regresión logística y SVM (0,2656 vs 0,2580 Standard y 0,2378 sin escalar); diferencias pequeñas frente a la desviación |
| 2026-10-06 | No usar PCA | 5 componentes dan 90% y 7 dan 95% de la varianza, pero el F1 no mejora (0,264 y 0,267 vs 0,269) y se pierde interpretación |
| 2026-10-06 | Regla de selección de rasgos cambiada: solo compiten métodos con selector dentro del pipeline, mínimo 3 rasgos y gana el de mayor F1 | La regla del plan dejaba solo `nbumps` (1 rasgo), con F1 de todos los métodos dentro del ruido de la CV; con 1 rasgo el 03 y la app pierden sentido. Decidido por Juan |
| 2026-10-06 | Rasgos elegidos: `gpuls`, `nbumps`, `nbumps2`, `nbumps3`, `energy` (SelectKBest con información mutua, k=5) | F1 CV 0,283 ± 0,068 y ROC AUC 0,768 (vs 0,744 de `nbumps` solo); selección sin filtración |
| 2026-10-06 | Top 3 para el notebook 04: Naive Bayes, regresión logística y SVM lineal, los tres con submuestreo | Mayor F1 en CV entre modelos distintos (0,309, 0,298 y 0,291); la SVM lineal no tiene `predict_proba`, así que si gana en el 04 se consulta al grupo |
| 2026-10-06 | La app pide solo los rasgos de `metadata_modelo.json` (hoy 5, todos numéricos) y muestra el riesgo en % más el veredicto con el umbral; el CSV solo necesita esas columnas (las demás se ignoran) | El formulario se arma solo a partir de `rasgos`, como pide el plan; así no se piden datos que el modelo no usa |
| 2026-10-06 | Cambio de enfoque en el notebook 04: hiperparámetros con `scoring='average_precision'` (no F1) y probabilidades calibradas (`CalibratedClassifierCV`, sigmoide, cv=5) antes de elegir el umbral | La primera versión (regresión logística `C=0,01`, F1 test 0,296) daba probabilidades pegadas a 0,5, inservibles para la app. Decidido por Juan: un modelo que funcione bien, aunque se salga del plan. Se detectó con datos de train, pero el test de la primera versión ya se había visto: **el test ya no es virgen y hay que decirlo en el informe** |
| 2026-10-06 | Modelo final: SVM lineal + submuestreo + calibración sigmoide (`C=0,1`), umbral 0,10 | Mayor F1 en CV con probabilidades calibradas (0,333 vs 0,323 de regresión logística y 0,309 de Naive Bayes; empate práctico). La calibración le da `predict_proba` a la SVM lineal |
| 2026-10-06 | Grilla de umbrales de 0,01 en 0,01 (en vez de 0,05 en 0,05 del plan) | Con probabilidades calibradas la mayoría quedan por debajo de 0,2 y se necesitan pasos finos |
| 2026-10-06 | Trabajo directo en `main`, sin ramas ni PR; en los commits solo aparece la persona (nunca un agente) | Pedido de Juan; quedó en AGENTS.md y PLAN.md |
| 2026-10-06 | El EDA y phik se hacen con los datos completos, antes de partir; el 01 no elige nada a partir de eso | Plan sección 5; la selección de rasgos se decide en el 02 solo con train |

## Resultados clave (se llenan a medida que avanzamos)

- **Filas del ARFF oficial:** 2.584 filas y 19 columnas (coincide con UCI), sin nulos. 170 peligrosos (6,58%); un modelo que siempre diga "sin peligro" saca 93,42% de accuracy
- **Duplicados eliminados:** 6 (todos de clase 0); quedan 2.578 filas
- **Columnas constantes eliminadas:** `nbumps6`, `nbumps7`, `nbumps89`; quedan 16 columnas (15 rasgos + `class`)
- **Partición 80/20 (estratificada, seed 42):** train 2.062 filas (136 peligrosos, 6,60%) y test 516 (34 peligrosos, 6,59%)
- **Mayor phik con la clase:** `nbumps` 0,350, `gpuls` 0,263, `genergy` 0,231, `shift` 0,216. Redundantes: `energy`/`maxenergy` 0,96, `genergy`/`gpuls` 0,89, `seismoacoustic`/`ghazard` 0,84
- **Escalador elegido:** `MinMaxScaler`. PCA descartado (5 comps = 90%, 7 = 95%)
- **Rasgos elegidos:** `gpuls`, `nbumps`, `nbumps2`, `nbumps3`, `energy` (información mutua, k=5; F1 CV 0,283 ± 0,068). Otros: `nbumps` solo 0,291 (la regla original del plan), SHAP 8 rasgos 0,282, los 15 rasgos 0,269. Todos los F1 caen dentro del ruido de la CV (0,258 a 0,291). Limitación: `shift` no quedó entre los rasgos aunque en el EDA tuvo gran diferencia de riesgo
- **Mejor combinación modelo + balanceo (CV):** Naive Bayes + submuestreo, F1 0,309 ± 0,085 (recall 0,615, precisión 0,207). Le siguen Naive Bayes + SMOTE 0,303, regresión logística + submuestreo 0,298, SVM lineal + submuestreo 0,291 y SVM + submuestreo 0,290; los cinco primeros no se distinguen del ruido (desviación ≈ 0,06). Sin balanceo el F1 promedio es 0,090 y con balanceo 0,234 a 0,250; el submuestreo fue la mejor estrategia en 8 de 12 modelos
- **Modelo final, umbral y F1 en test:** SVM lineal + submuestreo + calibración sigmoide, `C=0,1`, umbral 0,10. CV: F1 0,333, recall 0,537, precisión 0,242, ROC AUC 0,777. **Test:** F1 0,299, recall 0,471 (16 de 34 turnos peligrosos detectados), precisión 0,219 (57 falsas alarmas entre 482 turnos sin peligro), ROC AUC 0,744, Brier 0,059, accuracy 0,855. Un modelo que siempre dice "sin peligro" saca 0,934 de accuracy y F1 0. Primera versión descartada (regresión logística `C=0,01`): F1 test 0,296, 20 de 34 detectados y 81 falsas alarmas
- **Limitaciones del modelo final:** el Brier (0,059 en test) mejora solo un 6% frente a decir siempre el 6,6% (0,0616), y ni el grupo de mayor riesgo pasa de ~1 de cada 4 turnos peligrosos; sirve para ordenar turnos por riesgo, no para dar certeza. Las probabilidades sí son interpretables (media predicha en test 0,067 vs 0,066 real). Solo 34 positivos en test; umbral e hiperparámetros elegidos con la misma CV; el test ya se vio una vez

## Preguntas abiertas

- [ ] Nombres y roles de los integrantes (para el README y el informe).
- [ ] Confirmar con la profesora que un F1 modesto en la clase peligrosa es aceptable si está bien analizado (el problema es difícil de por sí).
- [ ] Formato y extensión del informe y duración de la presentación.
- [x] `.gitignore` tenía `models/*.joblib`; ya se quitó (2026-10-06).
- [x] **Probabilidades del modelo final:** resuelto el 2026-10-06 calibrándolas (sigmoide); la app puede mostrar el riesgo estimado como porcentaje.
- [x] **Rasgos del notebook 02:** se decidió usar los 5 de información mutua en vez de `nbumps` solo (2026-10-06).

## Bitácora

| Fecha | Quién | Qué se hizo | Resultado / siguiente paso |
| :--- | :--- | :--- | :--- |
| 2026-10-08 | Asistente | Se revisaron el código y las salidas guardadas del notebook 04 para explicar ajuste, calibración, umbral, evaluación y exportación | Se aclararon la precisión promedio, las métricas agrupadas de predicciones fuera de partición, los límites de la calibración y la consulta previa de test. Sin modificar ni ejecutar el notebook |
| 2026-10-07 | Asistente | Se revisaron las celdas y las salidas guardadas del notebook 03 para explicar las 41 combinaciones y la selección de candidatos | Explicación del catálogo, balanceo dentro del pipeline, métricas y gráficas. Se distinguió F1 cero de ausencia de alertas y se aclararon los límites de comparar promedios y desviaciones. Sin modificar ni ejecutar el notebook |
| 2026-10-07 | Asistente | Se revisaron el código y las salidas guardadas del notebook 02 para explicarlo paso a paso | Explicación de CV, escalado, PCA, selectores y decisión final; se aclaró que VarianceThreshold sí admite pipeline y que ajustarlo antes de CV usa información de las particiones de validación. Sin modificar ni ejecutar el notebook |
| 2026-10-06 | Asistente | Se leyeron todas las celdas y las salidas guardadas del notebook 01 para explicar su funcionamiento paso a paso | Explicación de carga, exploración, limpieza, phik, partición, codificación y exportación. Sin modificar ni volver a ejecutar el notebook |
| 2026-10-06 | Asistente | Se revisó la estructura, la documentación, la configuración de ejecución y la metadata para explicar el proyecto y su puesta en marcha | En esta copia se detectó uv 0.12.2 y `uv.lock`, pero no `.venv/Scripts/python.exe`. Los artefactos entrenados están presentes; el primer paso local es `uv sync`. No se instalaron dependencias ni se ejecutaron la app o los notebooks en esta revisión. `graphify-out/GRAPH_REPORT.md` describe otro proyecto (UJIIndoorLoc) |
| 2026-10-06 | Juan | Se rediseñó `app.py` con la identidad visual de la presentación (tema y CSS propios, encabezado con sismograma, medidor de riesgo animado, turnos reales de ejemplo y tarjetas de métricas) | Misma lógica y mismas pruebas de la sección 9.4 (probadas en navegador y por API). `evaluar_turno` devuelve ahora la tarjeta HTML y se quitó `etiquetas_probabilidad`. En Gradio 6 el tema y el CSS se pasan en `launch()`. Falta probar en Colab |
| 2026-10-06 | Juan | Se escribieron el informe final (HTML con estilo de impresión y PDF de 22 páginas) y la presentación interactiva en HTML (14 diapositivas con notas del orador, cronómetro, gráficas con los datos reales y un simulador que reproduce el modelo final con diferencia de 10⁻¹⁶ respecto a Python) | Revisados en pantalla y en navegador. Siguiente: ensayar la presentación, probar la app en Colab y que cada integrante corra los 4 notebooks de cero |
| 2026-10-06 | Juan | Se actualizó el README: resultados finales, calibración, 5 rasgos y nota de transparencia sobre el test | Quedan pendientes: probar la app en Colab, el informe y la presentación, y la revisión cruzada (correr los 4 notebooks de cero) |
| 2026-10-06 | Juan | Se creó `app.py` (Gradio, 3 pestañas) y se probó en local | Pruebas de la sección 9.4 de PLAN.md: pasan 1 a 5 y la prueba en local (misma probabilidad que el modelo del notebook 04 en los 516 turnos de test, 73 alarmas igual que el notebook, campo vacío y columna faltante con mensaje claro, CSV de 5 y de 516 filas, métricas de "Sobre el modelo" iguales a la metadata, botón de turno al azar con `.then`). **Falta probar en Colab** y subir el commit. El formulario tiene solo los 5 rasgos del modelo y muestra el riesgo estimado en % |
| 2026-10-06 | Juan | Se creó y ejecutó `04_Ajuste_Evaluacion_Exportacion.ipynb` completo, con cambio de enfoque (calibración de probabilidades) | Genera `modelo_final.joblib`, `metadata_modelo.json`, `ajuste_hiperparametros.csv` y 5 figuras `04_*.png`; el modelo cargado predice igual que el original. Siguiente: `app.py` |
| 2026-10-06 | Juan | Se creó y ejecutó `03_Balanceo_Comparacion_Modelos.ipynb` completo (41 combinaciones, unos 2 min) | Genera `resultados/comparacion_modelos.csv` y 4 figuras `03_*.png`. Siguiente: notebook 04 |
| 2026-10-06 | Juan | Se creó y ejecutó `02_Escalado_PCA_Seleccion.ipynb` completo | Genera `decisiones_preprocesamiento.joblib` (MinMaxScaler + 5 rasgos de información mutua), `resultados/comparacion_rasgos.csv` y 10 figuras `02_*.png`. Siguiente: notebook 03 |
| 2026-10-06 | Juan | Se creó y ejecutó `01_EDA_Limpieza.ipynb` completo y se subió a `main` | Genera `train.csv`, `test.csv`, `test_original.csv`, `mapeos_categoricas.joblib` y 8 figuras `01_*.png`. Se quitó `models/*.joblib` del `.gitignore` |
| 2026-10-05 | Juan | Se eligió el problema, se creó el plan, `AGENTS.md`, `CLAUDE.md`, `ESTADO.md` y el README | Siguiente: crear el repo, descargar el ARFF y arrancar el notebook 01 |
