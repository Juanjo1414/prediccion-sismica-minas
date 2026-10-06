# ESTADO.md — En qué va el proyecto

> Este archivo lo actualiza quien termine una tarea (persona o agente). Es la forma de que todos sepamos en qué va el proyecto sin tener que preguntar.
> Regla: al terminar, marca el avance, anota cualquier decisión nueva y agrega una línea a la bitácora (la más reciente arriba).

**Última actualización:** 2026-10-06

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
- [ ] 04 — Ajuste, evaluación final y exportación

### Publicación y entregables
- [ ] App Gradio funcionando en local
- [ ] App Gradio funcionando en Colab (enlace público)
- [ ] README con resultados finales
- [ ] Informe
- [ ] Presentación ensayada con demo

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
- **Modelo final, umbral y F1 en test:** _pendiente_

## Preguntas abiertas

- [ ] Nombres y roles de los integrantes (para el README y el informe).
- [ ] Confirmar con la profesora que un F1 modesto en la clase peligrosa es aceptable si está bien analizado (el problema es difícil de por sí).
- [ ] Formato y extensión del informe y duración de la presentación.
- [x] `.gitignore` tenía `models/*.joblib`; ya se quitó (2026-10-06).
- [x] **Rasgos del notebook 02:** se decidió usar los 5 de información mutua en vez de `nbumps` solo (2026-10-06).

## Bitácora

| Fecha | Quién | Qué se hizo | Resultado / siguiente paso |
| :--- | :--- | :--- | :--- |
| 2026-10-06 | Juan | Se creó y ejecutó `03_Balanceo_Comparacion_Modelos.ipynb` completo (41 combinaciones, unos 2 min) | Genera `resultados/comparacion_modelos.csv` y 4 figuras `03_*.png`. Siguiente: notebook 04 |
| 2026-10-06 | Juan | Se creó y ejecutó `02_Escalado_PCA_Seleccion.ipynb` completo | Genera `decisiones_preprocesamiento.joblib` (MinMaxScaler + 5 rasgos de información mutua), `resultados/comparacion_rasgos.csv` y 10 figuras `02_*.png`. Siguiente: notebook 03 |
| 2026-10-06 | Juan | Se creó y ejecutó `01_EDA_Limpieza.ipynb` completo y se subió a `main` | Genera `train.csv`, `test.csv`, `test_original.csv`, `mapeos_categoricas.joblib` y 8 figuras `01_*.png`. Se quitó `models/*.joblib` del `.gitignore` |
| 2026-10-05 | Juan | Se eligió el problema, se creó el plan, `AGENTS.md`, `CLAUDE.md`, `ESTADO.md` y el README | Siguiente: crear el repo, descargar el ARFF y arrancar el notebook 01 |