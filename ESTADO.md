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
- [ ] 02 — Escalado, PCA y selección de rasgos
- [ ] 03 — Balanceo y comparación de los 12 modelos
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
| 2026-10-06 | El EDA y phik se hacen con los datos completos, antes de partir; el 01 no elige nada a partir de eso | Plan sección 5; la selección de rasgos se decide en el 02 solo con train |

## Resultados clave (se llenan a medida que avanzamos)

- **Filas del ARFF oficial:** 2.584 filas y 19 columnas (coincide con UCI), sin nulos. 170 peligrosos (6,58%); un modelo que siempre diga "sin peligro" saca 93,42% de accuracy
- **Duplicados eliminados:** 6 (todos de clase 0); quedan 2.578 filas
- **Columnas constantes eliminadas:** `nbumps6`, `nbumps7`, `nbumps89`; quedan 16 columnas (15 rasgos + `class`)
- **Partición 80/20 (estratificada, seed 42):** train 2.062 filas (136 peligrosos, 6,60%) y test 516 (34 peligrosos, 6,59%)
- **Mayor phik con la clase:** `nbumps` 0,350, `gpuls` 0,263, `genergy` 0,231, `shift` 0,216. Redundantes: `energy`/`maxenergy` 0,96, `genergy`/`gpuls` 0,89, `seismoacoustic`/`ghazard` 0,84
- **Escalador elegido:** _pendiente_
- **Rasgos elegidos:** _pendiente_
- **Mejor combinación modelo + balanceo (CV):** _pendiente_
- **Modelo final, umbral y F1 en test:** _pendiente_

## Preguntas abiertas

- [ ] Nombres y roles de los integrantes (para el README y el informe).
- [ ] Confirmar con la profesora que un F1 modesto en la clase peligrosa es aceptable si está bien analizado (el problema es difícil de por sí).
- [ ] Formato y extensión del informe y duración de la presentación.
- [ ] `.gitignore` tiene `models/*.joblib`, pero AGENTS.md dice que los modelos sí se suben a Git. ¿Se quita esa línea para que la app funcione al clonar?

## Bitácora

| Fecha | Quién | Qué se hizo | Resultado / siguiente paso |
| :--- | :--- | :--- | :--- |
| 2026-10-06 | Claude Code (a cargo: Juan) | Se creó y ejecutó `01_EDA_Limpieza.ipynb` completo | Genera `train.csv`, `test.csv`, `test_original.csv`, `mapeos_categoricas.joblib` y 8 figuras `01_*.png`. Siguiente: notebook 02. Ojo: `.gitignore` ignora `models/*.joblib`, lo que contradice la decisión de versionar los modelos (ver preguntas abiertas) |
| 2026-10-05 | Juan | Se eligió el problema, se creó el plan, `AGENTS.md`, `CLAUDE.md`, `ESTADO.md` y el README | Siguiente: crear el repo, descargar el ARFF y arrancar el notebook 01 |