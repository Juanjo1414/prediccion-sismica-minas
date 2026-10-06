# ESTADO.md — En qué va el proyecto

> Este archivo lo actualiza quien termine una tarea (persona o agente). Es la forma de que todos sepamos en qué va el proyecto sin tener que preguntar.
> Regla: al terminar, marca el avance, anota cualquier decisión nueva y agrega una línea a la bitácora (la más reciente arriba).

**Última actualización:** 2026-10-05

## Avance

### Preparación
- [ ] Repositorio creado en GitHub y primer push
- [ ] Entorno con uv funcionando en local (`uv sync`)
- [ ] `data/seismic-bumps.arff` oficial descargado de UCI
- [ ] Notebooks de clase copiados en `referencia/`

### Notebooks
- [ ] 01 — Carga, EDA, limpieza, codificación y partición
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

## Resultados clave (se llenan a medida que avanzamos)

- **Filas del ARFF oficial:** _pendiente_ (UCI dice 2.584)
- **Duplicados eliminados:** _pendiente_
- **Columnas constantes eliminadas:** _pendiente_ (se esperan `nbumps6`, `nbumps7`, `nbumps89`)
- **Escalador elegido:** _pendiente_
- **Rasgos elegidos:** _pendiente_
- **Mejor combinación modelo + balanceo (CV):** _pendiente_
- **Modelo final, umbral y F1 en test:** _pendiente_

## Preguntas abiertas

- [ ] Nombres y roles de los integrantes (para el README y el informe).
- [ ] Confirmar con la profesora que un F1 modesto en la clase peligrosa es aceptable si está bien analizado (el problema es difícil de por sí).
- [ ] Formato y extensión del informe y duración de la presentación.

## Bitácora

| Fecha | Quién | Qué se hizo | Resultado / siguiente paso |
| :--- | :--- | :--- | :--- |
| 2026-10-05 | Juan | Se eligió el problema, se creó el plan, `AGENTS.md`, `CLAUDE.md`, `ESTADO.md` y el README | Siguiente: crear el repo, descargar el ARFF y arrancar el notebook 01 |