# Reglas del proyecto de abandono

Leer este archivo, ESTADO.md y solo la sección pertinente de PLAN.md. Trabajar únicamente en el paso solicitado; no avanzar sin indicación del usuario. Dar un cierre breve y actualizar este ESTADO.md.

## Alcance acordado

- Proyecto nuevo de clasificación binaria sobre Iranian Churn (UCI 563). Churn = 1 es abandono, 0 permanencia.
- Conservar intacto el proyecto sísmico padre. No reutilizar sus resultados, particiones ni modelos como resultados de este proyecto.
- Conservar filas repetidas y mantener perfiles idénticos juntos en las particiones. Definir los grupos usando entradas, nunca la etiqueta. El criterio exacto se documentará antes de reservar test.
- Reservar test una sola vez; usarlo únicamente para evaluación final. Mantener grupos también en validación cruzada.
- Revisar Status y Customer Value antes de decidir su uso; no asumir que sean válidas o filtraciones demostradas.
- F1 de abandono como métrica principal; precisión, recall, precisión promedio y priorización como complementos. Accuracy no decide. No prometer un F1 ni elegir particiones para mejorarlo.
- Escalado, codificación aprendida, selección y remuestreo dentro de pipelines. Ajustar umbral con train, nunca test.
- No modificar los datos originales. No eliminar extremos ni variables por decisiones heredadas del problema sísmico.

## Trabajo y estilo

- Python 3.11, uv, pyproject.toml y uv.lock propios. Nunca pip local. Consultar antes de agregar dependencias.
- Código de análisis en notebooks; app.py para Gradio cuando se solicite. Sin src/ ni módulos propios.
- Seguir las guías de clase antes de crear notebooks: configuración, importaciones, carga, análisis, proceso, resultados y conclusiones. Primeras dos celdas de código: configuración e importaciones. EN_COLAB=False en local.
- Español, markdown antes de cada bloque y conclusiones con números ejecutados. Tablas como DataFrame, no en markdown del notebook.
- Funciones con una responsabilidad y docstring; reciben lo necesario y devuelven resultados. Copias de funciones reutilizadas idénticas. Nombres X_train, y_train, rows, Pipeline y patrones de clase.
- Sin comprehensions, generadores, lambda, clases propias, decoradores ni *args/**kwargs en notebooks. Ciclos simples y random_state=42 cuando aplique. Excepción comentada *valores en app.py para Gradio.
- Guardar figuras con nombres descriptivos. Ejecutar completo con nbconvert todo notebook creado o editado. No afirmar validación en Colab si no se realizó.
- Preguntar ante ambigüedades metodológicas, cambios de alcance o celdas previstas de más de 20 minutos.
- No commits, push, ramas, PR ni Git anidado sin petición; trabajo en main. Si se autoriza commit, solo autoría humana y mensaje corto en español.

Las reglas específicas de sismos (ARFF, mapeos ordinales y selección de rasgos previa) no se trasladan a este problema. Las preferencias generales de clase se conservan. CLAUDE.md importa este archivo.
