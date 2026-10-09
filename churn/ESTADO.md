# Estado del proyecto

Actualizado: 2026-10-09.

- [x] Paso 1: alcance acordado.
- [x] Paso 2: estructura y entorno verificados.
- [x] Paso 3: descargar y auditar dataset; notebook ejecutado en local.
- [x] Paso 4: partición por grupos reservada y verificada.
- [x] Paso 6: notebook 01 completado y ejecutado en local; Colab pendiente.
- [ ] Paso 5: viabilidad con modelos sencillos, pendiente.
- [ ] Pasos 7–11: pendientes.

## Decisiones vigentes

- Proyecto independiente dentro del espacio de trabajo actual; sismos intacto.
- Dataset previsto: Iranian Churn, UCI 563; objetivo binario de abandono.
- Conservar repeticiones y separar perfiles idénticos por grupos, autorizado por el usuario.
- Status y Customer Value excluidas inicialmente, según acuerdo en chat. Se conservan las once entradas restantes, incluidos edad y grupo de edad; su redundancia se revisará en el notebook 02.
- Se conservan las siete filas con cargo de nivel 10, sin corregirlas ni eliminarlas sin evidencia.
- Partición existente reutilizada: 2.511 filas de entrenamiento y 639 de prueba, sin perfiles compartidos. Grupos definidos por las once entradas, nunca por abandono. Primera partición de StratifiedGroupKFold de cinco pliegues, semilla 42.
- EDA realizado únicamente sobre entrenamiento. No hay modelos entrenados ni evaluación predictiva de prueba.
- Cada paso requiere indicación del usuario. No crear notebooks o app durante la preparación del entorno.
- Carpeta `churn/`: nombre corto para evitar el límite de rutas de Windows. No se modificó la configuración del sistema.

## Bitácora

- 2026-10-09: revisado 01_EDA_Limpieza_Churn.ipynb a petición del usuario. Completadas explicaciones y conclusiones con resultados, verificación del original, cobertura de la partición y coherencia de CSV existentes. Documentada correspondencia edad/grupo de edad y guardadas tablas del EDA. Ejecución completa con nbconvert en local; 392 abandonos en entrenamiento (15,61 %), phik de quejas 0,747, segundos de uso 0,464 y frecuencia de uso 0,435. Reserva y CSV originales conservados. Windows requirió ejecución fuera del sandbox para iniciar Jupyter. Colab no probado.

- 2026-10-09: paso 3 completado. Dataset oficial conservado con hashes; auditoría en 00_Auditoria_Datos.ipynb y resultados/AUDITORIA.md. 3.150 filas, 495 abandonos, cero nulos, 300 repeticiones adicionales, 14 perfiles con etiquetas distintas y 7 cargos de nivel 10 fuera de la descripción. Copiadas dos referencias de clase sin modificarlas. Status/Customer Value siguen sin trazabilidad suficiente: propuesta de exclusión inicial, pendiente de acuerdo. No se dividieron datos ni entrenaron modelos. Colab no probado.

- 2026-10-09: paso 2 terminado. Python 3.11.9, entorno `.venv` propio y `uv sync --locked` completado. Verificados los 13 imports principales (incluidos scikit-learn, imbalanced-learn, SHAP, Gradio, ipykernel y nbconvert), misma lista de dependencias que el proyecto anterior y ausencia de CSV/notebooks. El nombre largo impedía cargar un componente de scikit-learn; se recreó el entorno en `churn/` y los imports pasaron. Sin descargas de datos, entrenamiento ni modificaciones del proyecto sísmico. Colab todavía no probado.

Siguiente tarea: paso 5, viabilidad con modelos sencillos y validación cruzada por grupos solo en entrenamiento. El usuario adelantó el notebook 01 respecto al orden inicial del plan. No avanzar sin indicación del usuario.
