# Auditoría del dataset — paso 3

Fecha: 2026-10-09. Reproducible ejecutando `00_Auditoria_Datos.ipynb` completo. No se crearon particiones ni se entrenaron modelos.

## Procedencia

Archivo oficial: `data/original/Customer Churn.csv`; ZIP conservado. Fuente: [UCI 563](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset), DOI 10.24432/C5JW3Z, CC BY 4.0. Los hashes SHA256 y la URL de descarga están en `data/procedencia.json`.

La ficha sitúa las entradas en los primeros nueve meses y la etiqueta al mes doce. No hay fechas individuales ni ID en el CSV que permitan comprobar seguimiento por cliente o generalización temporal. La mención de un ID en el texto de UCI no corresponde a una columna del archivo descargado.

## Resultados comprobados

- 3.150 filas y 14 columnas: 13 entradas más Churn.
- 495 abandonos (15,71 %) y 2.655 permanencias (84,29 %).
- Cero nulos, valores no finitos, columnas constantes o filas con valores negativos.
- 300 repeticiones adicionales de filas completas; 465 filas participan en repeticiones exactas.
- 2.836 perfiles únicos usando todas las entradas, sin Churn. Los 162 grupos repetidos contienen 476 filas; el mayor tiene 11.
- 14 perfiles tienen etiquetas distintas; abarcan 64 filas. Esto no demuestra errores: personas diferentes pueden compartir atributos y tener desenlaces diferentes. No reemplazar etiquetas por mayoría.
- Charge Amount presenta siete registros con valor 10, aunque la ficha indica niveles de 0 a 9. Se conservan sin cambios hasta acordar su tratamiento.
- Age tiene cinco valores distintos. Age Group también tiene cinco; la correspondencia se muestra en el notebook. No interpretar Age como edad exacta sin confirmar cómo se construyó.
- Hay espacios dobles en nombres de columnas. El original permanece igual; la normalización de nombres debe quedar explícita en la preparación.

## Disponibilidad de variables y filtración

**Status:** UCI indica activo/no activo y declara globalmente que las entradas corresponden a los primeros nueve meses. La ficha no define el criterio operativo de inactividad ni cuándo se actualizó este campo para cada cliente. No podemos afirmar filtración demostrada ni garantizar su disponibilidad en una aplicación real.

**Customer Value:** se describe como valor calculado del cliente. No se publica su fórmula, unidad ni detalle de entradas de cálculo en la ficha consultada; su disponibilidad requiere confirmación. No inferimos una fórmula a partir de los datos ni inventamos una interpretación monetaria.

**Recomendación para discutir antes del paso 4:** comenzar el piloto sin Status ni Customer Value por trazabilidad insuficiente, mantener una sola representación de edad si se confirma su equivalencia y conservar el nivel observado 10 de Charge Amount documentando la discrepancia. Son propuestas, no cambios ya aplicados ni una selección basada en rendimiento.

Los perfiles agrupados deberán definirse sobre las entradas que se acuerde usar. Excluir columnas puede unir perfiles antes distintos; los conteos de grupos deberán recalcularse antes de separar. Mantener los grupos también en validación cruzada.

## Límites de esta revisión

Se revisó el archivo completo para integridad y conteos, antes de reservar test. No se calcularon asociaciones predictivas ni rankings por rendimiento. Esta auditoría no certifica calidad causal, ausencia total de filtración o utilidad comercial. Se necesita evidencia externa para cerrar la trazabilidad de las variables ambiguas.

## Artefactos

- `auditoria_resumen.json`: conteos reproducibles.
- `auditoria_columnas.csv`: tipos, rangos, faltantes y variedad.
- `auditoria_dominios.csv`: contraste de códigos con la documentación.
- `auditoria_clases.csv`: distribución del objetivo.
- `diccionario_datos.csv`: significado y observaciones por columna.
- `../figuras/00_distribucion_clase_auditoria.png`: distribución de clases.

Validación local con nbconvert; Colab no probado. Las ocho celdas de código quedaron ejecutadas sin salidas de error; nbconvert terminó con código 0 y guardó el archivo. Hubo advertencias del bucle de eventos y TCP de Jupyter, y en la segunda ejecución un mensaje de libzmq al cerrar el kernel (socket 10038). Se verificaron después la estructura del notebook, sus salidas y el hash intacto del original; no se interpreta ese cierre como un error de las celdas.
