# Predicción de abandono de clientes

Proyecto académico de Fundamentos de Inteligencia Artificial, Universidad EIA.

Buscamos anticipar el abandono registrado al finalizar el mes doce usando información de los primeros nueve meses. Fuente prevista: [Iranian Churn, UCI 563](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset). Etiqueta: `Churn = 1` abandono, `0` permanencia.

Uso propuesto: priorizar contactos de retención. No estimamos si una oferta evitará el abandono ni prometemos ahorro económico.

## Estado

Pasos 1–3 completados: estructura, entorno y dataset oficial auditado. Ver `00_Auditoria_Datos.ipynb` y `resultados/AUDITORIA.md`. Todavía no hay particiones, modelos ni app. Consultar ESTADO.md antes de continuar.

## Entorno local

Desde la carpeta del proyecto sísmico, en PowerShell:

```powershell
cd churn
uv sync --locked
uv run python --version
```

Python 3.11 y dependencias reutilizadas del proyecto anterior, con lock propio. No se agregaron librerías. En VS Code abrir esta carpeta; cuando existan notebooks, seleccionar `.venv\Scripts\python.exe` como kernel. No hace falta activar manualmente el entorno al usar `uv run`.

Usamos el nombre corto `churn/` porque la ruta padre ya es larga y Windows tiene deshabilitadas las rutas largas. Con el nombre completo de carpeta algunos componentes de scikit-learn no podían importarse. Evitar trasladar el proyecto a una ruta más larga.

## Carpetas

- `data/`: futuros originales y particiones.
- `models/`: futuros artefactos entrenados.
- `resultados/`: futuras tablas de evaluación.
- `figuras/`: futuras gráficas.
- `referencia/`: espacio reservado para las guías de clase.
- `00_Auditoria_Datos.ipynb`: comprobación de esquema, calidad y procedencia, sin entrenamiento.
- `informe/` y `presentacion/`: futuros entregables.

La carpeta es independiente en entorno y archivos, pero está dentro del repositorio sísmico: no es un repositorio Git separado. El proyecto anterior se conserva intacto.
