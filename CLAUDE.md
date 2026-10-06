# CLAUDE.md

Las reglas de este proyecto están en `AGENTS.md`, que compartimos con Codex y Antigravity. No las dupliques aquí: si algo cambia, se cambia en `AGENTS.md`.

@AGENTS.md

## Solo para Claude Code

- Al iniciar una sesión, lee `ESTADO.md` y dime en una o dos líneas en qué va el proyecto antes de empezar.
- Para ejecutar notebooks usa siempre `uv run jupyter nbconvert ...` (comando en la sección 8 de `AGENTS.md`).
- Cuando termines una tarea, actualiza `ESTADO.md` antes de darme el resumen.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
