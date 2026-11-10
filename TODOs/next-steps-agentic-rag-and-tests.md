# Next steps — Agentic RAG + tests pendientes

> Nota personal de planificación (no es un spec para el agente).
> Objetivo: tener algo mostrable del agentic RAG el viernes. Los tests no son el
> cuello de botella del demo; el agentic RAG sí.

## Plan

- **Ahora → martes**: scaffold del agentic RAG + una corrida demo sobre un escenario
  multi-hop existente.
- **Miércoles**: multi-thread forget (2 YAMLs + runs) — es barato y suma un resultado.
- **Jueves**: pulir demo + slides (L2 vs L3).
- **Post-viernes**: derived obsolescence (y, si querés, durability/boomerang, que
  sigue siendo más barato que derived).

## Agentic RAG — scope mínimo para el demo

- Nuevo nivel `level_3_agentic_rag` (o `level_2_agentic`), registrado en `agents/__init__.py`.
- Grafo: `decide → retrieve(tool) → model → append`; reusar el `EpisodicMemoryBackend`
  inyectado por `configurable` (Zero Hardcoding).
- Loop acotado: 2–3 iteraciones (reformular query si el modelo pide otra búsqueda).
  No hace falta un ReAct completo para el demo.
- Escenario demo: reusar un multi-hop existente (`in_context_multihop_01`,
  `cross_thread_derived_01` o `top_k_saturation_01`) donde un solo retrieve no alcanza.
- Mostrar L2 vs L3: pass rate, # de llamadas a retrieve, tokens, latencia.

## Tests a implementar (breve)

### Multi-thread forget (miércoles)

- Adaptar `explicit_forget_01` a multi-sesión: en s1 se pide "olvidá X"; en s2/s3 se
  sondea X. L1 no lo tiene ni en contexto; L2 puede filtrarlo por el store append-only.
- Mismo patrón para `adversarial_forget_01/02`: la petición adversarial en s1 y el
  probe del secreto en s2/s3.
- Es edición de YAML (sesiones + assertions) + test de sanidad. Resultado P1/seguridad.

### Derived obsolescence (post-viernes)

- Hecho A se actualiza; el hecho B depende de A; en una sesión posterior se sondea B.
- Correcto: retractar la conclusión derivada vieja (razonamiento no monotónico).
- Diseño: A y B en chunks/sesiones distintos, pin de ambos chunks, separar fallo de
  retrieval de fallo de retracción. Es el más caro de los cuatro.

### Durability / boomerang (opcional, post-viernes; más barato que derived)

- Después del update, re-afirmar el valor viejo (turno posterior o re-ingestar el doc
  obsoleto) y volver a sondear.
- Pregunta conductual: ¿sostiene el update o revierte?

## Estado de los 4 tests de obsolescencia

- ✅ Authority vs recency — creado + corrido (`20261106_122830`).
- ✅ Effective-dates — creado + corrido (`20261106_142741`).
- ❌ Durability / boomerang.
- ❌ Derived cross-session.
