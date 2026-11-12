# TODO — Contrato de telemetría de retrieval para **L2b** (agentic naive-RAG)

> **Status:** ✅ IMPLEMENTED (2026-11-12) — `core/schemas.py`, `core/runner.py`,
> `core/metrics.py`, `agents/__init__.py`; tests in `tests/test_l2b_telemetry.py`
> (22) + the updated `tests/test_harness_m3.py` exact-dict assertion.
> `pytest -q` = **783 passed**. Recorded in
> `TODOs/l2b-sequencing-plan.md` §9.
> **Origen:** `TODOs/l2b-sequencing-plan.md` §1 (Phase 0) y §0 (por qué no "L2b first").
> **Alcance:** `core/schemas.py`, `core/runner.py`, `core/metrics.py`,
> `agents/__init__.py`, `tests/`. Sin cambios en el corpus, en los YAML ni en
> `_format_memory_block`.
> **Fuera de alcance:** store-level assertions (`must_store`), set-recall,
> aserciones numéricas — son los pasos H2/M* de `DIFFICULTY_AUDIT.md`, no este.

### As shipped (2026-11-12)

| Pieza | Qué quedó |
|---|---|
| `L2AgentState` / `TurnResult` | `retrieval_calls`, `retrieval_call_count`, `retrieval_skipped`, `tool_call_cap_hit` |
| `BenchmarkRunSummary` | `total_tool_calls`, `avg_calls_per_retrieval_turn`, `tool_call_cap_hits`, `calls_per_probe`, `retrieval_call_efficiency` |
| `core/metrics.py` | `union_retrieved_episodes()` (unión por `chunk_id`, primera ocurrencia gana), `pins_per_call()`, `mean_calls_per_probe()` |
| `core/runner.py` | lectura guardada (`[]`/`0`/`False`); `evaluate_assertions` recibe la **unión** de todas las llamadas; agregados por corrida |
| `agents/__init__.py` | `LEVEL_ALIASES = {"2b": "level_2b_agentic_rag"}` resuelto **a través** del registro → `--level 2b` falla limpio hoy y funciona el día que el agente exista |

Verificación de no-regresión (criterio §3): un `results.json` L2a pre-existente
(`20261112_132030`) valida contra el esquema nuevo y gana **solo** los campos nuevos en
cero — `pass_rate` 0.5714, `retrieval_recall` 0.6667 y `total_tokens` 5 576 intactos.

### Decisión de grafo adoptada

Un solo nodo (`agent_turn_node`) con el `for` acotado adentro, devolviendo las listas
completas del turno en un único update (§2.2). Preserva la semántica de sobrescritura y
evita el doble conteo por checkpointer que sí tendría un ciclo con reducer.

---

## 1. Por qué es bloqueante

| Problema medido | Evidencia en el repo |
|---|---|
| **`retrieved_memories` es un canal de sobrescritura, no de acumulación** | `core/schemas.py::L2AgentState` lo declara explícitamente: *"Every telemetry channel below is a **per-turn overwrite channel**: no reducer, so each graph invocation replaces the previous turn's value"*. Con un loop `model → tool → model`, **la segunda llamada pisa la primera** y `must_retrieve_chunk` evalúa solo la última ventana. |
| **El runner lee un set fijo de claves** | `core/runner.py::_read_l2_telemetry` hace `list(result.get("retrieved_memories") or [])`. Los tres modos de falla nuevos de L2b — *(i)* no llamar a la tool, *(ii)* llamarla con una query mala, *(iii)* llamadas desperdiciadas que llenan el contexto — **no son representables**. Se depuraría leyendo `results.json` a mano. |
| **Los runs previos quedan inservibles o hay que backfillear** | Precedente pagado: el incidente del denominador de `retrieval_recall` (notas F7) obligó a `scripts/backfill_retrieval_recall.py` y a una columna de "denominator drift". Cada corrida anterior al contrato repite ese costo. |
| **`--level 2b` no resuelve** | `agents/__init__.py::resolve_level_ids`: `prefix = f"level_{token}" if token.isdigit() else token` → `2b` no es dígito, se usa `2b` como prefijo y no matchea ninguna clave. (`--level 2` sí queda inequívoco: el regex `^level_2(?:_|$)` no matchea `level_2b…`.) |

---

## 2. El contrato a congelar

### 2.1 Canales de estado (`L2AgentState`)

| Canal | Tipo | Semántica |
|---|---|---|
| `retrieval_calls` | `list[dict]` | Una entrada **por llamada a la tool** en el turno: `{call_index, query, k, chunk_ids, top_score, latency_ms, embedding_tokens}` |
| `retrieval_call_count` | `int` | `len(retrieval_calls)`; **0 es válido y significa "no llamó"** (F0), no un error |
| `retrieval_skipped` | `bool` | `True` cuando el modelo decidió no llamar. Debe ser observable, no inferido |
| `tool_call_cap_hit` | `bool` | El loop se cortó por `max_tool_calls`, no por decisión del modelo |

`TurnResult` y `BenchmarkRunSummary` espejan los mismos campos; el resumen agrega
`total_tool_calls` y `avg_calls_per_retrieval_turn`.

### 2.2 Decisión de diseño recomendada: un nodo, no un ciclo

**Recomendación:** modelar el loop agéntico como **un solo nodo** (`agent_turn_node`) que
corre el `for` acotado en Python y devuelve las listas completas del turno en un único
update de estado.

- **Por qué:** preserva la semántica de sobrescritura existente (cero cambios en el
  reducer, cero riesgo de acumular a través de turnos de la sesión por el checkpointer),
  y `_read_l2_telemetry` solo necesita leer dos claves nuevas.
- **Alternativa** (si se quiere telemetría por nodo): ciclo con aristas condicionales
  `model → tool → model` + `Annotated[list[dict], operator.add]` en `retrieval_calls`.
  **Costo:** hay que resetear el canal al inicio de cada turno (el checkpointer acumula
  entre turnos de la misma sesión) — un bug silencioso de doble conteo, exactamente el que
  ya se documentó para `embedding_cache_hits` en `retrieve_node`. Si se elige esta vía, el
  reset debe ser explícito y testeado.

### 2.3 Runner

`_read_l2_telemetry` lee las claves nuevas con el **mismo patrón guardado**:
ausente ⇒ `[]` / `0` / `False`. L0/L1/L2 no escriben estos canales y no deben cambiar
**ni un byte** de su `results.json` (salvo los campos nuevos en cero).

### 2.4 Métricas (`core/metrics.py`)

| Métrica | Definición |
|---|---|
| `retrieval_recall` | Sin cambios de política (denominador declarado, unidad regla-por-turno, solo `must_retrieve_chunk`), pero el **match set pasa a ser la unión de todas las llamadas del turno** |
| `calls_per_probe` | `retrieval_call_count` por turno con aserciones positivas |
| `retrieval_call_efficiency` | pines satisfechos / llamadas hechas (0 si no hubo llamadas) |

**Trampa a evitar:** deduplicar `chunk_ids` entre llamadas parece limpio pero **oculta
las llamadas desperdiciadas**, que son justo lo que L2b puede empeorar. Unión para el
match set y conteo de llamadas para la eficiencia, sin colapsar ninguna de las dos.

### 2.5 CLI

Agregar un alias explícito (`{"2b": "level_2b_agentic_rag"}`) en `resolve_level_ids`
**antes** del fallback por prefijo, y un test que verifique: `--level 2b` → exactamente
`level_2b_agentic_rag`, y `--level 2` → exactamente `level_2_episodic_rag` (que `2` no
arrastre a `2b`).

---

## 3. Criterios de aceptación

- [ ] `pytest -q` verde, con tests nuevos para: canal ausente (L0/L1 → `0`/`False`, sin `KeyError`), turno con 3 llamadas → `retrieval_call_count == 3`, y `must_retrieve_chunk` matcheando chunks de la **llamada 1 y de la 3** (la unión, no solo la última).
- [ ] Una corrida L2a posterior al cambio es idéntica a una previa **salvo** los campos nuevos en cero (diff de `results.json`).
- [ ] `retrieval_recall` de un turno multi-llamada == valor calculado sobre la unión; el flag de **denominator drift** sigue funcionando.
- [ ] `--level 2b` resuelve; `--level 2` sigue resolviendo a un único agente.
- [ ] `scripts/difficulty_report.py` y `scripts/retrieval_recall.py` no se rompen con los campos nuevos (leen por clave, pero verificar igual).
- [ ] `_format_memory_block` **sin tocar**: el payload del prompt tiene que ser idéntico al de L2a para que la única variable del experimento sea la decisión agéntica.

---

## 4. Lo que NO hay que hacer

- **No** cambiar `_format_memory_block` ni el orden del payload (rompe el aislamiento L2a vs L2b y el prefix-cache).
- **No** agregar `add` a `retrieved_memories`: hay 4 corridas y 3 scripts leyendo esa clave; el contrato nuevo va en claves nuevas.
- **No** backfillear `retrieval_calls` en corridas viejas: no existió; un backfill inventaría llamadas.
- **No** mezclar esto con la reescritura del generador de corpus (M9): son dos compromisos de una sola vía, y este va primero.

---

## 5. Secuencia de implementación sugerida

1. `core/schemas.py` — canales nuevos en `L2AgentState` + espejo en `TurnResult`/`BenchmarkRunSummary`.
2. `core/runner.py` — lectura guardada + agregados por corrida.
3. `core/metrics.py` — unión del match set + `calls_per_probe` + `retrieval_call_efficiency`, con tests unitarios puros (sin modelos).
4. `agents/__init__.py` — alias `2b` + test de resolución.
5. `agents/level_2b_agentic_rag/` — el grafo (una vez que 1–4 están verdes).

Estimación: **½–1 día** para 1–4, sin gastar tokens de modelo.
