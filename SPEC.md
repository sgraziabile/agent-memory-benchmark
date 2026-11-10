# SPEC.md: Especificación Técnica del Arnés de Evaluación Experimental (agent-memory-benchmark)

Este documento constituye la **Fuente Única de Verdad (Single Source of Truth)** para el diseño, scaffolding e implementación del banco de pruebas experimental `agent-memory-benchmark`.

---

## 1. Contexto Académico e Hipótesis de Investigación

* **Proyecto:** Tesis de Licenciatura en Ciencias de la Computación (DCIC - Universidad Nacional del Sur).
* **Lugar de Trabajo:** Laboratorio ICIC (CONICET - UNS), Grupo de Representación de Conocimiento y Razonamiento (KRR).
* **Dirección:** Dr. Alejandro J. García | **Codirección:** Dr. Sebastián Gottifredi.
* **Metodología:** *Evaluation-Driven Development* (EDD).
* **Objetivo Teórico:** Medir empíricamente la degradación de consistencia lógica y evaluar mecanismos de memoria y revisión de creencias (*Belief Revision*) en arquitecturas deliberativas basadas en LLMs y LangGraph.

### Preguntas de Investigación Guía
1. **P1:** ¿Cuál es la arquitectura mínima necesaria para gestionar persistencia contextual entre múltiples sesiones (*cross-thread*) de forma fiable?
2. **P2:** ¿Cómo afecta el desacoplamiento de un mecanismo de reflexión (*LLM-as-a-Judge*) frente a información contradictoria o extinta ($A$ vs. $\neg A$)?
3. **P3:** ¿Qué penalización en latencia y sobrecosto de tokens introduce la persistencia estructurada frente a un baseline reactivo o amnésico?

---

## 2. Principio Arquitectónico del Arnés ("El Rompecabezas")

El sistema desacopla los componentes para garantizar reproducibilidad científica. Cada ejecución unitaria del benchmark se formaliza como una tupla inmutable:

$$\langle \text{Escenario / Conversación},\, \text{System Prompt},\, \text{Modelo LLM},\, \text{Nivel de Persistencia} \rangle$$

### Reglas de Diseño
1. **Zero Hardcoding:** Ningún grafo o nodo de LangGraph debe tener instanciado un modelo o prompt en su definición interna.
2. **Inyección Dinámica:** Todo parámetro de ejecución se pasa en tiempo de ejecución a través del argumento `config={"configurable": {...}}` (`RunnableConfig`).
3. **Aislamiento de la Persistencia:** La arquitectura de control y los bindings de herramientas se mantienen congelados al cambiar el nivel de memoria evaluado.

---

## 3. Estructura de Directorios del Repositorio

```
agent-memory-benchmark/
├── configs/
│   ├── models.yaml                 # Registro unificado de proveedores y modelos
│   └── prompts.yaml                # Banco de system prompts e invariantes epistémicas
├── datasets/
│   ├── documents/                  # Corpus documental (WS-D): fixtures .md + README + generated/
│   └── conversations/              # Casos YAML: <hilo>/<tema>/<categoría>/
│       ├── single_thread/          # Un solo session_id (18): L0/L1 pueden responderlos
│       └── multi_thread/           # Varios session_id (15): requieren memoria cross-thread
├── core/
│   ├── __init__.py
│   ├── dataset_layout.py            # Layout del dataset + clasificación hilo/tema/categoría
│   ├── documents.py                # Resolución, lectura y records de documentos (WS-D)
│   ├── model_factory.py            # Instanciador dinámico multi-proveedor (LLMs y Embeddings)
│   ├── schemas.py                  # Modelos Pydantic y TypedDicts para Estado y Tests
│   └── runner.py                   # Orquestador del producto cartesiano y logging
├── agents/
│   ├── __init__.py
│   ├── base.py                     # Interfaz abstracta BaseBenchmarkAgent
│   ├── level_0_reactive/           # L0: Baseline amnésico (Stateless)
│   │   ├── __init__.py
│   │   └── graph.py
│   ├── level_1_working_memory/     # L1: Memoria de trabajo intra-hilo (MemorySaver)
│   │   ├── __init__.py
│   │   └── graph.py
│   └── level_2_episodic_rag/       # L2: Memoria episódica cross-thread (Naive RAG)
│       ├── __init__.py
│       ├── graph.py                # retrieve → model → append
│       └── backends/               # Contrato EpisodicMemoryBackend + VectorMemoryBackend
├── outputs/
│   ├── runs/                       # Resultados crudos (JSON), consolidados (CSV) y resúmenes (MD)
│   └── by_theme/                   # Snapshot histórico congelado (ya no se regenera)
├── tests/                          # Pruebas deterministas de integración del arnés (pytest)
├── pyproject.toml                  # Dependencias y configuración de packaging
└── SPEC.md                         # Este documento de especificación
```

---

## 4. Escalera de Ablación de Persistencia (`L0`–`L4`)

El arnés evalúa agentes a través de cinco niveles progresivos donde cada nivel introduce un único mecanismo de memoria:

| Nivel | Identificador | Mecanismo de Persistencia | Alcance y Comportamiento | Estado |
|:-----:|---------------|---------------------------|--------------------------|:------:|
| **L0** | `level_0_reactive` | Ninguno (*Stateless*) | Baseline amnésico; solo dispone del turno actual. | ✅ Implementado |
| **L1** | `level_1_working_memory` | `MemorySaver` (*Checkpointer*) | Memoria de trabajo **intra-hilo** (`thread_id`); pierde el contexto entre sesiones. | ✅ Implementado |
| **L2** | `level_2_episodic_rag` | Memoria Episódica Acumulativa (*Naive RAG*) | Persistencia **inter-sesión** (*cross-thread*) **append-only** (sin edición ni borrado) sobre backend **VectorDB** (el backend `.md`/BM25 queda fuera de alcance). Ingesta dos canales: turnos de chat e **ingesta de documentos** (§5). | ✅ Implementado (backend `vector`) |
| **L3** | `level_3_reflective_memory` | Memoria con Mutación (CRUD) y Reflexión | Incorpora operadores de actualización/eliminación y un mecanismo *LLM-as-a-Judge* para resolver contradicciones ($A$ vs. $\neg A$). | 🔲 Planificado |
| **L4** | `level_4_belief_state` | Estado de Creencias Estructurado | Representación simbólica/estructurada de hechos y reglas con operadores explícitos de revisión de creencias. | 🔲 Planificado |

---

## 5. Ingesta de Documentos a Memoria (WS-D)

Los documentos son el **segundo canal de ingesta** del arnés: se fragmentan y
se anexan al backend episódico activo con `source="document"` **sin invocar
nunca al modelo de chat**. El costo de ingesta es exclusivamente de embeddings.

### 5.1 Canales

| Canal | Declaración YAML | Momento de ingesta | `session_id` / `turn_number` |
|---|---|---|---|
| **Precarga de corpus** | `documents:` (lista de `doc_id` + `path` \| `content`) | Una vez por tupla, **antes del turno 1** | `corpus` / `0` |
| **Evento de sistema** | `role: system_event` + `attachments:` (+ `content` inline) | Al alcanzar ese turno de la secuencia | sesión del turno / `turn_number` del turno |

```yaml
documents:
  - doc_id: policy_v1
    path: policy_v1.md            # relativo al YAML, luego datasets/documents/
    metadata: {title: "Access Policy"}

turns:
  - turn_number: 2
    session_id: s1
    role: system_event
    content: "Compliance publicó el Addendum 1; deroga la sección 3.2."
    attachments: ["addendum_1.md"]
```

### 5.2 IDs de chunk deterministas

| Canal | `chunk_id` |
|---|---|
| Precarga | `doc_<doc_id>` (el backend añade `_c<i>` al fragmentar) |
| `system_event` (archivo) | `doc_<stem>_t<turn>` |
| `system_event` (contenido inline) | `doc_evt_<session>_t<turn>` |

Los IDs son estables entre corridas, de modo que reingestar el mismo documento
es idempotente. Reingestar el mismo `doc_id` con **contenido distinto** viola la
restricción *append-only* del backend y falla de forma explícita: es
exactamente la colisión que L3 deberá resolver con operadores `UPDATE`/`DELETE`.

### 5.3 Semántica por nivel de ablación

| Nivel | Precarga | `system_event` | Consecuencia |
|---|---|---|---|
| **L0** | se ignora (+warning) | no-op histórico | los probes documentales fallan (piso de la ablación) |
| **L1** | se ignora (+warning) | no-op histórico | falla: el documento no vive en el hilo del checkpointer |
| **L2+** | anexa (`source="document"`) | anexa en ese turno | recuperable desde cualquier sesión posterior |

### 5.4 Telemetría y artefactos

- `TurnResult`: `ingested_documents`, `ingested_chars`, `ingested_chunk_ids`
  (además de `append_latency_ms` / `embedding_latency_ms` / `embedding_tokens`).
- `BenchmarkRunSummary`: `documents_ingested`, `ingested_chars`,
  `ingested_chunks`, `corpus_chunks`, `corpus_chars`, `ingestion_latency_ms`.
- Los turnos `system_event` **nunca** entran en el denominador de aserciones ni
  en `avg_latency_per_turn_ms` (solo turnos `role="user"`).
- Snapshot (`outputs/runs/<ts>/memory/<run_id>/`): `manifest.json` (incluye
  `by_source`, `document_chunks` y `run_metadata.documents` con `sha256`),
  `episodes.json` y `documents.json`.

### 5.5 Controles de CLI

`--top-k` (0 desactiva la recuperación), `--chunk-size`, `--chunk-overlap` y
`--no-documents` (corrida de control con corpus vacío). Los cuatro valores
quedan registrados en `results.json`/`summary.csv` como identidad del run.

### 5.6 Fixtures y generación

`datasets/documents/` contiene el corpus versionado (`policy_v1.md`,
`addendum_1.md`, `ops_handbook.md`) y su README. Para corpus grandes y
reproducibles, `scripts/generate_corpus.py` genera N documentos deterministas
a partir de una semilla, con una *needle* plantada en un documento conocido y
un `corpus_manifest.json` con `sha256` por archivo.


### 5.7 Aserciones de recuperación (opcionales)

Además de las aserciones textuales (`must_contain` / `must_not_contain`),
`AssertionRule` admite reglas evaluadas contra la telemetría de recuperación
del turno (`retrieved_memories`), lo que vuelve **causal** un fallo de
`top_k`:

| Tipo | Semántica |
|---|---|
| `must_retrieve_chunk` / `must_not_retrieve_chunk` | substring (case-insensitive) del `chunk_id`; `doc_policy_v1` también coincide con sus sub-chunks `doc_policy_v1_c2` |
| `must_retrieve_source` / `must_not_retrieve_source` | etiqueta exacta de origen (`chat` / `document`) |

En L0/L1 no hay recuperación: las reglas positivas fallan y las negativas
pasan de forma determinista. Los detalles exportados incluyen
`matched_chunks` con los IDs que satisfacieron la regla.

### 5.8 Métrica de recall de recuperación

`retrieval_recall` responde *"¿qué fracción de los pines de recuperación
declarados se satisfizo?"*:

```
retrieval_recall = pines must_retrieve_chunk satisfechos / pines declarados
```

Tres decisiones de política definen el número, y el arnés las aplica de forma
idéntica en la ejecución (`core/metrics.py`) y en el reporte posterior
(`scripts/retrieval_recall.py`):

| # | Decisión | Razón |
|---|---|---|
| **D1** | **Denominador declarado**: se cuentan los pines del YAML del escenario (`turns[*].assertions` con `role: user`), no las aserciones evaluadas. Un turno con `status: error` **permanece** en el denominador y cuenta como *miss*. | Los turnos con error nunca llegan a `evaluate_assertions`; con denominador evaluado un turno en blanco o truncado *mejora* el recall (artefacto F7 en `notes/2026-11-06_l2_authority_vs_recency.md`). |
| **D2** | **Unidad = una regla en un turno**. | Seis escenarios fijan el mismo documento oro en varios turnos a propósito (`doc_addendum_1` ×3 en `document_addendum_revision_01`); deduplicar por documento eliminaría turnos en lugar de medirlos. |
| **D3** | **Solo `must_retrieve_chunk`**: `must_not_retrieve_*` (guardas de precisión) y `must_retrieve_source` (canal, no ítem oro) quedan fuera del numerador y del denominador. | Incluir guardas negativas permitiría que un agente amnésico obtenga 100 % sin recuperar nada. |

**Alcance del número.** `must_retrieve_chunk` es un *substring* del `chunk_id`,
así que un sub-chunk hermano del documento oro satisface el pin: el resultado es
recall **a nivel documento**. El `Recall@k` **a nivel chunk** (el chunk
recuperado debe contener el token de respuesta) es otra métrica, calculada por
`scripts/lr_report.py`. No son intercambiables.

**Exportación.** Cada `BenchmarkRunSummary` incluye:

| Campo | Significado |
|---|---|
| `retrieval_assertions_total` | pines declarados (denominador) |
| `retrieval_assertions_evaluated` | pines efectivamente evaluados; `total − evaluated` son pines perdidos por error de infraestructura |
| `retrieval_assertions_passed` | pines satisfechos |
| `retrieval_recall` | `passed / total`, o `null` si el escenario no declara pines |
| `retrieval_recall_ci_low` / `_high` | intervalo de Wilson 95 % |

Los contadores viajan en `results.json`, `summary.csv`, en la columna
`Retrieval Recall` de `summary.md` (que solo aparece si algún run declara pines)
y en la salida de consola. L0/L1 con pines puntúan `0.0` sobre el denominador
declarado (no recuperan nada), nunca `null`.

Para runs históricos anteriores a estos campos:

```bash
python scripts/retrieval_recall.py --scenario-prefix document_ --verbose
python scripts/retrieval_recall.py --csv outputs/retrieval_recall.csv
```

El script recupera el denominador declarado desde los YAML cuando el
`results.json` no lo trae (columna `denom`: `run` / `yaml` / `evaluated`) y, con
`--verbose`, lista cada pin perdido (directorio del run, turno y `value`).

Un `!` junto al recall marca los runs cuyo escenario **creció** después de
ejecutarse (el YAML declara hoy más pines de los que el run evaluó, sin turnos
con error): el denominador de la ejecución no es recuperable del registro, la
cifra reportada es una **cota inferior** y no debe compararse con runs de la
versión actual del escenario. La columna `denominator_drift_runs` del CSV
cuenta cuántos runs del grupo están en esa situación.

**Runs anteriores a la métrica.** Los seis campos solo los escribe el runner al
exportar, así que los `results.json` previos no los tienen. El backfill los
añade en sitio, sin tocar ninguna otra clave:

```bash
python scripts/backfill_retrieval_recall.py --dry-run   # solo reporta
python scripts/backfill_retrieval_recall.py             # escribe los campos
```

Por cada directorio de run: inserta los seis campos en cada resumen de
`results.json` (después de `pass_rate`), reescribe `summary.csv` con el mismo
conjunto y orden de columnas que `_export_results`, guarda los originales como
`*.bak` (una sola vez, nunca se sobrescriben) y escribe
`retrieval_recall_backfill.json` con la fuente y los `run_id` con deriva de
denominador. `summary.md` es una vista previa y no se toca: se regenera en la
próxima ejecución real. Directorios sin pines quedan intactos; el proceso es
idempotente.
