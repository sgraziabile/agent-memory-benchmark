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
│   └── conversations/              # Casos de prueba YAML, organizados por tema
│       ├── logical_consistency/    # belief_revision, explicit_forget, in_context_control, negative_hallucination
│       ├── retention_persistence/  # attrition, in_context_needle, needle_haystack
│       ├── compositional_reasoning/ # in_context_multihop, temporal_multi_hop
│       ├── robustness_security/    # adversarial_forget
│       └── computational_cost/     # pruebas de sobrecosto de latencia y tokens (P3)
├── core/
│   ├── __init__.py
│   ├── model_factory.py            # Instanciador dinámico multi-proveedor (LLMs y Embeddings)
│   ├── schemas.py                  # Modelos Pydantic y TypedDicts para Estado y Tests
│   └── runner.py                   # Orquestador del producto cartesiano y logging
├── agents/
│   ├── __init__.py
│   ├── base.py                     # Interfaz abstracta BaseBenchmarkAgent
│   ├── level_0_reactive/           # L0: Baseline amnésico (Stateless)
│   │   ├── __init__.py
│   │   └── graph.py
│   └── level_1_working_memory/     # L1: Memoria de trabajo intra-hilo (MemorySaver)
│       ├── __init__.py
│       └── graph.py
├── outputs/
│   ├── runs/                       # Resultados crudos (JSON), consolidados (CSV) y resúmenes (MD)
│   └── by_theme/                   # Corridas reorganizadas por eje temático
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
| **L2** | `level_2_episodic_rag` | Memoria Episódica Acumulativa (*Naive RAG*) | Persistencia **inter-sesión** (*cross-thread*) **append-only** (sin edición ni borrado) sobre dos backends intercambiables: **VectorDB** y **archivos `.md`**. | 🚧 En planificación ([TODOs/phase-2-l2-plan.md](TODOs/phase-2-l2-plan.md)) |
| **L3** | `level_3_reflective_memory` | Memoria con Mutación (CRUD) y Reflexión | Incorpora operadores de actualización/eliminación y un mecanismo *LLM-as-a-Judge* para resolver contradicciones ($A$ vs. $\neg A$). | 🔲 Planificado |
| **L4** | `level_4_belief_state` | Estado de Creencias Estructurado | Representación simbólica/estructurada de hechos y reglas con operadores explícitos de revisión de creencias. | 🔲 Planificado |