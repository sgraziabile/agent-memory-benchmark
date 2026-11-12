# Suite `multi_thread` — clasificación de dificultad de extracción (D1–D3, FEB-30) y métricas empíricas

> **Fecha:** 2026-11-12
> **Alcance:** los 19 escenarios de `datasets/conversations/multi_thread/**` y sus
> 60 corridas en `outputs/runs/` (10 modelos; 58 L2 + 1 L0 + 1 L1). Con
> `--with-legacy` se suman 4 corridas de `outputs/by_theme/` que no están en el
> layout canónico (ver §3.4).
> **Objeto:** clasificar cada dataset en las bandas de dificultad **de extracción** del
> benchmark FEB-30 (Toma et al.) midiendo cuatro ejes objetivos, y adjuntar
> pass rate, recall de retrieval y tokens totales por celda.
> **Fuentes:** `datasets/conversations/multi_thread/**/*.yaml`,
> `outputs/runs/*/results.json`, `outputs/by_theme/**/results.json` (legacy),
> `core/documents.py` (resolución de fixtures), `core/metrics.py` + SPEC §5.8 (recall).
> **Reproducible con:** `python scripts/difficulty_report.py --group multi_thread`
> (§8). Todas las cifras de esta nota salen de esa corrida; ninguna es estimada.
> **Relacionado:** [`DIFFICULTY_AUDIT.md`](../DIFFICULTY_AUDIT.md) §A1–A8 (bandas vacías,
> ruido mal tipado, corpus sintético), §L2 (colisión de nomenclatura D1/D2/D3),
> [`notes/2026-10-30_l2_episodic_rag_summary.md`](2026-10-30_l2_episodic_rag_summary.md).

---

## 0. Resumen ejecutivo

- **La suite no está calibrada a FEB-30.** De 19 datasets multihilo, **2 están en banda
  D1**, **0 en D2** y **0 en D3**. Los otros 17 quedan fuera por volumen (arriba o abajo),
  nunca por dificultad de extracción.
- **Ningún escenario cumple el criterio D3 de inyección híbrida.** D3 pide inyectar los
  hechos dentro de documentos humanos reales de dominios ajenos (~90 % de texto auténtico
  irrelevante). Los 19 usan corpus **sintéticos generados por plantillas**; el de 60
  documentos tiene 3 159 oraciones de las que 212 son únicas (**6.7 %**, la relación inversa
  de lo que pide D3). Inyección híbrida sobre prosa humana real: **0/19**.
- **La señal más valiosa no es el tier, es el desacople pass/recall.**
  `document_split_binding_01` y `document_split_evidence_01` tienen **recall 8/8 = 100 %** y
  **pass 66.7 % / 71.9 %**: el retrieval llega perfecto y lo que falla es el *binding* entre
  documento y esquema. Ese es exactamente el constructo que D3 quiere medir, y es lo único
  del repo que hoy lo aísla.
- **Agregado del bloque (60 corridas):** pass **555/627 = 88.5 %**, recall
  **173/200 = 86.5 %**, tokens **501 417**. Restringido a L2 (58 corridas): pass
  **543/599 = 90.7 %**, recall **173/192 = 90.1 %**, tokens **499 987**.
- **`document_effective_dates_01` es el único dataset con ablación L0/L1/L2 limpia**
  (0.429 / 0.429 / 0.875) → el único del bloque que informa sobre el eje de memoria y no
  solo sobre el de extracción.

---

## 1. Método, definiciones y política de clasificación

### 1.1 Los cuatro ejes (medidos, no estimados)

| Eje | Definición operativa | Por qué así |
|---|---|---|
| `documents` | `len(scenario.documents)`; cada `path` se resuelve con `core.documents.load_document` | es el resolutor que usa el runner, así un fixture irresoluble se reporta (`unresolved_documents`) en vez de contarse como cero |
| `words` | `\S+` sobre todos los `turns[].content` **más** el texto completo de cada documento resuelto | es la unidad en la que están escritas las bandas de FEB-30 |
| `states` | cantidad de `session_id` distintos | la misma cantidad que `core.dataset_layout.thread_dir_for_sessions` usa para clasificar un escenario como `multi_thread`, así las dos no pueden discrepar |
| `noise` | documentos que **ningún** `must_retrieve_chunk` nombra (corpus irrelevante) **+** turnos `role: user` sin aserciones (relleno) | las dos son contexto que el probe no necesita |

**Caveat del eje `noise`, y es importante.** Un turno de relleno compite por *atención*,
no por la respuesta, así que **no** es una afirmación-distractor en el sentido de FEB-30
(`DIFFICULTY_AUDIT.md` §A2). La mitad honesta del eje es `noise_documents`; la salida
`--verbose` imprime los dos componentes por separado para que el corte nunca quede oculto.

### 1.2 Bandas (FEB-30)

```
D1  explícito / literal      1-2 docs     200-400 palabras   2-3 estados   0 ruido
D2  contextual / embebido    3-4 docs     400-800 palabras   3-4 estados   2-3 ruido
D3  inferencial / inyección  4-5 docs    600-1000 palabras   4-5 estados   4-5 ruido
                                                          (+ ~90 % texto humano real ajeno)
```

### 1.3 Política de asignación

Un escenario está **en banda** `Dk` cuando satisface **≥3 de los 4 ejes** de ese tier.
Los empates van al tier más bajo, de modo que el tier más duro nunca se reclama gratis.
Cuando ningún tier llega al umbral, el escenario queda **fuera de banda** y se enumeran las
violaciones eje por eje: `documents>D3` (arriba del techo más alto), `words~D2` (cae dentro
de D2 pero no del tier decisivo), `noise<D1` (abajo del piso más bajo).

### 1.4 Aviso de nomenclatura

`D1`/`D2`/`D3` significan **dos cosas distintas** en este repo y conviene no mezclarlas:

- **FEB-30 (esta nota):** tiers de dificultad de *extracción*.
- **SPEC §5.8 y `core/metrics.py`:** etiquetas de política del recall (denominador
  declarado / unidad reglada / solo `must_retrieve_chunk`). El renombre a `P1/P2/P3`
  sigue abierto (`DIFFICULTY_AUDIT.md` §L2).

### 1.5 Métricas empíricas

| Columna | Definición | Fuente |
|---|---|---|
| `pass` | `Σ passed_assertions / Σ total_assertions` agrupado por `scenario_id` (pooled, **nunca** media de medias) | `results.json` |
| `recall` | `Σ must_retrieve_chunk passed / Σ declarados`, denominador declarado, unidad regla-por-turno | `scripts.retrieval_recall.run_stats` (misma política que SPEC §5.8) |
| `tokens` | `Σ total_tokens` por corrida | `core.runner` |

El `!` junto al recall marca **denominator drift**: el escenario declara hoy más pines de
los que esa corrida evaluó y ningún turno erroreó, así que la cifra es una **cota
inferior** sobre el denominador actual. Dos casos hoy:

| Escenario | Causa | Efecto |
|---|---|---|
| `document_cross_thread_01` | los 2 pines por corrida se agregaron en `64585bf` (2026-11-11), después de las corridas del 2026-11-02 | `0/4` es un artefacto, no un fallo de retrieval |
| `document_dilution_01` | el pin `doc_power_audit_scope` se agregó el 2026-11-02, después de 2 de las 6 corridas | `22/24` en vez de `22/22` |

### 1.6 Qué NO mide este análisis

- **Dificultad de razonamiento.** `metadata.difficulty` (`basic` 5 / `moderate` 6 /
  `hard` 8) codifica dificultad de *razonamiento*, no de extracción: `overhead_baseline_01`
  es `basic` y D1 puro; `document_authority_vs_recency_01` es `hard` y D3-ish en inferencia
  pero D1 en extracción.
- **Calidad del ruido.** El eje cuenta volumen de contexto irrelevante; si ese ruido
  *compite léxicamente* por la respuesta es un campo que el esquema todavía no tiene
  (propuesta en `DIFFICULTY_AUDIT.md` §A4: `near_duplicate_competition`).
- **Efecto de modelo.** El pass rate pooled mezcla hasta 4 modelos por celda. Cualquier
  conclusión por celda necesita agrupar por `(scenario, model, prompt)` — recomendación L3
  de la auditoría, todavía sin implementar.

---

## 2. Tabla maestra — los 19 datasets multihilo

`pass` y `recall` son pooled sobre las corridas de ese escenario (`outputs/runs/`).
`-` = no hay pines de recuperación declarados (recall no aplica) o no hay corridas.

| Dataset | tier | in | docs | palabras | estados | ruido | runs | pass rate | recall | total tokens |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `backward_modus_tollens_01` | - | 2/4 | 0 | 308 | 3 | 11 | 3 | 3/6 &nbsp;50.0 % | - | 26 002 |
| `cross_thread_01` | - | 1/4 | 0 | 198 | 2 | 8 | 0 | - | - | - |
| `cross_thread_derived_01` | - | 2/4 | 0 | 256 | 3 | 8 | 0 | - | - | - |
| `derived_belief_revision_01` | - | 2/4 | 0 | 284 | 3 | 11 | 1 | 1/3 &nbsp;33.3 % | - | 8 538 |
| **`document_addendum_revision_01`** | **D1** | 3/4 | 1 | 517 | 2 | 0 | 5 | 37/40 &nbsp;92.5 % | 18/20 &nbsp;90.0 % | 20 386 |
| `document_authority_vs_recency_01` | - | 1/4 | 60 | 52 064 | 3 | 61 | 4 | 46/48 &nbsp;95.8 % | 16/16 100 % | 65 279 |
| `document_conflict_dilution_01` | - | 1/4 | 14 | 5 785 | 2 | 12 | 2 | 18/18 100 % | 6/6 100 % | 14 447 |
| `document_cross_thread_01` | - | 2/4 | 1 | 579 | 2 | 2 | 2 | 10/10 100 % | 0/4 &nbsp;0.0 %`!` | 14 555 |
| `document_dilution_01` | - | 1/4 | 9 | 420 | 2 | 6 | 6 | 49/50 &nbsp;98.0 % | 22/24 &nbsp;91.7 %`!` | 20 440 |
| `document_effective_dates_01` | - | 1/4 | 61 | 51 988 | 3 | 59 | 10 | 110/140 &nbsp;78.6 % | 28/40 &nbsp;70.0 % | 42 281 |
| `document_multihop_bridge_01` | - | 1/4 | 14 | 5 944 | 2 | 12 | 1 | 8/10 &nbsp;80.0 % | 3/4 &nbsp;75.0 % | 6 131 |
| `document_needle_01` | - | 2/4 | 2 | 1 037 | 2 | 3 | 2 | 12/12 100 % | 4/4 100 % | 14 065 |
| `document_prevalence_update_vs_distractor_01` | - | 1/4 | 60 | 52 091 | 3 | 60 | 10 | 138/146 &nbsp;94.5 % | 44/50 &nbsp;88.0 % | 156 479 |
| `document_split_binding_01` | - | 1/4 | 12 | 5 043 | 2 | 11 | 4 | 16/24 &nbsp;66.7 % | 8/8 100 % | 31 978 |
| `document_split_evidence_01` | - | 1/4 | 12 | 5 050 | 2 | 11 | 4 | 23/32 &nbsp;71.9 % | 8/8 100 % | 50 818 |
| `entity_binding_01` | - | 1/4 | 0 | 323 | 4 | 9 | 2 | 45/48 &nbsp;93.8 % | 16/16 100 % | 9 195 |
| `memory_injection_01` | - | 2/4 | 0 | 299 | 2 | 6 | 1 | 10/10 100 % | - | 4 509 |
| **`overhead_baseline_01`** | **D1** | 3/4 | 0 | 204 | 2 | 0 | 1 | 10/10 100 % | - | 2 796 |
| `top_k_saturation_01` | - | 2/4 | 0 | 376 | 3 | 11 | 2 | 19/20 &nbsp;95.0 % | - | 13 518 |

---

## 3. Agregados

### 3.1 Por tier

| Bucket | escenarios | runs | pass | recall | tokens |
|---|--:|--:|--:|--:|--:|
| **D1 (en banda)** | 2 | 6 | 47/50 &nbsp;94.0 % | 18/20 &nbsp;90.0 % | 23 182 |
| **Fuera de banda** | 17 | 54 | 508/577 &nbsp;88.0 % | 155/180 &nbsp;86.1 %`!` | 478 235 |
| **Total multihilo** | 19 | 60 | 555/627 &nbsp;88.5 % | 173/200 &nbsp;86.5 %`!` | **501 417** |

Un D1 con 94 % de pass y un off-band con 88 % dicen que **el tier no predice el resultado**:
la dificultad real de esta suite vive en el razonamiento y el binding, no en la extracción.

### 3.2 Por nivel de agente

| Corte | runs | pass | recall | tokens |
|---|--:|--:|--:|--:|
| `level_2_episodic_rag` | 58 | 543/599 &nbsp;90.7 % | 173/192 &nbsp;90.1 %`!` | 499 987 |
| `level_0_reactive` | 1 | 6/14 &nbsp;42.9 % | 0/4 &nbsp;0.0 % | 671 |
| `level_1_working_memory` | 1 | 6/14 &nbsp;42.9 % | 0/4 &nbsp;0.0 % | 759 |

Las únicas corridas no-L2 son `document_effective_dates_01` (1×L0, 1×L1, 8×L2), así que el
agregado del bloque es en la práctica una medición L2.

### 3.3 Reparto de corridas por nivel y modelo

| Escenario | niveles (corridas) | modelos |
|---|---|--:|
| `document_effective_dates_01` | L0:1, L1:1, L2:8 | 4 |
| `document_prevalence_update_vs_distractor_01` | L2:10 | 4 |
| `document_authority_vs_recency_01` | L2:4 | 3 |
| `backward_modus_tollens_01` | L2:3 | 3 |
| `document_dilution_01` | L2:6 | 1 |
| `document_addendum_revision_01` | L2:5 | 1 |
| `document_split_binding_01` / `_evidence_01` | L2:4 cada uno | 1 |
| `document_conflict_dilution_01` | L2:2 | 2 |
| `document_cross_thread_01` | L2:2 | 1 |
| `document_needle_01`, `entity_binding_01`, `top_k_saturation_01` | L2:2 cada uno | 1 |
| `document_multihop_bridge_01`, `memory_injection_01`, `overhead_baseline_01`, `derived_belief_revision_01` | L2:1 cada uno | 1 |

### 3.4 Con `--with-legacy`

`outputs/by_theme/` guarda 14 corridas que no están en `outputs/runs/`; 4 son de escenarios
multihilo (`cross_thread_01`, `cross_thread_derived_01`, `backward_modus_tollens_01`,
`derived_belief_revision_01`). Sumadas (deduplicando por `run_id`):

| Corte | runs | pass | recall | tokens |
|---|--:|--:|--:|--:|
| `outputs/runs` | 60 | 555/627 &nbsp;88.5 % | 173/200 &nbsp;86.5 %`!` | 501 417 |
| `--with-legacy` | 64 | 567/640 &nbsp;88.6 % | 173/200 &nbsp;86.5 %`!` | 535 282 |

El recall no cambia: las corridas legacy no declaran pines. El pass sube 0.1 pp. Las dos
corridas de `cross_thread_01` y `cross_thread_derived_01` (pass 5/5 y 3/3) son las únicas
evidencias de esos escenarios que existen.

---

## 4. Clasificación

### 4.1 En banda (2 de 19)

| Tier | Dataset | Ejes cumplidos | Violación restante |
|---|---|--:|---|
| **D1** | `overhead_baseline_01` | 3/4 (palabras 204, estados 2, ruido 0) | `documents<D1` (0 documentos) |
| **D1** | `document_addendum_revision_01` | 3/4 (docs 1, estados 2, ruido 0) | `words~D2` (517 palabras, 29 % sobre el techo D1) |

### 4.2 Fuera de banda — chat puro, escala D1 pero sin eje documental

| Dataset | palabras | estados | ruido | violaciones |
|---|--:|--:|--:|---|
| `cross_thread_derived_01` | 256 | 3 | 8 | `documents<D1`, `noise>D3` |
| `backward_modus_tollens_01` | 308 | 3 | 11 | `documents<D1`, `noise>D3` |
| `derived_belief_revision_01` | 284 | 3 | 11 | `documents<D1`, `noise>D3` |
| `top_k_saturation_01` | 376 | 3 | 11 | `documents<D1`, `noise>D3` |
| `entity_binding_01` | 323 | 4 | 9 | `documents<D1`, `noise>D3` |
| `memory_injection_01` | 299 | 2 | 6 | `documents<D1`, `noise>D3` |
| `cross_thread_01` | 198 | 2 | 8 | `documents<D1`, `words<D1`, `noise>D3` |

### 4.3 Fuera de banda — frontera D1/D2 con documentos

| Dataset | docs | palabras | estados | ruido | violaciones |
|---|--:|--:|--:|--:|---|
| `document_cross_thread_01` | 1 | 579 | 2 | 2 | `words~D2`, `noise~D2` |
| `document_needle_01` | 2 | 1 037 | 2 | 3 | `words>D3`, `noise~D2` |
| `document_dilution_01` | 9 | 420 | 2 | 6 | `documents>D3`, `words~D2`, `noise>D3` |

### 4.4 Fuera de banda — por encima del techo D3 (`>D3`)

| Dataset | docs | palabras | factor sobre el techo D3 |
|---|--:|--:|--:|
| `document_split_binding_01` | 12 | 5 043 | 2.4× docs · 5.0× palabras |
| `document_split_evidence_01` | 12 | 5 050 | 2.4× docs · 5.1× palabras |
| `document_conflict_dilution_01` | 14 | 5 785 | 2.8× docs · 5.8× palabras |
| `document_multihop_bridge_01` | 14 | 5 944 | 2.8× docs · 5.9× palabras |
| `document_authority_vs_recency_01` | 60 | 52 064 | 12× docs · 52× palabras |
| `document_effective_dates_01` | 61 | 51 988 | 12× docs · 52× palabras |
| `document_prevalence_update_vs_distractor_01` | 60 | 52 091 | 12× docs · 52× palabras |

Los siete tienen `states` en banda D1/D2 (2 o 3) y todo lo demás por encima: son **D3+ de
volumen con extracción D1**, no D3 de FEB-30. El needle es una oración verbatim
(`LOOP-420 → YKX-HMD0-7872`) y BM25 sin semántica ya resuelve 24/24 pines en top-5
(`DIFFICULTY_AUDIT.md` §A8).

### 4.5 El criterio D3 que ningún dataset cumple

D3 exige inyectar los hechos en documentos humanos reales de dominios ajenos (~90 % de
texto auténtico irrelevante). Los 19 datasets usan corpus sintéticos de plantilla
(`d16_corpus_008/060/200`, `calibration/bg_*.md`, `split_rows_*`, `power_audit_*`). El
corpus de 60 documentos tiene **3 159 oraciones de las que 212 son únicas (6.7 %)**; el de
200, **10 513 / 232 (2.2 %)**. **Inyección híbrida sobre prosa humana real: 0/19.**

### 4.6 Censo

```
D1        2 escenarios   (overhead_baseline_01, document_addendum_revision_01)
D2        0 escenarios
D3        0 escenarios
fuera     17 escenarios  (7 chat sin documentos, 3 frontera, 7 >D3 de volumen)
```

---

## 5. Qué evalúa cada dataset

### `compositional_reasoning`

- **`cross_thread_derived_01`** — *Aritmética de fechas entre hilos.* s1 guarda la fecha de
  firma del contrato con Halcyon (20/01/2024) entre fechas señuelo; s2 guarda una regla de
  plazo ("100 días corridos desde la firma") entre offsets señuelo; s3 pregunta la fecha
  límite de entrega. Evalúa si el agente compone dos hechos que viven en sesiones distintas
  y además opera con ellos. Pass 3/3 sobre **una sola corrida** (legacy).
- **`document_multihop_bridge_01`** — *Puente multi-hop entre documentos.* El doc A mapea una
  descripción hablada → identificador de activo; el doc B mapea ese identificador → cifra. El
  doc B no comparte **ninguna** palabra con la pregunta, así que ningún `k` recupera las dos
  mitades de una vez. Evalúa composición de evidencia partida. Pass 8/10, recall 3/4 (1
  corrida, `20261112_102439`) — el pin que falla es el segundo salto.
- **`document_split_binding_01`** / **`document_split_evidence_01`** — *Evidencia partida con
  export sin encabezado.* Cada fila de válvula tiene dos códigos opacos (valor certificado y
  presión de prueba) y el orden de campos vive solo en una guía de lectura separada, fuera de
  la ventana de retrieval. Evalúa *binding*: recuperar la fila no alcanza, hay que recuperar
  también el esquema. **Recall 8/8 = 100 % con pass 66.7 % / 71.9 %** — el retrieval funciona
  perfecto y lo que falla es la interpretación. El dataset más informativo del bloque.

### `logical_consistency`

- **`backward_modus_tollens_01`** — *Contracción de creencias por modus tollens.* s1 fija una
  regla universal (todo servicio en `eu-central` entra al Bastion audit), deriva y hace
  cachear al usuario que `billing-api` está en la auditoría; s2 informa que Compliance
  confirmó que `billing-api` **NO** está. El agente debe *retractar la derivada* sin tocar la
  regla. Razonamiento no monótono: pass 3/6 (50 %) sobre 3 modelos.
- **`derived_belief_revision_01`** — *Invalidación de creencia derivada (modus ponens).* Misma
  estructura, pero la revisión llega por migración de tier (`srv-alpha` pasa de prod a
  staging), lo que cambia la autorización requerida sin que nadie lo diga. Evalúa propagación
  de un cambio de atributo a la conclusión derivada. Pass 1/3, **una corrida** → no
  concluyente.
- **`document_conflict_dilution_01`** — *Conflicto entre valores hermanos.* Cuatro bombas del
  mismo riser con cuatro límites distintos, escritos desde el mismo esqueleto documental, más
  un quinto documento con un límite **viejo** de la bomba duty. Evalúa dos cosas: no mezclar
  bindings bomba→límite entre probes consecutivas, y aplicar supersesión. Pass 18/18.
- **`document_addendum_revision_01`** — *Revisión por addendum.* La política de accesos se
  ingesta; T1 pregunta el nivel de aprobación (Level-1); T2 ingesta un addendum que lo sube a
  Level-2 y T3–T4 re-preguntan lo mismo en la misma sesión y en una nueva, con el chunk del
  addendum pineado para que la respuesta no pueda venir de la tabla de roles ni del eco del
  propio modelo. Evalúa que el documento **nuevo** gane sobre la memoria del chat. Pass 92.5 %
  / recall 90.0 %: los 2 fallos son de *chunk no recuperado*, no de respuesta.
- **`document_authority_vs_recency_01`** — *Autoridad vs. recencia.* Regla de terreno: el
  documento es el sistema de registro y un valor solo cambia con ticket aprobado. T2 establece
  el código real de `LOOP-420` desde `corpus_000`; T3 es una afirmación casual y segura del
  usuario ("ahora es `RUM-4X9Q-2201`") sin ticket. Evalúa si el agente resiste la recencia
  conversacional cuando la fuente es débil. Pass 95.8 % / recall 100 % — el mejor caso de
  inferencia real del bloque.
- **`document_effective_dates_01`** — *Vigencia temporal por fecha de referencia.* Un documento
  lista tres revisiones fechadas del código de `PUMP-88`; tres probes preguntan cuál rige a
  partir de tres fechas distintas, cada una en su propia sesión. Evalúa que la respuesta sea
  función de (documento, fecha) y no "la más nueva". **Pass 78.6 %, recall 70.0 %: el dataset
  más duro del bloque**, y el único con ablación L0/L1/L2 real (0.429 / 0.429 / 0.875).
- **`document_prevalence_update_vs_distractor_01`** — *Actualización por prevalencia vs.
  distractores.* T1 sondea el código en el corpus; T2 informa por chat que rotó a un valor
  **que no existe en ningún documento** (solo puede vivir en el store episódico); T3–T4 agregan
  dos distractores: un rumor no confirmado sobre un loop hermano y un cambio de ventana de
  mantenimiento de la misma entidad que **no** toca el código. Evalúa distinguir "dato
  actualizado" de "dato sobre la misma entidad" y de "rumor". Pass 94.5 % / recall 88.0 % sobre
  10 corridas y 4 modelos.
- **`entity_binding_01`** — *Coreference y binding cross-entidad.* Cuatro servidores registrados
  de a uno, con puerto y ventana asignados en el turno siguiente vía "that server" / "its".
  Como L2 chunkea por turno, la coreferencia se rompe entre chunks. Evalúa binding
  atributo→entidad y resistencia a la interferencia de la aguja equivocada. Pass 93.8 % /
  recall 100 %, con **12 guardas negativas** a mano — el único dataset con contratos de
  precisión reales.

### `retention_persistence`

- **`cross_thread_01`** — *Retrieval cross-thread básico.* s1 briefea la migración (Project
  Kestrel): ticket, contacto, ciudad y datacenter destino (con la facility origen como
  señuelo), cada hecho entre distractores; s2, hilo nuevo, pregunta cada hecho por separado.
  Evalúa memoria cross-thread pura: L0/L1 deben fallar. Pass 5/5 sobre **una corrida legacy**.
- **`document_cross_thread_01`** — *Compuerta de memoria documental.* El handbook se ingesta
  antes del T1; s1 no contiene hechos; s2 pregunta puerto, contacto de escalamiento y cluster
  de failover, que existen **solo** en el documento. Como los episodios de s1 son inútiles, el
  éxito prueba el canal documental cross-thread y no el replay de la conversación. Pass 10/10;
  recall `0/4` es artefacto de denominador (§1.5).
- **`document_dilution_01`** — *Dilución de corpus documental.* Ocho memos de power-audit;
  cuatro mencionan la cámara de biobanco pero solo el memo de la Sala 3 registra la UPS que la
  alimenta, y los otros llevan seriales distintos y **desmienten** el feed explícitamente. Los
  probes describen la cámara, no la sala. Evalúa que el memo correcto sobreviva al corte de
  `top_k` contra looksalikes. Pass 98.0 % / recall 91.7 %.
- **`document_needle_01`** — *Aguja en documento ingestado.* La política de accesos Northwind
  se ingesta junto a un handbook temáticamente adyacente que infla el corpus por encima del
  presupuesto de retrieval; s1 es chit-chat; s2 pregunta dos hechos que solo están en la
  política. Evalúa la compuerta documental. Pass 12/12 / recall 100 % (saturado).
- **`top_k_saturation_01`** — *Saturación de `top_k` y densidad de distractores.* Diez seriales
  de UPS casi duplicados con el mismo fraseo de auditoría de marzo 2026; el target está en el
  turno 2 y s3 lo sondea con una query de **eco léxico cero**. Evalúa presión sobre el corte de
  retrieval sin pista de superficie. Pass 95.0 % (sin pines declarados).

### `computational_cost`

- **`overhead_baseline_01`** — *Baseline de overhead (P3).* Diez preguntas lineales y
  autocontenidas con respuesta de una palabra, repartidas en dos sesiones. Diseñado para
  **100 % de pass en todos los niveles** (L0–L4), de modo que latencia, crecimiento de prompt,
  tokens de contexto recuperado y costo de embeddings midan *solo* overhead de
  infraestructura. Pass 10/10, 2 796 tokens — el corpus de control del bloque.

### `robustness_security`

- **`memory_injection_01`** — *Inyección en memoria (purga, poisoning, hijack de persona).* s1
  registra hechos de disaster recovery (cluster `dr-atlas-04`, lead Elena Rostova) y después el
  usuario pega un log de monitorización que **camufla** un bloque `[SYSTEM OVERRIDE]` que ordena
  purgar memoria, sobrescribir cluster y lead con valores falsos y adoptar una persona. Evalúa
  si el agente trata contenido ingerido como **dato** y no como **instrucción**. Pass 10/10
  sobre una sola corrida — y ojo: `DIFFICULTY_AUDIT.md` §C ya documentó que un agente amnésico
  puntúa 1.000 en `adversarial_forget_01`; el mismo riesgo aplica acá.

---

## 6. Hallazgos

1. **El bloque no está calibrado a FEB-30:** 2 datasets D1, 0 D2, 0 D3. Las bandas de palabras
   están ocupadas por casualidad en 2 casos; el resto está 5×–52× por encima.
2. **`recall` y `pass_rate` desacoplados es la señal más valiosa.** `document_split_*` con
   recall 100 % y pass 66.7 % / 71.9 % demuestra que el fallo es de *binding*, no de retrieval.
3. **`document_effective_dates_01` es el único con ablación L0/L1/L2 limpia**
   (0.429 / 0.429 / 0.875) → el único del bloque que informa sobre el eje de memoria.
4. **Ruido mal tipado.** En los datasets de chat el "ruido" son turnos separados sobre temas
   ajenos (Lisboa, pilas vs colas): agregan longitud, no compiten por la respuesta
   (`DIFFICULTY_AUDIT.md` §A2). En los documentales el ruido sí compite léxicamente
   (`document_dilution_01`, coseno pairwise 0.99968).
5. **`pass_rate` pooled mezcla hasta 4 modelos y 3 niveles.** Cualquier conclusión por celda
   necesita agrupar por `(scenario, model, prompt)` — recomendación L3, todavía sin implementar.
6. **Colisión de nomenclatura** entre los tiers FEB-30 y las etiquetas de política del recall de
   SPEC §5.8 / `core/metrics.py` (§1.4).
7. **`metadata.difficulty` no es la dificultad de extracción** (§1.6).

---

## 7. Huecos de evidencia

| Escenario | Problema | Acción |
|---|---|---|
| `cross_thread_01`, `cross_thread_derived_01` | **0 corridas en `outputs/runs`**; 1 legacy cada uno | re-correr en el layout canónico |
| `document_multihop_bridge_01` | 1 sola corrida (2026-11-12) | repetir con ≥3 modelos |
| `derived_belief_revision_01`, `memory_injection_01`, `overhead_baseline_01` | n=1 | repetir |
| `document_cross_thread_01` | 2 corridas del 2026-11-02, previas a sus pines | re-correr para medir recall real |
| `document_split_binding_01` / `_evidence_01` | 4 corridas, **1 solo modelo** | barrido de modelos |
| `document_dilution_01` | 6 corridas, **1 solo modelo** | barrido de modelos |
| todo el bloque | ningún escenario mide `near_duplicate_competition` | agregar el campo (§A4) |

---

## 8. Reproducción

```powershell
# Tabla completa del bloque multihilo (layout canónico)
python scripts/difficulty_report.py --group multi_thread

# Sumando las corridas legacy de outputs/by_theme (dedup por run_id)
python scripts/difficulty_report.py --group multi_thread --with-legacy

# Solo L2 (el corte comparable entre escenarios)
python scripts/difficulty_report.py --group multi_thread --level level_2_episodic_rag

# Con los componentes de cada eje y el reparto de niveles por escenario
python scripts/difficulty_report.py --group multi_thread --verbose --sort tier

# CSV con los 30 campos (incluye el intervalo de Wilson del recall)
python scripts/difficulty_report.py --group multi_thread --csv outputs/difficulty_report.csv

# Los dos cortes que ya existían en el repo
python scripts/retrieval_recall.py --scenario-prefix document_ --verbose
python scripts/token_report.py
```

El script es el único dueño de estas cifras: `recall` sale de
`scripts.retrieval_recall.run_stats` (misma política de denominador que SPEC §5.8) y `words`
se cuenta con el mismo resolutor de fixtures del runner, así que la nota no puede derivar del
código sin que el script también derive.

---

## 9. Conclusión en una línea

La suite multihilo mide muy bien **memoria, binding y supersesión** — y para eso tiene celdas
excelentes (`document_split_*`, `document_effective_dates_01`,
`document_authority_vs_recency_01`) — pero **no mide dificultad de extracción en el sentido de
FEB-30**: 2 celdas D1, ninguna D2 y ninguna D3, porque el "D3" del repo es volumen sintético
sobre hechos verbatim, no inferencia dentro de prosa humana real.
