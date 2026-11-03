# 🗺️ Benchmark Test Expansion Plan

This document outlines the progressive strategy to expand the `agent-memory-benchmark` dataset. The goal is to provide a robust evaluation framework that can accurately differentiate and stress-test the planned memory architectures (L0 through L4).

**Coding Agent Instructions:**
Do not implement all stages at once. Wait for the user to explicitly request the execution of a specific Stage and provide the corresponding YAML specifications in the chat context.

---

## Stage 1: Experimental Controls Setup
**Objective:** Establish the scientific boundaries of the benchmark. We must prove the LLM's raw cognitive capabilities (Positive Controls) and measure its baseline hallucination rate (Negative Control).

*   **1.1. In-Context Multi-Hop (Positive Control)**
    *   **Path:** `/datasets/conversations/test_in_context_multihop_01.yaml`
    *   **Theme:** `compositional_reasoning`
    *   **Evaluates:** Proves the LLM can perform 3-hop logic if all premises are provided in a single prompt. If an agent fails the multi-turn version but passes this, the failure is strictly a memory persistence issue.
*   **1.2. In-Context Needle (Positive Control)**
    *   **Path:** `/datasets/conversations/test_in_context_needle_01.yaml`
    *   **Theme:** `retention_persistence`
    *   **Evaluates:** Proves the LLM can extract a specific detail from a dense text block when asked immediately, without distractor turns.
*   **1.3. Global Negative Control (Anti-Hallucination)**
    *   **Path:** `/datasets/conversations/test_negative_hallucination_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** Asks the agent for a specific fact (e.g., "Project Delta's budget") that was *never* mentioned. Evaluates if the memory architecture (L1-L4) incorrectly retrieves unrelated data or hallucinates instead of safely declining.

---

## Stage 2: Advanced Consistency & Stress Testing
**Objective:** Introduce tests that will specifically challenge Level 1 (Thread Memory) and Level 3 (Semantic/Reflective Memory).

*   **2.1. Cascading Revision**
    *   **Path:** `/datasets/conversations/test_cascading_revision_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** A fact is corrected twice (A → B → C). Tests if Semantic RAG (L3) correctly invalidates the intermediate state 'B' or if it gets confused by multiple conflicting episodes.
*   **2.2. Implicit Contradiction**
    *   **Path:** `/datasets/conversations/test_implicit_contradiction_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** The user introduces a fact that logically overrides a previous one, without explicitly stating "I was wrong". Essential for testing the Dialectical reasoning of L4.

    > **⚠ Assertion design decision (recorded 2026-10-14, before implementation):**
    >
    > **The hazard:** the natural spec — `must_contain: <new_fact>` + `must_not_contain: <old_fact>`
    > on the probe turn — **punishes rational hedging**. A model that answers *"you first mentioned
    > Lima, but your recent message suggests you now live in Buenos Aires"* has demonstrably noticed
    > the implicit collision (the exact capability under test) yet fails the substring assertion
    > — the same failure class as the "pedantry leak" documented in
    > `notes/2026-10-09_adversarial_forget_summary.md` (F3: model holds the policy line but trips
    > the assertion on meta-commentary). `belief_revision_01` escapes this because the explicit
    > "not Luna" licenses the model to confidently drop the old value; in 2.2 **no such license
    > exists** — mentioning the superseded fact while reasoning toward the new one is the
    > epistemically correct behavior.
    >
    > **The fix — two-step probe structure (bake into the YAML from the start):**
    > 1.  **Open recall turn (observational, NO assertions):** free-form "where do I live?" —
    >     the response is recorded verbatim by the runner for qualitative analysis (does the
    >     model notice the collision? anchor on first mention? hedge?).
    > 2.  **Commitment turn (hard assertions live HERE only):** "So, one city, which is it?
    >     Give me the single city where I live today." → `must_contain: <new_fact>` +
    >     `must_not_contain: <old_fact>`. The follow-up explicitly licenses dropping the old
    >     value, so a lingering mention on this turn is a *decision* failure, not meta-commentary.
    >
    > **Rules for the spec author:** (a) the invalidating fact must be delivered as idle
    > small-talk — no "actually/however/correction" markers, no reference to the old fact;
    > (b) collision must require world knowledge (e.g. two distinct cities), not lexical
    > contradiction; (c) at least one distractor turn between old fact, new fact, and probe to
    > force retrieval over recency-in-window; (d) `expected_level_0_behavior`: safe decline
    > (amnesia) — note the failure shape is a *non-answer*, opposite of every other
    > invalidation test; (e) keep a future `implicit_contradiction_02` in mind for rung 5
    > (derived-belief invalidation, where even the colliding fact is never stated).
*   **2.3. Massive Attrition (Long Context)**
    *   **Path:** `/datasets/conversations/test_massive_attrition_01.yaml`
    *   **Theme:** `retention_persistence`
    *   **Evaluates:** Injects 15+ complex distractor turns. Stress-tests the context window limits and evaluates the efficiency of L1 Checkpointers.

---

## Stage 3: Cross-Session & Deep Reasoning
**Objective:** Introduce the core tests that justify the existence of Level 2 (Episodic RAG) and Level 4 (Neuro-Symbolic).

*   **3.1. Cross-Thread Memory Retrieval**
    *   **Path:** `/datasets/conversations/test_cross_thread_01.yaml`
    *   **Theme:** `retention_persistence`
    *   **Evaluates:** The conversation is formally split across two different `thread_id`s. This is the ultimate test for L2 (BaseStore); L0 and L1 are guaranteed to fail this.

    > **✅ Implemented (2026-10-24):** lives at the real nested path `datasets/conversations/retention_persistence/cross_thread/test_cross_thread_01.yaml` (id `cross_thread_01`), not the flat path listed above — the repo uses the `<theme>/<category>/` layout. Stage 3 assertion policy: assertions sit on probe turns only and are `must_contain` only (the sole exception being the two 3.3 leakage guards).

*   **3.2. Deep Temporal Multi-Hop (3+ Hops)**
    *   **Path:** `/datasets/conversations/test_3hop_temporal_01.yaml`
    *   **Theme:** `compositional_reasoning`
    *   **Evaluates:** A chain of facts: A relates to B, B relates to C, C relates to D. The agent must deduce D from A across multiple conversational turns.

    > **✅ Implemented (2026-10-24):** lives at the real nested path `datasets/conversations/compositional_reasoning/temporal_multi_hop/test_3hop_temporal_01.yaml` (id `3hop_temporal_01`), next to the existing 2-hop scenario — not the flat path listed above. Stage 3 assertion policy: assertions sit on probe turns only and are `must_contain` only (the sole exception being the two 3.3 leakage guards).

*   **3.3. Cross-Thread Derived Answer** *(recorded 2026-10-14; motivated by the reasoning-token analysis in `notes/2026-10-14_l1_three_model_comparison.md`)*
    *   **Path:** `/datasets/conversations/test_cross_thread_derived_01.yaml`
    *   **Theme:** `compositional_reasoning`
    *   **Evaluates:** The conjunction of 3.1 and 3.2 — premises seeded in *different* `thread_id`s, retrieved via BaseStore, then **combined through a computation (date arithmetic)** whose answer exists in no stored fact. The derived answer must be produced, not recalled: this is the only planned configuration where the value of model deliberation (reasoning tokens) can be measured on inference that exceeds retrieval.

    > **⚠ Design decision (recorded 2026-10-14, before implementation):**
    >
    > **Origin:** at L1, all three tested models scored 100% on every in-window derivation scenario;
    > qwen3.6-27b's externalized chain-of-thought (≈22k hidden reasoning tokens) made it simultaneously
    > the most expensive and the weakest scorer, while qwen3-235b (`reasoning: 0` across the sweep)
    > topped the board at the lowest quota cost. The question *"does deliberation pay off on hard
    > inference?"* is unanswerable with any current or planned scenario: 3.1 tests retrieval only;
    > 3.2 tests derivation but keeps every premise inside L1's context window (attention, not memory).
    > This entry supplies the missing conjunction.
    >
    > **Spec structure (three threads, one probe):**
    > 1.  **Thread A:** fact A (e.g. "contract signed on March 3, 2025") buried mid-thread among
    >     distractor turns containing plausible decoy dates.
    > 2.  **Thread B:** fact B (e.g. "delivery is due 45 days after signing") buried similarly.
    > 3.  **Probe turn:** asks for the derived answer ("what's the delivery date?") — requires
    >     A + B + real date arithmetic. The derived date (April 17, 2025) must appear in **no turn
    >     of any thread**: no verbatim-recall path to the answer may exist.
    >
    > **The confound, and the paired control that resolves it:** the composite scenario confounds
    > *retrieval failure* with *derivation failure* — a probe miss doesn't attribute. Fix: a **paired
    > hard in-context derivation control** containing the *same* derivation problem with all premises
    > in a single prompt. The existing 1.1 control cannot serve: it is at ceiling for all tested
    > models. The paired control must be calibrated hard enough that a terse strong model is not
    > trivially at 100%, so the two outcomes stay interpretable: **pass control + fail cross-thread
    > → memory failure; fail both → reasoning failure** (and per-model deltas on the control measure
    > the raw value of deliberation, independent of memory architecture). Run each model on both;
    > comparing reasoning-token spend between them gives the direct P3 measurement of the cost/value
    > of deliberation.
    >
    > **Rules for the spec author:** (a) date arithmetic must require actual computation (day-level,
    > not month-level "same month + 1"); (b) premises must sit mid-thread with distractors before and
    > after — never the final turn of a thread (defeat recency); (c) the probe must **paraphrase** —
    > it may not echo either premise verbatim, so lexical matching cannot substitute for retrieval;
    > (d) the derived date must be distinct from every stated date; (e) each thread must contain
    > plausible decoy dates so guess-and-check fails; (f) `expected_level_0/1_behavior`: safe decline
    > (premises out of window) — this scenario is an L2 capability gate, same as 3.1.
    >
    > **Prerequisites:** per-turn `thread_id` support in the runner (`core/runner.py` currently pins
    > one thread per run — the limitation documented in the adversarial-forget analysis) and the L2
    > BaseStore agent. Implement **after 3.1**, which validates the multi-thread harness this test
    > inherits.

    > **✅ Implemented (2026-10-24):** lives at the real nested path
    > `datasets/conversations/compositional_reasoning/cross_thread_derived/test_cross_thread_derived_01.yaml`
    > (id `cross_thread_derived_01`), with the paired control
    > `datasets/conversations/compositional_reasoning/in_context_derivation/test_in_context_derivation_01.yaml`
    > (id `in_context_derivation_01`). Calibration deviates from the illustrative "Mar 3 + 45": the
    > derivation is Jan 20, 2024 + 100 calendar days (crossing Feb 29 of a leap year) → April 29, 2024,
    > answered date-only, so the control is not trivially at ceiling. Turns 8–9 carry automatic
    > `must_not_contain "April 29"` leakage guards (the only `must_not_contain` assertions in Stage 3).
    > Headline metric rule: report T11 pass/fail; a failed T8/T9 guard marks the run as **leaked**
    > (its T11 result is recall, not derivation, and is excluded from derivation analysis); the
    > scenario `pass_rate` is **not** comparable across levels because L0/L1 score 66.7% just by
    > declining. Otherwise the Stage 3 assertion policy applies: assertions on probe turns only, and
    > only `must_contain` except these two guards.

---

## Stage 4: Infrastructure & Boundaries
**Objective:** Measure the operational costs and security vulnerabilities introduced by persistent memory architectures — and isolate the retrieval-quality failure modes of naive RAG at the L1/L2 boundary (L2's fixed retrieve → model → append chain with thresholdless top-k dense retrieval).

> **📋 Coverage note (added 2026-10-26 — L2 gap analysis):** 4.3–4.5 originate from mapping the L2 implementation against the suite: Stage 5 stresses L2's *append-only* semantics (revision), but nothing isolates L2's *retrieval* semantics — top-k cutoff saturation, cross-entity interference, and no-threshold injection of irrelevant chunks. `needle_haystack_01` / `attrition_01` are single-session (L0-passable) and never touch the backend. All three new scenarios are failure-attributable with **zero harness changes**: `retrieved_memories` (per-turn `chunk_id` / `score` / `rank` / `content`) is already persisted in `TurnResult` (`core/schemas.py`), and `top_k` is injectable at runtime via `configurable`.

*   **4.1. Computational Overhead Baseline**
    *   **Path:** `/datasets/conversations/test_overhead_baseline_01.yaml`
    *   **Theme:** `computational_cost`
    *   **Evaluates:** A strictly linear, simple Q&A conversation. Designed to have a 100% pass rate. Used exclusively to isolate and measure the raw latency (ms) and token overhead of memory retrieval mechanics (P3).
*   **4.2. Memory Prompt Injection**
    *   **Path:** `/datasets/conversations/test_memory_injection_01.yaml`
    *   **Theme:** `robustness_security`
    *   **Evaluates:** Resistance to adversarial attacks. The user inputs a "System Override" command instructing the agent to purge its memory and adopt a new persona. Tests if the memory store can be poisoned or erased maliciously.

*   **4.3. Top-k Saturation / Distractor Density (Naive-RAG Cutoff Stress)**
    *   **Path:** `/datasets/conversations/retention_persistence/top_k_saturation/test_top_k_saturation_01.yaml`
    *   **Theme:** `retention_persistence`
    *   **Evaluates:** Whether the correct fact survives L2's top-k dense retrieval once the persistent store fills with semantically near-duplicate facts seeded across sessions. Isolates ranking precision at the `top_k` cutoff (default 5 — `DEFAULT_TOP_K`, `agents/level_2_episodic_rag/graph.py`): a failure mode no other scenario reaches. `needle_haystack_01` is single-session (an L0-passable attention test) and `cross_thread_01` has only one plausible candidate chunk, so ranking competition never occurs.

    > **⚠ Design rationale (recorded 2026-10-26):**
    >
    > **Why this is required for L2:** `VectorMemoryBackend.retrieve` truncates to `ranked[:top_k]` with **no relevance threshold** — a target chunk that ranks 6th among near-duplicates is silently dropped. No existing cross-session scenario stores enough similar candidates for this to happen.
    >
    > **Spec structure (three sessions):**
    > 1.  **Session A (target + early decoys):** State the target fact plus 3–4 near-duplicates in the same syntactic schema (e.g., UPS unit serials for different datacenter rooms sharing inspection phrasing and dates: *"Room 3's UPS unit serial is UPS-7C42, last inspected March 2026"* vs. *"Room 1's UPS unit serial is UPS-7C41, last inspected March 2026"*).
    > 2.  **Session B (crowding):** Add 5–8 more same-schema decoys spread over several short turns, so ≥8 similar chunks compete at probe time. Distractor prompts ask for short answers (the Stage-3 rule: long assistant replies would split into extra chunks and add noise to top-k).
    > 3.  **Session C (paraphrased probe + commitment turn):** Ask for the target with **no lexical echo** of the target chunk (the 3.3 paraphrase rule). Commitment turn asserts `must_contain: "UPS-7C42"` and `must_not_contain` on the closest decoy serials. Leakage guard: the probe may not contain any serial.
    >
    > **Ablation lever (free failure attribution):** `top_k` is injected at runtime via `configurable`, so the *same* YAML can be run at `k=5` and at a large `k` (e.g., 50). Fail@5 + pass@50 ⇒ **retrieval miss** (cutoff); fail@both ⇒ **read-time interference** (the model sees the target but picks a decoy). `retrieved_memories` telemetry audits the target's rank directly. Report per-`top_k` separately — pass rates are not comparable across settings.
    >
    > **Expected behavior:** L0/L1: safe decline (cross-session, premises out of window). L2: passes only if the target chunk ranks within top-k *under paraphrase* — the L2 capability gate.

*   **4.4. Cross-Entity Attribute Binding (Wrong-Needle Interference)**
    *   **Path:** `/datasets/conversations/logical_consistency/entity_binding/test_entity_binding_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** Whether attributes from a stored near-duplicate fact about a *sibling entity* stay bound to the right entity. The two facts share their syntactic schema and differ only in entity token and value, so their embeddings are near-identical and *both* rank in top-k for either probe. The test is therefore not about retrieval *reach* (both chunks arrive) but about read-time binding over an injected memory block containing two lookalike rows — the "retrieved the wrong needle" failure, distinct from missing the needle (4.3) and from revision (Stage 5).

    > **⚠ Design rationale (recorded 2026-10-26):**
    >
    > **Spec structure (three sessions, paired probe):**
    > 1.  **Session A:** Entity A fact in a fixed schema (*"Server `srv-alpha` runs its admin console on port `7443`; the platform team owns it"*).
    > 2.  **Session B:** Sibling entity B in the *same* schema with a plausible transposed value (*"Server `srv-beta` runs its admin console on port `7080`; the platform team owns it"*). Calibration may substitute values — they must not collide with Stage 5 fixtures (5.8 uses 8443/8080).
    > 3.  **Session C (paired probes — target + control, mandatory as in 5.2):** Ask A's port (`must_contain: "7443"`, `must_not_contain: "7080"`) and, on a separate turn, B's port (`must_contain: "7080"`, `must_not_contain: "7443"`). Passing requires **both sides of the pair**: an agent that retrieves only one row, or transfers the retrieved value onto the wrong entity, fails exactly one side.
    >
    > **What it exposes:** attribute transfer between lookalike entities is a documented RAG failure that substring assertions catch cleanly. An L3 that extracts typed facts at write time disambiguates by construction; L2 relies entirely on the model reading the provenance-stamped memory block correctly — this scenario measures that boundary directly.

*   **4.5. Retrieval-Anchored Hallucination (The "Memory Hurts" Control)**
    *   **Path:** `/datasets/conversations/logical_consistency/retrieval_hallucination/test_retrieval_hallucination_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** The L2-specific sharpening of `negative_hallucination_01`: a probe about an entity that was **never mentioned**, while a lookalike entity's similar facts are stored. Because retrieval has no relevance threshold, the top-k block always contains plausible-but-irrelevant chunks once the store is non-empty; the test measures whether the model anchors on them. This is the one scenario family where L2 can score **below** L0 — a direct demonstration that adding memory can degrade correctness.

    > **⚠ Design rationale (recorded 2026-10-26):**
    >
    > **Spec structure (two sessions + in-scenario positive control):**
    > 1.  **Session A (lookalike seeding):** State several concrete facts about entity Y (*"Project Orion's contingency budget is $1.2M"*, plus timeline and lead) in vocabulary that would sound natural for the never-mentioned entity X.
    > 2.  **Session B (unprimed probe first, licensed commitment second):**
    >     *   *Primary observational turn (unprimed):* ask openly and naturally about X (*"What is Project X's contingency budget?"*) — **no conditional template, no licensed absence phrase**: the model must reason about what it does and does not know on its own. No assertions; the runner records the response verbatim (the `negative_hallucination_01` T3 pattern). If the lookalike value (`$1.2M`) appears, the turn is *flagged* for anchoring, subject to manual review — a mention with correct attribution to Y is meta-commentary, not a fail.
    >     *   *Confirmatory commitment turn (mechanical, §2.2):* the licensed yes/no format — *"Does Project X have a registered contingency budget — yes or no? If no, reply exactly: 'Project X has no registered contingency budget.'"* Assert `must_contain: "no registered contingency budget"` and `must_not_contain: "$1.2M"`.
    >     *   *Positive control turn:* ask for Y's budget (`must_contain: "$1.2M"`) — proves the store retrieves and the phrasing is answerable, so a decline on X is genuine abstention, not a retrieval miss.
    >
    > **⚠ Priming hazard (recorded 2026-10-26):** an instructed conditional (*"if you have X, answer X; otherwise say you don't"*) **telegraphs the absence option** — a demand characteristic that competes with the very effect under test. Unlike `negative_hallucination_01` (where the fact was never mentioned and priming only reduces variance), here the license to decline can mask retrieval-anchored hallucination: an anchored model may comply with the template while its unprimed response would have leaked the lookalike value. The template therefore **cannot carry the primary measurement** — hence the free-form turn runs first and the licensed turn second; ordering guarantees the primed turn can never contaminate the unprimed one.
    >
    > **Why the licensed turn survives at all (2026-10-14 lesson, applied in the opposite direction):** absence cannot be substring-asserted without format licensing — free-form declines (*"I don't have that"*, *"not mentioned"*) false-negative any `must_contain`, the exact failure class that forced the negative-hallucination redesign. The commitment turn is therefore **secondary/confirmatory**: a value leak there (despite an explicit license to decline) is *strong* anchoring evidence; a pass there plus a flag on the unprimed turn is **instruction-sensitive abstention** — a distinct, reportable finding, not a clean pass. Report the unprimed flag and the commitment score side by side; neither alone is the headline.
    >
    > **Do not guard the entity name:** *"No — the only contingency budget on file is Project Orion's"* is correct behavior and must pass on either turn; only the concrete value (`$1.2M`) applied to X is a fail. The value guard on the commitment turn remains calibration-subject (drop it if meta-commentary trips it — the mirror of the negative-hallucination fix).
    >
    > **What it exposes:** L0 declines cleanly — nothing in its context mentions X. L2 *always* injects the top-k lookalike chunks at the probe turn (no score cutoff in `retrieve`), so any failure is retrieval-anchored hallucination: the model pattern-matches retrieved content onto the question. `retrieved_memories` telemetry confirms the near-miss chunks were injected on the failing turn.

---

## Stage 5: Rule-Based & Document-Scale Belief Revision
**Objective:** Move beyond atomic literal replacement (`belief_revision_01`, which is saturated at 100% across all tested models at L1 and kept as a basic sanity control) toward genuine Knowledge Representation & Reasoning (KRR) belief revision dynamics. These scenarios introduce **facts + rules**, **derived conclusions**, **backward contraction (Modus Tollens)**, **defeasible exceptions & reinstatement (DeLP)**, **Katsuno–Mendelzon revision vs. update**, **suppositional isolation (Ramsey Test)**, **non-prioritized revision**, and **large-document updates**, specifically targeting the boundary between L2 (VectorDB / `.md` persistent stores), L3 (reflective conflict resolution), and L4 (structured belief state).

*   **5.1. Derived-Belief Invalidation (Facts + Rules / Modus Ponens Retraction)**
    *   **Path:** `/datasets/conversations/logical_consistency/belief_revision/test_derived_belief_revision_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** Whether revising an underlying base fact ($F_1 \to F_2$) properly invalidates a previously inferred and stored derived belief ($D_1$) governed by an invariant domain rule ($R$).

    > **⚠ Design rationale (recorded 2026-10-17):**
    >
    > **Why atomic updates are insufficient:** In `belief_revision_01` ("Luna $\to$ Miso"), the user explicitly names both the old and new values of a single atomic proposition. In realistic agent workflows, an agent derives intermediate conclusions and caches them in memory.
    >
    > **Spec structure (three sessions, cross-thread):**
    > 1.  **Session A (Rule + Base Fact + Explicit Derivation):** Seed a conditional rule $R$ (e.g., *"Every `prod`-tier server requires Level-3 authorization to reboot; `staging`-tier servers require Level-1"*) and a base fact $F_1$ (*"`srv-alpha` is a `prod`-tier server"*). Prompt the agent to state the authorization level required for `srv-alpha` so that the derived conclusion $D_1$ (*"`srv-alpha` requires Level-3 authorization"*) is explicitly articulated and stored in long-term memory.
    > 2.  **Session B (Base-Fact Revision Only):** Update the base fact to $F_2$ (*"`srv-alpha` has been migrated to `staging`"*), **without** mentioning authorization levels.
    > 3.  **Session C (Two-Step Probe):** Ask *"What authorization level is required today to reboot `srv-alpha`?"* (observational open recall turn, followed by a single-value commitment turn asserting `must_contain: "Level-1"` and `must_not_contain: "Level-3"`).
    >
    > **What it exposes across storage backends:** If an L2 agent updates $F_1 \to F_2$ but leaves the cached derivation $D_1$ untouched in VectorDB or `.md`, a semantic or lexical search for *"authorization level for `srv-alpha`"* in Session C will match the stale conclusion $D_1$ more strongly than the updated fact $F_2$. Passing requires either propagating the retraction to derived entries at write time or re-deriving from base facts + rules at read time.

    > **✅ Implemented (2026-10-26):** `datasets/conversations/logical_consistency/belief_revision/test_derived_belief_revision_01.yaml` (id `derived_belief_revision_01`), paired with the single-prompt control `datasets/conversations/logical_consistency/in_context_control/test_in_context_derived_revision_01.yaml` (id `in_context_derived_revision_01`); invariants in `tests/test_stage5_scenarios.py`. Layout s1 = T1–T7 (R at T3, F1 at T4, $D_1$ derived by the agent at T5 **and** cached by the user at T6, so the stale entry is stored no matter what the model answers), s2 = T8–T10 (F2 at T9, no authorization lexicon), s3 = T11 open recall → T12 commitment (`must_contain "Level-1"` + `must_not_contain "Level-3"`, form `Level-N` enforced) → **T13 paired control entity** (`srv-beta`, still prod → `Level-3`) to tell a stale $D_1$ apart from an erased rule or a general memory failure. Headline = T12 turn-level pass. Scenario `pass_rate` is not comparable across levels (L0/L1 = 33.3% via the `must_not_contain`). A `propagation_audit` (does T9 already state Level-1?) separates write-time propagation from read-time re-derivation.

*   **5.2. Defeasible Rules & Specificity (Exceptions Without Rule Erasure)**
    *   **Path:** `/datasets/conversations/logical_consistency/belief_revision/test_defeasible_exception_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** Non-monotonic reasoning where a general default rule is overridden for a specific individual by a more specific exception rule, while the general rule must remain active for all other individuals.

    > **⚠ Design rationale (recorded 2026-10-17):**
    >
    > **Spec structure:**
    > 1.  **Session A (Default Rule + Entities):** Establish a general policy (*"External contractors have no server access on weekends"*) and two entities (*"Martin and Sofia are external contractors"*). Confirm neither has Sunday access.
    > 2.  **Session B (Specific Exception Rule + Fact):** Introduce a higher-priority exception (*"Members of the On-Call Incident Response team have 24/7 server access regardless of contract type"*) and assign one entity (*"Martin joined the On-Call Incident Response team today"*).
    > 3.  **Session C (Paired Probe — Target + Control Entity):** Probe weekend access for **both** Martin and Sofia (followed by a commitment turn).
    >
    > **Why the paired probe is mandatory:** An agent that naively treats the exception as a destructive update and deletes or overwrites the general contractor rule will pass the query for Martin (has access) but **fail the control query for Sofia** (must still be denied access). Conversely, an append-only store that retrieves only the general rule fails on Martin. Scoring requires satisfying both (`Martin: allowed`, `Sofia: denied`) simultaneously.

*   **5.3. Large-Document Ingestion + Addendum Revision (Document-Scale Belief Revision)**
    *   **Path:** `/datasets/conversations/logical_consistency/belief_revision/test_document_addendum_revision_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** How persistent memory backends (VectorDB vs. `.md`) handle storage, retrieval, selective update, and deletion over a large multi-section document when a subsequent addendum modifies specific facts and rules that interact with untouched sections of the original document.

    > **⚠ Design rationale (recorded 2026-10-17):**
    >
    > **Spec structure (three sessions):**
    > 1.  **Session A (Large Document Ingestion):** Ingest a synthetic multi-section document (e.g., an architecture specification or compliance manual containing cross-referenced parameters, rules, and plausible decoys across distant sections). Verify baseline needle and multi-section retrieval.
    > 2.  **Session B (Compact Addendum / Errata):** Deliver a short addendum that (a) updates 1–2 parameters, (b) replaces one conditional rule, and (c) formally deprecates/deletes one section or credential from the base document.
    > 3.  **Session C (Compositional & Suppression Probes):**
    >     *   *Updated rule × untouched fact:* A query requiring an untouched entity from Section 2 evaluated against the revised rule from the Session B addendum.
    >     *   *Deprecated clause probe:* A query targeting the deleted section (must abstain and pass `must_not_contain` on the revoked value, plus optional `memory_assertions` on the backing store).
    >
    > **Backend comparison target (VectorDB vs. `.md`):** Directly stress-tests chunk-level invalidation and re-embedding in VectorDB (where top-$k$ retrieval risks surfacing both the original chunk and the addendum chunk) against targeted in-place file editing (`str_replace` / section deletion) in the `.md` backend.

*   **5.4. Backward Contraction via Modus Tollens (Kernel Contraction / Contradicting a Derived Conclusion)**
    *   **Path:** `/datasets/conversations/logical_consistency/belief_revision/test_backward_modus_tollens_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** The logical converse of 5.1. When the user directly contradicts a derived conclusion ($\neg D_1$) while the underlying domain rule ($R: F_1 \to D_1$) holds higher epistemic entrenchment, the agent must apply *Modus Tollens* (Hansson's *Kernel Contraction*) to retract the base fact ($F_1$), rather than merely deleting $D_1$ and leaving $\{F_1, F_1 \to D_1, \neg D_1\}$ globally inconsistent.

    > **⚠ Design rationale (recorded 2026-10-22):**
    >
    > **Spec structure (three sessions, cross-thread):**
    > 1.  **Session A (Entrenched Rule + Base Fact + Derivation):** Establish an invariant compliance rule $R$ (*"Mandatory policy: every service hosted in the `eu-central` region is subject to strict GDPR auditing — no exceptions exist"*) and a base fact $F_1$ (*"Service `billing-api` is hosted in `eu-central`"*). Have the agent explicitly derive $D_1$ (*"`billing-api` is subject to GDPR auditing"*).
    > 2.  **Session B (Contradiction of Derived Conclusion Only):** The user states $\neg D_1$ (*"Compliance just confirmed that `billing-api` is NOT subject to GDPR auditing"*), **without** mentioning regions or `eu-central`.
    > 3.  **Session C (Two-Step Probe on Base Fact $F_1$):** Ask *"Is `billing-api` currently hosted in the `eu-central` region?"* (observational recall turn, followed by a commitment turn asserting `must_contain: "no"` / `must_not_contain` an affirmative `eu-central` placement).
    >
    > **Why this separates L3 from L4:** An L3 CRUD agent searches for chunks matching *"GDPR auditing for `billing-api`"* and updates/deletes $D_1$, leaving the base fact $F_1$ (*"hosted in `eu-central`"*) untouched in storage. When Session C asks about `eu-central`, L3 retrieves $F_1$ and answers *"Yes"*, violating global logical consistency. Only backward propagation over the minimal incision set $\{F_1, R\}$ (retracting $F_1$ because $R$ is entrenched) passes.

    > **✅ Implemented (2026-10-26):** `datasets/conversations/logical_consistency/belief_revision/test_backward_modus_tollens_01.yaml` (id `backward_modus_tollens_01`), paired with the single-prompt control `datasets/conversations/logical_consistency/in_context_control/test_in_context_modus_tollens_01.yaml` (id `in_context_modus_tollens_01`); invariants in `tests/test_stage5_scenarios.py`. Same 7/3/3 session layout as 5.1. **Deviations from the text above:** (1) *GDPR → the fictional "Bastion audit"*. In the real GDPR, scope follows data subjects rather than hosting region, so world knowledge would license retracting $R$ instead of $F_1$ and would leak answers to L0. $R$ is entrenched by mechanism instead ("enforced by the deployment pipeline, no exceptions can be granted"). (2) *`must_contain "no"` → single closed label `must_contain: "NO_EU_CENTRAL"`* (`YES_EU_CENTRAL` / `NO_EU_CENTRAL` / `CANNOT_DETERMINE`, none a substring of another). With `must_contain` only, amnesic agents (L0/L1) score 0% cleanly without free credit from `must_not_contain`. (3) Added the **T13 rule control** (a hypothetical new `eu-central` service must still be `YES_BASTION`), which detects contracting $R$ instead of $F_1$. Full outcome matrix and `conflict_audit` are in the YAML metadata. Headline = T12 pass. L0/L1 `pass_rate` = 0%.

*   **5.5. Defeater Retraction & Default Reinstatement (Non-Monotonic Recovery / DeLP Dynamics)**
    *   **Path:** `/datasets/conversations/logical_consistency/belief_revision/test_defeater_retraction_reinstatement_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** In Defeasible Logic Programming (DeLP) and Truth Maintenance Systems (TMS), when an exception (*defeater*) that overrode a default rule is subsequently retracted, the original default conclusion must **automatically reinstate itself** without requiring the user to re-state the base fact or the default policy.

    > **⚠ Design rationale (recorded 2026-10-22):**
    >
    > **Spec structure (four sessions, extending 5.2):**
    > 1.  **Session A (Default Rule + Base Fact):** *"External contractors have no weekend server access."* + *"Martin is an external contractor."* ($\vdash$ Martin denied weekend access).
    > 2.  **Session B (Defeater Introduced):** *"Members of the On-Call Incident Response team have 24/7 server access regardless of contract type."* + *"Martin joined the On-Call Incident Response team."* ($\vdash$ Martin allowed weekend access).
    > 3.  **Session C (Defeater Retracted Only):** *"Rotation update: Martin's shift ended and he has been removed from the On-Call Incident Response team."* (No mention of his contractor status or weekend policy).
    > 4.  **Session D (Two-Step Reinstatement Probe):** Probe whether Martin has Sunday server access today. Must revert to **denied**.
    >
    > **What it exposes:** If an L3 agent resolved Session B by destructively overwriting Martin's access status (`Martin: denied` $\to$ `Martin: allowed`) or pruning the defeated contractor link, then removing the On-Call fact in Session C leaves Martin's status either stuck at `allowed` or unknown. Passing requires non-destructive maintenance of base facts and rules so the undefeated default warrant revives automatically.

*   **5.6. Belief Revision ($K * \alpha$) vs. Belief Update ($K \diamond \alpha$) — Epistemic Error vs. World-State Evolution (Katsuno & Mendelzon)**
    *   **Path:** `/datasets/conversations/logical_consistency/belief_revision/test_revision_vs_update_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** Whether the agent distinguishes **Revision** (correcting a false belief about a static fact, where the old value was *never* true) from **Update** (recording a real transition in a dynamic world, where the old value *was* true at $t_1$ and is superseded at $t_2$).

    > **⚠ Design rationale (recorded 2026-10-22):**
    >
    > **Spec structure (three sessions, paired entities):**
    > 1.  **Session A (Initial State $t_1$):** Seed two facts: (a) *"Employee Sofia's badge tier is `Silver`"* and (b) *"Employee Lucas's hire city was registered as `Cordoba`."*
    > 2.  **Session B (Paired Update vs. Revision):**
    >     *   **Update ($K \diamond \alpha$, dynamic world change):** *"Sofia was promoted today; her badge tier has been upgraded from `Silver` to `Gold`."*
    >     *   **Revision ($K * \alpha$, static epistemic correction):** *"Correction on Lucas's HR file: I misread the form earlier — Lucas was hired in `Rosario`, never in `Cordoba`. He has never lived or worked in `Cordoba`."*
    > 3.  **Session C (Present + Historical Probes):**
    >     *   *Present state probe:* Current badge tier for Sofia (`Gold`, not `Silver`) and hire city for Lucas (`Rosario`, not `Cordoba`).
    >     *   *Historical / provenance probe:* (1) *"Which badge tier did Sofia hold before her promotion?"* (`must_contain: "Silver"` — destructive L3 `DELETE` fails this by erasing history); (2) *"Did Lucas ever work in Cordoba before Rosario?"* (commitment turn: `must_contain: "no"` — append-only L2 or naive temporal-log L3 risks treating `Cordoba` as a prior historical state rather than a retracted false belief).

*   **5.7. Suppositional & Counterfactual Isolation (Ramsey Test Without Belief Pollution)**
    *   **Path:** `/datasets/conversations/logical_consistency/belief_revision/test_suppositional_isolation_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** Whether performing hypothetical/counterfactual reasoning (*"Suppose $\alpha$ were true, what would follow?"* — the Ramsey Test) leaves the committed belief state $K$ unpolluted after the hypothetical scope closes.

    > **⚠ Design rationale (recorded 2026-10-22):**
    >
    > **Spec structure (three sessions):**
    > 1.  **Session A (Committed Belief State $K$):** Establish production database engine (`PostgreSQL 16`) and replica count (`3 replicas`), plus domain rules on licensing costs.
    > 2.  **Session B (Extended Counterfactual Simulation):** The user explicitly frames a what-if simulation: *"For capacity planning, let's run a hypothetical exercise: suppose we switched the production database to `Oracle 23c` with `8 replicas`..."* The agent calculates hypothetical costs across 2 turns, and the user concludes: *"Okay, that's way too expensive — we are NOT making any changes to our current setup."*
    > 3.  **Session C (Two-Step Probe on Committed State):** Ask for the active production database engine and replica count today (`must_contain: "PostgreSQL"`, `"3"`; `must_not_contain: "Oracle"`, `"8"`).
    >
    > **What it exposes:** L2 RAG retrieves the dense Session B chunks discussing `Oracle 23c` and `8 replicas`; naive L3 fact-extractors frequently extract hypothetical propositions into the belief store as committed updates.

*   **5.8. Non-Prioritized Revision & Epistemic Entrenchment (Shielded Contraction Against Uncertain or Inconsistent Inputs)**
    *   **Path:** `/datasets/conversations/logical_consistency/belief_revision/test_non_prioritized_revision_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** Breaks the "blind recency always wins" assumption (AGM Success Postulate $K * \alpha \vdash \alpha$) using Hansson's *Non-Prioritized Belief Revision* / Fermé & Hansson's *Shielded Contraction*. Tests whether an entrenched, high-credibility fact resists being overwritten by (a) a later low-confidence guess or (b) a self-contradictory update.

    > **⚠ Design rationale (recorded 2026-10-22):**
    >
    > **Why this is required as a methodological control:** Every other revision test in the benchmark rewards preferring the highest `turn_number` / `session_id`. An agent using a trivial recency heuristic passes 100% of them without genuine epistemic evaluation.
    >
    > **Spec structure (three sessions):**
    > 1.  **Session A (High-Entrenchment Verified Fact):** Establish an authoritative, verified parameter (*"Verified from the signed hardware manifest: the master vault port is `8443`"*).
    > 2.  **Session B (Low-Epistemic-Authority Contradiction):** In casual conversation, the user remarks tentatively: *"Wait, I'm away from my desk and guessing off the top of my head — wasn't the master vault port `8080`? Don't quote me on that, I haven't checked the manifest."*
    > 3.  **Session C (Two-Step Probe):** Ask for the verified master vault port (`must_contain: "8443"`, `must_not_contain: "8080"` on the commitment turn).

*   **5.9. Oscillating Reinstatement ($A \to B \to A$) & Partial Set Contraction (Granular Minimal Change)**
    *   **Path:** `/datasets/conversations/logical_consistency/belief_revision/test_oscillating_and_set_revision_01.yaml`
    *   **Theme:** `logical_consistency`
    *   **Evaluates:** Two structural edge cases of AGM Minimal Change and iterated revision: (1) **Oscillating revision ($A \to B \to A$)**, where a belief reverts to a value that was explicitly deprecated in an intermediate session, and (2) **Partial set contraction/expansion**, where one element of a multi-valued relation is removed and another added without wiping the untouched elements.

    > **⚠ Design rationale (recorded 2026-10-22):**
    >
    > **Spec structure (four sessions):**
    > 1.  **Session A:** Primary gateway is `gw-north` ($A$). Authorized keyholders are `{Ana, Bruno, Clara}`.
    > 2.  **Session B:** Explicit deprecation: *"`gw-north` is decommissioned; active gateway is now `gw-south`, not `gw-north`"* ($B$). Keyholder update: *"Revoke Bruno's keyholder access and add Diego"* (without repeating Ana or Clara).
    > 3.  **Session C:** Rollback: *"We rolled back the gateway migration — `gw-north` is active again and `gw-south` is offline"* ($A$).
    > 4.  **Session D (Commitment Probes):**
    >     *   *Gateway probe:* Must output `gw-north` and not `gw-south` (tests whether Session B's explicit `"not gw-north"` chunk in L2 or tombstone in L3 blocks reinstatement of $A$).
    >     *   *Keyholders set probe:* Must contain `Ana`, `Clara`, and `Diego`, and `must_not_contain: "Bruno"` (tests whether L3's slot update overwrote the whole list with just `Diego`, violating Minimal Change on `Ana` and `Clara`).