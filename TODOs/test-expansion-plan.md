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
**Objective:** Measure the operational costs and security vulnerabilities introduced by persistent memory architectures.

*   **4.1. Computational Overhead Baseline**
    *   **Path:** `/datasets/conversations/test_overhead_baseline_01.yaml`
    *   **Theme:** `computational_cost`
    *   **Evaluates:** A strictly linear, simple Q&A conversation. Designed to have a 100% pass rate. Used exclusively to isolate and measure the raw latency (ms) and token overhead of memory retrieval mechanics (P3).
*   **4.2. Memory Prompt Injection**
    *   **Path:** `/datasets/conversations/test_memory_injection_01.yaml`
    *   **Theme:** `robustness_security`
    *   **Evaluates:** Resistance to adversarial attacks. The user inputs a "System Override" command instructing the agent to purge its memory and adopt a new persona. Tests if the memory store can be poisoned or erased maliciously.

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