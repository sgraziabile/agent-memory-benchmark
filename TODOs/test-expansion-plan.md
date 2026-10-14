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
*   **3.2. Deep Temporal Multi-Hop (3+ Hops)**
    *   **Path:** `/datasets/conversations/test_3hop_temporal_01.yaml`
    *   **Theme:** `compositional_reasoning`
    *   **Evaluates:** A chain of facts: A relates to B, B relates to C, C relates to D. The agent must deduce D from A across multiple conversational turns.

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