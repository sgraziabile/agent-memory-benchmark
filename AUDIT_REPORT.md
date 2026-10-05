# 🔍 Repository Audit Report: `agent-memory-benchmark`

**Auditor role:** Principal Software Architect / Lead AI Engineer
**Date:** October 5, 2026
**Scope:** Full workspace — 26 tracked files, ~1,900 LOC of Python, 6 YAML scenarios, 3 config registries, 1 test suite, README + SPEC.

**Verification performed:** Full file-tree inspection, line-by-line review of `core/`, `agents/`, `run.py`, `tests/`, configs, and datasets; live execution of `pytest` (28 passed in 4.78s, no API keys required); live `pip install -e . --no-deps --dry-run` (fails with `OSError: Readme file does not exist: docs/SPEC.md`); dependency audit via `pip show`; git-tracking hygiene audit.

> **📌 Revision 2 (Oct 5, 2026):** Issues **§3.1** (broken packaging path) and the `[cite: N]` artifacts from **§4 / action item 11** have been **fixed and live-verified** — see status markers inline and the [Changelog](#changelog) at the end of this report.

---

## Table of Contents

1. [Executive Summary & Production Readiness Score](#1-executive-summary--production-readiness-score)
2. [Strengths (What Is Already Well Engineered)](#2-strengths-what-is-already-well-engineered)
3. [Areas for Improvement (Refactoring & Code Smells)](#3-areas-for-improvement-refactoring--code-smells)
4. [Critical Missing Elements (Gaps for Portfolio & CV Backing)](#4-critical-missing-elements-gaps-for-portfolio--cv-backing)
5. [Prioritized Action Plan (24–48 Hours)](#5-prioritized-action-plan-24-48-hours)
6. [Changelog (Revision History)](#changelog)

---

## 1. Executive Summary & Production Readiness Score

### Overall Score: **6.5 / 10**

**What it does well:** The repository demonstrates genuinely strong architectural thinking — a disciplined "Zero Hardcoding" injection pattern (`config["configurable"]`), modern LangGraph idioms (`Command` routing, `add_messages` reducers, ephemeral system messages), Pydantic v2 contracts everywhere, and an Evaluation-Driven Development methodology with a deterministic, API-key-free test suite that passes 28/28. The README is among the better ones I've seen: research questions, mermaid diagrams, ablation-ladder roadmap, and a sample-results section with actual scientific interpretation (positive control → "persistence problem, not reasoning problem").

**Single biggest blocker (at audit time):** The documented 1-command quickstart is **broken** — live-verified: `pip install -e ".[dev]"` fails with `OSError: Readme file does not exist: docs/SPEC.md` (`pyproject.toml:5` points at a nonexistent path). Combined with an undeclared runtime dependency (`python-dotenv`), zero CI, zero linting/typing gates, and no lockfile, the project fails the "clone → install → verify" recruiter smoke-test on the very first command. That is a fatal first impression for a portfolio piece, even though the underlying architecture would earn a 7.5–8 on its own.

**✅ Update (Rev 2, Oct 5, 2026):** The packaging blocker is **resolved** — `pyproject.toml` now points at the root-level `SPEC.md` and `pip install -e .` completes successfully (live-verified: `Preparing editable metadata ... finished with status 'done'`). The remaining top blockers are the undeclared `python-dotenv` with its silent fallback (§3.2), zero CI/linting/typing gates (§4), and error-turn metric contamination (§3.3). **Revised standing: 6.8 / 10** (from 6.5 — the fatal first-contact failure is gone, but nothing yet publicly attests to the tests passing).

---

## 2. Strengths (What Is Already Well Engineered)

- **✅ Rigorous Zero-Hardcoding injection architecture.** `agents/level_0_reactive/graph.py:71–82` — `model_node` resolves `model_id`, `system_prompt`, `tools`, and `model_kwargs` exclusively from `config["configurable"]`; nothing is bound at build time. The abstract contract in `agents/base.py:24–50` enforces this across all future agent levels (L1–L4). Textbook dependency inversion applied to agent graphs; directly supports the Cartesian-product sweep design.

- **✅ Modern, correct LangGraph idioms.**
  - `core/schemas.py:22–30` — `AgentState` uses `Annotated[Sequence[BaseMessage], add_messages]` (correct channel reducer; messages append, never overwrite).
  - `agents/level_0_reactive/graph.py:100–107` — conditional routing via node-returned `Command(update=..., goto=...)` instead of legacy `add_conditional_edges()`. This is the current post-0.2.60 idiom and shows framework fluency.
  - `graph.py:170–171` — L0 is compiled **explicitly without checkpointer/store**, and `tests/test_scaffolding.py:175–178` asserts `graph.checkpointer is None`, turning an architectural invariant into a regression test. Excellent EDD discipline.
  - `graph.py:88–89` — the system prompt is prepended as an *ephemeral* `SystemMessage` per invocation and never written into state, avoiding the classic "system prompt accumulation" bug across multi-turn runs. Subtle and correct.

- **✅ Multi-provider model factory with lazy imports.** `core/model_factory.py:39–45, 86–115` — a dispatch table (`PROVIDER_MAP`) + `importlib` lazy loading means a missing optional provider (e.g., `langchain-ollama`) degrades gracefully with an actionable error (`"Install it with: pip install langchain-ollama"`) instead of an import-time crash. Dual resolution (YAML registry vs. inline `provider:model`) is a nice ergonomic touch.

- **✅ Pydantic v2 contracts with full docstrings.** `core/schemas.py` — `Literal`-constrained assertion types (`must_contain`/`must_not_contain`), `Field(default_factory=list)` (correct mutable-default handling), and Google-style docstrings with `Args/Returns/Raises` on nearly every function. The file reads like an internal library, not a thesis script.

- **✅ Deterministic, hermetic test suite (verified live).** `pytest tests/ -q` → **28 passed in 4.78s, zero API calls, zero network**. Schema validation, assertion-engine semantics (case sensitivity both ways), text-block extraction, graph compilation with/without tools, checkpointer absence, and scenario loading are all covered. The README's claim "Run harness tests (no API keys needed)" is true — rare honesty in LLM repos.

- **✅ Measurement-aware result schemas.** `BenchmarkRunSummary`/`TurnResult` (`core/schemas.py:92–157`) capture latency, pass rate, and token accounting (input/output/total). The runner correctly prefers the standard `AIMessage.usage_metadata` and falls back to `response_metadata["usage"]` (`core/runner.py:277–280`), then normalizes both `input_tokens|prompt_tokens` naming variants during aggregation (`runner.py:330–337`). Cross-provider token accounting is handled properly.

- **✅ Experiment design maturity beyond the code.**
  - A **positive control** scenario (`test_in_context_control_01.yaml`) that establishes the in-context upper bound — real experimental methodology (isolate the variable: persistence vs. reasoning).
  - Per-scenario `metadata.expected_level_0_behavior` documents the *hypothesized* failure mode before running — falsifiable predictions, exactly what EDD should look like.
  - `thread_id` is derived from scenario + UUID (`runner.py:243`) so L1+ checkpointed runs stay isolated while the runner code remains identical across ablation levels — thoughtful forward compatibility.

- **✅ Multi-format export with sensible details.** `core/runner.py:446–553` — JSON (full fidelity), CSV with BOM (`utf-8-sig` for Excel compatibility — a detail that shows operational empathy), and a hand-formatted Markdown table with aligned numeric columns.

- **✅ Good secret hygiene.** `.env` is untracked (verified via `git ls-files`), `.env.example` documents all five provider keys, `outputs/runs/*` is ignored with a `.gitkeep` exception, and the working tree carries no committed artifacts.

- **✅ Strong README storytelling.** Research-question table (P1–P3), three mermaid diagrams (pipeline, topology, dynamic-injection sequence), scenario matrix mapped to research questions, and a "Sample Results" section with genuine interpretation rather than raw numbers.

---

## 3. Areas for Improvement (Refactoring & Code Smells)

### 3.1 🔴 Broken packaging metadata — ✅ RESOLVED (Rev 2)
- **Location:** `pyproject.toml:5` — `readme = "docs/SPEC.md"`.
- **Issue (at audit time):** `SPEC.md` lives at the repo root; `docs/` does not exist. **Live-verified:** `pip install -e . --dry-run` → `OSError: Readme file does not exist: docs/SPEC.md` → `metadata-generation-failed`. Every recruiter who follows the README quickstart hits an error before a single line of your code runs. It also means the wheel metadata is broken for any distribution.
- **Resolution (Oct 5, 2026):** Changed to `readme = "SPEC.md"` (root-level SPEC retained as the project readme per maintainer's choice, rather than switching to `README.md` as originally suggested). **Live-verified:** `pip install -e . --no-deps --dry-run` now reports `finished with status 'done'` and `Would install agent-memory-benchmark-0.1.0`.
- **Follow-up:** a CI step running `pip install -e ".[dev]"` (action item 3) would have caught this automatically and will guard against regressions.

### 3.2 🔴 Undeclared runtime dependency with silent fallback (`python-dotenv`)
- **Location:** `run.py:30–35`; `pyproject.toml:11–20` (dependency list).
- **Issue:** `run.py` does `try: from dotenv import load_dotenv ... except ImportError: pass`. `python-dotenv` is **not** in `pyproject.toml` (verified: it's only installed in your global interpreter by accident). In a fresh venv, the `except ImportError: pass` swallows the problem and `.env` is **silently never loaded** — the user's API key is ignored, and they get a confusing provider auth error despite having done exactly what the README said. A silent except around an *expected* code path is a harness leak: it hides a missing dependency by design.
- **Suggested refactor:**
  ```toml
  # pyproject.toml — declare it
  dependencies = [
      "python-dotenv>=1.0.0",
      ...
  ]
  ```
  ```python
  # run.py — import unconditionally once declared; fail loudly, not silently
  from dotenv import load_dotenv
  load_dotenv()
  ```
  If you genuinely want dotenv optional, at least warn: `logger.warning("python-dotenv not installed — .env will NOT be loaded")`.

### 3.3 🔴 Error turns contaminate benchmark metrics
- **Location:** `core/runner.py:283–290` (exception handler inside `run_single`) and `runner.py:292–309`.
- **Issue:** Any per-turn exception (API outage, rate limit, malformed response) is converted into `ai_response = f"[ERROR] {type(exc).__name__}: {exc}"` and then **fed through `evaluate_assertions`** as if it were a model response. An infrastructure failure is scored as a 0% benchmark result and exported to CSV/JSON indistinguishably from genuine agent failure. For a scientific harness this is invalid data: your `pass_rate` conflates model behavior with harness reliability.
- **Suggested refactor:** Add an explicit status to `TurnResult` and exclude errored turns from `pass_rate`:
  ```python
  # schemas.py
  class TurnResult(BaseModel):
      ...
      status: Literal["ok", "error"] = "ok"
      error: str | None = None

  # runner.py — inside the except block:
  turn_results.append(TurnResult(
      turn_number=turn.turn_number,
      user_input=turn.content,
      agent_response="",
      status="error",
      error=f"{type(exc).__name__}: {exc}",
      latency_ms=latency_ms,
  ))
  continue  # skip assertion evaluation entirely

  # and in run_matrix's tuple-level except (runner.py:428-436): append a
  # BenchmarkRunSummary with status="error" (or a FailedRunRecord) so failed
  # tuples appear in exports instead of silently vanishing.
  ```

### 3.4 🔴 Per-turn model construction pollutes the latency metric (P3 validity threat)
- **Location:** `agents/level_0_reactive/graph.py:82` (`model = create_model(model_id, **model_kwargs)`) + `core/model_factory.py:161` (registry path re-reads and re-parses `models.yaml` **on every call**) — both execute *inside* `agent_graph.invoke()`, which sits between `t_start`/`t_end` in `runner.py:259–266`.
- **Issue:** You are measuring disk IO (YAML load), import machinery, and HTTP-client pool construction per turn and reporting it as "agent latency." Since P3 is literally *"What latency overhead does persistence introduce?"*, this systematically inflates the baseline you'll compare against. It also wastes IO on every single turn of every run in the 216-tuple matrix.
- **Suggested refactor:** Cache instances in the factory (models are stateless per-config):
  ```python
  # model_factory.py
  from functools import lru_cache

  @lru_cache(maxsize=32)
  def _cached_create(model_id: str, temperature: float, kwargs_key: tuple) -> BaseChatModel:
      return _create_model_impl(model_id, temperature, dict(kwargs_key))

  def create_model(model_id, temperature=0.0, config_path=..., **kwargs):
      return _cached_create(model_id, temperature, tuple(sorted(kwargs.items())))
  ```
  Alternatively, keep instantiation but load the registry once at `BenchmarkRunner.__init__` and pass entries via `configurable["model_kwargs"]`.

### 3.5 🟠 Methodological contradiction: `strict_epistemic` prompt vs. `must_not_contain` assertions
- **Location:** `configs/prompts.yaml:36–37` (rule 4: *"reference when they were introduced or corrected … e.g., 'You initially said X, then corrected it to Y'"*) vs. `datasets/conversations/test_belief_revision_01.yaml:50–52, 64–66` (`must_not_contain: "Luna"` on turns 4–5).
- **Issue:** A model that *correctly obeys* the strict prompt will say something like "You initially said **Luna**, then corrected it to **Miso**" — and your assertion engine will score it as a **failure**. The benchmark penalizes exactly the behavior the prompt mandates. Any reviewer with LLM-eval experience will spot this in five minutes, and it undermines the headline `28.6%` result for `strict_epistemic` runs.
- **Suggested refactor:** Either (a) soften the prompt for benchmark runs (drop the "reference when introduced" rule), or (b) switch to regex/semantic assertions that target the *final answer clause* rather than the whole response, or (c) minimum viable fix — scope assertions:
  ```yaml
  - type: must_not_contain
    value: "Luna"
    match: final_answer   # evaluate only the last sentence / quoted answer
  ```
  Even a `match_scope` field (Pydantic `Literal["full_response","final_sentence"]`) shows you understand eval gaming.

### 3.6 🟠 Substring-only assertion engine — gameable and coarse
- **Location:** `core/schemas.py:36–48` (`AssertionRule`), `core/runner.py:159–195` (`evaluate_assertions`).
- **Issue:** Only two operators (`must_contain`/`must_not_contain`), substring semantics. `must_contain: "Miso"` passes if the model rambles "Miso? No, I don't recall a Miso." — precision is unmeasured. There's no regex, no numeric tolerance, no per-turn weighting. For a thesis that claims to measure "conflict resolution accuracy," substring matching is the weakest link. Also: case-insensitivity uses `.lower()` (`runner.py:176–177`) instead of `.casefold()` — subtle i18n bug (e.g., `ß` vs `SS`, Turkish dotless i).
- **Suggested refactor:** Extend the discriminated union — this is where Pydantic v2 shines:
  ```python
  class SubstringRule(BaseModel):
      type: Literal["must_contain", "must_not_contain"] = Field(alias="type")
      value: str
      case_sensitive: bool = False

  class RegexRule(BaseModel):
      type: Literal["must_match_regex"]
      pattern: str

  AssertionRule = Annotated[Union[SubstringRule, RegexRule], Field(discriminator="type")]
  ```
  And replace `.lower()` with `.casefold()` in the matcher.

### 3.7 🟠 Docstring/code drift in the assertion engine
- **Location:** `core/runner.py:171–172` (docstring: "keys: `rule_type`, `value`, `passed`, `actual_found`") vs. `runner.py:191` (actual key: `found_in_response`).
- **Issue:** The documented contract doesn't match the emitted dict. Small, but this is exactly the kind of drift that makes a code reviewer distrust the rest of the docs. The untyped `list[dict[str, Any]]` return also breaks the otherwise-consistent "everything is a Pydantic model" story.
- **Suggested refactor:**
  ```python
  class AssertionOutcome(BaseModel):
      rule_type: Literal["must_contain", "must_not_contain"]
      value: str
      passed: bool
      found_in_response: bool
  ```
  Return `list[AssertionOutcome]` and `model_dump()` at the export boundary only.

### 3.8 🟠 `system_event` turns are a silent no-op
- **Location:** `core/runner.py:248–251` (`if turn.role == "system_event": ... continue`) vs. `core/schemas.py:56–58` ("`system_event` for injected environmental changes").
- **Issue:** The schema advertises environmental injection; the runner only logs the event and discards it — nothing is injected anywhere. Currently it's dead weight that misleads dataset authors into thinking system events do something. In a *memory* benchmark, a system event should mutate the persisted context (the interesting test case).
- **Suggested refactor:** Either implement the semantics (for L1+: write to the thread's store/checkpointer; for L0: inject as a `SystemMessage` into the message list) or remove the role until L2 lands. Do not ship a contract you don't honor.

### 3.9 🟠 Silent dataset loss in `load_scenarios`
- **Location:** `core/runner.py:111–115` — `except Exception as exc: logger.warning("Skipping invalid scenario ...")`.
- **Issue:** A typo in one YAML file silently shrinks the experimental matrix. You'd run 5 scenarios believing you ran 6, and the Cartesian product quietly changes size. For a scientific harness, partial data loss must be loud.
- **Suggested refactor:** Fail fast by default, opt out explicitly:
  ```python
  def load_scenarios(self, filter_ids=None, *, strict: bool = True) -> list[BenchmarkScenario]:
      ...
      except ValidationError as exc:
          if strict:
              raise ValueError(f"Invalid scenario file {yaml_path.name}: {exc}") from exc
          logger.warning(...)
  ```

### 3.10 🟠 Leaky abstraction: CLI calls a private method
- **Location:** `run.py:198` — `run_dir = runner._export_results(results)`.
- **Issue:** The CLI reaches into `BenchmarkRunner._export_results` (underscore-private) for the inline-model path. It works today, breaks silently the day you refactor the export internals.
- **Suggested refactor:** Add a public API that composes both: `results, run_dir = runner.run_single_and_export(...)` — or expose `export_results()` publicly and make `_export_results` an alias.

### 3.11 🟡 Loose typing at the architectural seams
- **Location:** `core/runner.py:205, 372` (`agent_graph: Any`), `agents/base.py:25, 41` (`build_graph(...) -> Any`), `agents/level_0_reactive/graph.py:110–121` (`make_tools_node` — untyped params/return).
- **Issue:** The most important contract in the system — "a compiled graph the runner can invoke" — is typed as `Any`, defeating static analysis at exactly the boundary where a typo would only surface at runtime. Everything else is carefully typed, which makes these `Any`s conspicuous.
- **Suggested refactor:**
  ```python
  from langgraph.graph.state import CompiledStateGraph

  def build_graph(self, tools: list[BaseTool] | None = None) -> CompiledStateGraph: ...
  def run_single(self, ..., agent_graph: CompiledStateGraph, ...) -> BenchmarkRunSummary: ...
  ```

### 3.12 🟡 Brittle tests coupled to config values
- **Location:** `tests/test_scaffolding.py:100–102` — `assert cfg.get("timeout") == 60.0`, `max_retries == 2`, `max_tokens == 300` for *every* model.
- **Issue:** These assert current YAML *content*, not invariants. Change one model's `timeout` to 120 and the suite goes red for a legitimate configuration change. Tests should pin behavior and schema, not data values.
- **Suggested refactor:** Assert structural invariants only: keys exist, types are correct (`isinstance(cfg["timeout"], (int, float))`), provider is in `PROVIDER_MAP` (that one is good — keep it).

### 3.13 🟡 Missing `model_id` raises bare `KeyError`
- **Location:** `agents/level_0_reactive/graph.py:74` — `configurable["model_id"]`.
- **Issue:** A misconfigured invocation surfaces as `KeyError: 'model_id'` deep inside graph execution instead of an actionable message.
- **Suggested refactor:**
  ```python
  try:
      model_id = configurable["model_id"]
  except KeyError as exc:
      raise ValueError(
          "Missing 'model_id' in config['configurable'] — "
          "the runner must inject it at invoke time (Zero Hardcoding, SPEC §2)"
      ) from exc
  ```

### 3.14 🟡 Unhandled tool-loop termination + UTC/local time inconsistency
- **Location:** `agents/level_0_reactive/graph.py:101–102` (model emits `tool_calls` but `tools` is falsy → silently routes to END, dropping the calls); `core/runner.py:455, 537` (`datetime.now()` local time for dir names / "Generated at") vs. `runner.py:225` (UTC ISO for data).
- **Issue:** (a) A model that wants tools in a tool-less run produces a truncated, silently-wrong conversation — worth at least a `logger.warning`. (b) Timestamps are UTC in `results.json` but local in filenames and `summary.md`, so output artifacts can disagree with the data they summarize by hours.
- **Suggested refactor:** Log a warning on dropped tool calls; use `datetime.now(timezone.utc)` consistently (or record both `timestamp_utc` and `timestamp_local` explicitly).

### 3.15 🟡 `_export_results` violates SRP (~110 lines)
- **Location:** `core/runner.py:446–553`.
- **Issue:** One method serializes JSON, writes CSV, and hand-rolls a Markdown table formatter (lines 519–548 are a mini text-alignment library). It's the largest method in the codebase and will grow with each new format (you'll want Parquet/JSONL soon for statistical analysis).
- **Suggested refactor:** Extract `core/exporters.py` with `JsonExporter`, `CsvExporter`, `MarkdownExporter` classes behind a small `Exporter` protocol; the runner just fans out. This also makes exporters unit-testable in isolation.

---

## 4. Critical Missing Elements (Gaps for Portfolio & CV Backing)

### Tooling & CI/CD — **entirely absent**
- ❌ **No CI pipeline.** No `.github/workflows/` exists (verified). There is no public, third-party-attested proof that the tests pass. A recruiter's first credibility check — the green badge at the top of the README — is missing. For a repo whose whole thesis is *"assertions validate architecture,"* the repo itself is unvalidated in public view. This is the highest-leverage gap in the portfolio.
- ❌ No linter/formatter config (`ruff`), no type checker (`mypy` or `pyright`), no `pre-commit-config.yaml`, no `Makefile`/`justfile` task runner, no `pytest-cov` coverage gate.
- ❌ **No lockfile** — no `uv.lock`, `poetry.lock`, or `requirements.txt` (verified). All deps are lower-bound-only (`langgraph>=0.2.0`, `pydantic>=2.0`). The environment you benchmarked with cannot be reproduced; a re-clone next month may resolve to different library majors (note: you're actually running langgraph 1.2.12 / langchain-core 1.6.5, well past the declared floors — the pyproject doesn't reflect reality). For a *benchmark*, non-reproducible environments undermine every number in `outputs/`.
- ❌ No `[project.scripts]` console entry point (`agent-memory-benchmark = "run:main"`); invocation depends on CWD-relative imports (`python run.py`).
- ❌ `.pytest_cache/` is not in `.gitignore` (it's currently untracked by luck — a single `git add .` would commit cache internals).

### Testing & E2E Validation
- ❌ **Zero LLM mocking.** There is no `FakeChatModel`/`GenericFakeChatModel` (available in `langchain_core.language_models.fake_chat_models`) or stubbed `create_model` fixture. Consequently, the most business-critical code paths — `run_single` turn loop, latency capture, token aggregation, `system_event` skip logic, error-path handling, and all of `_export_results` — have **no test coverage at all**. Your 28 tests cover the scaffolding; nothing covers an actual benchmark execution end-to-end.
- ❌ No edge-case tests: empty scenario, zero-assertion scenario (pass_rate division is guarded at `runner.py:349` but untested), unknown scenario ID, all-turns-errored summary, CSV/Markdown export content assertions.
- ❌ No regression test for the packaging bug (~~a trivial `pip install -e . --dry-run` CI step would have caught it~~ **the bug itself is fixed — Rev 2, §3.1** — but the guard step still needs to land with CI, action item 3).
- ⚠️ Single-shot measurements: every tuple runs **n=1** with no repeats, no variance, no confidence intervals. The README's latency table (e.g., "10,929 ms" for belief_revision) presents one sample as a finding. For P3 (latency overhead), n≥3 with mean±σ is the minimum bar for the claim to survive scrutiny.

### Documentation & Artifacts
- ✅ **RESOLVED (Rev 2):** `SPEC.md` raw `[cite: 1, 2]`-style citation artifacts (lines 9–18) — all 8 instances stripped and verified clean via full-codebase search. ~~Visible residue of an AI-assisted drafting workflow in a document read by hiring managers.~~
- ❌ `SPEC.md` is Spanish-only while the README is English-only; there's no bridge (even a two-line "Spec (ES) — English summary coming" note). International reviewers can't evaluate the single-source-of-truth document.
- ❌ No GIF/asciinema demo of a run; the terminal report (`run.py:220–254`, which is genuinely nice output) is invisible to anyone who doesn't clone and execute.
- ⚠️ **Partially resolved (Rev 2):** the broken `docs/SPEC.md` readme path is fixed (`pyproject.toml` now points at root `SPEC.md`; install live-verified), but the project readme still targets the Spanish SPEC rather than the polished English README — acceptable as an intentional choice, though an English executive summary in `SPEC.md` (action item 11) is still recommended.
- ⚠️ Committed sample outputs exist only on disk under `outputs/runs/` but are git-ignored — so GitHub visitors see none of them; the README's sample table is the only visible artifact (that's fine, but pin it: the table cites `gemini-3.5-flash-lite` while `run.py`'s default is `gemini-3.8-flash` — a reviewer trying to reproduce the README numbers with the default CLI flags gets different results).
- ⚠️ No documented statistical methodology (temperature=0 is set in the factory default but never stated in the README's "reproducibility" story; no seed/versioning of provider models is recorded in `BenchmarkRunSummary` — you should capture `model_name` resolved *and* the provider-reported model version in run metadata).

---

## 5. Prioritized Action Plan (24–48 Hours)

### 🔴 High impact — do these first (~6 hours total)

1. ~~**Fix the packaging bug (15 min).**~~ ✅ **DONE (Rev 2, Oct 5, 2026)** — fixed as `readme = "SPEC.md"` (root-level SPEC retained as project readme per maintainer's choice); `pip install -e . --dry-run` live-verified green. Remaining follow-up: run `pip install -e ".[dev]" && pytest tests/ -q` in a clean venv once `python-dotenv` is declared (§3.2 / item 2).
2. **Declare `python-dotenv` and remove the silent except (15 min).** Add to `dependencies`, import unconditionally in `run.py`. Verify with a clean venv: `python -m venv .venv && .venv\Scripts\activate && pip install -e ".[dev]"`.
3. **Add GitHub Actions CI (45 min).** `.github/workflows/ci.yml`: matrix over `python 3.10/3.11/3.12`, steps = `pip install -e ".[dev]"`, `ruff check .`, `mypy core agents run.py`, `pytest tests/ -v`. Add the status badge as the **first line** of the README. This is the single highest portfolio-ROI item: it converts your private test pass into public, attested proof.
4. **Add `ruff` + `mypy` + `pre-commit` (1 h).** `pyproject.toml` config: `ruff` (line-length 88, target `py310`), `mypy` with `disallow_untyped_defs = true` for `core`/`agents`. Tighten the `Any`s from §3.11 to `CompiledStateGraph`. Wire a 4-hook `.pre-commit-config.yaml` (ruff, ruff-format, mypy, end-of-file-fixer). Your codebase is already clean enough that these will pass nearly green — free credibility.
5. **Mock-LLM end-to-end tests (2–3 h).** Add `tests/test_runner_e2e.py` using `langchain_core.language_models.fake_chat_models.GenericFakeChatModel`, monkeypatching `core.model_factory.create_model` (or injecting via `configurable["model_id"]` with a registry fixture). Cover: full `run_single` happy path with scripted responses, `system_event` skip, token aggregation arithmetic, error-turn handling (after §3.3 refactor), and export file contents (assert the CSV header row and the summary.md table cells). These tests turn your runner from "probably works" into "provably works" — and they're the tests a Staff interviewer will ask about.
6. **Fix metric contamination (1 h).** Implement §3.3 (`status: Literal["ok","error"]` on `TurnResult`, skip assertion evaluation on error turns, record failed tuples in `run_matrix` exports). Pair with an `error_rate` field in `BenchmarkRunSummary`.

### 🟠 Medium impact — same day (~4 hours)

7. **Lock the environment (30 min).** Adopt `uv` (`uv lock`, commit `uv.lock`) or export `requirements.lock` via `pip-tools`. Update README quickstart to `uv sync`. Record resolved library versions in every run's `results.json` (add `environment: {langgraph: ..., langchain_core: ..., pydantic: ...}` to the run metadata) — a benchmark that doesn't record its stack isn't reproducible.
8. **Cache model instances (§3.4, 30 min)** so latency numbers measure the agent, not YAML parsing. Note the change in the README methodology section.
9. **Resolve the prompt/assertion contradiction (§3.5, 1 h)** — at minimum, add a `match_scope` or regex assertion option (§3.6), and re-run the `strict_epistemic` scenario to update the README table with corrected numbers. Being able to *write about* this fix in the README ("we discovered our assertion engine penalized epistemically correct responses and here's how we fixed it") is itself portfolio gold — it demonstrates eval-design maturity.
10. **Write the exporter module (§3.15, 1.5 h)** — `core/exporters.py` with a `Protocol`, split JSON/CSV/MD classes, and add per-exporter unit tests.
11. **Clean `SPEC.md` (30 min)** — ✅ **Partially done (Rev 2):** all `[cite: N]` artifacts stripped and codebase-search-verified. Still pending: add an English executive summary section at the top.
12. **README reproducibility pass (30 min):** pin the sample-results section to the exact CLI command that generated it, add a "Methodology" subsection (temperature=0, n runs, environment versions), and note n=1 limitations honestly.

### 🟡 Low impact — polish (~2 hours, or next window)

13. **GIF/asciinema demo (30 min):** record `run.py --scenario all` terminal output, embed in README under Quickstart.
14. **Extend the CLI (45 min):** `--matrix` flag to run the full 6×3×12 product via `run_matrix`; `--repeats N` flag for basic variance; `[project.scripts]` entry point.
15. **Hygiene sweep (30 min):** add `.pytest_cache/` and `.ruff_cache/` to `.gitignore`; `casefold()` in the assertion matcher (§3.6); fix the docstring drift (§3.7); warning on dropped tool calls (§3.14); make the failed-scenario loader strict (§3.9).
16. **Optional statistical floor (stretch):** with `--repeats 3`, report mean±std latency in `summary.md` — this upgrades the P3 claims from anecdote to evidence.

---

## Bottom Line

This is a **well-architected research harness trapped in pre-production packaging**. The thinking is legitimately strong — injection-based agent construction, checkpointer-aware forward design, falsifiable scenario metadata, and a hermetic test suite that actually passes (verified live). But at the hiring bar, none of that survives first contact with a broken `pip install`, and the absence of CI means the repo *publicly attests to nothing*. Items 1–6 of the action plan are ~6 hours of work and would move this from **6.5 → 8.5/10**: the distance between "promising thesis code" and "this person engineers benchmarks the way we engineer production systems." The remaining items are what turn 8.5 into the elite, undeniable proof-of-work piece you're aiming for.

> **Rev 2 note:** the fatal `pip install` failure is fixed (§3.1 ✅) and the SPEC citation residue is gone (item 11, partially ✅) — revised standing **6.8/10**. Items 2–6 of the action plan now define the path to 8.5.

---

## Changelog

### Rev 2 — Oct 5, 2026
- **✅ §3.1 RESOLVED — packaging path fixed.** `pyproject.toml:5` changed from `readme = "docs/SPEC.md"` → `readme = "SPEC.md"` (root-level SPEC retained as the project readme per maintainer's choice). Live-verified: `pip install -e . --no-deps --dry-run` → `Preparing editable metadata (pyproject.toml): finished with status 'done'`, `Would install agent-memory-benchmark-0.1.0`. The original `OSError: Readme file does not exist` is gone.
- **✅ Action item 1 (partial → done).** Packaging bug closed; remaining follow-up is the clean-venv verification (`pip install -e ".[dev]" && pytest tests/ -q`), which should land together with the `python-dotenv` declaration (§3.2 / item 2).
- **✅ §4 Documentation (SPEC.md cites) RESOLVED.** All 8 raw `[cite: N]` citation artifacts stripped from `SPEC.md` (lines 9–18: project header + research questions P1–P3). Verified clean via full-codebase regex search; the only remaining `[cite` mentions in the repo are intentional quotations inside this report. The still-open half of item 11 (English executive summary at the top of `SPEC.md`) is unchanged.
- **⚠️ §4 pyproject readme target — reclassified.** The crash is fixed, but the readme still points at the Spanish `SPEC.md` instead of the English `README.md`; retained as an intentional maintainer choice and downgraded from ❌ to ⚠️.
- **Score: 6.5 → 6.8.** Rationale: the single fatal blocker (broken quickstart on first command) is removed, but the highest-ROI gap (CI with public test attestation, item 3) and the hermetic-blocker pair (§3.2 dotenv, §3.3 metric contamination) remain open.
- **Standing instruction:** this report is a living document — every fix applied to the repository that touches a finding here must be reflected in the corresponding section, its status marker, the affected action-plan item, and a new Changelog entry (with live verification evidence where applicable).

### Rev 1 — Oct 5, 2026
- Initial audit report issued. 28 findings and gaps identified across architecture, testing, tooling, and documentation; overall score **6.5/10**.







