"""Dataset-wide assertion-hygiene invariants (TODOs/assertion-hygiene-pass.md).

Eight checks over every scenario YAML under ``datasets/conversations/``:

- **H1** — a ``must_contain`` value must not appear in its own probe (R10),
  unless the turn is a *guarded forced-choice* probe (the probe prints a menu
  and the rejected labels are guarded) or the hit is an accepted exemption.
- **H2** — an assertion value carries ≥3 characters, or sets
  ``case_sensitive: true``, or is an accepted exemption.
- **H3** — ``metadata.kind`` is declared and the kind justifies the assertion
  mix (a privacy cell may be negative-only, a recall cell may not).
- **H4** — a scenario with ``documents:`` carries ≥1 positive retrieval pin.
- **H5** — a scenario with ``documents:`` carries ≥1 precision pin
  (``must_not_retrieve_*``).
- **H6** — no ``must_not_*`` value is a substring of a ``must_*`` value on the
  same turn (a pair that can never pass).
- **H7** — ``metadata.expected`` is machine-readable for levels 0–4.
- **H8** — every asserting turn declares ``probe.query_kind``.

The ``EXEMPT`` table is the single allow-list. Every entry names the check,
the scenario, the turn and value, the reason and the finding, and the ratchet
test fails when an entry stops being necessary or stops being justified by a
declared control kind — so exemptions cannot rot.

Everything here is offline and reads the YAML as it stands: no API keys, no
model calls, no runner invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from core.dataset_layout import scenario_id_from_filename
from core.runner import _normalize_for_matching
from core.schemas import BenchmarkScenario

REPO_ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS = REPO_ROOT / "datasets" / "conversations"
SCENARIO_FILES = sorted(CONVERSATIONS.rglob("test_*.yaml"))

#: Kinds declared by H3 and the contract each one carries.
KINDS: frozenset[str] = frozenset(
    {
        "recall",
        "revision",
        "privacy",
        "security",
        "scaling",
        "control_p3",
        "control_in_context",
    }
)
CONTROL_KINDS: frozenset[str] = frozenset({"control_p3", "control_in_context"})

LEVELS: tuple[str, ...] = tuple(f"level_{n}" for n in range(5))
EXPECTED_VALUES: frozenset[str] = frozenset(
    {"pass", "fail", "partial", "control", "n/a"}
)

POSITIVE_TYPES: frozenset[str] = frozenset(
    {"must_contain", "must_retrieve_chunk", "must_retrieve_source"}
)
TEXT_POSITIVE = "must_contain"
TEXT_NEGATIVE = "must_not_contain"
QUERY_KINDS: frozenset[str] = frozenset({"lookup", "semantic", "guard"})


@dataclass(frozen=True)
class Hit:
    """One raw invariant violation, before exemptions are applied."""

    scenario: str
    turn: int
    value: str


@dataclass(frozen=True)
class Exemption:
    """A documented, justified exception to one hygiene check."""

    check: str
    scenario: str
    turn: int | None
    value: str
    reason: str
    finding: str


@dataclass(frozen=True)
class ScenarioDoc:
    """Raw YAML view of one scenario (kept unnormalized on purpose)."""

    path: Path
    data: dict[str, Any]

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.data.get("metadata") or {}

    @property
    def kind(self) -> str | None:
        return self.metadata.get("kind")

    @property
    def turns(self) -> list[dict[str, Any]]:
        return self.data.get("turns") or []

    @property
    def documents(self) -> list[dict[str, Any]]:
        return self.data.get("documents") or []

#: Accepted violations. H1: an in-context control carries the whole context in
#: the probe by construction, so the positive token is necessarily echoed;
#: ``overhead_baseline_01`` T7 asks for a one-letter symbol. H2: the P3 cost
#: control's numeric answers are saturated by any digit; tightening them would
#: only add false negatives on a metric the cell does not claim.
EXEMPT: tuple[Exemption, ...] = (
    Exemption(
        check="H1",
        scenario="in_context_control_01",
        turn=1,
        value="Thursday",
        reason="In-context ceiling control: the schedule note is inside the probe.",
        finding="The control turn echoes its own probe verbatim (R10 cannot apply "
        "to a context that is deliberately part of the question).",
    ),
    Exemption(
        check="H1",
        scenario="in_context_control_01",
        turn=1,
        value="108",
        reason="In-context ceiling control: the room number is inside the probe.",
        finding="Second echoed token of the same control note.",
    ),
    Exemption(
        check="H1",
        scenario="in_context_multihop_01",
        turn=1,
        value="Berlin",
        reason="In-context control: the whole premise is inside the probe.",
        finding="Echo measures reading comprehension, not memory (that is the point).",
    ),
    Exemption(
        check="H1",
        scenario="in_context_needle_01",
        turn=1,
        value="DELTA-7749",
        reason="In-context control: the needle is in the same turn as the question.",
        finding="Echo is by construction; the cell isolates attention, not persistence.",
    ),
    Exemption(
        check="H1",
        scenario="in_context_derived_revision_01",
        turn=1,
        value="Level-1",
        reason="In-context control: the whole revision log is inside the probe.",
        finding="Echo is by construction; the cell isolates revision reasoning.",
    ),
    Exemption(
        check="H1",
        scenario="in_context_modus_tollens_01",
        turn=1,
        value="NO_EU_CENTRAL",
        reason="In-context control: the forced-label menu is inside the probe.",
        finding="Echo is by construction; the cell isolates Modus Tollens reasoning.",
    ),
    Exemption(
        check="H1",
        scenario="overhead_baseline_01",
        turn=7,
        value="O",
        reason="P3 cost control: accuracy is not its claim and the answer is one letter.",
        finding="'O' also occurs in 'on the periodic table', so the check is vacuous.",
    ),
    Exemption(
        check="H2",
        scenario="overhead_baseline_01",
        turn=2,
        value="6",
        reason="P3 cost control: the pass rate is designed to be 100% at every level.",
        finding="Any response containing a digit satisfies the assertion.",
    ),
    Exemption(
        check="H2",
        scenario="overhead_baseline_01",
        turn=4,
        value="0",
        reason="P3 cost control: the pass rate is designed to be 100% at every level.",
        finding="Any response containing a zero satisfies the assertion.",
    ),
    Exemption(
        check="H2",
        scenario="overhead_baseline_01",
        turn=6,
        value="7",
        reason="P3 cost control: the pass rate is designed to be 100% at every level.",
        finding="Any response containing a seven satisfies the assertion.",
    ),
    Exemption(
        check="H2",
        scenario="overhead_baseline_01",
        turn=9,
        value="24",
        reason="P3 cost control: the pass rate is designed to be 100% at every level.",
        finding="Any response containing '24' satisfies the assertion.",
    ),
    Exemption(
        check="H2",
        scenario="overhead_baseline_01",
        turn=10,
        value="9",
        reason="P3 cost control: the pass rate is designed to be 100% at every level.",
        finding="Any response containing a nine satisfies the assertion.",
    ),
)


def _norm(text: Any) -> str:
    """Same folding the runner applies, minus case sensitivity."""
    return _normalize_for_matching(str(text)).casefold()


def _rules(turn: dict[str, Any]) -> list[dict[str, Any]]:
    return turn.get("assertions") or []


def _asserting_turns(doc: ScenarioDoc) -> Iterator[dict[str, Any]]:
    for turn in doc.turns:
        if _rules(turn):
            yield turn


def _values(turn: dict[str, Any], rule_type: str) -> list[str]:
    return [str(rule["value"]) for rule in _rules(turn) if rule["type"] == rule_type]


def _type_values(doc: ScenarioDoc, rule_type: str) -> list[str]:
    return [
        str(rule["value"])
        for turn in doc.turns
        for rule in _rules(turn)
        if rule["type"] == rule_type
    ]


def _has_positive_text(doc: ScenarioDoc) -> bool:
    return any(_values(turn, TEXT_POSITIVE) for turn in doc.turns)


@pytest.fixture(scope="module")
def scenarios() -> list[ScenarioDoc]:
    docs = [
        ScenarioDoc(path=path, data=yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in SCENARIO_FILES
    ]
    assert docs, f"no scenario YAMLs under {CONVERSATIONS}"
    return docs


@pytest.fixture(scope="module")
def by_id(scenarios: list[ScenarioDoc]) -> dict[str, ScenarioDoc]:
    return {doc.id: doc for doc in scenarios}


def _exemptions(check: str) -> dict[tuple[str, int | None, str], Exemption]:
    return {
        (entry.scenario, entry.turn, entry.value): entry
        for entry in EXEMPT
        if entry.check == check
    }


def _echo_hits(doc: ScenarioDoc) -> list[Hit]:
    hits: list[Hit] = []
    for turn in _asserting_turns(doc):
        probe = _norm(turn.get("content", ""))
        for rule in _rules(turn):
            if rule["type"] != TEXT_POSITIVE:
                continue
            value = _norm(rule["value"])
            if value and value in probe:
                hits.append(Hit(doc.id, int(turn["turn_number"]), str(rule["value"])))
    return hits


def _short_hits(doc: ScenarioDoc) -> list[Hit]:
    hits: list[Hit] = []
    for turn in _asserting_turns(doc):
        for rule in _rules(turn):
            value = str(rule["value"])
            if len(value) < 3 and not rule.get("case_sensitive", False):
                hits.append(Hit(doc.id, int(turn["turn_number"]), value))
    return hits


def _is_guarded_forced_choice(turn: dict[str, Any], value: str) -> bool:
    """A menu probe whose rejected labels are guarded on the same turn.

    The probe prints the candidate labels, so the accepted one necessarily
    appears in the question. The assertion is only non-vacuous when the
    rejected labels are ruled out by ``must_not_contain`` guards (audit M6),
    which is what makes "print no menu, or print the whole menu" fail.
    """
    probe = _norm(turn.get("content", ""))
    if _norm(value) not in probe:
        return False
    alternatives = [
        _norm(rule["value"])
        for rule in _rules(turn)
        if rule["type"] == TEXT_NEGATIVE and _norm(rule["value"]) in probe
    ]
    return any(alt and alt != _norm(value) for alt in alternatives)


# ── H1 — no echo (R10) ───────────────────────────────────────────────────────


class TestH1NoEcho:
    def test_no_unguarded_positive_assertion_echoes_its_own_probe(self, scenarios):
        """A ``must_contain`` value bound to its own probe measures nothing.

        Two ways a hit is legitimate: the accepted answer and a rejected
        alternative are both printed in the probe and the alternative is
        guarded (a forced-choice menu, audit M6), or the scenario declares a
        control kind and the hit is listed in ``EXEMPT``.
        """
        exempt = _exemptions("H1")
        failures: list[str] = []
        for doc in scenarios:
            for turn in _asserting_turns(doc):
                probe = _norm(turn.get("content", ""))
                for rule in _rules(turn):
                    if rule["type"] != TEXT_POSITIVE:
                        continue
                    value = str(rule["value"])
                    if _norm(value) not in probe:
                        continue
                    key = (doc.id, int(turn["turn_number"]), value)
                    if key in exempt or _is_guarded_forced_choice(turn, value):
                        continue
                    failures.append(f"{doc.id} T{turn['turn_number']} {value!r}")
        assert not failures, (
            "must_contain values satisfied by the probe itself (R10): "
            + ", ".join(failures)
        )

    def test_unrelated_positive_values_are_not_affected(self, scenarios):
        """Sanity guard on the check itself: derived answers stay unflagged."""
        flagged = {
            (doc.id, int(turn["turn_number"]), str(rule["value"]))
            for doc in scenarios
            for turn in _asserting_turns(doc)
            for rule in _rules(turn)
            if rule["type"] == TEXT_POSITIVE
            and _norm(rule["value"]) in _norm(turn.get("content", ""))
        }
        assert ("cross_thread_derived_01", 11, "April 29") not in flagged


# ── H2 — entropy floor ──────────────────────────────────────────────────────


class TestH2EntropyFloor:
    def test_short_assertion_values_are_case_sensitive_or_exempt(self, scenarios):
        """``"0"`` must not pass on any response containing a zero.

        Values shorter than three characters are only acceptable when case
        matters (``Au`` is unambiguous, ``au`` is every 'because') or when a
        declared control lists the finding in ``EXEMPT``.
        """
        exempt = _exemptions("H2")
        failures: list[str] = []
        for doc in scenarios:
            for hit in _short_hits(doc):
                if (hit.scenario, hit.turn, hit.value) in exempt:
                    continue
                failures.append(f"{hit.scenario} T{hit.turn} {hit.value!r}")
        assert not failures, (
            "assertion values shorter than 3 characters without "
            "case_sensitive (H2): " + ", ".join(failures)
        )

    def test_chemical_symbols_are_case_sensitive(self, by_id):
        """The two symbol assertions that survived H2 must pin their case."""
        attrition = by_id["attrition_01"]
        gold_turn = next(t for t in attrition.turns if t["turn_number"] == 6)
        rule = next(r for r in gold_turn["assertions"] if r["value"] == "Au")
        assert rule["case_sensitive"] is True


# ── exemption ratchet ───────────────────────────────────────────────────────


class TestExemptionRatchet:
    def test_every_exemption_is_still_a_raw_violation(self, scenarios):
        """Fail when an exemption stops being necessary (the data was fixed)."""
        raw = {
            "H1": {
                (hit.scenario, hit.turn, hit.value)
                for doc in scenarios
                for hit in _echo_hits(doc)
            },
            "H2": {
                (hit.scenario, hit.turn, hit.value)
                for doc in scenarios
                for hit in _short_hits(doc)
            },
        }
        for entry in EXEMPT:
            key = (entry.scenario, entry.turn, entry.value)
            assert key in raw[entry.check], (
                f"stale {entry.check} exemption {key}: the underlying violation "
                "is gone — delete the entry"
            )

    def test_every_exemption_is_justified_by_a_control_kind(self, by_id):
        """Exemptions are a declared control contract, not an ad-hoc allow-list."""
        for entry in EXEMPT:
            doc = by_id[entry.scenario]
            assert doc.kind in CONTROL_KINDS, (
                f"{entry.check} exemption for {entry.scenario} (type "
                f"{doc.kind!r}): only control_p3/control_in_context cells may "
                "hold H1/H2 exemptions — fix the assertion instead"
            )
            assert doc.metadata.get("controls"), (
                f"{entry.scenario} must declare what it controls (H3)"
            )

    def test_checks_are_known(self):
        assert {entry.check for entry in EXEMPT} <= {"H1", "H2"}

    def test_exemptions_are_unique(self):
        keys = [(entry.check, entry.scenario, entry.turn, entry.value) for entry in EXEMPT]
        assert len(keys) == len(set(keys)), "duplicate exemption entry"


# ── H3 — declared kind ──────────────────────────────────────────────────────


class TestH3DeclaredKind:
    def test_every_scenario_declares_a_known_kind(self, scenarios):
        missing = [doc.id for doc in scenarios if not doc.kind]
        assert not missing, f"scenarios without metadata.kind (H3): {missing}"
        unknown = sorted({doc.kind for doc in scenarios} - KINDS)
        assert not unknown, f"unknown metadata.kind values: {unknown}"

    def test_quality_eligibility_is_declared(self, scenarios):
        for doc in scenarios:
            assert isinstance(doc.metadata.get("quality_eligible"), bool), (
                f"{doc.id}: metadata.quality_eligible must be a declared bool"
            )

    def test_recall_and_revision_cells_carry_a_positive_rule(self, scenarios):
        """An amnesic agent must not be able to pass a recall/revision cell."""
        failures = [
            doc.id
            for doc in scenarios
            if doc.kind in {"recall", "revision"}
            and (
                not _has_positive_text(doc)
                or doc.metadata.get("amnesia_passes") is not False
                or doc.metadata.get("quality_eligible") is not True
            )
        ]
        assert not failures, (
            "recall/revision cells must have >=1 must_contain, "
            f"amnesia_passes: false and quality_eligible: true: {failures}"
        )

    def test_negative_only_privacy_cells_declare_the_amnesia_contract(self, scenarios):
        """A privacy cell may be negative-only — but only if it says so."""
        for doc in scenarios:
            if doc.kind not in {"privacy", "security"} or _has_positive_text(doc):
                continue
            assert doc.metadata.get("amnesia_passes") is True, (
                f"{doc.id}: negative-only {doc.kind} cell must declare "
                "amnesia_passes: true"
            )
            assert doc.metadata.get("quality_eligible") is False, (
                f"{doc.id}: negative-only {doc.kind} cell must set "
                "quality_eligible: false (it is not an accuracy score)"
            )

    def test_excluded_cells_declare_why(self, scenarios):
        for doc in scenarios:
            if doc.metadata.get("quality_eligible") is not False:
                continue
            justified = (
                doc.kind in CONTROL_KINDS and bool(doc.metadata.get("controls"))
            ) or (
                doc.kind in {"privacy", "security"}
                and doc.metadata.get("amnesia_passes") is True
            )
            assert justified, (
                f"{doc.id}: quality_eligible: false needs a declared reason "
                "(controls: text for a control kind, or amnesia_passes: true "
                "for a privacy/security cell)"
            )

    def test_control_cells_declare_what_they_control(self, scenarios):
        for doc in scenarios:
            if doc.kind not in CONTROL_KINDS:
                continue
            controls = doc.metadata.get("controls")
            assert isinstance(controls, str) and len(controls) >= 20, (
                f"{doc.id}: control cells must declare metadata.controls "
                "(what the cell controls for), not just a kind label"
            )

    def test_scaling_cells_declare_the_lexical_baseline(self, scenarios):
        scaling = [doc for doc in scenarios if doc.kind == "scaling"]
        assert scaling, "no scaling cell declares kind: scaling"
        for doc in scaling:
            difficulty = doc.metadata.get("corpus_difficulty") or {}
            hit = difficulty.get("lexical_baseline_hit_at_5")
            assert isinstance(hit, (int, float)) and 0.0 <= hit <= 1.0, (
                f"{doc.id}: scaling cells must declare "
                "metadata.corpus_difficulty.lexical_baseline_hit_at_5 in [0, 1]"
            )



# ── H4/H5 — document coverage and precision ─────────────────────────────────


class TestH4H5DocumentRetrieval:
    def test_document_scenarios_pin_at_least_one_positive_retrieval(self, scenarios):
        missing = [
            doc.id
            for doc in scenarios
            if doc.documents
            and not (
                _type_values(doc, "must_retrieve_chunk")
                or _type_values(doc, "must_retrieve_source")
            )
        ]
        assert not missing, (
            "document cells without a positive retrieval pin (H4) — a document "
            f"that is never ingested would pass: {missing}"
        )

    def test_document_scenarios_pin_at_least_one_precision_decoy(self, scenarios):
        missing = [
            doc.id
            for doc in scenarios
            if doc.documents
            and not (
                _type_values(doc, "must_not_retrieve_chunk")
                or _type_values(doc, "must_not_retrieve_source")
            )
        ]
        assert not missing, (
            "document cells without a must_not_retrieve_* pin (H5) — context "
            f"blending stays a text guard instead of a metric: {missing}"
        )


# ── H6 — no self-contradiction ──────────────────────────────────────────────


def _contradictions(doc: ScenarioDoc) -> list[str]:
    failures: list[str] = []
    for turn in _asserting_turns(doc):
        text_pairs = [
            (neg, pos)
            for neg in _values(turn, TEXT_NEGATIVE)
            for pos in _values(turn, TEXT_POSITIVE)
            if _norm(neg) and _norm(neg) in _norm(pos)
        ]
        chunk_pairs = [
            (neg, pos)
            for neg in _values(turn, "must_not_retrieve_chunk")
            for pos in _values(turn, "must_retrieve_chunk")
            if _norm(neg) and _norm(neg) in _norm(pos)
        ]
        source_pairs = [
            (neg, pos)
            for neg in _values(turn, "must_not_retrieve_source")
            for pos in _values(turn, "must_retrieve_source")
            if _norm(neg) == _norm(pos)
        ]
        for neg, pos in text_pairs + chunk_pairs + source_pairs:
            failures.append(
                f"{doc.id} T{turn['turn_number']}: must_not {neg!r} is inside "
                f"must {pos!r}"
            )
    return failures


class TestH6NoSelfContradiction:
    def test_no_negative_rule_can_contradict_a_positive_rule(self, scenarios):
        failures = [msg for doc in scenarios for msg in _contradictions(doc)]
        assert not failures, "unsatisfiable assertion pairs (H6): " + "; ".join(
            failures
        )



# ── H7 — machine-readable expectations ──────────────────────────────────────


class TestH7MachineReadableExpectations:
    def test_every_scenario_declares_every_level(self, scenarios):
        failures: list[str] = []
        for doc in scenarios:
            expected = doc.metadata.get("expected")
            if not isinstance(expected, dict):
                failures.append(f"{doc.id}: metadata.expected missing")
                continue
            missing = [level for level in LEVELS if level not in expected]
            extra = sorted(set(expected) - set(LEVELS))
            if missing:
                failures.append(f"{doc.id}: missing {missing}")
            if extra:
                failures.append(f"{doc.id}: unknown levels {extra}")
        assert not failures, "machine-readable expectations (H7): " + "; ".join(
            failures
        )

    def test_values_come_from_the_closed_vocabulary(self, scenarios):
        bad = [
            f"{doc.id}.{level}={value!r}"
            for doc in scenarios
            for level, value in (doc.metadata.get("expected") or {}).items()
            if value not in EXPECTED_VALUES
        ]
        assert not bad, f"expectations outside {sorted(EXPECTED_VALUES)}: {bad}"

    def test_control_cells_expect_control_at_level_0(self, scenarios):
        bad = [
            doc.id
            for doc in scenarios
            if doc.kind in CONTROL_KINDS
            and (doc.metadata.get("expected") or {}).get("level_0") != "control"
        ]
        assert not bad, f"control cells must expect 'control' at L0: {bad}"

    def test_privacy_cells_expect_an_amnesic_pass_at_level_0(self, scenarios):
        bad = [
            doc.id
            for doc in scenarios
            if doc.kind == "privacy"
            and doc.metadata.get("amnesia_passes") is True
            and (doc.metadata.get("expected") or {}).get("level_0") != "pass"
        ]
        assert not bad, (
            f"privacy cells that pass by amnesia must expect 'pass' at L0: {bad}"
        )

    def test_quality_cells_do_not_claim_a_level_0_pass(self, scenarios):
        """A quality-eligible cell claiming an L0 pass would score amnesia."""
        bad = [
            doc.id
            for doc in scenarios
            if doc.metadata.get("quality_eligible") is True
            and (doc.metadata.get("expected") or {}).get("level_0") == "pass"
        ]
        assert not bad, (
            "quality-eligible cells (recall/revision/scaling) must not expect a "
            f"level-0 pass — amnesia would be scored as success: {bad}"
        )



# ── H8 — probe kind (for L2b) ───────────────────────────────────────────────


class TestH8ProbeKind:
    def test_every_asserting_turn_declares_a_query_kind(self, scenarios):
        missing = [
            f"{doc.id} T{turn['turn_number']}"
            for doc in scenarios
            for turn in _asserting_turns(doc)
            if not (turn.get("probe") or {}).get("query_kind")
        ]
        assert not missing, (
            "asserting turns without probe.query_kind (H8) — L2b cannot tell a "
            f"reformulated query from a handed-over key: {missing}"
        )

    def test_query_kind_comes_from_the_closed_vocabulary(self, scenarios):
        bad = [
            f"{doc.id} T{turn['turn_number']}={kind!r}"
            for doc in scenarios
            for turn in _asserting_turns(doc)
            for kind in [(turn.get("probe") or {}).get("query_kind")]
            if kind not in QUERY_KINDS
        ]
        assert not bad, f"query_kind outside {sorted(QUERY_KINDS)}: {bad}"

    def test_guard_labels_only_cover_guard_only_turns(self, scenarios):
        bad = [
            f"{doc.id} T{turn['turn_number']}"
            for doc in scenarios
            for turn in _asserting_turns(doc)
            if (turn.get("probe") or {}).get("query_kind") == "guard"
            and any(rule["type"] in POSITIVE_TYPES for rule in _rules(turn))
        ]
        assert not bad, (
            "probe.query_kind: 'guard' is a leakage-guard label, not an escape "
            f"hatch for an unlabelled probe: {bad}"
        )

    def test_ungraded_turns_need_no_label(self, scenarios):
        for doc in scenarios:
            for turn in doc.turns:
                if _rules(turn):
                    continue
                probe = turn.get("probe") or {}
                if probe:
                    assert probe.get("query_kind") in QUERY_KINDS

    def test_schema_loads_the_probe_and_rejects_unknown_kinds(self, scenarios):
        """``probe`` must be a schema contract, not a silently kept YAML key."""
        for doc in scenarios:
            model = BenchmarkScenario(**doc.data)
            raw_by_turn = {turn["turn_number"]: turn for turn in doc.turns}
            for turn in model.turns:
                raw_kind = (raw_by_turn[turn.turn_number].get("probe") or {}).get(
                    "query_kind"
                )
                declared = turn.probe.query_kind if turn.probe else None
                assert declared == raw_kind

        data = dict(scenarios[0].data)
        data["turns"] = [
            {
                "turn_number": 1,
                "role": "user",
                "content": "probe?",
                "probe": {"query_kind": "telepathic"},
            }
        ]
        with pytest.raises(Exception):
            BenchmarkScenario(**data)


# ── dataset wiring ──────────────────────────────────────────────────────────


class TestDatasetWiring:
    def test_every_scenario_file_is_covered(self, scenarios):
        expected = {scenario_id_from_filename(path.name) for path in SCENARIO_FILES}
        loaded = {doc.id for doc in scenarios}
        assert loaded == expected, (
            "hygiene fixture/runner disagreement: " f"{sorted(loaded ^ expected)}"
        )
        for doc in scenarios:
            assert doc.path.is_relative_to(CONVERSATIONS)

