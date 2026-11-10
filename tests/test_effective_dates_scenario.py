"""Offline soundness checks for the effective-dates scenario.

The scenario ``document_effective_dates_01`` measures whether a model applies a
document's **effective dates** to the reference date named by the probe, instead
of answering with the newest revision or echoing an answer from another session.
Everything here runs offline (no models, embeddings or network): it loads the
scenario through the real runner and reads the corpus files it references.

Key invariants:
- the revision-history document is the ONLY source of the three ``P88-*``
  tokens: they occur in no file under ``datasets/documents/d16_corpus_060/``,
  and the entity ``PUMP-88`` is absent from the corpus as well, so no probe can
  be satisfied by the reused corpus;
- the dated revisions live in a DOCUMENT (inline ``content``), never in a chat
  turn — chat-only dated statements would measure chat temporal memory, not the
  validity semantics of document content;
- every date probe forces a format (``"Reply with ONLY ..."``), which is exactly
  what makes the ``must_not_contain`` guards on the other two revisions fair
  (stress-plan rule R1);
- the three date probes sit in three different sessions, so no answer can anchor
  the next probe in-context;
- each probe pins its gold chunk (``revisions_pump88``; ``corpus_007`` for the
  untouched control), so "the document never arrived" is distinguishable from
  "the wrong revision was picked";
- no ``must_contain`` value is echoed by its own probe (R10 style).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.documents import DEFAULT_DOCUMENTS_DIR, resolve_document_path
from core.runner import BenchmarkRunner
from core.schemas import BenchmarkScenario

SCENARIO_ID = "document_effective_dates_01"
CORPUS_DIR = DEFAULT_DOCUMENTS_DIR / "d16_corpus_060"
def _corpus_scenario_path() -> Path:
    """The 60-document scenario whose ``documents:`` block is reused VERBATIM.

    Located by scenario ID rather than a pinned path, so a dataset re-layout
    (``<single_thread|multi_thread>/<theme>/<category>/``) cannot break it.
    """
    hits = sorted(
        (DEFAULT_DOCUMENTS_DIR.parent / "conversations").rglob(
            "test_document_corpus_size_060_01.yaml"
        )
    )
    assert len(hits) == 1, (
        f"document_corpus_size_060_01: expected one dataset file, got {hits}"
    )
    return hits[0]

REVISION_DOC = "revisions_pump88"
ENTITY = "PUMP-88"
ALPHA = "P88-ALPHA-1000"
BETA = "P88-BETA-2000"
GAMMA = "P88-GAMMA-3000"
LOOP420 = "YKX-HMD0-7872"
LOOP969 = "BQR-485R-6500"
RACK923 = "UVA-046I-7504"

#: revision value -> the effective date the document attaches to it.
REVISIONS = {ALPHA: "2025-01-01", BETA: "2026-09-14", GAMMA: "2027-01-01"}

#: turn_number -> (session, reference date, expected value, forbidden values).
DATE_PROBES = {
    1: ("s1", "2026-08-01", ALPHA, (BETA, GAMMA)),
    2: ("s2", "2026-10-15", BETA, (ALPHA, GAMMA)),
    3: ("s3", "2027-03-01", GAMMA, (ALPHA, BETA)),
}

#: doc_id -> the corpus needle that must appear in that document.
EXPECTED_DOC_TOKENS = {
    "corpus_000": LOOP420,
    "corpus_045": LOOP969,
    "corpus_007": RACK923,
}


@pytest.fixture(scope="module")
def scenario():
    scenarios = {s.id: s for s in BenchmarkRunner().load_scenarios()}
    assert SCENARIO_ID in scenarios, f"{SCENARIO_ID} not found in datasets/"
    return scenarios[SCENARIO_ID]


def _turns(scenario):
    return {t.turn_number: t for t in scenario.turns}


def _rules(turn, rule_type):
    return [r for r in turn.assertions if r.type == rule_type]


def _document_text(scenario, doc_id):
    """Text of a scenario document: inline ``content`` or its resolved path."""
    spec = next(d for d in scenario.documents if d.doc_id == doc_id)
    if spec.content is not None:
        return spec.content
    path = resolve_document_path(
        spec.path, scenario_dir=Path(scenario.source_path).parent
    )
    return path.read_text(encoding="utf-8")


def _corpus_files():
    files = sorted(CORPUS_DIR.glob("*.md"))
    assert len(files) == 60, f"expected the 60-document corpus, found {len(files)}"
    return files


class TestScenarioShape:
    def test_validates_against_the_schema(self, scenario):
        """The raw YAML must satisfy ``BenchmarkScenario.model_validate``."""
        raw = yaml.safe_load(Path(scenario.source_path).read_text(encoding="utf-8"))
        validated = BenchmarkScenario.model_validate(raw)
        assert validated.id == SCENARIO_ID
        assert validated.category == "document_revision"
        assert [t.turn_number for t in validated.turns] == [1, 2, 3, 4]

    def test_loads_the_60_corpus_docs_plus_one_revision_doc(self, scenario):
        assert scenario.category == "document_revision"
        assert scenario.source_path is not None
        assert len(scenario.documents) == 61
        expected_ids = [f"corpus_{i:03d}" for i in range(60)] + [REVISION_DOC]
        assert [d.doc_id for d in scenario.documents] == expected_ids
        assert scenario.metadata["corpus_dir"] == "d16_corpus_060"
        assert scenario.metadata["corpus_docs"] == 61
        assert scenario.metadata["chunk_size"] == 1200
        assert scenario.metadata["default_top_k"] == 8
        assert scenario.metadata["sessions"] == ["s1", "s2", "s3"]

    def test_corpus_entries_are_reused_verbatim(self, scenario):
        """The 60 document entries must be the committed D16 corpus, untouched."""
        source = yaml.safe_load(_corpus_scenario_path().read_text(encoding="utf-8"))
        reused = [
            (d["doc_id"], d["path"], d.get("metadata"))
            for d in source["documents"]
        ]
        assert len(reused) == 60
        here = [
            (d.doc_id, d.path, dict(d.metadata))
            for d in scenario.documents
            if d.doc_id != REVISION_DOC
        ]
        assert here == reused

    def test_revision_doc_is_inline_and_dated(self, scenario):
        spec = next(d for d in scenario.documents if d.doc_id == REVISION_DOC)
        assert spec.path is None
        assert spec.content is not None
        assert spec.metadata["title"] == f"{ENTITY} calibration code revision history"
        text = spec.content
        assert text.count("Effective ") == 3
        for value, effective in REVISIONS.items():
            assert f"Effective {effective}: calibration code {value}." in text


class TestRevisionValuesAreDocumentOnly:
    def test_revision_tokens_live_in_the_revision_doc(self, scenario):
        text = _document_text(scenario, REVISION_DOC)
        for value in REVISIONS:
            assert value in text, f"{value} missing from {REVISION_DOC}"

    def test_revision_tokens_are_absent_from_every_corpus_file(self):
        tokens = [ENTITY, *REVISIONS]
        leaked = [
            f"{path.name}:{token}"
            for path in _corpus_files()
            for token in tokens
            if token in path.read_text(encoding="utf-8")
        ]
        assert leaked == [], f"revision values leaked into the corpus: {leaked}"

    def test_entity_is_absent_from_the_corpus_docs(self, scenario):
        hits = [
            d.doc_id
            for d in scenario.documents
            if d.doc_id != REVISION_DOC
            and ENTITY in _document_text(scenario, d.doc_id)
        ]
        assert hits == [], f"{ENTITY} leaked into corpus documents: {hits}"

    def test_revisions_are_not_chat_turns(self, scenario):
        """The dated revisions must be document content, never chat statements."""
        for turn in scenario.turns:
            assert turn.role == "user"
            assert turn.attachments == []
            for value in REVISIONS:
                assert value not in turn.content, (
                    f"T{turn.turn_number} states a revision value in chat"
                )
            for effective in REVISIONS.values():
                assert effective not in turn.content, (
                    f"T{turn.turn_number} states a revision effective date in chat"
                )


class TestCorpusAnchors:
    def test_corpus_tokens_live_in_their_documents(self, scenario):
        for doc_id, token in EXPECTED_DOC_TOKENS.items():
            assert token in _document_text(scenario, doc_id), (
                f"{token} not found in {doc_id}"
            )


class TestReferenceDateProbes:
    @pytest.mark.parametrize("number", sorted(DATE_PROBES))
    def test_probe_names_the_reference_date_and_forces_a_format(
        self, scenario, number
    ):
        content = _turns(scenario)[number].content
        _, reference_date, _, _ = DATE_PROBES[number]
        assert f"As of {reference_date}" in content
        assert ENTITY in content
        # The forced format is what makes the negation guards fair (rule R1).
        assert "ONLY" in content

    @pytest.mark.parametrize("number", sorted(DATE_PROBES))
    def test_probe_grades_the_date_effective_revision(self, scenario, number):
        session, _, expected, forbidden = DATE_PROBES[number]
        turn = _turns(scenario)[number]
        assert turn.session_id == session
        assert turn.role == "user"
        assert [r.value for r in _rules(turn, "must_contain")] == [expected]
        assert {r.value for r in _rules(turn, "must_not_contain")} == set(forbidden)

    @pytest.mark.parametrize("number", sorted(DATE_PROBES))
    def test_probe_pins_the_revision_doc(self, scenario, number):
        turn = _turns(scenario)[number]
        assert [r.value for r in _rules(turn, "must_retrieve_chunk")] == [
            REVISION_DOC
        ]

    def test_each_date_probe_owns_its_session(self, scenario):
        """Fresh threads: no probe can be answered by echoing an earlier one."""
        sessions: dict[str, list[int]] = {}
        for turn in scenario.turns:
            sessions.setdefault(turn.session_id, []).append(turn.turn_number)
        assert sessions == {"s1": [1], "s2": [2], "s3": [3, 4]}

    def test_metadata_records_the_same_sweep(self, scenario):
        meta = scenario.metadata
        assert meta["effective_dates_doc"] == REVISION_DOC
        assert {
            revision["value"]: revision["effective"]
            for revision in meta["revisions"]
        } == REVISIONS
        assert meta["reference_dates"] == {
            f"t{number}": dates[1] for number, dates in DATE_PROBES.items()
        }
        for reference_date, expected in [
            (dates[1], dates[2]) for dates in DATE_PROBES.values()
        ]:
            # Effective dates and reference dates must never coincide, or the
            # probe could not discriminate "not yet effective" from "retired".
            assert REVISIONS[expected] != reference_date


class TestControlProbe:
    def test_control_pins_the_untouched_corpus_document(self, scenario):
        turn = _turns(scenario)[4]
        assert turn.session_id == "s3"
        assert turn.role == "user"
        assert [r.value for r in _rules(turn, "must_contain")] == [RACK923]
        assert [r.value for r in _rules(turn, "must_retrieve_chunk")] == [
            "corpus_007"
        ]
        assert _rules(turn, "must_not_contain") == []

    def test_control_entity_metadata_matches_the_probe(self, scenario):
        control = scenario.metadata["control_entity"]
        assert control == {
            "token": RACK923,
            "doc_id": "corpus_007",
            "entity": "RACK-923",
        }
        assert control["entity"] in _turns(scenario)[4].content


class TestAssertionHygiene:
    def test_no_must_contain_echoes_its_own_probe(self, scenario):
        """R10: an assertion satisfied by echoing the question measures nothing."""
        for turn in scenario.turns:
            probe = turn.content.casefold()
            for rule in _rules(turn, "must_contain"):
                assert rule.value.casefold() not in probe, (
                    f"T{turn.turn_number}: {rule.value!r} is echoed by its probe"
                )

    def test_must_not_contain_is_reserved_for_the_forced_format_probes(
        self, scenario
    ):
        turns_with_negation = [
            turn.turn_number
            for turn in scenario.turns
            if _rules(turn, "must_not_contain")
        ]
        assert turns_with_negation == [1, 2, 3]
        for number in turns_with_negation:
            assert "ONLY" in _turns(scenario)[number].content

    def test_every_probe_pins_exactly_one_chunk(self, scenario):
        for turn in scenario.turns:
            assert len(_rules(turn, "must_retrieve_chunk")) == 1, (
                f"T{turn.turn_number} must pin exactly one gold chunk"
            )

    def test_probe_verdict_is_declared_and_cannot_be_earned_by_guards(self, scenario):
        """The headline metric is the probe verdict: value present AND gold chunk hit.

        ``must_not_contain`` guards pass for free when a model declines (the
        ``--no-documents`` control scores 6/14 assertions but 0/4 probes), so no
        probe may be graded by guards alone — every turn needs a positive
        ``must_contain`` and a retrieval pin.
        """
        assert scenario.metadata["headline_metric"] == "probe_verdict"
        rule = scenario.metadata["probe_verdict_rule"].casefold()
        assert "must_contain" in rule
        assert "must_retrieve_chunk" in rule
        assert "guards pass for free" in rule
        for turn in scenario.turns:
            assert _rules(turn, "must_contain"), (
                f"T{turn.turn_number} has no positive assertion"
            )
            assert _rules(turn, "must_retrieve_chunk"), (
                f"T{turn.turn_number} has no retrieval pin"
            )

