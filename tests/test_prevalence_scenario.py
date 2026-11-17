"""Offline soundness checks for the document-prevalence scenario.

The scenario ``document_prevalence_update_vs_distractor_01`` measures whether a
conversation update supersedes a document fact while near-miss distractors leave
other document facts intact. Everything here runs offline (no models, embeddings
or network): it only loads the scenario through the real runner and reads the
corpus files it references.

Key invariants:
- the superseding token (``QZX-7P4K-1180``) is CHAT-EXCLUSIVE — absent from every
  corpus document — so a correct cross-session answer can only come from the
  episodic chat store;
- the document tokens live in their expected documents;
- the supersession is a ``user`` (chat-channel) turn, not a document attachment;
- T7 stays lenient (no ``must_not_contain`` on the stale value, rule R1) while
  the isolated T10 strict probe forbids historical values and may grade the stale
  code absent;
- no ``must_contain`` value is echoed by its own probe (R10 style).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.documents import resolve_document_path
from core.runner import BenchmarkRunner

SCENARIO_ID = "document_prevalence_update_vs_distractor_01"
NEW = "QZX-7P4K-1180"
OLD = "YKX-HMD0-7872"
LOOP969 = "BQR-485R-6500"
RACK923 = "UVA-046I-7504"

#: doc_id -> the needle value that must appear in that document.
EXPECTED_DOC_TOKENS = {
    "corpus_000": OLD,
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


def _doc_text(scenario, doc_id):
    spec = next(d for d in scenario.documents if d.doc_id == doc_id)
    path = resolve_document_path(
        spec.path, scenario_dir=Path(scenario.source_path).parent
    )
    return path.read_text(encoding="utf-8")


class TestScenarioShape:
    def test_loads_with_the_full_60_document_corpus(self, scenario):
        assert scenario.category == "document_revision"
        assert scenario.source_path is not None
        assert len(scenario.documents) == 60
        assert scenario.documents[0].doc_id == "corpus_000"
        assert scenario.metadata["corpus_dir"] == "d16_corpus_060"
        assert [t.turn_number for t in scenario.turns] == list(range(1, 11))

    def test_new_token_is_chat_exclusive(self, scenario):
        """QZX-7P4K-1180 must exist in NO document (causal cross-session proof)."""
        hits = [
            d.doc_id
            for d in scenario.documents
            if NEW in _doc_text(scenario, d.doc_id)
        ]
        assert hits == [], f"superseding token leaked into documents: {hits}"

    def test_document_tokens_live_in_their_documents(self, scenario):
        for doc_id, token in EXPECTED_DOC_TOKENS.items():
            assert token in _doc_text(scenario, doc_id), (
                f"{token} not found in {doc_id}"
            )


class TestSupersessionChannel:
    def test_supersession_is_a_chat_turn_not_a_document(self, scenario):
        turn = _turns(scenario)[2]
        assert turn.role == "user"
        assert turn.session_id == "s1"
        assert turn.attachments == []
        assert turn.assertions == []
        assert NEW in turn.content

    def test_scenario_declares_no_system_event_ingestion(self, scenario):
        assert all(t.role == "user" for t in scenario.turns)

    def test_distractors_introduce_no_confirmed_value(self, scenario):
        turns = _turns(scenario)
        for number in (3, 4):
            content = turns[number].content
            assert NEW not in content
            assert OLD not in content
            assert LOOP969 not in content
        # T3 names the sibling loop (the near-miss) but only as a rumor.
        assert "LOOP-969" in turns[3].content
        assert "rumor" in turns[3].content.casefold()
        # T4 is a same-entity, different-attribute change.
        assert "LOOP-420" in turns[4].content
        assert "maintenance window" in turns[4].content.casefold()


class TestPrevalenceAssertions:
    def test_baseline_probe_pins_the_document(self, scenario):
        turn = _turns(scenario)[1]
        assert [r.value for r in _rules(turn, "must_contain")] == [OLD]
        assert [r.value for r in _rules(turn, "must_retrieve_chunk")] == ["corpus_000"]

    def test_same_session_update_is_lenient(self, scenario):
        turn = _turns(scenario)[5]
        assert turn.session_id == "s1"
        assert [r.value for r in _rules(turn, "must_contain")] == [NEW]
        assert _rules(turn, "must_not_contain") == []

    def test_cross_session_update_is_lenient_and_pins_chat(self, scenario):
        """R1: T7 must NOT forbid the stale value (a correct answer may cite it)."""
        turn = _turns(scenario)[7]
        assert turn.session_id == "s2"
        assert [r.value for r in _rules(turn, "must_contain")] == [NEW]
        assert _rules(turn, "must_not_contain") == []
        assert [r.value for r in _rules(turn, "must_retrieve_chunk")] == ["ep_s1_t2"]

    def test_strict_probe_is_isolated_and_forbids_history(self, scenario):
        turn = _turns(scenario)[10]
        assert turn.session_id == "s3"
        # s3 contains only this probe -> T7/its echoes cannot anchor it.
        assert [t.turn_number for t in scenario.turns if t.session_id == "s3"] == [10]
        assert [r.value for r in _rules(turn, "must_contain")] == [NEW]
        assert [r.value for r in _rules(turn, "must_not_contain")] == [OLD]
        assert "no earlier or historical values" in turn.content

    @pytest.mark.parametrize("number", [6, 8])
    def test_not_superseded_probes_preserve_the_document(self, scenario, number):
        turn = _turns(scenario)[number]
        assert [r.value for r in _rules(turn, "must_contain")] == [LOOP969]
        assert [r.value for r in _rules(turn, "must_not_contain")] == [NEW]
        assert [r.value for r in _rules(turn, "must_retrieve_chunk")] == ["corpus_045"]

    def test_control_probe_pins_untouched_document(self, scenario):
        turn = _turns(scenario)[9]
        assert [r.value for r in _rules(turn, "must_contain")] == [RACK923]
        assert [r.value for r in _rules(turn, "must_retrieve_chunk")] == ["corpus_007"]

    def test_no_must_contain_echoes_its_own_probe(self, scenario):
        """R10: an assertion satisfied by echoing the question measures nothing."""
        for turn in scenario.turns:
            probe = turn.content.casefold()
            for rule in _rules(turn, "must_contain"):
                assert rule.value.casefold() not in probe, (
                    f"T{turn.turn_number}: {rule.value!r} is echoed by its probe"
                )

    def test_cross_session_probes_are_in_s2(self, scenario):
        turns = _turns(scenario)
        for number in (7, 8, 9):
            assert turns[number].session_id == "s2"
