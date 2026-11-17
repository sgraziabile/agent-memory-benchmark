"""Offline soundness checks for the authority-vs-recency scenario.

The scenario ``document_authority_vs_recency_01`` measures whether a recent,
confident chat claim can be mistaken for an authoritative update: the corpus
documents are the system of record, and only an APPROVED change ticket may
supersede a documented value. Everything here runs offline (no models, no
embeddings, no network): it loads the scenario through the real runner and
reads the corpus files it references.

Key invariants:
- both graded tokens are CHAT-EXCLUSIVE — ``RUM-4X9Q-2201`` (casual claim) and
  ``AUT-8K2M-5530`` (approved update) appear in no corpus document, so a
  cross-session answer can only come from the episodic chat store;
- the documented tokens live in their expected documents (LOOP-420 ->
  ``corpus_000`` -> ``YKX-HMD0-7872``, RACK-923 -> ``corpus_007`` ->
  ``UVA-046I-7504``, plus the unused near-miss ``BQR-485R-6500`` in
  ``corpus_045``, kept as an integrity check on the reused corpus);
- the casual claim is a ``user`` (chat-channel) turn with no hedge, no ticket
  wording and no ``system_event`` ingestion, so it cannot be excused as an
  authoritative document;
- ``must_not_contain`` is used ONLY on the two forced-format probes (rule R1);
  the post-ticket probe stays lenient because a correct answer may legitimately
  mention the superseded values;
- the cross-session probe T5 carries a **source-level anchor**
  (``must_retrieve_source: document``) and deliberately carries NO
  ``must_not_retrieve_source: chat``: the casual-claim chunk is supposed to
  compete for the same top_k slots, so forbidding chat would fail by
  construction;
- T1's in-context leak (its retrieval block can contain ``doc_corpus_000_c0``,
  the chunk that carries the documented value) is an ACCEPTED confound: T1
  forbids restating codes, T2 keeps its ``corpus_000`` retrieval pin so the
  document evidence is graded independently of any echo, T4 is in-context by
  design, and T5/T7/T8 run in fresh sessions. The scenario metadata records it;
- no ``must_contain`` value is echoed by its own probe (R10 style).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.documents import resolve_document_path
from core.runner import BenchmarkRunner

SCENARIO_ID = "document_authority_vs_recency_01"
DOCUMENTED = "YKX-HMD0-7872"
CASUAL = "RUM-4X9Q-2201"
OFFICIAL = "AUT-8K2M-5530"
TICKET = "CHG-4471"
LOOP969 = "BQR-485R-6500"
RACK923 = "UVA-046I-7504"

#: doc_id -> a needle value that must appear in that document.
EXPECTED_DOC_TOKENS = {
    "corpus_000": DOCUMENTED,
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


def _chat_chunk_id(turns, turn_number):
    """``ep_<session>_t<n>`` for a turn: ``n`` counts human turns of its thread."""
    turn = turns[turn_number]
    ordinal = sum(
        1
        for t in turns.values()
        if t.session_id == turn.session_id
        and t.role == "user"
        and t.turn_number <= turn.turn_number
    )
    return f"ep_{turn.session_id}_t{ordinal}"


class TestScenarioShape:
    def test_loads_with_the_full_60_document_corpus(self, scenario):
        assert scenario.category == "document_revision"
        assert scenario.source_path is not None
        assert len(scenario.documents) == 60
        assert scenario.documents[0].doc_id == "corpus_000"
        assert scenario.documents[-1].doc_id == "corpus_059"
        assert scenario.metadata["corpus_dir"] == "d16_corpus_060"
        assert scenario.metadata["corpus_docs"] == 60
        assert [t.turn_number for t in scenario.turns] == list(range(1, 9))

    def test_metadata_pins_the_run_configuration(self, scenario):
        assert scenario.metadata["difficulty"] == "hard"
        assert scenario.metadata["stage"] == "prevalence/authority-vs-recency"
        assert scenario.metadata["target_research_question"] == "P2"
        assert scenario.metadata["chunk_size"] == 1200
        assert scenario.metadata["default_top_k"] == 8
        assert scenario.metadata["sessions"] == ["s1", "s2", "s3"]
        # Change #2 (source anchor) and change #4 (accepted T1 leak) are recorded
        # in the scenario itself, so a reader of the YAML cannot miss them.
        assert "must_retrieve_source: document" in scenario.metadata["source_anchor"]
        assert "doc_corpus_000_c0" in scenario.metadata["in_context_leak_note"]

    def test_session_layout(self, scenario):
        per_session: dict[str, list[int]] = {}
        for turn in scenario.turns:
            per_session.setdefault(turn.session_id, []).append(turn.turn_number)
        assert per_session == {"s1": [1, 2, 3, 4], "s2": [5, 6], "s3": [7, 8]}

    def test_chat_tokens_are_chat_exclusive(self, scenario):
        """RUM-4X9Q-2201 / AUT-8K2M-5530 must exist in NO document."""
        leaks = {
            doc_id: token
            for doc_id in (d.doc_id for d in scenario.documents)
            for token in (CASUAL, OFFICIAL)
            if token in _doc_text(scenario, doc_id)
        }
        assert leaks == {}, f"chat-only tokens leaked into documents: {leaks}"

    def test_document_tokens_live_in_their_documents(self, scenario):
        for doc_id, token in EXPECTED_DOC_TOKENS.items():
            assert token in _doc_text(scenario, doc_id), (
                f"{token} not found in {doc_id}"
            )

    def test_no_system_event_ingestion(self, scenario):
        """Every turn is a chat turn: nothing is smuggled in as a document."""
        assert all(t.role == "user" for t in scenario.turns)
        assert all(t.attachments == [] for t in scenario.turns)


class TestAuthorityClarification:
    def test_t1_states_the_authority_rule_without_assertions(self, scenario):
        turn = _turns(scenario)[1]
        assert turn.session_id == "s1"
        assert turn.role == "user"
        assert turn.assertions == []
        content = turn.content.casefold()
        assert "system of record" in content
        assert "authoritative" in content
        assert "approved change ticket" in content

    def test_t1_forbids_enumerating_codes(self, scenario):
        """Anti-leak instruction: T1's retrieval block may already hold the value."""
        content = _turns(scenario)[1].content.casefold()
        assert "don't restate any codes" in content

    def test_t1_leaks_no_code_value(self, scenario):
        content = _turns(scenario)[1].content
        for token in (DOCUMENTED, CASUAL, OFFICIAL, LOOP969, RACK923):
            assert token not in content

    def test_in_context_leak_is_documented_and_bounded(self, scenario):
        """The T1 leak is accepted, recorded, and never credited as evidence."""
        meta = scenario.metadata
        note = meta["in_context_leak_note"]
        assert "doc_corpus_000" in note
        assert "fresh sessions" in note
        turns = _turns(scenario)
        # Bounded by: T2 pins the chunk, T4 is explicitly in-context, T5+ are s2/s3.
        assert [r.value for r in _rules(turns[2], "must_retrieve_chunk")] == [
            "corpus_000"
        ]
        assert turns[4].session_id == "s1"
        assert turns[5].session_id == "s2"
        assert turns[7].session_id == "s3"


class TestCasualClaimChannel:
    def test_t3_is_a_confident_user_turn_without_assertions(self, scenario):
        turn = _turns(scenario)[3]
        assert turn.session_id == "s1"
        assert turn.role == "user"  # chat channel — never a system_event
        assert turn.attachments == []
        assert turn.assertions == []
        assert CASUAL in turn.content

    @pytest.mark.parametrize("word", ["official", "not sure", "ticket"])
    def test_t3_carries_no_hedge_or_ticket_wording(self, scenario, word):
        """A hedge would pre-solve the test; the claim must sound authoritative."""
        assert word not in _turns(scenario)[3].content.casefold()

    def test_t3_never_states_the_documented_or_approved_value(self, scenario):
        content = _turns(scenario)[3].content
        assert DOCUMENTED not in content
        assert OFFICIAL not in content


class TestForcedFormatProbes:
    @pytest.mark.parametrize("number", [4, 5])
    def test_probes_force_the_format_and_forbid_the_casual_claim(
        self, scenario, number
    ):
        turn = _turns(scenario)[number]
        assert turn.role == "user"
        # Rule R1: only a format-forcing probe may grade the stale value absent.
        assert "ONLY" in turn.content
        assert [r.value for r in _rules(turn, "must_contain")] == [DOCUMENTED]
        assert [r.value for r in _rules(turn, "must_not_contain")] == [CASUAL]

    def test_probes_split_same_session_and_cross_session(self, scenario):
        turns = _turns(scenario)
        assert turns[4].session_id == "s1"
        assert turns[5].session_id == "s2"
        assert "New session" in turns[5].content

    def test_cross_session_probe_pins_the_document_chunk(self, scenario):
        turn = _turns(scenario)[5]
        assert [r.value for r in _rules(turn, "must_retrieve_chunk")] == [
            "corpus_000"
        ]

    def test_cross_session_probe_anchors_a_document_source(self, scenario):
        """Source-level anchor: an all-chat pool must never satisfy T5.

        ``must_not_retrieve_source: chat`` is deliberately NOT used — the
        casual-claim chunk is supposed to compete for the same slots, so that
        rule would fail by construction (every observed T5 pool mixed both).
        """
        turn = _turns(scenario)[5]
        assert [r.value for r in _rules(turn, "must_retrieve_source")] == ["document"]
        assert _rules(turn, "must_not_retrieve_source") == []
        assert "source=\"document\"" not in turn.content

    def test_same_session_probe_pins_no_chunk(self, scenario):
        """T4 is answered from the thread; its grading must not depend on RAG."""
        assert _rules(_turns(scenario)[4], "must_retrieve_chunk") == []
        assert _rules(_turns(scenario)[4], "must_retrieve_source") == []


class TestOfficialUpdateChannel:
    def test_t6_names_the_ticket_and_the_new_value(self, scenario):
        turn = _turns(scenario)[6]
        assert turn.session_id == "s2"
        assert turn.role == "user"
        assert turn.assertions == []
        content = turn.content.casefold()
        assert "ticket" in content
        assert TICKET.casefold() in content
        assert "approved" in content
        assert OFFICIAL in turn.content

    def test_official_update_is_exactly_the_chunk_pinned_by_t7(self, scenario):
        turns = _turns(scenario)
        assert _chat_chunk_id(turns, 6) == "ep_s2_t2"
        pinned = [r.value for r in _rules(turns[7], "must_retrieve_chunk")]
        assert pinned == [_chat_chunk_id(turns, 6)]

    def test_authority_metadata_matches_the_turns(self, scenario):
        meta = scenario.metadata
        turns = _turns(scenario)
        assert meta["documented_value"]["token"] == DOCUMENTED
        assert meta["documented_value"]["doc_id"] == "corpus_000"
        assert meta["casual_claim"] == {"token": CASUAL, "turn": 3}
        assert meta["official_update"] == {
            "token": OFFICIAL,
            "turn": 6,
            "ticket": TICKET,
        }
        assert CASUAL in turns[meta["casual_claim"]["turn"]].content
        assert OFFICIAL in turns[meta["official_update"]["turn"]].content
        assert "system of record" in meta["authority_rule"].casefold()
        assert "approved change ticket" in meta["authority_rule"].casefold()


class TestLenientPostTicketProbe:
    def test_t7_adopts_the_approved_value(self, scenario):
        turn = _turns(scenario)[7]
        assert turn.session_id == "s3"
        assert [r.value for r in _rules(turn, "must_contain")] == [OFFICIAL]

    def test_t7_is_lenient_about_superseded_values(self, scenario):
        """R1: no must_not_contain — a correct answer may cite the old values."""
        turn = _turns(scenario)[7]
        assert _rules(turn, "must_not_contain") == []
        assert DOCUMENTED not in turn.content
        assert CASUAL not in turn.content

    def test_t7_asks_for_the_current_value(self, scenario):
        assert "current" in _turns(scenario)[7].content.casefold()


class TestControlProbe:
    def test_t8_pins_the_untouched_document(self, scenario):
        turn = _turns(scenario)[8]
        assert turn.session_id == "s3"
        assert [r.value for r in _rules(turn, "must_contain")] == [RACK923]
        assert [r.value for r in _rules(turn, "must_retrieve_chunk")] == [
            "corpus_007"
        ]
        assert _rules(turn, "must_not_contain") == []


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
        assert turns_with_negation == [4, 5]

