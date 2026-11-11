
"""Invariants for the document conflict & dilution cell.

Pins the contract introduced with ``document_conflict_dilution_01``:

1. the scenario loads, spans two sessions and declares the 14-document corpus;
2. the corpus sits in the band where the competitors fit the ``top_k`` window
   (35-70 chunks) — the opposite constraint from the bridge cell, which needs
   the competitors to be *unreachable*;
3. each of the four pump limits, and the superseded value, occurs in exactly one
   chunk, so every answer has exactly one legitimate source;
4. the current record never states the stale value and the superseded record
   never states the current one — otherwise precedence would be decided by
   which document happened to be read;
5. the four live pump records are written from one skeleton: identical
   headings, identical paragraph count, and every paragraph except the title
   and the asset body is identical once identifiers, values, roles and dates
   are masked (this is what makes them compete);
6. at ``k=5`` the composed window really does contain the gold record, the
   superseded record and the three siblings — i.e. the cell has competition,
   measured lexically;
7. the assertions target the right document each time, the paired commitment
   probe guards against binding bleed, and the guard turn is negative-only;
8. the cell is namespace-isolated from ``document_multihop_bridge_01`` so the
   two corpora cannot contradict each other.

Everything is offline: same splitter as ``VectorMemoryBackend`` (1200/200) and a
dependency-free BM25. No API keys, no model calls.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.schemas import BenchmarkScenario

REPO_ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS = REPO_ROOT / "datasets" / "conversations"
DOCUMENTS = REPO_ROOT / "datasets" / "documents"

SCENARIO_ID = "document_conflict_dilution_01"
SCENARIO_PATH = (
    CONVERSATIONS
    / "multi_thread"
    / "logical_consistency"
    / "document_conflict"
    / f"test_{SCENARIO_ID}.yaml"
)

CURRENT = "HLX-2C7V-9043"      # PMP-310, January 2026 cycle
SIBLING_02 = "RKV-9T4B-6602"   # PMP-335, second standby
STALE = "SWG-1F9K-5520"        # PMP-310, June 2025 cycle
JOCKEY = "HLX-6D3P-1175"       # PMP-348, guard target
PUMPS = (
    "conflict_gold_01",
    "conflict_sibling_01",
    "conflict_sibling_02",
    "conflict_sibling_03",
)
SUPERSEDED = "conflict_superseded_01"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
CHUNK_BAND = (35, 70)
DOC_BAND = (12, 16)
MIN_JACCARD = 0.85

#: Cues that would hand the model the precedence decision it is being tested on.
#: They may appear in the *documents* (that is the evidence) but never in a user
#: turn: a probe that says "the superseded one" or "the latest value" defuses the
#: trap before retrieval starts.
PRECEDENCE_CUES = (
    "supersed",
    "stale",
    "outdated",
    "obsolete",
    "deprecat",
    "previous",
    "previously",
    "latest",
    "current",
    "newer",
    "older",
    "no longer",
    "revis",
    "changed",
    "still the same",
    "earlier",
    "used to",
)

_TOKEN = re.compile(r"[a-z0-9]+")
_MASKS = (
    (re.compile(r"pmp-\d+"), "<id>"),
    (re.compile(r"\b[a-z]{3}-[a-z0-9]{4}-\d{4}\b"), "<val>"),
    (re.compile(r"the (second standby|duty|standby|jockey) pump"), "the <role> pump"),
    (re.compile(r"\b(january 2026|june 2025|2026|2025)\b"), "<date>"),
)


def _scenario() -> BenchmarkScenario:
    return BenchmarkScenario.model_validate(
        yaml.safe_load(SCENARIO_PATH.read_text(encoding="utf-8"))
    )


@pytest.fixture(scope="module")
def scenario() -> BenchmarkScenario:
    return _scenario()


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def _doc_text(doc_id: str) -> str:
    spec = next(s for s in _scenario().documents if s.doc_id == doc_id)
    return (DOCUMENTS / spec.path).read_text(encoding="utf-8")


def _chunks(scenario: BenchmarkScenario) -> list[tuple[str, str]]:
    splitter = _splitter()
    out: list[tuple[str, str]] = []
    for spec in scenario.documents:
        text = (DOCUMENTS / spec.path).read_text(encoding="utf-8")
        for part in splitter.split_text(text):
            out.append((spec.doc_id, part))
    return out


def _turn(scenario: BenchmarkScenario, number: int) -> Any:
    return next(t for t in scenario.turns if t.turn_number == number)


def _mask(text: str) -> str:
    out = " ".join(text.casefold().split())
    for pattern, replacement in _MASKS:
        out = pattern.sub(replacement, out)
    return out


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


class TestScenarioShape:
    def test_loads_with_the_expected_identity(self, scenario):
        assert scenario.id == SCENARIO_ID
        assert scenario.metadata["kind"] == "revision"
        assert scenario.metadata["quality_eligible"] is True
        assert scenario.metadata["amnesia_passes"] is False

    def test_two_sessions_and_sequential_turns(self, scenario):
        assert {t.session_id for t in scenario.turns} == {"s1", "s2"}
        assert [t.turn_number for t in scenario.turns] == [1, 2, 3, 4, 5]

    def test_declares_corpus_and_chunking(self, scenario):
        assert DOC_BAND[0] <= len(scenario.documents) <= DOC_BAND[1]
        assert scenario.metadata["chunk_size"] == CHUNK_SIZE
        assert scenario.metadata["default_top_k"] == 5
        for spec in scenario.documents:
            assert (DOCUMENTS / spec.path).is_file(), spec.path

    def test_reuses_filler_and_exactly_one_decoy(self, scenario):
        calibration = scenario.metadata["calibration"]
        assert calibration["reused_filler"] == [f"bg_0{n}" for n in range(1, 9)]
        assert calibration["reused_decoy"] == "decoy_01"
        ids = {spec.doc_id for spec in scenario.documents}
        assert {"decoy_01"} <= ids
        assert not ids & {"decoy_02", "decoy_03", "decoy_04"}


class TestCorpusInvariants:
    def test_chunk_count_is_in_the_competitive_band(self, scenario):
        chunks = _chunks(scenario)
        low, high = CHUNK_BAND
        assert low <= len(chunks) <= high, len(chunks)
        assert len(chunks) == scenario.metadata["calibration"]["chunks_measured"]

    def test_every_document_is_a_plausible_length(self, scenario):
        for spec in scenario.documents:
            chars = len((DOCUMENTS / spec.path).read_text(encoding="utf-8"))
            assert 1200 <= chars <= 4000, f"{spec.doc_id}: {chars}"

    @pytest.mark.parametrize("token", [CURRENT, SIBLING_02, STALE, JOCKEY])
    def test_each_value_occurs_in_exactly_one_chunk(self, scenario, token):
        hits = [(doc, i) for i, (doc, text) in enumerate(_chunks(scenario)) if token in text]
        assert len(hits) == 1, f"{token}: {hits}"

    def test_current_and_superseded_records_do_not_leak_into_each_other(self):
        assert STALE not in _doc_text("conflict_gold_01")
        assert CURRENT not in _doc_text(SUPERSEDED)

    def test_precedence_is_inferable_from_date_and_tense_alone(self):
        """No "superseded" label: the reader must compare cycles and tense.

        The label was removed deliberately — it performed the precedence
        reasoning this cell is meant to measure. What remains is what a real
        archive gives: a dated record written in the past tense, and a later
        record written in the present tense for the same asset.
        """
        old = " ".join(_doc_text(SUPERSEDED).casefold().split())
        new = " ".join(_doc_text("conflict_gold_01").casefold().split())
        assert "supersed" not in old, "the label must not be in the document"
        assert "june 2025" in old and " was " in old
        assert "january 2026" in new and " is " in new
        assert "2026" not in old, "the old record must not point forward at the newer cycle"


class TestSkeleton:
    def test_all_pump_records_share_the_same_structure(self):
        """Every pump record uses the same section skeleton (the H1 differs by role)."""
        structures = {}
        for doc in PUMPS:
            paragraphs = _paragraphs(_doc_text(doc))
            structures[doc] = [
                p.splitlines()[0].strip()
                for p in paragraphs
                if p.lstrip().startswith("## ")
            ]
        assert len({tuple(v) for v in structures.values()}) == 1, structures
        assert len(structures[PUMPS[0]]) == 5, structures[PUMPS[0]]

    def test_paragraphs_are_identical_apart_from_title_and_asset_body(self):
        base = [_mask(p) for p in _paragraphs(_doc_text(PUMPS[0]))]
        for doc in PUMPS[1:]:
            masked = [_mask(p) for p in _paragraphs(_doc_text(doc))]
            assert len(masked) == len(base), doc
            differing = [i for i, (a, b) in enumerate(zip(base, masked)) if a != b]
            assert differing == [0, 2], (
                f"{doc}: only the title (0) and the asset body (2) may differ, "
                f"found {differing}"
            )

    def test_lexical_similarity_between_gold_and_each_sibling(self):
        gold = set(_TOKEN.findall(_mask(_doc_text(PUMPS[0]))))
        for doc in PUMPS[1:]:
            other = set(_TOKEN.findall(_mask(_doc_text(doc))))
            jaccard = len(gold & other) / len(gold | other)
            assert jaccard >= MIN_JACCARD, f"{doc}: jaccard {jaccard:.3f}"


class TestCrossCellIsolation:
    def test_bridge_namespace_is_absent_from_this_corpus(self, scenario):
        blob = " ".join(text for _, text in _chunks(scenario))
        for token in ("QMV-T3R8-4402", "NBL-420", "asset_registry_01"):
            assert token not in blob, token

    def test_this_namespace_is_absent_from_the_bridge_corpus(self):
        bridge_docs = (
            REPO_ROOT
            / "datasets/conversations/multi_thread/compositional_reasoning/document_bridge"
            / "test_document_multihop_bridge_01.yaml"
        )
        bridge = BenchmarkScenario.model_validate(
            yaml.safe_load(bridge_docs.read_text(encoding="utf-8"))
        )
        blob = " ".join(
            (DOCUMENTS / spec.path).read_text(encoding="utf-8")
            for spec in bridge.documents
        )
        for token in ("PMP-310", CURRENT, STALE):
            assert token not in blob, token


class TestCompetition:
    """The window must contain the answer *and* its distractors.

    This is the mirror image of the bridge cell's lexical gate: there the gold
    had to be unreachable; here it has to be surrounded.
    """

    @staticmethod
    def _bm25(chunks: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        docs = [_TOKEN.findall(text.casefold()) for _, text in chunks]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(doc))
        avgdl = sum(len(doc) for doc in docs) / len(docs)
        total = len(docs)

        def rank(query: str) -> list[int]:
            terms = _TOKEN.findall(query.casefold())
            scored = []
            for index, doc in enumerate(docs):
                tf = Counter(doc)
                score = 0.0
                for term in terms:
                    if term not in tf:
                        continue
                    idf = math.log(1 + (total - df[term] + 0.5) / (df[term] + 0.5))
                    score += idf * tf[term] * (k1 + 1) / (
                        tf[term] + k1 * (1 - b + b * len(doc) / avgdl)
                    )
                scored.append((score, index))
            scored.sort(key=lambda pair: -pair[0])
            return [index for _, index in scored]

        return rank

    @staticmethod
    def _best_doc_rank(order: list[int], chunks, doc_id: str) -> int:
        positions = {
            index: rank + 1 for rank, index in enumerate(order)
        }
        ranks = [positions[i] for i, (doc, _) in enumerate(chunks) if doc == doc_id]
        return min(ranks)

    def test_k5_window_holds_the_gold_its_siblings_and_the_superseded_record(
        self, scenario
    ):
        chunks = _chunks(scenario)
        order = self._bm25(chunks)(_turn(scenario, 2).content)
        window = [chunks[i][0] for i in order[:5]]
        assert window[0] == "conflict_gold_01", window
        assert SUPERSEDED in window, window
        assert sum(1 for doc in window if doc in PUMPS[1:]) >= 2, window

    def test_declared_ranks_match_the_measurement(self, scenario):
        chunks = _chunks(scenario)
        order = self._bm25(chunks)(_turn(scenario, 2).content)
        declared = scenario.metadata["corpus_difficulty"]
        assert self._best_doc_rank(order, chunks, "conflict_gold_01") == declared[
            "measured_gold_rank"
        ]
        assert self._best_doc_rank(order, chunks, SUPERSEDED) == declared[
            "measured_superseded_rank"
        ]
        ranks = sorted(
            self._best_doc_rank(order, chunks, doc) for doc in PUMPS[1:]
        )
        assert ranks == declared["measured_sibling_ranks"]

    def test_off_topic_filler_stays_out_of_the_window(self, scenario):
        chunks = _chunks(scenario)
        order = self._bm25(chunks)(_turn(scenario, 2).content)
        window = [chunks[i][0] for i in order[:5]]
        assert "bg_02" not in window, window


class TestAssertions:
    def test_t2_targets_the_current_record(self, scenario):
        turn = _turn(scenario, 2)
        assert [r.value for r in turn.assertions if r.type == "must_contain"] == [
            CURRENT
        ]
        assert "conflict_gold_01" in [
            r.value for r in turn.assertions if r.type == "must_retrieve_chunk"
        ]

    def test_t2_carries_a_precision_pin(self, scenario):
        turn = _turn(scenario, 2)
        assert [
            r.value for r in turn.assertions if r.type == "must_not_retrieve_chunk"
        ]

    def test_t3_is_a_paired_commitment_with_a_bleed_guard(self, scenario):
        turn = _turn(scenario, 3)
        assert [r.value for r in turn.assertions if r.type == "must_contain"] == [
            SIBLING_02
        ]
        assert CURRENT in [
            r.value for r in turn.assertions if r.type == "must_not_contain"
        ], "the sibling probe must forbid the duty pump's value"
        assert "conflict_sibling_02" in [
            r.value for r in turn.assertions if r.type == "must_retrieve_chunk"
        ]

    def test_t4_asks_for_history_and_pins_the_superseded_record(self, scenario):
        turn = _turn(scenario, 4)
        assert [r.value for r in turn.assertions if r.type == "must_contain"] == [
            STALE
        ]
        assert "conflict_superseded_01" in [
            r.value for r in turn.assertions if r.type == "must_retrieve_chunk"
        ]

    def test_guard_turn_is_negative_only(self, scenario):
        turn = _turn(scenario, 5)
        assert turn.probe is not None and turn.probe.query_kind == "guard"
        assert {r.type for r in turn.assertions} == {"must_not_contain"}

    def test_no_probe_leaks_an_asserted_value(self, scenario):
        for turn in scenario.turns:
            for rule in turn.assertions:
                assert rule.value.casefold() not in turn.content.casefold(), (
                    f"T{turn.turn_number}: {rule.value!r} is echoed by its own probe"
                )

    def test_every_asserting_turn_declares_a_query_kind(self, scenario):
        for turn in scenario.turns:
            if turn.assertions:
                assert turn.probe is not None, f"T{turn.turn_number}"

    def test_no_user_turn_signals_precedence(self, scenario):
        """The probe must not tell the model that a precedence decision exists.

        "Before the 2026 revision…" or "the latest value…" would do the
        disambiguation the cell is supposed to measure, and would also make the
        superseded document easier to recognise than it is in real tickets.
        """
        offenders = [
            (turn.turn_number, turn.content, cue)
            for turn in scenario.turns
            if turn.role == "user"
            for cue in PRECEDENCE_CUES
            if cue in turn.content.casefold()
        ]
        assert not offenders, offenders

    def test_the_history_probe_names_the_cycle_not_the_relationship(self, scenario):
        """T4 asks by cycle, never by "old" / "previous" / "superseded"."""
        content = _turn(scenario, 4).content.casefold()
        assert "2025" in content
        for cue in ("before", "supersed", "previous", "old", "changed", "revis"):
            assert cue not in content, cue
