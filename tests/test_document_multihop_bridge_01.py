
"""Invariants for the document multi-hop bridge cell (compositional retrieval).

Pins the contract introduced with ``document_multihop_bridge_01`` and its
in-context twin ``in_context_bridge_01``:

1. the scenario loads, spans two sessions and declares the document corpus;
2. the corpus is big enough for retrieval to be selective and small enough
   that the competitors survive the ``top_k`` window (35-70 chunks);
3. the description occurs in exactly one chunk and the identifier in exactly
   two chunks, so hop 1 has a unique target;
4. the answer token occurs in exactly one chunk and is never inside a probe;
5. the register shares no content word with the composed probe, so **BM25
   cannot reach it in one hop** - the finding that motivated this cell
   (the ``d16`` needles were solvable by a keyword engine at 24/24);
6. the identifier query *can* reach it, so the cell is solvable in two hops;
7. the in-context twin carries the same question and the same assertion, so a
   failure here separates composition from reading.

Everything is offline: the splitter mirrors ``VectorMemoryBackend`` exactly
(``chunk_size=1200``/``chunk_overlap=200``) and the lexical baseline is a
dependency-free BM25. No API keys, no model calls, no runner invocation.
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

BRIDGE_ID = "document_multihop_bridge_01"
TWIN_ID = "in_context_bridge_01"
BRIDGE_PATH = (
    CONVERSATIONS
    / "multi_thread"
    / "compositional_reasoning"
    / "document_bridge"
    / f"test_{BRIDGE_ID}.yaml"
)
TWIN_PATH = (
    CONVERSATIONS
    / "single_thread"
    / "compositional_reasoning"
    / "in_context_bridge"
    / f"test_{TWIN_ID}.yaml"
)

ANSWER = "QMV-T3R8-4402"
IDENTIFIER = "NBL-420"
DESCRIPTION = "west-aisle cooling loop"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
CHUNK_BAND = (35, 70)
DOC_BAND = (12, 16)

#: Words that would let a single query reach the register without the
#: identifier. The register must stay free of them - that is hop 2's lock.
REGISTER_FORBIDDEN = ("west", "aisle", "cooling", "loop", "figure", "recorded")

#: Cues that would do the composition for the model. A probe that said "the
#: document where the value is held" or "the latest figure" would hand over the
#: hop the cell is measuring, so they must not appear in any user turn.
COMPOSITION_CUES = (
    "held in",
    "where the value",
    "register",
    "schedule",
    "supersed",
    "latest",
    "current",
    "previous",
)

_TOKEN = re.compile(r"[a-z0-9]+")


def _scenario(path: Path) -> BenchmarkScenario:
    return BenchmarkScenario.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


@pytest.fixture(scope="module")
def scenario() -> BenchmarkScenario:
    return _scenario(BRIDGE_PATH)


@pytest.fixture(scope="module")
def twin() -> BenchmarkScenario:
    return _scenario(TWIN_PATH)


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def _doc_text(scenario: BenchmarkScenario, doc_id: str) -> str:
    spec = next(s for s in scenario.documents if s.doc_id == doc_id)
    return (DOCUMENTS / spec.path).read_text(encoding="utf-8")


def _corpus(scenario: BenchmarkScenario) -> list[tuple[str, int, str]]:
    """``(doc_id, index, text)`` for every chunk."""
    splitter = _splitter()
    out: list[tuple[str, int, str]] = []
    for spec in scenario.documents:
        text = (DOCUMENTS / spec.path).read_text(encoding="utf-8")
        for index, part in enumerate(splitter.split_text(text)):
            out.append((spec.doc_id, index, part))
    return out


def _chunk_ids(scenario: BenchmarkScenario) -> list[str]:
    """Chunk ids exactly as ``VectorMemoryBackend`` would build them."""
    splitter = _splitter()
    ids: list[str] = []
    for spec in scenario.documents:
        text = (DOCUMENTS / spec.path).read_text(encoding="utf-8")
        if len(text) <= CHUNK_SIZE:
            ids.append(f"doc_{spec.doc_id}")
        else:
            for index, _ in enumerate(splitter.split_text(text)):
                ids.append(f"doc_{spec.doc_id}_c{index}")
    return ids


def _turn(scenario: BenchmarkScenario, number: int) -> Any:
    return next(t for t in scenario.turns if t.turn_number == number)


class TestScenarioShape:
    def test_loads_and_matches_its_file_name(self, scenario):
        assert scenario.id == BRIDGE_ID

    def test_two_sessions_so_a_checkpointer_cannot_answer_it(self, scenario):
        sessions = {t.session_id for t in scenario.turns}
        assert sessions == {"s1", "s2"}
        assert [t.turn_number for t in scenario.turns] == [1, 2, 3, 4, 5]

    def test_declares_the_document_corpus(self, scenario):
        assert DOC_BAND[0] <= len(scenario.documents) <= DOC_BAND[1]
        ids = [spec.doc_id for spec in scenario.documents]
        assert len(ids) == len(set(ids)), "duplicate doc_id"

    def test_declares_chunking_and_target_k(self, scenario):
        meta = scenario.metadata
        assert meta["chunk_size"] == CHUNK_SIZE
        assert meta["chunk_overlap"] == CHUNK_OVERLAP
        assert meta["default_top_k"] == 5

    def test_document_paths_resolve(self, scenario):
        for spec in scenario.documents:
            assert (DOCUMENTS / spec.path).is_file(), spec.path

    def test_declares_the_calibration_block(self, scenario):
        calibration = scenario.metadata["calibration"]
        assert calibration["hop_1_doc"] == "asset_registry_01"
        assert calibration["hop_2_doc"] == "calibration_register_02"
        assert calibration["chunks_measured"] == len(_corpus(scenario))
        assert set(calibration["expected_arms"]) == {"L2a_k5", "L2a_k15", "L2b_3x5"}


class TestCorpusInvariants:
    def test_chunk_count_is_in_the_discriminating_band(self, scenario):
        chunks = _corpus(scenario)
        low, high = CHUNK_BAND
        assert low <= len(chunks) <= high, (
            f"{len(chunks)} chunks: below {low} the whole corpus fits the window, "
            f"above {high} the competitors stop reaching it"
        )

    def test_every_document_is_a_plausible_length(self, scenario):
        for spec in scenario.documents:
            chars = len((DOCUMENTS / spec.path).read_text(encoding="utf-8"))
            assert 1200 <= chars <= 4000, f"{spec.doc_id}: {chars} chars"

    def test_description_occurs_in_exactly_one_chunk(self, scenario):
        hits = [
            (doc, index)
            for doc, index, text in _corpus(scenario)
            if DESCRIPTION in text.casefold()
        ]
        assert len(hits) == 1, f"description is not unique: {hits}"
        assert hits[0][0] == "asset_registry_01"

    def test_identifier_occurs_in_exactly_two_documents(self, scenario):
        docs = [
            spec.doc_id
            for spec in scenario.documents
            if IDENTIFIER in (DOCUMENTS / spec.path).read_text(encoding="utf-8")
        ]
        assert sorted(docs) == ["asset_registry_01", "calibration_register_02"], docs

    def test_answer_token_occurs_in_exactly_one_chunk(self, scenario):
        hits = [(doc, i) for doc, i, text in _corpus(scenario) if ANSWER in text]
        assert len(hits) == 1, f"answer token is not unique: {hits}"
        assert hits[0][0] == "calibration_register_02"

    def test_register_is_free_of_the_probe_vocabulary(self, scenario):
        text = _doc_text(scenario, "calibration_register_02").casefold()
        leaked = [word for word in REGISTER_FORBIDDEN if word in text]
        assert not leaked, (
            "the register shares words with the composed probe, so a single "
            f"query could reach hop 2 directly: {leaked}"
        )

    def test_retrieval_pins_match_real_chunk_ids(self, scenario):
        ids = _chunk_ids(scenario)
        for turn in scenario.turns:
            for rule in turn.assertions:
                if not rule.type.endswith("_chunk"):
                    continue
                assert any(rule.value in chunk_id for chunk_id in ids), (
                    f"T{turn.turn_number}: pin {rule.value!r} matches no chunk id"
                )


class TestAssertions:
    def test_composed_probe_pins_both_hops(self, scenario):
        """Both hops, pinned at CHUNK level.

        A document-level pin is satisfied by any chunk of the document, and
        `calibration_register_02` splits into [145, 1113, 759] chars where only
        c1 carries the figure — c0 is a header. Pinning the document therefore
        certifies "hop 2 arrived" from a chunk that cannot answer the probe.
        """
        turn = _turn(scenario, 2)
        pins = {r.value for r in turn.assertions if r.type == "must_retrieve_chunk"}
        assert pins == {"asset_registry_01_c0", "calibration_register_02_c1"}
        positives = {r.value for r in turn.assertions if r.type == "must_contain"}
        assert positives == {ANSWER}

    def test_each_pin_names_the_chunk_that_carries_its_evidence(self, scenario):
        """The hop pins must track the corpus, not a hand-maintained id.

        Recomputed from the chunker: the chunk containing the description and
        the chunk containing the answer *are* what T2 needs in the prompt. If an
        edit to the corpus moves the table into a different chunk, the pin value
        has to move with it or this test fails.
        """
        chunks = _corpus(scenario)
        expected = set()
        for doc_id, token, casefold in (
            ("asset_registry_01", DESCRIPTION, True),
            ("calibration_register_02", ANSWER, False),
        ):
            hits = [
                index
                for doc, index, text in chunks
                if doc == doc_id and token in (text.casefold() if casefold else text)
            ]
            assert len(hits) == 1, f"{token!r} is not in exactly one chunk: {hits}"
            expected.add(f"{doc_id}_c{hits[0]}")

        pins = {
            r.value
            for r in _turn(scenario, 2).assertions
            if r.type == "must_retrieve_chunk"
        }
        assert pins == expected

    def test_composed_probe_carries_a_precision_pin(self, scenario):
        turn = _turn(scenario, 2)
        negatives = [
            r.value for r in turn.assertions if r.type == "must_not_retrieve_chunk"
        ]
        assert negatives, "a document cell needs at least one precision pin (H5)"

    def test_no_probe_leaks_the_identifier(self, scenario):
        for turn in scenario.turns:
            if not turn.assertions:
                continue
            assert IDENTIFIER not in turn.content, (
                f"T{turn.turn_number} hands over the identifier, removing hop 1"
            )

    def test_no_probe_leaks_the_answer_token(self, scenario):
        for turn in scenario.turns:
            if not turn.assertions:
                continue
            assert ANSWER not in turn.content

    def test_hop_one_is_asserted_in_isolation(self, scenario):
        turn = _turn(scenario, 3)
        assert {r.value for r in turn.assertions if r.type == "must_contain"} == {
            IDENTIFIER
        }

    def test_single_hop_control_uses_a_corpus_exclusive_token(self, scenario):
        turn = _turn(scenario, 4)
        assert [r.value for r in turn.assertions if r.type == "must_contain"] == [
            "six weeks"
        ]
        sources = [
            spec.doc_id
            for spec in scenario.documents
            if "six weeks"
            in (DOCUMENTS / spec.path).read_text(encoding="utf-8").casefold()
        ]
        assert sources == ["bg_07"], sources

    def test_guard_turn_is_negative_only(self, scenario):
        turn = _turn(scenario, 5)
        assert turn.probe is not None and turn.probe.query_kind == "guard"
        assert {r.type for r in turn.assertions} == {"must_not_contain"}

    def test_every_asserting_turn_declares_a_query_kind(self, scenario):
        for turn in scenario.turns:
            if not turn.assertions:
                continue
            assert turn.probe is not None, f"T{turn.turn_number} has no probe block"

    def test_no_user_turn_does_the_composition_for_the_model(self, scenario):
        """The probe must not name the second document or the precedence rule.

        Saying "the limit schedule" or "the latest figure" would hand over the hop
        this cell measures — and would also make the acknowledgement turn a map of
        the corpus, which an agentic arm could exploit without retrieving.
        """
        offenders = [
            (turn.turn_number, turn.content, cue)
            for turn in scenario.turns
            if turn.role == "user"
            for cue in COMPOSITION_CUES
            if cue in turn.content.casefold()
        ]
        assert not offenders, offenders


class TestLexicalGate:
    """BM25 must not close the hop; the identifier must.

    This gate is what separates this cell from the ``d16`` corpora, where a
    keyword engine retrieved the gold chunk 24/24 at k=5 because the probe
    named the unique lookup token.
    """

    @staticmethod
    def _bm25(chunks: list[tuple[str, int, str]], k1: float = 1.5, b: float = 0.75):
        docs = [_TOKEN.findall(text.casefold()) for _, _, text in chunks]
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
    def _hop_indices(chunks: list[tuple[str, int, str]]) -> tuple[int, int]:
        hop_one = next(
            i
            for i, (doc, _, text) in enumerate(chunks)
            if doc == "asset_registry_01" and DESCRIPTION in text.casefold()
        )
        hop_two = next(
            i
            for i, (doc, _, text) in enumerate(chunks)
            if doc == "calibration_register_02" and ANSWER in text
        )
        return hop_one, hop_two

    def test_bm25_ranks_hop_one_first_and_cannot_reach_hop_two(self, scenario):
        chunks = _corpus(scenario)
        hop_one, hop_two = self._hop_indices(chunks)
        order = self._bm25(chunks)(_turn(scenario, 2).content)
        assert order.index(hop_one) == 0, "hop 1 must be trivially reachable"
        assert order.index(hop_two) + 1 > 5, (
            "hop 2 sits inside the k=5 window for a lexical query: the cell "
            "would be solvable in one hop"
        )

    def test_identifier_query_does_reach_hop_two(self, scenario):
        chunks = _corpus(scenario)
        _, hop_two = self._hop_indices(chunks)
        order = self._bm25(chunks)(IDENTIFIER)
        assert order.index(hop_two) == 0, "hop 2 is unreachable even with the identifier"

    def test_declared_baseline_matches_the_measurement(self, scenario):
        declared = scenario.metadata["corpus_difficulty"]["lexical_baseline_hit_at_5"]
        chunks = _corpus(scenario)
        _, hop_two = self._hop_indices(chunks)
        order = self._bm25(chunks)(_turn(scenario, 2).content)
        measured = 0.0 if order.index(hop_two) + 1 > 5 else 1.0
        assert declared == measured
        assert declared <= 0.6


class TestInContextTwin:
    def test_twin_loads_in_one_session(self, twin):
        assert twin.id == TWIN_ID
        assert len({t.session_id for t in twin.turns}) == 1
        assert len(twin.turns) == 1

    def test_twin_declares_what_it_controls(self, twin):
        assert twin.metadata["kind"] == "control_in_context"
        assert twin.metadata["paired_with"] == BRIDGE_ID
        assert twin.metadata["controls"]

    def test_twin_asks_the_same_question_with_the_same_assertion(self, twin, scenario):
        probe = " ".join(_turn(scenario, 2).content.split())
        content = " ".join(twin.turns[0].content.split())
        assert content.endswith(probe[-40:]), (
            "the twin must end with the bridge cell's question verbatim"
        )
        assert [r.value for r in twin.turns[0].assertions] == [ANSWER]

    def test_twin_carries_both_hop_rows(self, twin):
        text = " ".join(twin.turns[0].content.split())
        assert DESCRIPTION in text.casefold()
        assert IDENTIFIER in text
        assert ANSWER in text
