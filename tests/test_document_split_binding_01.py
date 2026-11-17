"""Invariants for the isolated binding cell (one value probe, asked first).

`document_split_binding_01` is the DECISIVE half of the split-evidence pair. It
exists because the two value probes used to share a thread and the second one
read the field position out of the first one's answer: the binding verdict could
not be attributed to the corpus. This file pins the three things that keep it
clean:

1. **exactly one value probe**, and it is the first assertion-bearing turn — a
   second one would re-open the leak (`failure_taxonomy.t2_leaks_the_field_order`
   in the sibling cell documents the four-variant experiment);
2. the probe names **no term of art**, so a wrong field cannot be excused as a
   reading of the question;
3. it is **strict**: a single token with a guard on the other code, because a
   hedge satisfies must_contain while certifying nothing.

Plus the shared corpus invariants (severance, four fields per record, guide out
of the window) and the ordering rule that the format probe runs last — its
answer IS the field order.

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

SCENARIO_ID = "document_split_binding_01"
SCENARIO_PATH = (
    CONVERSATIONS
    / "multi_thread"
    / "compositional_reasoning"
    / "document_split"
    / f"test_{SCENARIO_ID}.yaml"
)

LEGEND = "split_legend_01"
ROWS_1 = "split_rows_01"
ROWS_2 = "split_rows_02"

#: North-riser record: the certified value is the THIRD field, so the model's
#: "report the second field" default answers the pressure instead. A third-riser
#: record cannot show this — its second field *is* the certified value.
V320_VALUE = "MVB-XNJL-1944"
V320_PRESSURE = "TRQ-HJZS-1262"

#: Third-riser record, kept as the counter-example pinned by a test.
V247_VALUE = "KVN-8T2D-4417"
V247_PRESSURE = "TST-3M6Q-9058"

FIELD_NAMES = ("asset identifier", "certified value", "test pressure", "verification date")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
CHUNK_BAND = (35, 70)

#: Words the decisive probe must not use. Naming any of them either points at one
#: field (making the question decidable without the corpus) or invites the model
#: to argue about the wording instead of binding the row.
TERMS_OF_ART = (
    "limit", "rating", "certified", "value", "pressure", "date", "order", "field",
)

#: Cues that would do the compositional work for the model.
COMPOSITION_CUES = ("held in", "where the value", "reading guide", "field order",
                    "schedule", "legend")

_TOKEN = re.compile(r"[a-z0-9]+")


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


def _data_rows(doc_id: str) -> list[list[str]]:
    rows = []
    for line in _doc_text(doc_id).splitlines():
        if line.count("|") == 3 and line.split("|")[0].strip().startswith("VLV-"):
            rows.append([field.strip() for field in line.split("|")])
    return rows


def _turn(scenario: BenchmarkScenario, number: int) -> Any:
    return next(t for t in scenario.turns if t.turn_number == number)


def _asserting_turns(scenario: BenchmarkScenario) -> list[Any]:
    return [t for t in scenario.turns if t.assertions]



class TestScenarioShape:
    def test_loads_with_the_expected_identity(self, scenario):
        assert scenario.id == SCENARIO_ID
        assert scenario.category == "document_split"
        assert scenario.metadata["kind"] == "recall"
        assert scenario.metadata["quality_eligible"] is True
        assert scenario.metadata["amnesia_passes"] is False

    def test_two_sessions_three_sequential_turns(self, scenario):
        assert {t.session_id for t in scenario.turns} == {"s1", "s2"}
        assert [t.turn_number for t in scenario.turns] == [1, 2, 3]
        assert _turn(scenario, 1).session_id == "s1"
        assert _turn(scenario, 2).session_id == "s2"

    def test_the_same_twelve_documents_as_the_diagnostic_cell(self, scenario):
        """The pair is comparable only if both ingest the identical corpus."""
        sibling = BenchmarkScenario.model_validate(
            yaml.safe_load(
                (SCENARIO_PATH.parent / "test_document_split_evidence_01.yaml")
                .read_text(encoding="utf-8")
            )
        )
        assert [d.doc_id for d in scenario.documents] == [
            d.doc_id for d in sibling.documents
        ]
        assert len(scenario.documents) == scenario.metadata["documents_count"] == 12

    def test_chunk_count_matches_the_declared_measurement(self, scenario):
        counted = len(_chunks(scenario))
        assert counted == scenario.metadata["calibration"]["chunks_measured"]
        low, high = CHUNK_BAND
        assert low <= counted <= high

    def test_document_paths_resolve(self, scenario):
        for spec in scenario.documents:
            assert (DOCUMENTS / spec.path).is_file(), spec.path


class TestSeverance:
    """The record must be reachable while the schema is not."""

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
    def _best_rank(order: list[int], chunks, doc_id: str) -> int:
        positions = {index: rank + 1 for rank, index in enumerate(order)}
        ranks = [positions[i] for i, (doc, _) in enumerate(chunks) if doc == doc_id]
        return min(ranks) if ranks else -1

    def test_the_binding_probe_reaches_the_record_but_not_the_guide(self, scenario):
        chunks = _chunks(scenario)
        order = self._bm25(chunks)(_turn(scenario, 2).content)
        assert self._best_rank(order, chunks, ROWS_2) <= 2
        assert self._best_rank(order, chunks, LEGEND) > 5, (
            "the guide is inside the k=5 window, so the row can be decoded in one hop"
        )

    def test_the_format_probe_reaches_the_guide(self, scenario):
        chunks = _chunks(scenario)
        order = self._bm25(chunks)(_turn(scenario, 3).content)
        assert self._best_rank(order, chunks, LEGEND) <= 5, (
            "the guide is unreachable even when the probe names the format: "
            "the cell would be unsolvable"
        )

    def test_declared_bounds_hold(self, scenario):
        chunks = _chunks(scenario)
        declared = scenario.metadata["corpus_difficulty"]["measured_ranks_bm25"]

        binding = self._bm25(chunks)(_turn(scenario, 2).content)
        gate = declared["t_binding_probe"]
        assert self._best_rank(binding, chunks, ROWS_2) == gate["rows_02_best"]
        assert self._best_rank(binding, chunks, LEGEND) > gate["legend_01_at_least"]

        fmt = self._bm25(chunks)(_turn(scenario, 3).content)
        assert (
            self._best_rank(fmt, chunks, LEGEND)
            <= declared["t_format_probe"]["legend_01_at_most"]
        )


class TestAssertions:
    def test_the_binding_probe_is_strict_and_guards_the_other_code(self, scenario):
        turn = _turn(scenario, 2)
        assert [r.value for r in turn.assertions if r.type == "must_contain"] == [
            V320_VALUE
        ]
        assert [r.value for r in turn.assertions if r.type == "must_not_contain"] == [
            V320_PRESSURE
        ], "a hedge must not certify a binding"
        assert ROWS_2 in [
            r.value for r in turn.assertions if r.type == "must_retrieve_chunk"
        ]
        assert "code only" in turn.content.casefold(), (
            "the guard is only fair while the probe asks for a single token"
        )

    def test_the_binding_probe_pins_the_second_hop(self, scenario):
        """The row alone cannot certify the probe: the guide is half the answer.

        A row carries two opaque codes and nothing that says which is the
        certified value — only `split_legend_01` does. While the pin covered just
        the row, `retrieval_recall` read 8/8 = 100 % with 16/24 assertions: the
        missing retrieval was unmetered, so the failure was unattributable and an
        agent that found the second chunk could not be told apart from one that
        never looked for it.
        """
        pins = sorted(
            r.value
            for r in _turn(scenario, 2).assertions
            if r.type == "must_retrieve_chunk"
        )
        assert pins == [LEGEND, ROWS_2], pins

    def test_the_binding_probe_names_no_term_of_art(self, scenario):
        content = _turn(scenario, 2).content.casefold()
        named = [word for word in TERMS_OF_ART if word in content]
        assert not named, named

    def test_the_binding_probe_stands_alone(self, scenario):
        """No "And valve VLV-247?": the probe must not lean on a prior turn."""
        content = _turn(scenario, 2).content.casefold().strip()
        assert not content.startswith(("and ", "also ", "same ", "what about")), content
        assert "vlv-320" in content

    def test_it_is_the_first_asserting_turn(self, scenario):
        first = _asserting_turns(scenario)[0]
        assert first.turn_number == 2, (
            "asking anything before the binding probe risks handing it the schema"
        )
        assert _turn(scenario, 1).assertions == []

    def test_exactly_one_value_probe_in_the_scenario(self, scenario):
        """The leak is only closed while there is a single value probe.

        A value probe is one that guards a code from the other field. Two of them
        in one thread was the original defect: the first answer carries the field
        identity into the second (`failure_taxonomy.t2_leaks_the_field_order`).
        """
        probes = [
            t.turn_number
            for t in scenario.turns
            if any(
                r.type == "must_not_contain" and r.value.startswith(("TST-", "KVN-", "MVB-", "TRQ-"))
                for r in t.assertions
            )
        ]
        assert probes == [2], probes

    def test_no_probe_runs_after_the_format_probe(self, scenario):
        """The format probe's answer IS the field order."""
        last = max(t.turn_number for t in _asserting_turns(scenario))
        assert last == 3
        assert LEGEND in [
            r.value for r in _turn(scenario, 3).assertions
            if r.type == "must_retrieve_chunk"
        ]

    def test_the_format_probe_asserts_the_order_not_a_value(self, scenario):
        turn = _turn(scenario, 3)
        assert [r.value for r in turn.assertions if r.type == "must_contain"] == [
            "certified value"
        ]
        assert V320_VALUE not in turn.content
        assert V320_PRESSURE not in turn.content


class TestCorpusContract:
    def test_each_asserted_value_lives_in_exactly_one_document(self, scenario):
        for token, expected in ((V320_VALUE, ROWS_2), (V320_PRESSURE, ROWS_2)):
            holders = [
                spec.doc_id
                for spec in scenario.documents
                if token in (DOCUMENTS / spec.path).read_text(encoding="utf-8")
            ]
            assert holders == [expected], (token, holders)

    def test_exports_carry_no_field_names(self, scenario):
        for doc_id in (ROWS_1, ROWS_2):
            text = _doc_text(doc_id).casefold()
            for name in FIELD_NAMES:
                assert name not in text, (doc_id, name)

    def test_every_record_carries_four_fields(self):
        for doc_id in (ROWS_1, ROWS_2):
            rows = _data_rows(doc_id)
            assert rows, doc_id
            assert all(len(row) == 4 for row in rows), doc_id

    def test_the_probe_targets_the_family_where_position_two_is_wrong(self):
        """This is why the binding cell probes the north riser, not the third.

        On a third-riser line the second field is the certified value, so
        "report the second field" — the model's measured default — is right by
        accident and the cell would pass with the guide out of reach.
        """
        north = [row for row in _data_rows(ROWS_2) if row[0] == "VLV-320"][0]
        assert north[1] == V320_PRESSURE and north[2] == V320_VALUE, (
            "the certified value must NOT sit at position 2 in the probed family"
        )
        third = [row for row in _data_rows(ROWS_1) if row[0] == "VLV-247"][0]
        assert third[1] == V247_VALUE and third[2] == V247_PRESSURE, (
            "counter-example: here position 2 IS the certified value"
        )

    def test_the_guide_states_the_order_the_probe_needs(self):
        guide = _doc_text(LEGEND)
        assert "certified value" in guide
        assert "north-riser" in guide.casefold()


class TestProbeHygiene:
    def test_no_probe_leaks_an_asserted_value(self, scenario):
        for turn in _asserting_turns(scenario):
            for rule in turn.assertions:
                if rule.type != "must_contain":
                    continue
                assert rule.value not in turn.content, (
                    f"T{turn.turn_number}: {rule.value!r} is echoed by its own probe"
                )

    def test_no_probe_does_the_compositional_work(self, scenario):
        for turn in _asserting_turns(scenario):
            content = turn.content.casefold()
            for cue in COMPOSITION_CUES:
                assert cue not in content, (turn.turn_number, cue)

    def test_every_asserting_turn_declares_a_query_kind(self, scenario):
        for turn in _asserting_turns(scenario):
            assert turn.probe is not None, f"T{turn.turn_number}"
            assert turn.probe.query_kind, f"T{turn.turn_number}"

    def test_other_cells_namespaces_are_absent_here(self, scenario):
        blob = " ".join(text for _, text in _chunks(scenario))
        for token in ("PMP-310", "NBL-420", "HLX-2C7V-9043", "SWG-1F9K-5520"):
            assert token not in blob, token
