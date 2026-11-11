
"""Invariants for the split-evidence cell (headerless export + external schema).

Pins the contract introduced with ``document_split_evidence_01``:

1. two exports carry one record per line and **no header** — verified by asserting
   that no chunk of either export contains a field name;
2. the only place the field order exists is ``split_legend_01``, and it lists the
   two riser families **in different orders**, so decoding a record with the other
   family's order yields a syntactically valid, semantically wrong answer;
3. the certified value of each asserted valve occurs in exactly one chunk;
4. for the value probes the record ranks first and the guide stays **outside** the
   ``top_k`` window (measured), so a single retrieval cannot decode the row;
5. for the format probe the guide **is** reachable, which is what makes the cell
   solvable in two hops (the solvability control);
6. the probes never name the format or the guide, and the guide probe runs *after*
   the value probes so the answer cannot be carried forward in the thread;
7. the cell is namespace-isolated from the bridge and conflict cells.

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

SCENARIO_ID = "document_split_evidence_01"
SCENARIO_PATH = (
    CONVERSATIONS
    / "multi_thread"
    / "compositional_reasoning"
    / "document_split"
    / f"test_{SCENARIO_ID}.yaml"
)

LEGEND = "split_legend_01"
ROWS_1 = "split_rows_01"   # third-riser export
ROWS_2 = "split_rows_02"   # north-riser export, different field order

V220_VALUE = "KVN-4C8R-1194"
V220_PRESSURE = "TST-9F2M-6630"
V247_VALUE = "KVN-8T2D-4417"
V247_PRESSURE = "TST-3M6Q-9058"
V331_VALUE = "MVB-2K9P-6634"
V331_PRESSURE = "TRQ-5N8C-2271"

FIELD_NAMES = ("asset identifier", "certified value", "test pressure", "verification date")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
CHUNK_BAND = (35, 70)
DOC_BAND = (10, 16)

PRECEDENCE_CUES = (
    "supersed", "stale", "outdated", "obsolete", "deprecat", "previous",
    "previously", "latest", "current", "newer", "older", "no longer", "revis",
    "changed", "earlier", "used to",
)
COMPOSITION_CUES = (
    "held in", "where the value", "reading guide", "field order", "schedule",
    "legend", "supersed", "latest", "current", "previous",
)

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
    """Records of an export: lines that carry values separated by pipes."""
    rows = []
    for line in _doc_text(doc_id).splitlines():
        if line.count("|") == 3 and line.split("|")[0].strip().startswith("VLV-"):
            rows.append([field.strip() for field in line.split("|")])
    return rows


def _turn(scenario: BenchmarkScenario, number: int) -> Any:
    return next(t for t in scenario.turns if t.turn_number == number)


class TestScenarioShape:
    def test_loads_with_the_expected_identity(self, scenario):
        assert scenario.id == SCENARIO_ID
        assert scenario.metadata["kind"] == "recall"
        assert scenario.metadata["quality_eligible"] is True
        assert scenario.metadata["amnesia_passes"] is False

    def test_two_sessions_and_sequential_turns(self, scenario):
        assert {t.session_id for t in scenario.turns} == {"s1", "s2"}
        assert [t.turn_number for t in scenario.turns] == [1, 2, 3, 4]

    def test_declares_one_guide_two_exports_and_filler(self, scenario):
        ids = [spec.doc_id for spec in scenario.documents]
        assert DOC_BAND[0] <= len(ids) <= DOC_BAND[1]
        assert ids.count(LEGEND) == 1
        assert {ROWS_1, ROWS_2} <= set(ids)
        assert scenario.metadata["chunk_size"] == CHUNK_SIZE
        assert scenario.metadata["default_top_k"] == 5

    def test_document_paths_resolve(self, scenario):
        for spec in scenario.documents:
            assert (DOCUMENTS / spec.path).is_file(), spec.path


class TestCorpusInvariants:
    def test_chunk_count_matches_the_declared_measurement(self, scenario):
        chunks = _chunks(scenario)
        low, high = CHUNK_BAND
        assert low <= len(chunks) <= high, len(chunks)
        assert len(chunks) == scenario.metadata["calibration"]["chunks_measured"]

    def test_every_document_is_a_plausible_length(self, scenario):
        for spec in scenario.documents:
            chars = len((DOCUMENTS / spec.path).read_text(encoding="utf-8"))
            assert 1200 <= chars <= 4000, f"{spec.doc_id}: {chars}"

    @pytest.mark.parametrize("token", [V220_VALUE, V247_VALUE, V331_VALUE])
    def test_each_asserted_value_lives_in_exactly_one_document(self, scenario, token):
        """One legitimate source; two chunks at most, because 200-char overlap
        duplicates a record that sits on a chunk boundary."""
        hits = [(doc, i) for i, (doc, text) in enumerate(_chunks(scenario)) if token in text]
        assert {doc for doc, _ in hits} == {ROWS_1 if token != V331_VALUE else ROWS_2}
        assert len(hits) <= 2, f"{token}: {hits}"

    def test_exports_carry_no_field_names(self, scenario):
        """The severance itself: a record never says which field is which."""
        for doc_id in (ROWS_1, ROWS_2):
            for index, (doc, text) in enumerate(_chunks(scenario)):
                if doc != doc_id:
                    continue
                leaked = [name for name in FIELD_NAMES if name in text.casefold()]
                assert not leaked, f"{doc_id} chunk {index}: {leaked}"

    def test_the_guide_carries_no_record(self):
        text = _doc_text(LEGEND)
        assert "VLV-" not in text, "the guide must not contain records"
        for token in (V220_VALUE, V247_VALUE, V331_VALUE):
            assert token not in text

    def test_the_guide_states_both_riser_orders(self):
        text = " ".join(_doc_text(LEGEND).casefold().split())
        assert "third-riser assembly" in text
        assert "north-riser assembly" in text
        assert "asset identifier, certified value, test pressure, verification date" in text
        assert "asset identifier, test pressure, certified value, verification date" in text


class TestOrderTrap:
    def test_the_same_value_sits_at_different_positions_per_family(self):
        third = next(r for r in _data_rows(ROWS_1) if r[0] == "VLV-220")
        north = next(r for r in _data_rows(ROWS_2) if r[0] == "VLV-331")
        assert third[1] == V220_VALUE and third[2] == V220_PRESSURE
        assert north[1] == V331_PRESSURE and north[2] == V331_VALUE, (
            "the two families must disagree on which position holds the certified value"
        )

    def test_every_record_carries_four_fields(self):
        for doc_id in (ROWS_1, ROWS_2):
            rows = _data_rows(doc_id)
            assert rows, doc_id
            assert all(len(row) == 4 for row in rows), doc_id

    def test_records_are_unique_per_export_and_disjoint_across_exports(self):
        first = {row[0] for row in _data_rows(ROWS_1)}
        second = {row[0] for row in _data_rows(ROWS_2)}
        assert len(first) == len(_data_rows(ROWS_1))
        assert not first & second, sorted(first & second)


class TestSeverance:
    """The record must be reachable while the schema is not.

    Mirror of the bridge cell's gate, with the construct changed: there the
    identifier was missing, here the field order is.
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
    def _best_rank(order: list[int], chunks, doc_id: str) -> int:
        positions = {index: rank + 1 for rank, index in enumerate(order)}
        ranks = [positions[i] for i, (doc, _) in enumerate(chunks) if doc == doc_id]
        return min(ranks) if ranks else -1

    def test_a_value_probe_reaches_the_record_but_not_the_guide(self, scenario):
        chunks = _chunks(scenario)
        order = self._bm25(chunks)(_turn(scenario, 2).content)
        assert self._best_rank(order, chunks, ROWS_1) == 1
        assert self._best_rank(order, chunks, LEGEND) > 5, (
            "the guide is inside the k=5 window, so the row can be decoded in one hop"
        )

    def test_the_format_probe_reaches_the_guide(self, scenario):
        chunks = _chunks(scenario)
        order = self._bm25(chunks)(_turn(scenario, 4).content)
        assert self._best_rank(order, chunks, LEGEND) <= 5, (
            "the guide is unreachable even when the probe names the format: "
            "the cell would be unsolvable"
        )

    def test_declared_bounds_hold(self, scenario):
        """Ranks are asserted as bounds, not exact values.

        Chunks that match none of the probe's terms all score zero, so their
        relative order is a tie broken by document order — declaring an exact rank
        for them would pin an artefact. What matters is asserted instead: the
        record is first, the guide is outside the window, and the guide is
        reachable when the probe names the format.
        """
        chunks = _chunks(scenario)
        declared = scenario.metadata["corpus_difficulty"]["measured_ranks_bm25"]

        t2 = self._bm25(chunks)(_turn(scenario, 2).content)
        assert self._best_rank(t2, chunks, ROWS_1) == declared["t2_probe"]["rows_01_best"]
        assert self._best_rank(t2, chunks, LEGEND) > declared["t2_probe"]["legend_01_at_least"]

        t4 = self._bm25(chunks)(_turn(scenario, 4).content)
        assert self._best_rank(t4, chunks, LEGEND) <= declared["t4_probe"]["legend_01_at_most"]


class TestCrossCellIsolation:
    def test_other_cells_namespaces_are_absent_here(self, scenario):
        blob = " ".join(text for _, text in _chunks(scenario))
        for token in (
            "PMP-310", "NBL-420", "HLX-2C7V-9043", "SWG-1F9K-5520", "QMV-T3R8-4402",
        ):
            assert token not in blob, token

    def test_this_namespace_is_absent_from_the_other_cells(self):
        others = [
            REPO_ROOT
            / "datasets/conversations/multi_thread/compositional_reasoning"
            / "document_bridge/test_document_multihop_bridge_01.yaml",
            REPO_ROOT
            / "datasets/conversations/multi_thread/logical_consistency"
            / "document_conflict/test_document_conflict_dilution_01.yaml",
        ]
        for path in others:
            other = BenchmarkScenario.model_validate(
                yaml.safe_load(path.read_text(encoding="utf-8"))
            )
            blob = " ".join(
                (DOCUMENTS / spec.path).read_text(encoding="utf-8")
                for spec in other.documents
            )
            for token in ("VLV-220", V220_VALUE, LEGEND):
                assert token not in blob, (path.name, token)


class TestAssertions:
    def test_t2_is_a_strict_single_token_probe(self, scenario):
        """T2 must not be passable by listing both codes.

        Run 20261111_155622 passed it with the guide absent, because the answer
        enumerated both fields with the right one first; 20261111_154640 failed
        the same probe by committing to the wrong field. The format constraint
        plus the guard turn that difference into a measurement.
        """
        turn = _turn(scenario, 2)
        assert [r.value for r in turn.assertions if r.type == "must_contain"] == [
            V220_VALUE
        ]
        assert [r.value for r in turn.assertions if r.type == "must_not_contain"] == [
            V220_PRESSURE
        ], "the other field of the same row must be guarded"
        assert ROWS_1 in [
            r.value for r in turn.assertions if r.type == "must_retrieve_chunk"
        ]
        assert [r.value for r in turn.assertions if r.type == "must_not_retrieve_chunk"]
        assert "code only" in turn.content.casefold(), (
            "the single-token constraint is what makes the guard fair"
        )

    def test_the_decisive_probe_lives_in_its_own_scenario(self, scenario):
        """This cell must hold exactly ONE value probe, and it must come first.

        Measured (archive/2026-11-11_document_split_t2_leak_experiment): a value
        probe's answer contains the decode key, and a shared session puts the
        prior answer in the prompt verbatim, so a second value probe in the same
        thread reads the position out of its own conversation. The binding probe
        therefore lives in `document_split_binding_01`; adding one back here
        would silently re-open that hole.
        """
        value_probes = [
            t.turn_number
            for t in scenario.turns
            if any(
                r.type == "must_not_contain" and r.value.startswith(("TST-", "KVN-"))
                for r in t.assertions
            )
        ]
        assert value_probes == [2], value_probes

    def test_t2_noun_is_never_bound_to_an_asset_in_the_corpus(self, scenario):
        """The confound is real: "limit" exists here, but never for a valve.

        bg_03 ("hard limit") and bg_08 ("to limit who may call") use the word in
        unrelated senses. Pinned so that an edit which starts binding "limit" to
        an asset — and would silently turn T2 into a decidable probe — fails here.
        """
        blob = " ".join(text for _, text in _chunks(scenario))
        bound = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+", blob)
            if re.search(r"\blimit", sentence, re.IGNORECASE)
            and re.search(r"VLV|valve|riser", sentence, re.IGNORECASE)
        ]
        assert not bound, bound

    def test_the_diagnostic_probe_guards_the_other_code(self, scenario):
        """A two-code answer must fail, or a guess can pass as knowledge."""
        guard = [
            r.value
            for r in _turn(scenario, 2).assertions
            if r.type == "must_not_contain"
        ]
        assert V220_PRESSURE in guard

    def test_t4_asserts_the_format_and_pins_the_guide(self, scenario):
        turn = _turn(scenario, 4)
        assert [r.value for r in turn.assertions if r.type == "must_contain"] == [
            "certified value"
        ]
        assert LEGEND in [
            r.value for r in turn.assertions if r.type == "must_retrieve_chunk"
        ]

    def test_t3_binds_the_family_and_keeps_the_value_guard(self, scenario):
        """T3 must be able to *fail* when the family is wrong.

        As a negative-only guard it could not: run 20261111_153740 answered "No"
        to "is VLV-331 on the north riser?" — wrong, and passing. Note what the
        probe actually measures now: the family label lives only in the export's
        first chunk, so this is a third instance of severed evidence, not a test
        about field positions (run 20261111_154640 abstained because the title
        chunk never arrived, which is the intended failure class).
        """
        turn = _turn(scenario, 3)
        assert [r.value for r in turn.assertions if r.type == "must_contain"] == [
            "north-riser"
        ]
        assert [r.value for r in turn.assertions if r.type == "must_not_contain"] == [
            "MVB-2K9P-6634"
        ]
        assert "north-riser" not in turn.content.casefold(), (
            "the probe must not name the family it asks for"
        )

    def test_the_family_label_is_severed_from_its_rows(self):
        """The premise of T5: the label sits in the first chunk, the rows in later ones."""
        splitter = _splitter()
        for doc_id, label in ((ROWS_1, "third-riser"), (ROWS_2, "north-riser")):
            parts = splitter.split_text(_doc_text(doc_id))
            carrying = [i for i, part in enumerate(parts) if label in part.casefold()]
            with_rows = [i for i, part in enumerate(parts) if "VLV-" in part]
            assert carrying == [0], (doc_id, carrying)
            assert 0 not in with_rows, (doc_id, with_rows)

    def test_the_guide_probe_runs_after_the_value_probes(self, scenario):
        """Otherwise the field order sits in the thread when the rows are read."""
        guide_turn = next(
            t.turn_number
            for t in scenario.turns
            for r in t.assertions
            if r.type == "must_retrieve_chunk" and r.value == LEGEND
        )
        value_turns = [
            t.turn_number
            for t in scenario.turns
            if any(
                r.type == "must_retrieve_chunk" and r.value == ROWS_1
                for r in t.assertions
            )
        ]
        assert guide_turn > max(value_turns)

    def test_no_user_turn_does_the_work_for_the_model(self, scenario):
        offenders = [
            (turn.turn_number, turn.content, cue)
            for turn in scenario.turns
            if turn.role == "user"
            for cue in PRECEDENCE_CUES + COMPOSITION_CUES
            if cue in turn.content.casefold()
        ]
        assert not offenders, offenders

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
