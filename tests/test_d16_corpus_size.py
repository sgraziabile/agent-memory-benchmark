"""D16 — corpus-size curve: generator needles, scenario builder, L-R report.

All tests are offline. They pin the two contracts D16 depends on:

1. **Non-inferability.** Every needle answer is an opaque token that occurs
   exactly once in the corpus and never in its own probe, so a correct
   response requires the gold chunk (stress-plan rule R10).
2. **Backwards compatibility.** Multi-needle mode must not perturb document
   bodies: the no-needle corpus for seed 7 must stay byte-identical to the
   output produced before ``--needles`` existed.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from core.schemas import BenchmarkScenario
from scripts import build_d16_scenarios, generate_corpus, lr_report

#: sha256 of the concatenated bodies for ``--docs 8 --chars-per-doc 6000
#: --seed 7`` produced by the generator BEFORE multi-needle support was added.
NO_NEEDLE_8X6000_SEED7_SHA256 = (
    "36f0e4de5717706dde28dd2d6971773362bb09190892abb0c440583c2f859cf6"
)


def _read_corpus(directory: Path) -> str:
    """Concatenate every document body in stable name order (no separator)."""
    return "".join(
        path.read_text(encoding="utf-8")
        for path in sorted(directory.glob("*.md"))
    )


# ── Generator: opaque needles ────────────────────────────────────────────────


class TestGeneratorNeedles:
    def test_manifest_has_one_opaque_needle_per_document(self, tmp_path):
        out = tmp_path / "corpus"
        assert (
            generate_corpus.main(
                [
                    "--docs", "8",
                    "--chars-per-doc", "6000",
                    "--seed", "7",
                    "--needles", "8",
                    "--out", str(out),
                ]
            )
            == 0
        )

        manifest = json.loads((out / "corpus_manifest.json").read_text("utf-8"))
        needles = manifest["needles"]
        assert len(needles) == 8
        assert manifest["needle"] is None  # legacy single-needle key unused

        doc_ids = [n["doc_id"] for n in needles]
        uniq = {n["answer"] for n in needles}
        entities = {n["entity"] for n in needles}
        assert doc_ids == sorted(doc_ids)
        assert len(set(doc_ids)) == 8
        assert len(uniq) == 8
        assert len(entities) == 8

        corpus = _read_corpus(out)
        for needle in needles:
            assert generate_corpus.ANSWER_RE.match(needle["answer"])
            # Exactly one occurrence → the answer cannot be reconstructed from
            # another document, and the gold chunk is unavoidable.
            assert corpus.count(needle["answer"]) == 1
            assert corpus.count(needle["entity"]) == 1
            # The probe names the entity but never the answer.
            assert needle["answer"] not in needle["probe"]
            assert needle["entity"] in needle["probe"]
            assert needle["sentence"] in corpus

    def test_no_needle_corpus_is_byte_identical_to_legacy(self, tmp_path):
        out = tmp_path / "corpus"
        assert (
            generate_corpus.main(
                [
                    "--docs", "8",
                    "--chars-per-doc", "6000",
                    "--seed", "7",
                    "--needle", "",  # disable the legacy default needle
                    "--out", str(out),
                ]
            )
            == 0
        )
        digest = hashlib.sha256(_read_corpus(out).encode("utf-8")).hexdigest()
        assert digest == NO_NEEDLE_8X6000_SEED7_SHA256

        manifest = json.loads((out / "corpus_manifest.json").read_text("utf-8"))
        assert manifest["needle"] is None
        assert manifest["needles"] == []
        assert not any(entry["has_needle"] for entry in manifest["files"])

    def test_multi_needle_documents_are_deterministic(self, tmp_path):
        """Same seed + same flags ⇒ byte-identical corpora and manifests."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        args = ["--docs", "8", "--chars-per-doc", "6000", "--seed", "7", "--needles", "8"]
        assert generate_corpus.main(args + ["--out", str(first)]) == 0
        assert generate_corpus.main(args + ["--out", str(second)]) == 0

        for name in sorted(p.name for p in first.glob("*.md")):
            assert (first / name).read_bytes() == (second / name).read_bytes()
        assert (first / "corpus_manifest.json").read_text(
            encoding="utf-8"
        ) == (second / "corpus_manifest.json").read_text(encoding="utf-8")

    def test_needle_placement_is_evenly_spread(self, tmp_path):
        """Needles must not cluster at the start of a large corpus."""
        out = tmp_path / "corpus"
        assert (
            generate_corpus.main(
                [
                    "--docs", "60",
                    "--chars-per-doc", "2000",
                    "--seed", "7",
                    "--needles", "8",
                    "--out", str(out),
                ]
            )
            == 0
        )
        manifest = json.loads((out / "corpus_manifest.json").read_text("utf-8"))
        indices = [int(n["doc_index"]) for n in manifest["needles"]]
        assert indices == [1 + (k * 60) // 8 for k in range(8)]
        assert indices[0] < indices[-1]
        assert indices[-1] > 40  # spread reaches the last third of the corpus

    def test_needle_answers_differ_between_corpus_sizes(self, tmp_path):
        """Each corpus owns its answer tokens (no cross-size reuse)."""
        manifests = []
        for docs in (8, 60):
            out = tmp_path / f"c{docs}"
            assert (
                generate_corpus.main(
                    [
                        "--docs", str(docs),
                        "--chars-per-doc", "2000",
                        "--seed", "7",
                        "--needles", "8",
                        "--out", str(out),
                    ]
                )
                == 0
            )
            manifests.append(json.loads((out / "corpus_manifest.json").read_text("utf-8")))
        small = {n["answer"] for n in manifests[0]["needles"]}
        large = {n["answer"] for n in manifests[1]["needles"]}
        assert small.isdisjoint(large)

    def test_needles_cannot_be_combined_with_single_needle(self, tmp_path):
        args = [
            "--docs", "8",
            "--chars-per-doc", "2000",
            "--needles", "2",
            "--needle", "custom",
            "--out", str(tmp_path),
        ]
        assert generate_corpus.main(args) == 2

    def test_needle_count_cannot_exceed_documents(self, tmp_path):
        args = [
            "--docs", "2",
            "--chars-per-doc", "2000",
            "--needles", "3",
            "--out", str(tmp_path),
        ]
        assert generate_corpus.main(args) == 2


# ── Scenario builder ────────────────────────────────────────────────────────


class TestScenarioBuilder:
    def _generate(self, root: Path, docs: int) -> None:
        assert (
            generate_corpus.main(
                [
                    "--docs", str(docs),
                    "--chars-per-doc", "6000",
                    "--seed", "7",
                    "--needles", "8",
                    "--out", str(root / f"d16_corpus_{docs:03d}"),
                ]
            )
            == 0
        )

    def test_builds_loadable_scenarios_with_unique_probes(self, tmp_path):
        corpus_root = tmp_path / "documents"
        out_dir = tmp_path / "scenarios"
        self._generate(corpus_root, 8)

        assert (
            build_d16_scenarios.main(
                [
                    "--corpus-root", str(corpus_root),
                    "--sizes", "8",
                    "--out-dir", str(out_dir),
                ]
            )
            == 0
        )
        yaml_path = out_dir / "test_document_corpus_size_008_01.yaml"
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        scenario = BenchmarkScenario(**raw)

        assert scenario.id == "document_corpus_size_008_01"
        assert scenario.category == "document_corpus_size"
        assert len(scenario.documents) == 8
        assert len(scenario.turns) == 8
        assert {t.session_id for t in scenario.turns} == {"s1"}

        doc_ids = {d.doc_id for d in scenario.documents}
        answers = []
        for turn in scenario.turns:
            must = [a for a in turn.assertions if a.type == "must_contain"]
            chunk = [a for a in turn.assertions if a.type == "must_retrieve_chunk"]
            assert len(must) == 1 and len(chunk) == 1
            # Non-inferability: the answer is never in its own probe.
            assert must[0].value.casefold() not in turn.content.casefold()
            # Gold is a real document of this scenario.
            assert chunk[0].value in doc_ids
            answers.append(must[0].value)
        assert len(set(answers)) == 8

    def test_builder_is_deterministic(self, tmp_path):
        corpus_root = tmp_path / "documents"
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        self._generate(corpus_root, 8)
        for out in (out_a, out_b):
            assert (
                build_d16_scenarios.main(
                    [
                        "--corpus-root", str(corpus_root),
                        "--sizes", "8",
                        "--out-dir", str(out),
                    ]
                )
                == 0
            )
        name = "test_document_corpus_size_008_01.yaml"
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()

    def test_missing_manifest_is_a_clear_error(self, tmp_path):
        assert (
            build_d16_scenarios.main(
                [
                    "--corpus-root", str(tmp_path),
                    "--sizes", "8",
                    "--out-dir", str(tmp_path / "out"),
                ]
            )
            == 2
        )


# ── L-R report ──────────────────────────────────────────────────────────────


class TestLrMetrics:
    def test_wilson_interval_bounds(self):
        low, high = lr_report.wilson_ci(8, 8)
        assert 0.6 < low < 1.0
        assert high == 1.0
        low0, high0 = lr_report.wilson_ci(0, 8)
        assert low0 == 0.0
        assert 0.0 < high0 < 0.4
        low4, high4 = lr_report.wilson_ci(4, 8)
        assert low4 < 0.5 < high4
        assert lr_report.wilson_ci(0, 0) == (0.0, 0.0)

    def test_chunk_and_doc_rank(self):
        episodes = [
            {"rank": 1, "chunk_id": "doc_corpus_001_c0", "score": 0.71, "content": "filler"},
            {"rank": 2, "chunk_id": "doc_corpus_000_c3", "score": 0.69, "content": "answer ARC-1234-5678"},
            {"rank": 3, "chunk_id": "doc_corpus_000_c4", "score": 0.68, "content": "other"},
        ]
        assert lr_report.gold_chunk_rank(episodes, "ARC-1234-5678") == 2
        assert lr_report.gold_doc_rank(episodes, "corpus_000") == 2
        assert lr_report.gold_chunk_rank(episodes, "NOPE") is None
        # Δcos: gold 0.69 vs best non-gold 0.71 → negative margin.
        assert lr_report.delta_cos(episodes, "ARC-1234-5678") == pytest.approx(-0.02)


class TestLrReportEndToEnd:
    SCENARIO = "document_corpus_size_008_01"

    def _write_results(self, root: Path, hits: int) -> None:
        gold = lr_report.load_gold_map(self.SCENARIO)[self.SCENARIO]
        turns = []
        for turn_number in sorted(gold):
            answer = gold[turn_number]["answer"]
            gold_doc = gold[turn_number]["doc"]
            episodes = [
                {"rank": 1, "chunk_id": "doc_corpus_999_c0", "score": 0.9, "content": "filler"}
            ]
            if turn_number <= hits:
                episodes.append(
                    {
                        "rank": 2,
                        "chunk_id": f"doc_{gold_doc}_c1",
                        "score": 0.8,
                        "content": f"the code is {answer}.",
                    }
                )
            turns.append(
                {
                    "turn_number": turn_number,
                    "role": "user",
                    "status": "ok",
                    "retrieved_memories": episodes,
                    "assertion_details": [
                        {
                            "rule_type": "must_contain",
                            "value": answer,
                            "passed": turn_number <= hits,
                        }
                    ],
                }
            )
        run = {
            "scenario_id": self.SCENARIO,
            "model_id": "unit-test-model",
            "chunk_size": 600,
            "top_k": 5,
            "corpus_chunks": 100,
            "documents_ingested": 8,
            "ingestion_latency_ms": 61_000.0,
            "turn_results": turns,
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / "results.json").write_text(json.dumps([run]), encoding="utf-8")

    def test_recall_mrr_and_coverage(self, tmp_path):
        self._write_results(tmp_path, hits=6)
        gold = lr_report.load_gold_map(self.SCENARIO)
        assert self.SCENARIO in gold

        buckets = lr_report.collect(gold, tmp_path)
        ((key, bucket),) = buckets.items()
        assert key[1] == 600 and key[2] == 5

        summary = lr_report.summarize(bucket)
        assert summary["probes"] == 8
        assert summary["recall"] == pytest.approx(6 / 8)
        assert summary["ci_low"] < summary["recall"] < summary["ci_high"]
        assert summary["mrr"] == pytest.approx((6 * 0.5) / 8)  # gold at rank 2
        assert summary["coverage"] == pytest.approx(5 / 100)
        assert summary["p_correct_given_hit"] == pytest.approx(1.0)
        assert summary["ingest_s"] == pytest.approx(61.0)
        assert bucket["documents_ingested"] == 8


