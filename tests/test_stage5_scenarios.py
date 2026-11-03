"""Offline invariants for the Stage 5 belief-revision scenarios (5.1 and 5.4).

Spec: ``TODOs/test-expansion-plan.md`` §Stage 5 — 5.1 *Derived-Belief
Invalidation* (``derived_belief_revision_01``) and 5.4 *Backward Contraction
via Modus Tollens* (``backward_modus_tollens_01``).

These tests pin the scenario *design* (sessions, premise placement, probe
lexicon, assertion shape, ground truth) so that later edits to the YAMLs
cannot silently break what the scenarios measure. Everything here is fully
offline: no API keys, no model calls. The real assertion evaluator
(``BenchmarkRunner.evaluate_assertions``) is exercised against canned
responses to document which answer shapes pass and fail.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pytest

from core.runner import BenchmarkRunner

try:
    from scripts.organize_datasets import scenario_id_from_filename
except ImportError:  # scripts/ not importable — apply the same 2-line rule inline
    def scenario_id_from_filename(filename: str) -> str | None:
        if not (filename.startswith("test_") and filename.endswith(".yaml")):
            return None
        return filename[len("test_") : -len(".yaml")]


REPO_ROOT = Path(__file__).resolve().parent.parent

STAGE5_IDS = ["derived_belief_revision_01", "backward_modus_tollens_01"]
STAGE5_CONTROL_IDS = [
    "in_context_derived_revision_01",
    "in_context_modus_tollens_01",
]
PAIRED_CONTROLS = {
    "derived_belief_revision_01": "in_context_derived_revision_01",
    "backward_modus_tollens_01": "in_context_modus_tollens_01",
}
_CONTROL_PROBE_MARKER = "Based strictly on this log:"

STAGE5_PATHS = {
    "derived_belief_revision_01": (
        "datasets/conversations/logical_consistency/belief_revision/"
        "test_derived_belief_revision_01.yaml"
    ),
    "backward_modus_tollens_01": (
        "datasets/conversations/logical_consistency/belief_revision/"
        "test_backward_modus_tollens_01.yaml"
    ),
    "in_context_derived_revision_01": (
        "datasets/conversations/logical_consistency/in_context_control/"
        "test_in_context_derived_revision_01.yaml"
    ),
    "in_context_modus_tollens_01": (
        "datasets/conversations/logical_consistency/in_context_control/"
        "test_in_context_modus_tollens_01.yaml"
    ),
}

# Shared session layout: s1 = T1-T7, s2 = T8-T10, s3 = T11-T13.
SESSION_LAYOUT = {
    "s1": list(range(1, 8)),
    "s2": list(range(8, 11)),
    "s3": list(range(11, 14)),
}
OPEN_RECALL_TURN = 11
COMMITMENT_TURN = 12
CONTROL_TURN = 13

# Expected assertions on commitment turn (T12) and control turn (T13).
EXPECTED_ASSERTIONS = {
    "derived_belief_revision_01": {
        "commit": [("must_contain", "Level-1"), ("must_not_contain", "Level-3")],
        "control": "Level-3",
    },
    "backward_modus_tollens_01": {
        "commit": [("must_contain", "NO_EU_CENTRAL")],
        "control": "YES_BASTION",
    },
}

# Premise turns per scenario — none may be the last turn of its session.
PREMISE_TURNS = {
    "derived_belief_revision_01": {"R": 3, "F1": 4, "D1_agent": 5, "D1_user": 6, "F2": 9},
    "backward_modus_tollens_01": {"R": 3, "F1": 4, "D1_agent": 5, "D1_user": 6, "notD1": 9},
}

MT_T12_LABELS = ["YES_EU_CENTRAL", "NO_EU_CENTRAL", "CANNOT_DETERMINE"]
MT_T13_LABELS = ["YES_BASTION", "NO_BASTION", "CANNOT_DETERMINE"]


@pytest.fixture(scope="module")
def stage5():
    """All Stage 5 scenarios and paired controls, keyed by id."""
    runner = BenchmarkRunner()
    scenarios = runner.load_scenarios(
        filter_ids=STAGE5_IDS + STAGE5_CONTROL_IDS
    )
    return {s.id: s for s in scenarios}


def _turn(scenario, turn_number):
    return next(t for t in scenario.turns if t.turn_number == turn_number)


def _turn_passes(response: str, rules) -> bool:
    """Turn-level pass = every assertion on the turn passes (headline rule)."""
    results = BenchmarkRunner.evaluate_assertions(response, rules)
    return all(r["passed"] for r in results)


# ── Loading & structure (both scenarios) ─────────────────────────────────────


class TestStage5Structure:
    """Shared structure for 5.1 and 5.4."""

    def test_both_ids_load(self, stage5):
        for sid in STAGE5_IDS:
            assert sid in stage5, (
                f"scenario {sid!r} did not load — check for a silent "
                f"'Skipping invalid scenario' warning"
            )

    def test_id_matches_filename(self, stage5):
        for sid, rel_path in STAGE5_PATHS.items():
            path = REPO_ROOT / rel_path
            assert path.is_file(), f"{rel_path} is missing"
            assert scenario_id_from_filename(path.name) == sid
            assert stage5[sid].id == sid

    def test_category_is_belief_revision(self, stage5):
        for sid in STAGE5_IDS:
            assert stage5[sid].category == "belief_revision"

    def test_turn_numbers_global_and_sequential(self, stage5):
        for sid in STAGE5_IDS:
            numbers = [t.turn_number for t in stage5[sid].turns]
            assert numbers == list(range(1, 14)), (
                f"{sid}: turn_number must be global 1..13, never reset per session"
            )

    def test_session_layout(self, stage5):
        for sid in STAGE5_IDS:
            layout: dict[str, list[int]] = {}
            for t in stage5[sid].turns:
                layout.setdefault(t.session_id, []).append(t.turn_number)
            assert layout == SESSION_LAYOUT, sid
            assert stage5[sid].metadata["sessions"] == ["s1", "s2", "s3"]

    def test_all_turns_are_user_turns(self, stage5):
        for sid in STAGE5_IDS:
            assert {t.role for t in stage5[sid].turns} == {"user"}, sid

    def test_premises_never_last_turn_of_session(self, stage5):
        last_turns = {turns[-1] for turns in SESSION_LAYOUT.values()}
        for sid, premises in PREMISE_TURNS.items():
            for name, n in premises.items():
                assert n not in last_turns, (
                    f"{sid}: premise {name} (T{n}) must sit mid-thread, "
                    f"not on a session's final turn (recency)"
                )

    def test_metadata_documents_scoring(self, stage5):
        required = {
            "headline_metric",
            "outcome_matrix",
            "expected_level_0_behavior",
            "expected_level_1_behavior",
            "expected_level_2_behavior",
            "target_research_question",
        }
        for sid in STAGE5_IDS:
            missing = required - set(stage5[sid].metadata)
            assert not missing, f"{sid}: metadata missing {sorted(missing)}"

    def test_registered_in_scenario_maps(self):
        # Text check so the test does not depend on scripts/ being importable.
        for script in ("scripts/organize_datasets.py", "scripts/reorganize_outputs.py"):
            text = (REPO_ROOT / script).read_text(encoding="utf-8")
            for sid in STAGE5_IDS:
                entry = f'"{sid}": ("logical_consistency", "belief_revision")'
                assert entry in text, f"{script}: SCENARIO_MAP lacks {sid}"


# ── Assertion policy (two-step probe) ────────────────────────────────────────


class TestStage5AssertionPolicy:
    """Assertions only on s3: T11 observational, T12 commitment, T13 control."""

    def test_no_assertions_outside_commitment_and_control(self, stage5):
        for sid in STAGE5_IDS:
            for t in stage5[sid].turns:
                if t.turn_number not in (COMMITMENT_TURN, CONTROL_TURN):
                    assert t.assertions == [], (
                        f"{sid} T{t.turn_number} must have no assertions"
                    )

    def test_open_recall_turn_is_observational(self, stage5):
        for sid in STAGE5_IDS:
            assert _turn(stage5[sid], OPEN_RECALL_TURN).assertions == []

    def test_commitment_turn_shape(self, stage5):
        for sid in STAGE5_IDS:
            rules = _turn(stage5[sid], COMMITMENT_TURN).assertions
            shape = [(r.type, r.value) for r in rules]
            expected = EXPECTED_ASSERTIONS[sid]["commit"]
            assert shape == expected, sid

    def test_control_turn_shape(self, stage5):
        for sid in STAGE5_IDS:
            rules = _turn(stage5[sid], CONTROL_TURN).assertions
            shape = [(r.type, r.value) for r in rules]
            assert shape == [("must_contain", EXPECTED_ASSERTIONS[sid]["control"])], sid

    def test_must_not_contain_only_on_5_1_commitment_turn(self, stage5):
        hits = {
            (sid, t.turn_number)
            for sid in STAGE5_IDS
            for t in stage5[sid].turns
            for r in t.assertions
            if r.type == "must_not_contain"
        }
        assert hits == {("derived_belief_revision_01", COMMITMENT_TURN)}

    def test_all_assertions_case_insensitive(self, stage5):
        for sid in STAGE5_IDS:
            for t in stage5[sid].turns:
                for r in t.assertions:
                    assert r.case_sensitive is False, f"{sid} T{t.turn_number}"

    def test_commitment_values_do_not_overlap(self, stage5):
        # A correct answer must never trip the must_not_contain by substring.
        want = "level-1"
        avoid = "level-3"
        assert want not in avoid and avoid not in want

    def test_assertion_values_absent_from_s2(self, stage5):
        # s2 is the revision session: it must not hand over the probe answer.
        for sid in STAGE5_IDS:
            want = EXPECTED_ASSERTIONS[sid]["commit"][0][1]
            for n in SESSION_LAYOUT["s2"]:
                content = _turn(stage5[sid], n).content.casefold()
                assert want.casefold() not in content, f"{sid} T{n} leaks {want!r}"


# ── 5.1 Derived-Belief Invalidation ──────────────────────────────────────────


class TestDerivedBeliefRevision01:
    """5.1: F1 -> F2 must retract the cached derived belief D1."""

    SID = "derived_belief_revision_01"

    @pytest.fixture
    def sc(self, stage5):
        return stage5[self.SID]

    def test_rule_turn_states_both_tiers(self, sc):
        rule = _turn(sc, PREMISE_TURNS[self.SID]["R"]).content
        for needle in ("prod-tier", "Level-3", "staging-tier", "Level-1"):
            assert needle in rule, f"rule R lacks {needle!r}"

    def test_base_fact_f1(self, sc):
        f1 = _turn(sc, PREMISE_TURNS[self.SID]["F1"]).content
        assert "srv-alpha is a prod-tier" in f1
        assert "srv-beta" in f1, "control entity must be seeded with F1"

    def test_derived_belief_cached_by_user(self, sc):
        # D1 must reach the store independently of the model's T5 answer.
        d1 = _turn(sc, PREMISE_TURNS[self.SID]["D1_user"]).content
        assert "srv-alpha" in d1 and "Level-3" in d1

    def test_level_1_only_inside_rule(self, sc):
        # The correct answer exists only as a rule clause: it must be derived.
        hits = [t.turn_number for t in sc.turns if "level-1" in t.content.casefold()]
        assert hits == [PREMISE_TURNS[self.SID]["R"]]

    def test_level_3_only_in_session_a(self, sc):
        hits = [t.turn_number for t in sc.turns if "level-3" in t.content.casefold()]
        assert hits == [3, 6], "Level-3 belongs to R (T3) and the cached D1 (T6) only"

    def test_revision_turn_avoids_authorization_lexicon(self, sc):
        f2 = _turn(sc, PREMISE_TURNS[self.SID]["F2"]).content.casefold()
        assert "srv-alpha" in f2 and "staging" in f2
        for needle in ("authoriz", "level", "reboot"):
            assert needle not in f2, f"F2 must not mention {needle!r}"

    def test_s2_never_mentions_authorization(self, sc):
        for n in SESSION_LAYOUT["s2"]:
            low = _turn(sc, n).content.casefold()
            assert "authoriz" not in low, f"T{n} mentions authorization"

    def test_probes_use_derived_lexicon_not_revised_fact(self, sc):
        for n in (OPEN_RECALL_TURN, COMMITMENT_TURN):
            low = _turn(sc, n).content.casefold()
            assert "srv-alpha" in low and "authorization level" in low, f"T{n}"
            for needle in ("staging", "prod", "tier", "migrat", "dev"):
                assert needle not in low, f"probe T{n} echoes F2/F1 word {needle!r}"

    def test_control_entity_untouched(self, sc):
        hits = [t.turn_number for t in sc.turns if "srv-beta" in t.content]
        assert hits == [4, CONTROL_TURN]

    def test_ground_truth(self, sc):
        rule = {"prod": "Level-3", "staging": "Level-1", "dev": None}
        assert rule["staging"] == EXPECTED_ASSERTIONS[self.SID]["commit"][0][1]  # R + F2
        assert rule["prod"] == EXPECTED_ASSERTIONS[self.SID]["commit"][1][1]  # stale D1
        assert rule["prod"] == EXPECTED_ASSERTIONS[self.SID]["control"]  # srv-beta

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("Level-1", True),
            ("level-1", True),
            ("Level\u20111", True),  # non-breaking hyphen is normalized
            ("Level-3", False),  # stale derived belief
            ("Level-1 (it was Level-3 before the migration)", False),  # hedge
            ("I don't have any information about srv-alpha.", False),  # decline
            ("Level 1", False),  # format miss — see metadata.format_audit
        ],
    )
    def test_commitment_answer_shapes(self, sc, response, expected):
        rules = _turn(sc, COMMITMENT_TURN).assertions
        assert _turn_passes(response, rules) is expected

    def test_decline_gets_partial_credit_only(self, sc):
        # Documents why scenario pass_rate is not comparable across levels.
        rules = _turn(sc, COMMITMENT_TURN).assertions
        results = BenchmarkRunner.evaluate_assertions("I don't know.", rules)
        assert [r["passed"] for r in results] == [False, True]


# ── 5.4 Backward Contraction via Modus Tollens ───────────────────────────────


class TestBackwardModusTollens01:
    """5.4: not-D1 with entrenched R must retract the base fact F1."""

    SID = "backward_modus_tollens_01"

    @pytest.fixture
    def sc(self, stage5):
        return stage5[self.SID]

    def test_rule_is_entrenched_and_exceptionless(self, sc):
        rule = _turn(sc, PREMISE_TURNS[self.SID]["R"]).content.casefold()
        for needle in ("eu-central", "bastion audit", "no exceptions", "pipeline"):
            assert needle in rule, f"rule R lacks {needle!r}"

    def test_base_fact_f1(self, sc):
        f1 = _turn(sc, PREMISE_TURNS[self.SID]["F1"]).content
        assert "billing-api is hosted in eu-central" in f1

    def test_derived_belief_cached_by_user(self, sc):
        d1 = _turn(sc, PREMISE_TURNS[self.SID]["D1_user"]).content
        assert "billing-api" in d1 and "Bastion audit" in d1
        assert "NOT" not in d1

    def test_contradiction_targets_d1_only(self, sc):
        not_d1 = _turn(sc, PREMISE_TURNS[self.SID]["notD1"]).content
        assert "billing-api" in not_d1 and "NOT enrolled" in not_d1
        low = not_d1.casefold()
        for needle in ("eu-central", "us-east", "region", "hosted"):
            assert needle not in low, f"not-D1 must not mention {needle!r}"

    def test_s2_never_mentions_regions(self, sc):
        for n in SESSION_LAYOUT["s2"]:
            low = _turn(sc, n).content.casefold()
            for needle in ("eu-central", "us-east", "region"):
                assert needle not in low, f"T{n} mentions {needle!r}"

    def test_probes_use_base_fact_lexicon_not_contradiction(self, sc):
        for n in (OPEN_RECALL_TURN, COMMITMENT_TURN):
            low = _turn(sc, n).content.casefold()
            assert "billing-api" in low and "eu-central" in low, f"T{n}"
            for needle in ("bastion", "audit", "compliance", "enrol"):
                assert needle not in low, f"probe T{n} echoes not-D1 word {needle!r}"

    def test_no_real_world_regulation(self, sc):
        # Plan deviation: GDPR replaced by a stipulated audit (see metadata).
        for t in sc.turns:
            assert "gdpr" not in t.content.casefold(), f"T{t.turn_number}"

    def test_labels_listed_in_probes_and_metadata(self, sc):
        t12 = _turn(sc, COMMITMENT_TURN).content
        t13 = _turn(sc, CONTROL_TURN).content
        for label in MT_T12_LABELS:
            assert label in t12
        for label in MT_T13_LABELS:
            assert label in t13
        assert sc.metadata["answer_labels"] == {"T12": MT_T12_LABELS, "T13": MT_T13_LABELS}

    @pytest.mark.parametrize("labels", [MT_T12_LABELS, MT_T13_LABELS])
    def test_labels_are_substring_free(self, labels):
        folded = [label.casefold() for label in labels]
        for a, b in product(folded, repeat=2):
            if a != b:
                assert a not in b, f"label {a!r} is a substring of {b!r}"

    def test_labels_absent_before_probe_session(self, sc):
        for t in sc.turns:
            if t.session_id != "s3":
                for label in MT_T12_LABELS + MT_T13_LABELS:
                    assert label not in t.content, f"T{t.turn_number} leaks {label!r}"

    def test_modus_tollens_ground_truth(self):
        # Every model of {R: F1 -> D1, not-D1} has F1 false.
        models = [
            (f1, d1)
            for f1, d1 in product([True, False], repeat=2)
            if ((not f1) or d1) and not d1
        ]
        assert models == [(False, False)]
        # Deleting D1 alone leaves {F1, R, not-D1} with no model at all.
        assert not [
            (f1, d1)
            for f1, d1 in product([True, False], repeat=2)
            if f1 and ((not f1) or d1) and not d1
        ]

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("NO_EU_CENTRAL", True),
            ("no_eu_central", True),
            ("YES_EU_CENTRAL", False),  # F1 survived (globally inconsistent)
            ("CANNOT_DETERMINE", False),  # conflict unresolved
            ("I don't know.", False),  # decline
        ],
    )
    def test_commitment_answer_shapes(self, sc, response, expected):
        rules = _turn(sc, COMMITMENT_TURN).assertions
        assert _turn_passes(response, rules) is expected

    @pytest.mark.parametrize(
        ("response", "expected"),
        [("YES_BASTION", True), ("NO_BASTION", False), ("CANNOT_DETERMINE", False)],
    )
    def test_control_answer_shapes(self, sc, response, expected):
        rules = _turn(sc, CONTROL_TURN).assertions
        assert _turn_passes(response, rules) is expected

    def test_plan_literal_no_would_reward_decline(self):
        # Pins the deviation rationale: the plan's must_contain "no" passes on
        # a safe decline, because "no" is a substring of "know".
        from core.schemas import AssertionRule

        rule = AssertionRule(type="must_contain", value="no")
        assert BenchmarkRunner.evaluate_assertions("I don't know.", [rule])[0]["passed"]

    def test_decline_gets_zero_credit(self, sc):
        # With must_contain only, amnesic decline scores 0% cleanly.
        rules = _turn(sc, COMMITMENT_TURN).assertions
        results = BenchmarkRunner.evaluate_assertions("I don't know.", rules)
        assert [r["passed"] for r in results] == [False]


# ── Paired single-prompt controls (5.1c and 5.4c) ────────────────────────────


class TestStage5PairedControls:
    """Invariants for in_context_derived_revision_01 and in_context_modus_tollens_01."""

    def test_controls_load_and_match_filenames(self, stage5):
        for cid in STAGE5_CONTROL_IDS:
            assert cid in stage5, f"control {cid!r} did not load"
            path = REPO_ROOT / STAGE5_PATHS[cid]
            assert path.is_file(), f"{STAGE5_PATHS[cid]} is missing"
            assert scenario_id_from_filename(path.name) == cid
            assert stage5[cid].id == cid
            assert stage5[cid].category == "in_context_control"

    def test_controls_are_single_turn_default_session(self, stage5):
        for cid in STAGE5_CONTROL_IDS:
            turns = stage5[cid].turns
            assert len(turns) == 1, cid
            assert turns[0].turn_number == 1
            assert turns[0].session_id == "default"
            assert turns[0].role == "user"

    def test_mutual_pairing_metadata(self, stage5):
        for sid, cid in PAIRED_CONTROLS.items():
            assert stage5[sid].metadata.get("paired_control") == cid
            assert stage5[cid].metadata.get("paired_with") == sid
            assert stage5[cid].metadata.get("premise_source_turns") == [3, 4, 6, 9]

    def test_controls_registered_in_scenario_maps(self):
        for script in ("scripts/organize_datasets.py", "scripts/reorganize_outputs.py"):
            text = (REPO_ROOT / script).read_text(encoding="utf-8")
            for cid in STAGE5_CONTROL_IDS:
                entry = f'"{cid}": ("logical_consistency", "in_context_control")'
                assert entry in text, f"{script}: SCENARIO_MAP lacks {cid}"

    def test_control_assertions_identical_to_commitment_turn(self, stage5):
        for sid, cid in PAIRED_CONTROLS.items():
            commit_rules = _turn(stage5[sid], COMMITMENT_TURN).assertions
            control_rules = stage5[cid].turns[0].assertions
            assert control_rules == commit_rules, (sid, cid)

    def test_control_probe_question_identical_to_commitment_turn(self, stage5):
        for sid, cid in PAIRED_CONTROLS.items():
            commit_text = " ".join(_turn(stage5[sid], COMMITMENT_TURN).content.split())
            control_content = stage5[cid].turns[0].content
            assert _CONTROL_PROBE_MARKER in control_content, cid
            _, probe_part = control_content.split(_CONTROL_PROBE_MARKER, 1)
            norm_probe = " ".join(probe_part.split())
            assert norm_probe == commit_text, (
                f"{cid}: probe question after {_CONTROL_PROBE_MARKER!r} "
                f"must equal {sid} T{COMMITMENT_TURN}"
            )

    @pytest.mark.parametrize(
        ("cid", "required_clauses"),
        [
            (
                "in_context_derived_revision_01",
                [
                    "rebooting a prod-tier server requires Level-3 authorization, rebooting a staging-tier server requires Level-1 authorization, and dev-tier servers can be rebooted with no authorization at all.",
                    "Fleet inventory for the runbook: srv-alpha is a prod-tier server, srv-beta is prod-tier as well, and srv-gamma is a dev-tier box.",
                    "srv-alpha needs Level-3 authorization for any reboot. I'm writing that into the runbook now.",
                    "Inventory update: as of this morning, srv-alpha has been migrated to the staging tier.",
                ],
            ),
            (
                "in_context_modus_tollens_01",
                [
                    "every service hosted in the eu-central region is automatically enrolled in the Bastion audit at deploy time. Enrollment is enforced by the deployment pipeline itself, so no exceptions exist and none can ever be granted.",
                    "From our service inventory spreadsheet: billing-api is hosted in eu-central, and search-api is hosted in us-east.",
                    "billing-api is enrolled in the Bastion audit. I'm adding it to the audit tracker now.",
                    "Update from the Compliance team: after a full review, they confirmed in writing that billing-api is NOT enrolled in the Bastion audit. I've taken it off the audit tracker.",
                ],
            ),
        ],
    )
    def test_control_premise_parity_with_multi_session(self, stage5, cid, required_clauses):
        sid = stage5[cid].metadata["paired_with"]
        multi_text = " ".join(
            _turn(stage5[sid], n).content for n in [3, 4, 6, 9]
        )
        control_premise_block = stage5[cid].turns[0].content.split(
            _CONTROL_PROBE_MARKER, 1
        )[0]
        for clause in required_clauses:
            assert clause in multi_text, f"{sid} missing {clause!r}"
            assert clause in control_premise_block, f"{cid} missing {clause!r}"

