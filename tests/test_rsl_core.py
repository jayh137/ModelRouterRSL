import copy
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.rsl import EvaluationThresholds, RSLStore, build_candidate, evaluate_candidate


class RSLCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = RSLStore(Path(self.temp.name) / "rsl.sqlite3")
        self.counter = 0
        self.history = {"profiles": {"general": {"models": {
            "expensive": {"cases": 10, "scored_cases": 10, "score_avg": .65, "pass_rate": .9, "bad_output_rate": .1},
        }, "summary": {"score_avg": .65, "pass_rate": .9}}}, "metadata": {"existing_setting": True}}
        self.baseline = self.store.ensure_baseline("p", self.history)

    def attempt(self, *, profile="p", group=None, status="completed", kind="generation", request_id=None):
        self.counter += 1
        decision = self.store.record_decision(
            request_id=request_id or f"request-{self.counter}", profile_id=profile, task="general", model="cheap",
            candidates=["cheap", "expensive"], features={"input_chars": 42, "language": "zh", "stream": False},
            group_id=group or f"train-group-{self.counter}",
            policy_version=self.store.get_active(profile)["version_id"] if self.store.get_active(profile) else None)
        return self.store.record_attempt(attempt_id=f"attempt-{self.counter}", decision_id=decision["decision_id"],
                                         model="cheap", status=status, kind=kind, latency_ms=20, cost=None)

    def label(self, attempt, score=.95, **kwargs):
        return self.store.record_feedback(attempt_id=attempt["attempt_id"], source=kwargs.pop("source", "objective"),
                                          source_id=kwargs.pop("source_id", f"feedback-{self.counter}"), score=score,
                                          passed=kwargs.pop("passed", True), **kwargs)

    def candidate(self, **kwargs):
        for _ in range(5):
            self.label(self.attempt())
        return build_candidate(self.store, "p", **kwargs)

    def cases(self, count=20):
        return [{"case_id": f"holdout-{i}", "group_id": f"holdout-group-{i}", "task": "general",
                 "messages": [{"role": "user", "content": f"held-out task {i}"}],
                 "features": {"input_chars": 20}, "allowed_models": ["cheap", "expensive"],
                 "models": {"cheap": {"score": .95, "cost": 1, "latency_ms": 30},
                            "expensive": {"score": .9, "cost": 2, "latency_ms": 100}}} for i in range(count)]

    @staticmethod
    def rank(history, case):
        models = history.get("profiles", {}).get(case["task"], {}).get("models", {})
        return max(case["allowed_models"], key=lambda model: models.get(model, {}).get("score_avg", .5))

    def evaluate(self, version, **kwargs):
        return evaluate_candidate(self.store, "p", version["version_id"], kwargs.pop("cases", self.cases()),
                                  kwargs.pop("rank", self.rank), **kwargs)

    def test_request_ids_do_not_collide_and_decision_replay_is_checked(self):
        first = self.attempt(request_id="client-reused-id")
        second = self.attempt(request_id="client-reused-id")
        self.assertNotEqual(first["decision_id"], second["decision_id"])
        decision = self.store.get_decision(first["decision_id"])
        replay = {key: value for key, value in decision.items() if key != "created_at"}
        self.assertEqual(decision, self.store.record_decision(**replay))
        with self.assertRaises(ValueError):
            self.store.record_decision(**{**replay, "task": "different"})

    def test_attempt_replay_and_unknown_cost(self):
        attempt = self.attempt()
        self.assertIsNone(attempt["cost"])
        replay = {key: value for key, value in attempt.items() if key != "created_at"}
        self.assertEqual(attempt, self.store.record_attempt(**replay))
        with self.assertRaises(ValueError):
            self.store.record_attempt(**{**replay, "cost": 2})

    def test_feedback_is_idempotent_and_conflicting_replay_rejected(self):
        attempt = self.attempt()
        first = self.label(attempt, source_id="source-event")
        self.assertEqual(first, self.label(attempt, source_id="source-event"))
        with self.assertRaises(ValueError):
            self.label(attempt, score=.1, source_id="source-event")
        other = self.attempt()
        with self.assertRaises(ValueError):
            self.label(other, source_id="source-event")

    def test_feedback_only_for_completed_generation(self):
        for status in ("error", "cancelled"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.label(self.attempt(status=status))
        with self.assertRaises(ValueError):
            self.label(self.attempt(kind="judge"))
        with self.assertRaises(ValueError):
            self.store.record_feedback(attempt_id="missing", source="user", source_id="missing", score=1)

    def test_nonfinite_and_out_of_range_feedback_rejected(self):
        attempt = self.attempt()
        for field in ("score", "confidence"):
            for value in (float("nan"), float("inf"), -1, 1.1, True, "0.5"):
                values = {"attempt_id": attempt["attempt_id"], "source": "user", "source_id": "test", "score": .5, "confidence": .9}
                values[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.store.record_feedback(**values)
        with self.assertRaises(ValueError):
            self.label(attempt, passed=1)

    def test_attempt_numbers_and_private_features_validated(self):
        attempt = self.attempt()
        replay = {key: value for key, value in attempt.items() if key != "created_at"}
        for field in ("cost", "latency_ms"):
            for value in (-1, float("nan"), float("inf"), True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.store.record_attempt(**{**replay, field: value})
        decision = self.store.get_decision(attempt["decision_id"])
        replay = {key: value for key, value in decision.items() if key != "created_at"}
        for features in ({"prompt": "private"}, {"messages": []}, {"language": "x" * 65}, {"input_chars": float("nan")}):
            with self.subTest(features=features), self.assertRaises(ValueError):
                self.store.record_decision(**{**replay, "features": features})

    def test_read_apis_and_limits(self):
        attempt = self.attempt()
        decision = self.store.list_decisions("p")[0]
        self.assertEqual([attempt], self.store.list_attempts(decision["decision_id"]))
        self.assertEqual([], self.store.list_decisions("unknown"))
        for invalid in (0, 1001, True):
            with self.assertRaises(ValueError):
                self.store.list_decisions("p", invalid)
        self.assertIsNone(self.store.get_active("unknown"))

    def test_baseline_is_immutable_and_idempotent(self):
        self.assertEqual(self.baseline, self.store.ensure_baseline("p", {"changed": True}))
        self.baseline["history"]["profiles"].clear()
        self.assertIn("general", self.store.get_active("p")["history"]["profiles"])

    def test_training_deduplicates_labels_and_smooths_fixed_prior(self):
        for _ in range(5):
            attempt = self.attempt()
            self.label(attempt, score=1, source="objective")
            self.label(attempt, score=0, source="judge", source_id=f"judge-{self.counter}")
        candidate = build_candidate(self.store, "p")
        row = candidate["history"]["profiles"]["general"]["models"]["cheap"]
        self.assertEqual(5, row["scored_cases"])
        self.assertAlmostEqual(.75, row["score_avg"])
        self.assertEqual(5, len(self.store.get_lineage(candidate["version_id"])))
        self.assertEqual(self.history["profiles"]["general"]["models"]["expensive"],
                         candidate["history"]["profiles"]["general"]["models"]["expensive"])
        self.assertTrue(candidate["history"]["metadata"]["existing_setting"])
        self.assertAlmostEqual((.75 * 5 + .65 * 10) / 15, candidate["history"]["profiles"]["general"]["summary"]["score_avg"])
        self.assertEqual(self.baseline["version_id"], self.store.get_active("p")["version_id"])

    def test_low_confidence_unknown_source_and_correlated_groups_cannot_train(self):
        for _ in range(6):
            self.label(self.attempt(), confidence=.5)
            self.label(self.attempt(), source="untrusted")
            self.label(self.attempt(group="one-repeated-request"))
        with self.assertRaisesRegex(ValueError, "distinct groups"):
            build_candidate(self.store, "p")
        self.assertEqual(1, len(self.store.list_versions("p")))

    def test_unobserved_pass_is_not_recorded_as_failure(self):
        for _ in range(5):
            self.label(self.attempt(), passed=None)
        row = build_candidate(self.store, "p")["history"]["profiles"]["general"]["models"]["cheap"]
        self.assertIsNone(row["pass_rate"])
        self.assertEqual(0, row["bad_output_rate"])

    def test_cutoff_and_training_arguments(self):
        self.candidate()
        old = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "insufficient"):
            build_candidate(self.store, "p", cutoff=old)
        for kwargs in ({"min_samples": True}, {"prior_strength": float("nan")}, {"min_confidence": 1.1},
                       {"cutoff": "2020-01-01"}, {"cutoff": "2999-01-01T00:00:00Z"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_candidate(self.store, "p", **kwargs)

    def test_evaluate_activate_rollback_and_context(self):
        candidate = self.candidate()
        evaluation = self.evaluate(candidate, context_fingerprint="config-sha256")
        self.assertTrue(evaluation["passed"])
        self.assertEqual("config-sha256", evaluation["report"]["context_fingerprint"])
        self.assertEqual(20, evaluation["report"]["metrics"]["changed_decisions"])
        self.store.activate("p", candidate["version_id"], evaluation["evaluation_id"])
        self.assertEqual(candidate["version_id"], self.store.get_active("p")["version_id"])
        with self.assertRaisesRegex(ValueError, "cannot predate"):
            build_candidate(self.store, "p", cutoff=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat())
        again = build_candidate(self.store, "p")
        self.assertEqual(candidate["history"]["profiles"], again["history"]["profiles"])
        self.store.rollback("p", self.baseline["version_id"])
        self.assertEqual(self.baseline["version_id"], self.store.get_active("p")["version_id"])
        self.assertEqual(["baseline", "activate", "rollback"], [row["action"] for row in self.store.list_activations("p")])

    def test_quality_cost_latency_and_task_regression_gate_activation(self):
        candidate = self.candidate()
        for field, value in (("score", .1), ("cost", 3), ("latency_ms", 1000)):
            cases = self.cases()
            for case in cases:
                case["models"]["cheap"][field] = value
            evaluation = self.evaluate(candidate, cases=cases)
            with self.subTest(field=field):
                self.assertFalse(evaluation["passed"])
                with self.assertRaisesRegex(ValueError, "release gates"):
                    self.store.activate("p", candidate["version_id"], evaluation["evaluation_id"])
        cases = self.cases()
        cases[0]["task"] = "small-critical-task"
        cases[0]["models"]["cheap"]["score"] = .8
        evaluation = self.evaluate(candidate, cases=cases, rank=lambda history, case: "cheap" if "cheap" in history["profiles"]["general"]["models"] else "expensive")
        self.assertGreater(evaluation["report"]["metrics"]["quality_delta"], 0)
        self.assertFalse(evaluation["passed"])

    def test_stale_evaluation_and_profile_isolation(self):
        first = self.candidate()
        second = build_candidate(self.store, "p")
        eval_first, eval_second = self.evaluate(first), self.evaluate(second)
        self.store.activate("p", first["version_id"], eval_first["evaluation_id"])
        with self.assertRaisesRegex(ValueError, "stale"):
            self.store.activate("p", second["version_id"], eval_second["evaluation_id"])
        self.store.ensure_baseline("another", {})
        with self.assertRaises(ValueError):
            self.store.rollback("another", first["version_id"])
        with self.assertRaises(ValueError):
            self.store.activate("another", first["version_id"], eval_first["evaluation_id"])
        with self.assertRaises(ValueError):
            self.store.rollback("p", second["version_id"])
        with self.assertRaises(ValueError):
            evaluate_candidate(self.store, "another", first["version_id"], self.cases(), self.rank)

    def test_no_boolean_or_unevaluated_activation(self):
        candidate = self.candidate()
        for value in (True, "passed", "missing"):
            with self.assertRaises(ValueError):
                self.store.activate("p", candidate["version_id"], value)

    def test_missing_models_unknown_cost_and_invalid_metrics_never_skipped(self):
        candidate = self.candidate()
        for field, value in (("score", float("nan")), ("cost", None), ("latency_ms", -1)):
            cases = self.cases()
            cases[0]["models"]["cheap"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.evaluate(candidate, cases=cases)
        cases = self.cases()
        del cases[0]["models"]["cheap"]
        with self.assertRaisesRegex(ValueError, "missing observations"):
            self.evaluate(candidate, cases=cases)
        with self.assertRaisesRegex(ValueError, "unobserved"):
            self.evaluate(candidate, rank=lambda history, case: "missing")

    def test_labels_are_not_exposed_to_rank_and_messages_are_not_persisted(self):
        candidate = self.candidate()
        cases = self.cases()
        for case in cases:
            case["gold_answer"] = "secret-gold-label"
        def rank(history, case):
            self.assertNotIn("models", case)
            self.assertNotIn("gold_answer", case)
            self.assertIn("messages", case)
            return self.rank(history, case)
        evaluation = self.evaluate(candidate, cases=cases, rank=rank)
        self.assertNotIn("held-out task", str(evaluation))
        self.assertNotIn("secret-gold-label", str(evaluation))
        raw = Path(self.store.db_path).read_bytes()
        self.assertNotIn(b"held-out task", raw)

    def test_training_and_heldout_overlap_and_duplicates_are_rejected(self):
        candidate = self.candidate()
        lineage = self.store.get_lineage(candidate["version_id"])[0]
        for field, value in (("group_id", lineage["group_id"]), ("case_id", lineage["request_id"]),
                             ("case_id", lineage["decision_id"])):
            cases = self.cases()
            cases[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "overlap"):
                self.evaluate(candidate, cases=cases)
        cases = self.cases()
        cases[1]["group_id"] = cases[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.evaluate(candidate, cases=cases)
        with self.assertRaisesRegex(ValueError, "at least 20"):
            self.evaluate(candidate, cases=self.cases(19))

    def test_reserved_holdout_groups_cannot_enter_later_training(self):
        candidate = self.candidate()
        self.evaluate(candidate)
        for i in range(5):
            self.label(self.attempt(group=f"holdout-group-{i}"), score=0)
        next_version = build_candidate(self.store, "p")
        groups = {row["group_id"] for row in self.store.get_lineage(next_version["version_id"])}
        self.assertFalse(any(group.startswith("holdout") for group in groups))
        self.assertEqual(5, next_version["history"]["profiles"]["general"]["models"]["cheap"]["cases"])

    def test_inherited_lineage_remains_excluded_after_activation(self):
        first = self.candidate()
        evaluation = self.evaluate(first)
        self.store.activate("p", first["version_id"], evaluation["evaluation_id"])
        for _ in range(5):
            self.label(self.attempt())
        second = build_candidate(self.store, "p")
        self.assertEqual(10, len(self.store.get_lineage(second["version_id"])))
        cases = self.cases()
        cases[0]["group_id"] = self.store.get_lineage(first["version_id"])[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.evaluate(second, cases=cases)

    def test_thresholds_validate_and_rank_errors_are_not_success(self):
        candidate = self.candidate()
        for thresholds in (EvaluationThresholds(min_cases=0), {"max_cost_ratio": float("inf")}, {"quality_tolerance": True}):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                self.evaluate(candidate, thresholds=thresholds)
        def broken(history, case):
            raise RuntimeError("router unavailable")
        with self.assertRaisesRegex(ValueError, "router failed"):
            self.evaluate(candidate, rank=broken)


if __name__ == "__main__":
    unittest.main()
