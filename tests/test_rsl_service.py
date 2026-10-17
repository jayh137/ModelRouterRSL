import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin.rsl import router
from app.rsl.evaluation import activate_router, evaluate_router
from app.rsl.learning import build_candidate
from app.rsl.store import RSLStore
from app.rsl.worker import demo_cases, demo_context, demo_training, run_demo


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = RSLStore(Path(self.temp.name) / "rsl.db")
        self.recommender, self.engine, self.policy = demo_context()
        self.baseline = demo_training(self.store, self.recommender, "__global__")
        self.candidate = build_candidate(self.store, "__global__")

    def evaluate(self, cases=None):
        return evaluate_router(self.store, "__global__", self.candidate["version_id"],
                               demo_cases() if cases is None else cases, self.recommender,
                               self.engine, self.policy, dataset_id="test-only")

    def test_actual_recommender_changes_selection_without_reading_gold_labels(self):
        report = self.evaluate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["report"]["metrics"]["changed_decisions"], 24)
        self.assertTrue(all(r["candidate_model"] == "demo-b" for r in report["report"]["decisions"]))
        self.assertEqual(self.recommender.performance_history, {})

    def test_renaming_training_messages_does_not_bypass_holdout_check(self):
        cases = demo_cases()
        cases[0]["messages"][0]["content"] = "Synthetic training example 0"
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.evaluate(cases)

    def test_cannot_hide_unobserved_candidates_by_omitting_allowed_models(self):
        cases = demo_cases()
        del cases[0]["models"]["demo-b"]
        with self.assertRaisesRegex(ValueError, "every eligible"):
            self.evaluate(cases)

    def test_configuration_changes_invalidate_activation(self):
        report = self.evaluate()
        self.recommender.model_traits["demo-a"] = {"operator_boost": 100}
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            activate_router(self.store, "__global__", self.candidate["version_id"], report["evaluation_id"],
                            self.recommender, self.engine, self.policy)

    def test_scalar_weight_changes_invalidate_activation(self):
        report = self.evaluate()
        self.recommender.price_cost_weight = 0.123
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            activate_router(self.store, "__global__", self.candidate["version_id"], report["evaluation_id"],
                            self.recommender, self.engine, self.policy)

    def test_excluded_models_do_not_require_observations(self):
        self.policy.recommendation_excluded_models.append("demo-a")
        cases = demo_cases()
        for case in cases:
            del case["models"]["demo-a"]
        self.assertTrue(self.evaluate(cases)["passed"])

    def test_api_auth_feedback_idempotency_and_release_cycle(self):
        app = FastAPI()
        app.include_router(router, prefix="/admin")
        app.state.rsl_store = self.store
        app.state.recommender = self.recommender
        app.state.router = self.engine
        app.state.db = SimpleNamespace()
        app.state.admin_key = "test-admin-key"
        app.state.enable_admin_api = True
        with TestClient(app) as client:
            self.assertEqual(client.get("/admin/rsl/versions").status_code, 401)
            client.headers["Authorization"] = "Bearer test-admin-key"
            response = client.get("/admin/rsl/decisions")
            self.assertEqual(response.status_code, 200)
            decision_id = response.json()[0]["decision_id"]
            detail = client.get(f"/admin/rsl/decisions/{decision_id}").json()
            attempt_id = detail["attempts"][0]["attempt_id"]
            feedback = {"attempt_id": attempt_id, "source": "manual", "source_id": "api-test", "score": 0.8}
            first = client.post("/admin/rsl/feedback", json=feedback)
            self.assertEqual(first.status_code, 201, first.text)
            self.assertEqual(first.json(), client.post("/admin/rsl/feedback", json=feedback).json())
            changed = client.post("/admin/rsl/feedback", json={**feedback, "score": 0.3})
            self.assertEqual(changed.status_code, 422)
            report = client.post("/admin/rsl/evaluate", json={"version_id": self.candidate["version_id"],
                                "dataset_id": "api-test", "cases": demo_cases()})
            self.assertEqual(report.status_code, 201, report.text)
            result = report.json()
            self.assertTrue(result["passed"])
            activation = client.post("/admin/rsl/activate", json={"version_id": self.candidate["version_id"],
                                      "evaluation_id": result["evaluation_id"]})
            self.assertEqual(activation.status_code, 200, activation.text)
            rollback = client.post("/admin/rsl/rollback", json={"version_id": self.baseline["version_id"]})
            self.assertEqual(rollback.status_code, 200, rollback.text)
            self.assertEqual(rollback.json()["version_id"], self.baseline["version_id"])
            app.state.rsl_store = None
            self.assertEqual(client.get("/admin/rsl/versions").status_code, 404)

    def test_synthetic_demo_completes_without_providers(self):
        result = run_demo()
        self.assertTrue(result["evaluation_passed"])
        self.assertTrue(result["rollback_verified"])
        self.assertFalse(result["real_providers_called"])


if __name__ == "__main__":
    unittest.main()
