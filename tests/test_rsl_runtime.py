import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat import router as chat_router
from app.cascade.local_judge import LocalJudgeResult
from app.core.cascade import CascadeEngine
from app.core.exceptions import ProviderError
from app.core.recommender import ModelRecommender
from app.core.request_context import RequestContext
from app.core.routing_policy import RoutingPolicy
from app.rsl.runtime import RSLTrace, message_group_id, prepare_runtime
from app.rsl.store import RSLStore
from app.schemas.chat import ChatChoice, ChatMessage, ChatRequest, ChatResponse, Delta, StreamChunk, StreamChoice, Usage


def reply(model="internal", content="A complete answer to the question.", usage=True):
    return ChatResponse(id="reply", created=1, model=model,
                        choices=[ChatChoice(message=ChatMessage(role="assistant", content=content), finish_reason="stop")],
                        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30) if usage else None)


class Provider:
    provider_name = "test-provider"

    def __init__(self, *, error=False, stream_error=False, content="A complete answer to the question.", usage=True):
        self.error, self.stream_error, self.content, self.usage = error, stream_error, content, usage

    async def chat_completions(self, body, config):
        if self.error:
            raise ProviderError(self.provider_name, 404, "model unavailable")
        return reply(config.name, self.content, self.usage)

    async def chat_completions_stream(self, body, config):
        yield StreamChunk(id="chunk", created=1, model=config.name,
                          choices=[StreamChoice(delta=Delta(content=self.content), finish_reason="stop")])
        if self.stream_error:
            raise ProviderError(self.provider_name, 503, "stream interrupted")
        if self.usage:
            yield StreamChunk(id="chunk", created=1, model=config.name, choices=[],
                              usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30))


def model(name):
    return SimpleNamespace(name=name, pricing=SimpleNamespace(prompt=1.0, completion=2.0))


def recommender(root):
    return ModelRecommender(model_traits_path=str(root / "absent-traits.yaml"),
                            task_profiles_path=str(root / "absent-tasks.yaml"),
                            runtime_signals_path=str(root / "absent-signals.yaml"),
                            performance_history_path=str(root / "absent-history.yaml"))


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = RSLStore(self.root / "rsl.sqlite")
        self.base = self.store.ensure_baseline("1", history={})
        self.messages = [ChatMessage(role="user", content="private input text")]

    async def trace(self):
        trace = RSLTrace(self.store, "1", self.base["version_id"], self.messages, "general", "collect", "upstream-id")
        await trace.start("routing-a", ["routing-a", "routing-b"])
        return trace

    def test_enable_requires_explicit_recommendation(self):
        self.assertFalse(RoutingPolicy().rsl_enabled)
        with self.assertRaises(ValueError):
            RoutingPolicy(rsl_enabled=True)
        with self.assertRaises(ValueError):
            RoutingPolicy(rsl_enabled=True, recommendation_enabled=False)
        self.assertEqual(RoutingPolicy(rsl_enabled=True, recommendation_enabled=True).rsl_mode, "collect")

    def test_history_snapshot_copies_metadata_and_does_not_mutate_shared(self):
        original = recommender(self.root)
        history = {"metadata": {"performance_adjustment": {"weight": 0.7, "min_scored_cases": 1}}, "profiles": {}}
        snapshot = original.with_performance_history(history)
        self.assertEqual(snapshot.performance_history_weight, 0.7)
        self.assertEqual(snapshot.performance_history_min_scored_cases, 1)
        self.assertEqual(original.performance_history_weight, 0.12)
        history["profiles"]["changed"] = {}
        self.assertNotIn("changed", snapshot.performance_history["profiles"])

    async def test_collect_does_not_load_active_or_change_recommender(self):
        original = recommender(self.root)
        policy = RoutingPolicy(rsl_enabled=True, recommendation_enabled=True)
        with patch.object(self.store, "get_active", side_effect=AssertionError("collect must not load active")):
            selected, trace = await prepare_runtime(self.store, original, policy, "1", self.messages, "request-id")
        self.assertIsNot(selected, original)
        self.assertEqual(selected.performance_history, original.performance_history)
        self.assertEqual(selected.performance_history_weight, original.performance_history_weight)
        self.assertEqual(trace.version_id, self.base["version_id"])
        self.assertEqual(self.store.list_decisions("1"), [])  # Preview creates no request data.

    async def test_active_snapshot_pins_history_and_reads_pointer_once(self):
        original = recommender(self.root)
        policy = RoutingPolicy(rsl_enabled=True, recommendation_enabled=True, rsl_mode="active")
        version = {"version_id": "active-v1", "history": {"metadata": {"weight": 0.9}}}
        with patch.object(self.store, "get_active", return_value=version) as get_active:
            selected, trace = await prepare_runtime(self.store, original, policy, "1", self.messages)
        version["history"]["metadata"]["weight"] = 0.1
        self.assertEqual(selected.performance_history_weight, 0.9)
        self.assertEqual(selected.performance_history["metadata"]["weight"], 0.9)
        self.assertEqual(trace.version_id, "active-v1")
        get_active.assert_called_once_with("1")

    async def test_collect_declines_mismatched_baseline_after_config_reload(self):
        original = recommender(self.root).with_performance_history({"metadata": {"weight": 0.8}})
        policy = RoutingPolicy(rsl_enabled=True, recommendation_enabled=True)
        with self.assertLogs("model_router.rsl", level="WARNING"):
            selected, trace = await prepare_runtime(self.store, original, policy, "1", self.messages)
        self.assertIs(selected, original)
        self.assertIsNone(trace)

    def test_learned_history_cannot_override_hard_requirements(self):
        original = recommender(self.root)
        original.task_profiles = {"general": {"default": True, "hard_requires": ["coding"]}}
        original.model_traits = {"routing-a": {"coding": 0}, "routing-b": {"coding": 100}}
        history = {"metadata": {"weight": 100}, "profiles": {"general": {
            "summary": {"score_avg": 0.5, "pass_rate": 0.5}, "models": {
                "routing-a": {"scored_cases": 30, "score_avg": 1, "pass_rate": 1},
                "routing-b": {"scored_cases": 30, "score_avg": 0, "pass_rate": 0},
            }}}}
        engine = SimpleNamespace(providers={"p": Provider()}, model_map={
            "routing-a": ("p", model("a")), "routing-b": ("p", model("b"))})
        result = original.with_performance_history(history).recommend(
            messages=self.messages, router_engine=engine, policy=RoutingPolicy(),
            allowed_models={"routing-a", "routing-b"})
        self.assertEqual(result.selected_model, "routing-b")
        self.assertEqual(result.eligible_models, ["routing-b"])

    async def test_disabled_runtime_never_touches_store(self):
        original = recommender(self.root)
        with patch.object(self.store, "ensure_baseline", side_effect=AssertionError("disabled")):
            selected, trace = await prepare_runtime(self.store, original, RoutingPolicy(), "1", self.messages)
        self.assertIs(selected, original)
        self.assertIsNone(trace)

    async def test_records_routing_key_private_features_and_known_cost(self):
        trace = await self.trace()
        await trace.complete(Provider(), ChatRequest(model="auto", messages=self.messages), model("provider-model"), "routing-a")
        decision = self.store.get_decision(trace.decision_id)
        attempt = self.store.list_attempts(trace.decision_id)[0]
        self.assertEqual(decision["request_id"], "upstream-id")
        self.assertEqual(attempt["model"], "routing-a")
        self.assertEqual(attempt["model_version"], "provider-model")
        self.assertEqual(attempt["status"], "completed")
        self.assertAlmostEqual(attempt["cost"], 0.00005)
        self.assertNotIn("private input text", json.dumps(decision))
        self.assertEqual(message_group_id(self.messages), message_group_id([m.model_dump() for m in self.messages]))

    async def test_unknown_usage_and_errors_do_not_report_zero_cost(self):
        trace = await self.trace()
        body = ChatRequest(model="auto", messages=self.messages)
        await trace.complete(Provider(usage=False), body, model("a"), "routing-a")
        with self.assertRaises(ProviderError):
            await trace.complete(Provider(error=True), body, model("b"), "routing-b")
        attempts = self.store.list_attempts(trace.decision_id)
        self.assertEqual([x["status"] for x in attempts], ["completed", "error"])
        self.assertEqual([x["cost"] for x in attempts], [None, None])
        self.assertIsNone(trace.total_cost)

    async def test_stream_completion_error_and_early_close_are_distinct(self):
        body = ChatRequest(model="auto", messages=self.messages)
        trace = await self.trace()
        async for _ in trace.stream(Provider(), body, model("a"), "routing-a"):
            pass
        with self.assertRaises(ProviderError):
            async for _ in trace.stream(Provider(stream_error=True), body, model("a"), "routing-a"):
                pass
        chunks = trace.stream(Provider(), body, model("a"), "routing-a")
        await anext(chunks)
        await chunks.aclose()
        self.assertEqual([x["status"] for x in self.store.list_attempts(trace.decision_id)],
                         ["completed", "error", "cancelled"])

    async def test_cancelled_completion_is_recorded(self):
        class Cancelled(Provider):
            async def chat_completions(self, body, config):
                raise asyncio.CancelledError()
        trace = await self.trace()
        with self.assertRaises(asyncio.CancelledError):
            await trace.complete(Cancelled(), ChatRequest(model="auto", messages=self.messages), model("a"), "routing-a")
        self.assertEqual(self.store.list_attempts(trace.decision_id)[0]["status"], "cancelled")

    async def test_judge_failure_is_not_positive_feedback_and_rules_are_weak(self):
        trace = await self.trace()
        body = ChatRequest(model="auto", messages=self.messages)
        await trace.complete(Provider(), body, model("a"), "routing-a")
        attempt_id = trace.last_generation_attempt_id
        with patch.object(self.store, "record_feedback", wraps=self.store.record_feedback) as record:
            await trace.feedback(attempt_id, True, "judge_error_pass", LocalJudgeResult(True, 1.0, "judge_error_pass"))
            record.assert_not_called()
            await trace.feedback(attempt_id, True, "rule_pass", None)
            self.assertEqual(record.call_args.kwargs["confidence"], 0.4)
            await trace.complete(Provider(), body, model("b"), "routing-b")
            await trace.feedback(trace.last_generation_attempt_id, True, "rule_pass", None)
            self.assertEqual(record.call_count, 2)
        await trace.complete(Provider(), body, model("judge"), "judge-routing", kind="judge")
        self.assertEqual(self.store.list_attempts(trace.decision_id)[-1]["kind"], "judge")

    async def test_storage_outage_does_not_break_provider_result(self):
        trace = await self.trace()
        with patch.object(self.store, "record_attempt", side_effect=OSError("storage unavailable")):
            with self.assertLogs("model_router.rsl", level="WARNING"):
                response = await trace.complete(Provider(), ChatRequest(model="auto", messages=self.messages), model("a"), "routing-a")
        self.assertEqual(response.id, "reply")


class ChatIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.store = RSLStore(root / "rsl.sqlite")
        self.providers = {"pa": Provider(), "pb": Provider()}
        unavailable = set()
        self.engine = SimpleNamespace(
            providers=self.providers,
            model_map={"routing-a": ("pa", model("actual-a")), "routing-b": ("pb", model("actual-b"))},
            is_model_available=lambda key: key not in unavailable,
            resolve_model_target=lambda key, **kwargs: "routing-a" if key == "auto" else key,
            mark_model_error=lambda key, *_: unavailable.add(key) is None,
        )
        async def route(key):
            provider_name, config = self.engine.model_map[key]
            return self.providers[provider_name], config
        self.engine.route = route
        self.db = SimpleNamespace(insert_usage=AsyncMock(), insert_cascade_stat=AsyncMock())
        app = FastAPI()
        app.include_router(chat_router)
        app.state.db = self.db
        app.state.router = self.engine
        app.state.cascade = CascadeEngine(self.db, self.engine)
        app.state.recommender = recommender(root)
        app.state.rsl_store = self.store
        self.policy = RoutingPolicy(rsl_enabled=True, recommendation_enabled=True)
        self.context = RequestContext(subject_type="api_key", subject_id="1", routing_profile_id="1", request_id="upstream-request")
        for target, value in (
            ("resolve_request_context", AsyncMock(return_value=self.context)),
            ("load_routing_policy", AsyncMock(return_value=self.policy)),
            ("load_effective_routing_settings", AsyncMock(return_value={"keyword_routing_enabled": False, "semantic_routing_enabled": False, "cascade_enabled": False})),
            ("log_request", lambda row: None),
        ):
            patcher = patch("app.api.chat." + target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def send(self, **extra):
        return self.client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hello"}], **extra})

    def attempts(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return self.store.list_attempts(response.headers["X-RSL-Decision-Id"])

    def test_automatic_retry_records_both_actual_models_under_one_version(self):
        self.providers["pa"] = Provider(error=True)
        response = self.send()
        attempts = self.attempts(response)
        self.assertEqual([x["model"] for x in attempts], ["routing-a", "routing-b"])
        self.assertEqual([x["status"] for x in attempts], ["error", "completed"])
        self.assertEqual(response.json()["actual_model"], "routing-b")
        decision = self.store.get_decision(response.headers["X-RSL-Decision-Id"])
        self.assertEqual(decision["policy_version"], response.headers["X-RSL-Policy-Version"])

    def test_ordinary_stream_records_actual_alias_target(self):
        response = self.send(stream=True)
        self.assertEqual(self.attempts(response)[0]["model"], "routing-a")
        self.assertIn("data: [DONE]", response.text)

    def test_cascade_records_every_generation_and_total_cost(self):
        self.providers["pa"] = Provider(content="short")
        response = self.send(cascade={"enabled": True, "chain": ["routing-a", "routing-b"]})
        attempts = self.attempts(response)
        self.assertEqual([x["model"] for x in attempts], ["routing-a", "routing-b"])
        self.assertAlmostEqual(self.db.insert_cascade_stat.call_args.args[0]["total_cost"], 0.0001)

    def test_stream_cascade_records_each_attempt(self):
        self.providers["pa"] = Provider(content="short")
        response = self.send(stream=True, cascade={"enabled": True, "chain": ["routing-a", "routing-b"]})
        self.assertEqual([x["model"] for x in self.attempts(response)], ["routing-a", "routing-b"])

    def test_cascade_judge_is_costed_separately(self):
        self.providers["pb"] = Provider(content='{"valid": true, "score": 0.9, "reason": "good"}')
        response = self.send(cascade={"enabled": True, "chain": ["routing-a", "routing-b"], "judge_model": "routing-b"})
        attempts = self.attempts(response)
        self.assertEqual([x["kind"] for x in attempts], ["generation", "judge"])
        self.assertAlmostEqual(sum(x["cost"] for x in attempts), 0.0001)

    def test_allowed_models_still_constrain_recommendation(self):
        self.context.allowed_models = {"routing-b"}
        response = self.send()
        self.assertEqual([x["model"] for x in self.attempts(response)], ["routing-b"])

    def test_disabled_profile_creates_no_rsl_data(self):
        self.policy.rsl_enabled = False
        response = self.send()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("X-RSL-Decision-Id", response.headers)
        self.assertEqual(self.store.list_decisions("1"), [])


if __name__ == "__main__":
    unittest.main()
