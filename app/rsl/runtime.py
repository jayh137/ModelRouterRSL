"""Request-scoped, best-effort RSL telemetry. No prompt/answer text is persisted."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import aclosing
from copy import copy, deepcopy

logger = logging.getLogger("model_router.rsl")
RSL_IO_TIMEOUT_SECONDS = 1.0


async def _io(function, *args, **kwargs):
    # SQLite work is off-loop and bounded; a slow telemetry database must not
    # hold up inference for SQLite's full busy timeout. The thread may finish later.
    return await asyncio.wait_for(asyncio.to_thread(function, *args, **kwargs), RSL_IO_TIMEOUT_SECONDS)


def _messages(messages):
    return [
        {"role": m.role, "content": m.content} if hasattr(m, "role")
        else {"role": m.get("role", ""), "content": m.get("content", "")}
        for m in messages
    ]


def message_group_id(messages) -> str:
    """Canonical conversation fingerprint shared by collection and holdout checks."""
    canonical = json.dumps(_messages(messages), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def usage_cost(model_config, usage):
    pricing = getattr(model_config, "pricing", None)
    if usage is None or pricing is None:
        return None
    return (usage.prompt_tokens * pricing.prompt + usage.completion_tokens * pricing.completion) / 1_000_000


async def prepare_runtime(store, recommender, policy, profile_id, messages, request_id=None):
    """Pin one version before routing; collect mode keeps the existing recommender."""
    if not policy.rsl_enabled or store is None or recommender is None:
        return recommender, None
    configured_recommender = recommender
    profile_id = str(profile_id) if profile_id is not None else "__global__"
    try:
        baseline = await _io(store.ensure_baseline, profile_id, history=recommender.performance_history)
        if policy.rsl_mode == "active":
            version = await _io(store.get_active, profile_id)
            if version is None:
                version = baseline
            recommender = recommender.with_performance_history(version["history"])
        else:
            version = baseline
            if baseline["history"] != recommender.performance_history:
                logger.warning("RSL collection disabled: configured history differs from the frozen baseline for profile %s", profile_id)
                return recommender, None
            recommender = copy(recommender)
            recommender.performance_history = deepcopy(recommender.performance_history)
        trace = RSLTrace(store, profile_id, version["version_id"], messages,
                         recommender.profile_task(messages).name, policy.rsl_mode, request_id)
        return recommender, trace
    except Exception:
        logger.warning("RSL snapshot unavailable; using configured routing", exc_info=True)
        return configured_recommender, None


class RSLTrace:
    def __init__(self, store, profile_id, version_id, messages, task, mode, request_id=None):
        self.store = store
        self.profile_id = profile_id
        self.version_id = version_id
        self.request_id = request_id or str(uuid.uuid4())
        self.decision_id = None
        self.task = task
        self.group_id = message_group_id(messages)
        rows = _messages(messages)
        self.features = {
            "message_count": len(rows),
            "input_chars": sum(len(str(row["content"])) for row in rows),
            "has_system": any(row["role"] == "system" for row in rows),
            "has_tool_message": any(row["role"] == "tool" for row in rows),
            "rsl_mode": mode,
            "selection_probability": 1.0,
        }
        self.last_generation_attempt_id = None
        self.costs = []

    @property
    def total_cost(self):
        return None if not self.costs or any(value is None for value in self.costs) else sum(self.costs)

    @property
    def headers(self):
        if self.decision_id is None:
            return {}
        return {"X-RSL-Decision-Id": self.decision_id, "X-RSL-Policy-Version": self.version_id}

    async def _write(self, method, **kwargs):
        try:
            return await _io(getattr(self.store, method), **kwargs)
        except Exception:
            logger.warning("RSL %s failed; request execution continues", method, exc_info=True)
            return None

    async def start(self, model, candidates):
        row = await self._write(
            "record_decision", request_id=self.request_id, profile_id=self.profile_id,
            task=self.task, model=model, candidates=list(dict.fromkeys([*candidates, model])),
            features=self.features, policy_version=self.version_id, group_id=self.group_id,
        )
        if row:
            self.decision_id = row["decision_id"]

    async def _attempt(self, attempt_id, model, provider, model_config, status, started, usage, kind):
        # A failed or cancelled stream may contain only a partial usage report.
        cost = usage_cost(model_config, usage) if status == "completed" else None
        self.costs.append(cost)
        if kind == "generation":
            self.last_generation_attempt_id = attempt_id
        if self.decision_id is None:
            return
        await self._write(
            "record_attempt", attempt_id=attempt_id, decision_id=self.decision_id,
            model=model, provider=getattr(provider, "provider_name", provider.__class__.__name__),
            model_version=getattr(model_config, "name", None), kind=kind,
            status=status, latency_ms=int((time.monotonic() - started) * 1000), cost=cost,
        )

    async def complete(self, provider, body, model_config, routing_key, kind="generation"):
        attempt_id, started = str(uuid.uuid4()), time.monotonic()
        status, usage = "cancelled", None
        try:
            response = await provider.chat_completions(body, model_config)
            usage, status = response.usage, "completed"
            return response
        except Exception:
            status = "error"
            raise
        finally:
            await self._attempt(attempt_id, routing_key, provider, model_config, status, started, usage, kind)

    async def stream(self, provider, body, model_config, routing_key):
        attempt_id, started = str(uuid.uuid4()), time.monotonic()
        status, usage = "cancelled", None
        try:
            async with aclosing(provider.chat_completions_stream(body, model_config)) as chunks:
                async for chunk in chunks:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    yield chunk
            status = "completed"
        except Exception:
            status = "error"
            raise
        finally:
            await self._attempt(attempt_id, routing_key, provider, model_config, status, started, usage, "generation")

    async def feedback(self, attempt_id, is_valid, reason, judge_result):
        if not attempt_id or not self.decision_id:
            return
        if judge_result is not None:
            # Upstream deliberately fails open on judge failure. That is not a label.
            if judge_result.reason == "judge_error_pass":
                return
            source, score, confidence = "judge", judge_result.score, 0.6
        else:
            # Length/refusal/finish checks do not establish answer correctness.
            source, score, confidence = "rule", float(is_valid), 0.4
        await self._write(
            "record_feedback", attempt_id=attempt_id, source=source,
            source_id=f"cascade-v1:{attempt_id}", score=score, passed=is_valid, confidence=confidence,
        )
