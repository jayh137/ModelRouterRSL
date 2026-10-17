import json
import logging
import re
from dataclasses import dataclass

from app.schemas.chat import ChatChoice, ChatMessage, ChatRequest, ChatResponse

logger = logging.getLogger("model_router.cascade.local_judge")


DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "You are a strict answer-quality judge for cascade routing. "
    "Your job is to decide whether the candidate answer is already good enough to return to the user, "
    "or whether the router should escalate to a stronger model. "
    "Judge the answer conservatively across relevance, completeness, correctness, directness, and requested format adherence. "
    "If the answer is vague, partial, evasive, or misses the user's requested output shape, prefer invalid. "
    "Return raw JSON only with no markdown, no code fences, and no extra commentary."
)


@dataclass
class LocalJudgeResult:
    valid: bool
    score: float
    reason: str


class LocalJudge:
    def __init__(
        self,
        router,
        judge_model: str,
        threshold: float = 0.7,
        system_prompt: str | None = None,
    ):
        self.router = router
        self.judge_model = judge_model
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.system_prompt = system_prompt or DEFAULT_JUDGE_SYSTEM_PROMPT
        self.rsl_trace = None

    def _build_messages(self, request: ChatRequest, response: ChatResponse) -> list[ChatMessage]:
        original_messages = [message.model_dump() for message in request.messages]
        candidate_choice = response.choices[0] if response.choices else None
        candidate_content = candidate_choice.message.content if candidate_choice else ""
        candidate_finish_reason = candidate_choice.finish_reason if candidate_choice else None
        user_content = (
            "Review the candidate response for the original conversation.\n"
            "Return JSON with exactly these keys: "
            '{"valid": boolean, "score": number, "reason": string, "dimension_scores": object}.\n'
            "The score must be between 0 and 1.\n"
            "Keep the reason short and concrete.\n"
            "dimension_scores should contain numeric values between 0 and 1 for: "
            "relevance, completeness, correctness, directness, and format.\n"
            f"Mark valid=true only if the answer is ready to ship to the user now. "
            f"Scores below {self.threshold:.2f} should be invalid. "
            f"Scores at or above {self.threshold:.2f} should still be invalid if the answer has an important flaw.\n\n"
            f"Original conversation:\n{json.dumps(original_messages, ensure_ascii=False, indent=2)}\n\n"
            "Candidate response:\n"
            f"{json.dumps({'content': candidate_content, 'finish_reason': candidate_finish_reason}, ensure_ascii=False, indent=2)}"
        )
        return [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_content),
        ]

    def _extract_json_object(self, content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if match:
                return match.group(0)
        raise ValueError("Judge did not return valid JSON")

    def _parse_bool(self, value) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return None

    def _parse_score(self, value) -> float | None:
        score = None
        if isinstance(value, (int, float)):
            score = float(value)
        elif isinstance(value, str):
            try:
                score = float(value.strip())
            except ValueError:
                score = None
        if score is None:
            return None
        if 1.0 < score <= 100.0:
            score = score / 100.0
        return max(0.0, min(1.0, score))

    def _parse_result(self, response: ChatResponse) -> LocalJudgeResult:
        content = response.choices[0].message.content if response.choices else ""
        payload = json.loads(self._extract_json_object(content))

        score = self._parse_score(payload.get("score"))

        valid = self._parse_bool(payload.get("valid"))
        if valid is None and score is not None:
            valid = score >= self.threshold
        if valid is None:
            raise ValueError("Judge JSON is missing a usable 'valid' or 'score' field")

        if score is None:
            score = 1.0 if valid else 0.0

        reason = str(payload.get("reason") or "").strip() or "judge_no_reason"
        return LocalJudgeResult(valid=valid, score=score, reason=reason)

    async def evaluate(self, response: ChatResponse, request: ChatRequest) -> LocalJudgeResult:
        try:
            provider_name, model_config = self.router.model_map[self.judge_model]
            provider = self.router.providers[provider_name]
            judge_request = request.model_copy(
                update={
                    "model": self.judge_model,
                    "messages": self._build_messages(request, response),
                    "stream": False,
                    "cascade": None,
                    "ensemble": None,
                    "temperature": 0.0,
                    "max_tokens": 300,
                }
            )
            if self.rsl_trace:
                judge_response = await self.rsl_trace.complete(
                    provider, judge_request, model_config, self.judge_model, kind="judge",
                )
            else:
                judge_response = await provider.chat_completions(judge_request, model_config)
            return self._parse_result(judge_response)
        except Exception as exc:
            logger.warning("Local judge validation failed for %s: %s", self.judge_model, exc)
            return LocalJudgeResult(valid=True, score=1.0, reason="judge_error_pass")
