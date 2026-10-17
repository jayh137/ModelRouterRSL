"""SQLite persistence for auditable RSL snapshots.

All operations are synchronous. Async applications should use ``asyncio.to_thread``.
Only derived request features and identifiers are persisted, never request/response text.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def number(value: Any, name: str, minimum: float = 0, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{name} must be a nonempty identifier of at most 512 characters")
    return value


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def _features(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("features must be a dictionary of derived scalars")
    forbidden = {"prompt", "messages", "content", "output", "response", "answer", "text", "query", "input"}
    for key, item in value.items():
        if not isinstance(key, str) or len(key) > 64 or key.lower() in forbidden:
            raise ValueError("features may contain only derived feature names, not request/response text")
        if isinstance(item, str):
            if len(item) > 64:
                raise ValueError("feature labels must be at most 64 characters")
        elif item is not None and not isinstance(item, bool):
            number(item, f"feature {key}", minimum=-1e12, maximum=1e12)
    return value


def _report_passes(report: dict) -> bool:
    """Recompute the gate from measured metrics; there is no caller-supplied pass flag."""
    try:
        if report["schema"] != 1 or report["integrity"] != "complete":
            return False
        config, metrics = report["thresholds"], report["metrics"]
        if metrics["cases"] < config["min_cases"] or metrics["cases"] != metrics["expected_cases"]:
            return False
        if metrics["missing_observations"] or metrics["overlapping_groups"] or metrics["overlapping_cases"]:
            return False
        if metrics["quality_delta"] < -config["quality_tolerance"]:
            return False
        if metrics["candidate_cost"] > metrics["baseline_cost"] * config["max_cost_ratio"] + config["max_cost_increase"]:
            return False
        if metrics["candidate_p95_latency_ms"] > metrics["baseline_p95_latency_ms"] * config["max_latency_ratio"] + config["max_latency_increase_ms"]:
            return False
        return all(v["quality_delta"] >= -config["max_task_quality_drop"] for v in report["tasks"].values())
    except (KeyError, TypeError, ValueError):
        return False


class RSLStore:
    def __init__(self, db_path: str | Path):
        if str(db_path) == ":memory:":
            raise ValueError("use a filesystem database (including a temporary file), not :memory:")
        self.db_path = str(Path(db_path))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY, request_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL, task TEXT NOT NULL, model TEXT NOT NULL,
                    features TEXT NOT NULL, candidates TEXT NOT NULL, policy_version TEXT,
                    group_id TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
                    model TEXT NOT NULL, status TEXT NOT NULL, latency_ms REAL NOT NULL,
                    cost REAL, provider TEXT, model_version TEXT, kind TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
                    source TEXT NOT NULL, source_id TEXT NOT NULL, score REAL NOT NULL,
                    passed INTEGER, confidence REAL NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(source, source_id)
                );
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, kind TEXT NOT NULL,
                    parent_version_id TEXT REFERENCES versions(version_id), cutoff TEXT NOT NULL,
                    history TEXT NOT NULL, history_hash TEXT NOT NULL, metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_lineage (
                    version_id TEXT NOT NULL REFERENCES versions(version_id),
                    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
                    decision_id TEXT NOT NULL, request_id TEXT NOT NULL, group_id TEXT NOT NULL,
                    PRIMARY KEY(version_id, attempt_id)
                );
                CREATE TABLE IF NOT EXISTS active_versions (
                    profile_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES versions(version_id)
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
                    version_id TEXT NOT NULL REFERENCES versions(version_id),
                    baseline_version_id TEXT NOT NULL REFERENCES versions(version_id),
                    report TEXT NOT NULL, report_hash TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS activations (
                    activation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL, version_id TEXT NOT NULL REFERENCES versions(version_id),
                    previous_version_id TEXT, action TEXT NOT NULL, evaluation_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS heldout_reservations (
                    profile_id TEXT NOT NULL, group_hash TEXT NOT NULL, case_hash TEXT NOT NULL,
                    evaluation_id TEXT NOT NULL REFERENCES evaluations(evaluation_id), created_at TEXT NOT NULL,
                    PRIMARY KEY(profile_id,group_hash,case_hash)
                );
                CREATE INDEX IF NOT EXISTS feedback_attempt_idx ON feedback(attempt_id);
                CREATE INDEX IF NOT EXISTS decisions_profile_idx ON decisions(profile_id);
                CREATE INDEX IF NOT EXISTS versions_profile_idx ON versions(profile_id);
                CREATE INDEX IF NOT EXISTS activations_profile_idx ON activations(profile_id);
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        for key in ("features", "candidates", "history", "metadata", "report"):
            if key in result:
                result[key] = json.loads(result[key])
        if "passed" in result and result["passed"] is not None:
            result["passed"] = bool(result["passed"])
        return result

    def record_decision(self, *, request_id: str, profile_id: str, task: str, model: str,
                        features: dict, candidates: list[str], policy_version: str | None = None,
                        group_id: str | None = None, decision_id: str | None = None) -> dict:
        decision_id = identifier(decision_id or uuid.uuid4().hex, "decision_id")
        for name, value in (("request_id", request_id), ("profile_id", profile_id), ("task", task), ("model", model)):
            identifier(value, name)
        if (not isinstance(candidates, list) or not candidates or any(not isinstance(item, str) for item in candidates)
                or len(set(candidates)) != len(candidates)):
            raise ValueError("candidates must be a nonempty list of distinct model IDs")
        for candidate in candidates:
            identifier(candidate, "candidate model")
        if model not in candidates:
            raise ValueError("selected model must belong to the candidate snapshot")
        if policy_version is not None:
            identifier(policy_version, "policy_version")
        # Callers should provide a digest of the logical request, stable across retries.
        group_id = identifier(group_id or decision_id, "group_id")
        values = (decision_id, request_id, profile_id, task, model, encode(_features(features)),
                  encode(candidates), policy_version, group_id)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if policy_version is not None:
                version = conn.execute("SELECT profile_id FROM versions WHERE version_id=?", (policy_version,)).fetchone()
                if version is None or version["profile_id"] != profile_id:
                    raise ValueError("policy_version must exist in the same profile")
            existing = conn.execute("SELECT * FROM decisions WHERE decision_id=?", (decision_id,)).fetchone()
            if existing:
                if tuple(existing)[:-1] != values:
                    raise ValueError("decision_id already belongs to a different decision")
                return self._row(existing)
            conn.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?)", (*values, utcnow()))
            return self._row(conn.execute("SELECT * FROM decisions WHERE decision_id=?", (decision_id,)).fetchone())

    def record_attempt(self, *, attempt_id: str, decision_id: str, model: str,
                       status: str, latency_ms: float, cost: float | None = None,
                       provider: str | None = None, model_version: str | None = None,
                       kind: str = "generation") -> dict:
        identifier(attempt_id, "attempt_id")
        identifier(decision_id, "decision_id")
        identifier(model, "model")
        if status not in {"completed", "error", "cancelled"}:
            raise ValueError("status must be completed, error or cancelled")
        if kind not in {"generation", "judge"}:
            raise ValueError("kind must be generation or judge")
        for name, value in (("provider", provider), ("model_version", model_version)):
            if value is not None:
                identifier(value, name)
        values = (attempt_id, decision_id, model, status, number(latency_ms, "latency_ms"),
                  None if cost is None else number(cost, "cost"), provider, model_version, kind)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            decision = conn.execute("SELECT candidates FROM decisions WHERE decision_id=?", (decision_id,)).fetchone()
            if decision is None:
                raise ValueError("unknown decision")
            if kind == "generation" and model not in json.loads(decision["candidates"]):
                raise ValueError("generation model is absent from the candidate snapshot")
            existing = conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if existing:
                if tuple(existing)[:-1] != values:
                    raise ValueError("attempt_id already belongs to a different attempt")
                return self._row(existing)
            conn.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?)", (*values, utcnow()))
            return self._row(conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone())

    def record_feedback(self, *, attempt_id: str, source: str, source_id: str, score: float,
                        passed: bool | None = None, confidence: float = 1.0) -> dict:
        for name, value in (("attempt_id", attempt_id), ("source", source), ("source_id", source_id)):
            identifier(value, name)
        if passed is not None and type(passed) is not bool:
            raise ValueError("passed must be a boolean or null")
        values = (attempt_id, source, source_id, number(score, "score", maximum=1),
                  passed, number(confidence, "confidence", maximum=1))
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            attempt = conn.execute("SELECT status,kind FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if attempt is None or attempt["status"] != "completed" or attempt["kind"] != "generation":
                raise ValueError("feedback requires an existing, completed generation attempt")
            existing = conn.execute("SELECT * FROM feedback WHERE source=? AND source_id=?", (source, source_id)).fetchone()
            if existing:
                if tuple(existing)[1:-1] != values:
                    raise ValueError("source/source_id was already used for different feedback")
                return self._row(existing)
            feedback_id = uuid.uuid4().hex
            conn.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?)", (feedback_id, *values, utcnow()))
            return self._row(conn.execute("SELECT * FROM feedback WHERE feedback_id=?", (feedback_id,)).fetchone())

    def get_decision(self, decision_id: str) -> dict | None:
        with self._connection() as conn:
            return self._row(conn.execute("SELECT * FROM decisions WHERE decision_id=?", (decision_id,)).fetchone())

    def get_attempt(self, attempt_id: str) -> dict | None:
        with self._connection() as conn:
            return self._row(conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone())

    def list_decisions(self, profile_id: str, limit: int = 100) -> list[dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        with self._connection() as conn:
            return [self._row(row) for row in conn.execute(
                "SELECT * FROM decisions WHERE profile_id=? ORDER BY created_at DESC,decision_id LIMIT ?", (profile_id, limit))]

    def list_attempts(self, decision_id: str) -> list[dict]:
        with self._connection() as conn:
            return [self._row(row) for row in conn.execute(
                "SELECT * FROM attempts WHERE decision_id=? ORDER BY created_at,attempt_id", (decision_id,))]

    def ensure_baseline(self, profile_id: str, history: dict | None = None) -> dict:
        identifier(profile_id, "profile_id")
        history = history if history is not None else {"profiles": {}, "summary": {"models": {}}}
        if not isinstance(history, dict):
            raise ValueError("history must be a dictionary")
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM versions WHERE profile_id=? AND kind='baseline'", (profile_id,)).fetchone()
            if existing:
                return self._row(existing)
            version_id, now = uuid.uuid4().hex, utcnow()
            conn.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?)",
                         (version_id, profile_id, "baseline", None, now, encode(history), digest(history), "{}", now))
            conn.execute("INSERT INTO active_versions VALUES (?,?)", (profile_id, version_id))
            conn.execute("INSERT INTO activations(profile_id,version_id,action,created_at) VALUES (?,?,?,?)",
                         (profile_id, version_id, "baseline", now))
            return self._row(conn.execute("SELECT * FROM versions WHERE version_id=?", (version_id,)).fetchone())

    def get_active(self, profile_id: str) -> dict | None:
        with self._connection() as conn:
            return self._row(conn.execute("SELECT v.* FROM versions v JOIN active_versions a USING(version_id) WHERE a.profile_id=?", (profile_id,)).fetchone())

    def get_version(self, version_id: str) -> dict | None:
        with self._connection() as conn:
            return self._row(conn.execute("SELECT * FROM versions WHERE version_id=?", (version_id,)).fetchone())

    def list_versions(self, profile_id: str) -> list[dict]:
        with self._connection() as conn:
            return [self._row(row) for row in conn.execute("SELECT * FROM versions WHERE profile_id=? ORDER BY created_at,version_id", (profile_id,))]

    def get_lineage(self, version_id: str) -> list[dict]:
        with self._connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM training_lineage WHERE version_id=? ORDER BY attempt_id", (version_id,))]

    def feedback_rows(self, profile_id: str, cutoff: str) -> list[dict]:
        with self._connection() as conn:
            return [dict(row) for row in conn.execute("""
                SELECT f.*,a.model,d.task,d.decision_id,d.request_id,d.group_id
                FROM feedback f JOIN attempts a USING(attempt_id)
                JOIN decisions d USING(decision_id)
                WHERE d.profile_id=? AND a.status='completed' AND a.kind='generation' AND f.created_at<=?
                ORDER BY f.created_at,f.feedback_id
            """, (profile_id, cutoff))]

    def heldout_hashes(self, profile_id: str) -> dict[str, set[str]]:
        with self._connection() as conn:
            rows = conn.execute("SELECT group_hash,case_hash FROM heldout_reservations WHERE profile_id=?", (profile_id,)).fetchall()
        return {"groups": {row["group_hash"] for row in rows}, "cases": {row["case_hash"] for row in rows}}

    def _create_candidate(self, *, profile_id: str, parent_version_id: str, cutoff: str,
                          history: dict, metadata: dict, lineage: list[dict]) -> dict:
        version_id, now = uuid.uuid4().hex, utcnow()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            parent = conn.execute("SELECT profile_id FROM versions WHERE version_id=?", (parent_version_id,)).fetchone()
            if parent is None or parent["profile_id"] != profile_id:
                raise ValueError("candidate parent must exist in the same profile")
            reserved = conn.execute("SELECT group_hash,case_hash FROM heldout_reservations WHERE profile_id=?", (profile_id,)).fetchall()
            groups = {row["group_hash"] for row in reserved}
            cases = {row["case_hash"] for row in reserved}
            if any(digest(row["group_id"]) in groups or digest(row["decision_id"]) in cases or digest(row["request_id"]) in cases for row in lineage):
                raise ValueError("training data became reserved for held-out evaluation; rebuild the candidate")
            conn.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?)",
                         (version_id, profile_id, "candidate", parent_version_id, cutoff, encode(history), digest(history), encode(metadata), now))
            conn.execute("""INSERT INTO training_lineage SELECT ?,attempt_id,decision_id,request_id,group_id
                            FROM training_lineage WHERE version_id=?""", (version_id, parent_version_id))
            for row in lineage:
                conn.execute("INSERT OR IGNORE INTO training_lineage VALUES (?,?,?,?,?)",
                             (version_id, row["attempt_id"], row["decision_id"], row["request_id"], row["group_id"]))
            return self._row(conn.execute("SELECT * FROM versions WHERE version_id=?", (version_id,)).fetchone())

    def _save_evaluation(self, *, profile_id: str, version_id: str, baseline_version_id: str,
                         report: dict) -> dict:
        # Internal: learning.evaluate_candidate computes the metrics from complete paired cases.
        report = json.loads(encode(report))
        report.pop("passed", None)
        evaluation_id = uuid.uuid4().hex
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for item in (version_id, baseline_version_id):
                row = conn.execute("SELECT profile_id FROM versions WHERE version_id=?", (item,)).fetchone()
                if row is None or row["profile_id"] != profile_id:
                    raise ValueError("evaluation versions must belong to the same profile")
            conn.execute("INSERT INTO evaluations VALUES (?,?,?,?,?,?,?)",
                         (evaluation_id, profile_id, version_id, baseline_version_id, encode(report), digest(report), utcnow()))
            for decision in report["decisions"]:
                conn.execute("INSERT OR IGNORE INTO heldout_reservations VALUES (?,?,?,?,?)",
                             (profile_id, decision["group_hash"], decision["case_hash"], evaluation_id, utcnow()))
        return self.get_evaluation(evaluation_id)

    def get_evaluation(self, evaluation_id: str) -> dict | None:
        with self._connection() as conn:
            result = self._row(conn.execute("SELECT * FROM evaluations WHERE evaluation_id=?", (evaluation_id,)).fetchone())
        if result:
            result["passed"] = result["report_hash"] == digest(result["report"]) and _report_passes(result["report"])
        return result

    def activate(self, profile_id: str, version_id: str, evaluation_id: str) -> dict:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("SELECT * FROM versions WHERE version_id=? AND profile_id=?", (version_id, profile_id)).fetchone()
            active = conn.execute("SELECT version_id FROM active_versions WHERE profile_id=?", (profile_id,)).fetchone()
            evaluation = conn.execute("SELECT * FROM evaluations WHERE evaluation_id=? AND profile_id=? AND version_id=?",
                                      (evaluation_id, profile_id, version_id)).fetchone()
            if version is None or version["kind"] != "candidate" or active is None or evaluation is None:
                raise ValueError("activation requires a candidate and its independent evaluation in this profile")
            report = json.loads(evaluation["report"])
            if digest(report) != evaluation["report_hash"] or not _report_passes(report):
                raise ValueError("independent evaluation did not pass all release gates")
            if evaluation["baseline_version_id"] != active["version_id"]:
                raise ValueError("stale evaluation: evaluate again against the current active version")
            if report.get("candidate_history_hash") != version["history_hash"]:
                raise ValueError("evaluation does not match the candidate snapshot")
            active_version = conn.execute("SELECT history_hash FROM versions WHERE version_id=?", (active["version_id"],)).fetchone()
            if report.get("baseline_history_hash") != active_version["history_hash"]:
                raise ValueError("evaluation does not match the active baseline snapshot")
            conn.execute("UPDATE active_versions SET version_id=? WHERE profile_id=?", (version_id, profile_id))
            conn.execute("INSERT INTO activations(profile_id,version_id,previous_version_id,action,evaluation_id,created_at) VALUES (?,?,?,?,?,?)",
                         (profile_id, version_id, active["version_id"], "activate", evaluation_id, utcnow()))
        return self.get_version(version_id)

    def rollback(self, profile_id: str, version_id: str) -> dict:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute("SELECT version_id FROM active_versions WHERE profile_id=?", (profile_id,)).fetchone()
            known = conn.execute("SELECT 1 FROM activations WHERE profile_id=? AND version_id=?", (profile_id, version_id)).fetchone()
            if not active or not known:
                raise ValueError("rollback target must be a previously activated version in the same profile")
            if active["version_id"] == version_id:
                raise ValueError("rollback target is already active")
            conn.execute("UPDATE active_versions SET version_id=? WHERE profile_id=?", (version_id, profile_id))
            conn.execute("INSERT INTO activations(profile_id,version_id,previous_version_id,action,created_at) VALUES (?,?,?,?,?)",
                         (profile_id, version_id, active["version_id"], "rollback", utcnow()))
        return self.get_version(version_id)

    def list_activations(self, profile_id: str) -> list[dict]:
        with self._connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM activations WHERE profile_id=? ORDER BY activation_id", (profile_id,))]
