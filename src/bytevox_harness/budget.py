"""Thread-safe run and per-agent limits."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


def estimate_tokens(value: Any) -> int:
    """Conservative, dependency-free text estimate for budget enforcement."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return max(1, (len(text) + 3) // 4) if text else 0


@dataclass(frozen=True)
class Limits:
    max_steps: int = 32
    max_seconds: float = 60.0
    max_tokens: int = 12_000
    max_agent_steps: int = 24
    max_agent_seconds: float = 20.0
    max_agent_tokens: int = 4_000
    max_completion_tokens: int = 1_024
    max_lead_completion_tokens: int = 1_536


@dataclass
class Reservation:
    agent_id: str
    input_tokens: int
    completion_cap: int
    started_at: float


class RunBudget:
    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.started = time.perf_counter()
        self.steps = 0
        self.tokens = 0
        self._reserved_tokens = 0
        self._agent_steps: dict[str, int] = {}
        self._agent_tokens: dict[str, int] = {}
        self._agent_started: dict[str, float] = {}
        self._lock = threading.Lock()
        self.reason: str | None = None

    def _fail(self, message: str) -> None:
        self.reason = self.reason or message
        raise BudgetExceeded(message)

    def check_time(self, agent_id: str | None = None) -> float:
        now = time.perf_counter()
        elapsed = now - self.started
        if elapsed >= self.limits.max_seconds:
            self._fail("run time limit reached")
        remaining = self.limits.max_seconds - elapsed
        if agent_id and agent_id in self._agent_started:
            agent_elapsed = now - self._agent_started[agent_id]
            if agent_elapsed >= self.limits.max_agent_seconds:
                self._fail(f"agent time limit reached: {agent_id}")
            remaining = min(remaining, self.limits.max_agent_seconds - agent_elapsed)
        return max(0.0, remaining)

    def start_agent(self, agent_id: str) -> None:
        with self._lock:
            self._agent_started.setdefault(agent_id, time.perf_counter())

    def begin_step(
        self,
        agent_id: str,
        *,
        input_payload: Any = None,
        model: bool = False,
        minimum_completion_cap: int | None = None,
    ) -> Reservation:
        input_tokens = estimate_tokens(input_payload) if model else 0
        now = time.perf_counter()
        with self._lock:
            if now - self.started >= self.limits.max_seconds:
                self._fail("run time limit reached")
            agent_started = self._agent_started.setdefault(agent_id, now)
            if now - agent_started >= self.limits.max_agent_seconds:
                self._fail(f"agent time limit reached: {agent_id}")
            agent_steps = self._agent_steps.get(agent_id, 0)
            if self.steps >= self.limits.max_steps:
                self._fail("run step limit reached")
            if agent_steps >= self.limits.max_agent_steps:
                self._fail(f"agent step limit reached: {agent_id}")
            available = self.limits.max_tokens - self.tokens - self._reserved_tokens
            agent_available = self.limits.max_agent_tokens - self._agent_tokens.get(agent_id, 0)
            cap = 0
            if model:
                remaining = min(available, agent_available)
                if input_tokens >= remaining:
                    self._fail("token limit reached before model call")
                completion_limit = (
                    self.limits.max_lead_completion_tokens
                    if agent_id in {"lead_planner", "lead_synthesizer"}
                    else self.limits.max_completion_tokens
                )
                cap = min(completion_limit, remaining - input_tokens)
                if cap < 1:
                    self._fail("token limit reached before model call")
                if minimum_completion_cap is not None and cap < minimum_completion_cap:
                    self._fail(
                        f"token limit cannot provide the required {minimum_completion_cap}-token completion cap"
                    )
                self._reserved_tokens += input_tokens + cap
            self.steps += 1
            self._agent_steps[agent_id] = agent_steps + 1
            return Reservation(agent_id, input_tokens, cap, now)

    def can_fit_model_sequence(self, requests: list[tuple[str, Any, int]]) -> tuple[bool, str | None]:
        """Check whether complete model calls fit without reserving budget or steps.

        Each request contains its agent ID, the serialized request envelope, and
        the full completion cap that must be available. The estimate is the same
        character-based approximation used by ``begin_step``.
        """
        with self._lock:
            now = time.perf_counter()
            if now - self.started >= self.limits.max_seconds:
                return False, "run time limit reached"
            if self.steps + len(requests) > self.limits.max_steps:
                return False, "run step limit reached"

            run_required = 0
            agent_required: dict[str, int] = {}
            agent_step_required: dict[str, int] = {}
            for agent_id, payload, completion_cap in requests:
                if completion_cap < 1:
                    return False, "planned completion cap must be positive"
                agent_started = self._agent_started.get(agent_id)
                if agent_started is not None and now - agent_started >= self.limits.max_agent_seconds:
                    return False, f"agent time limit reached: {agent_id}"
                input_tokens = estimate_tokens(payload)
                run_required += input_tokens + completion_cap
                agent_required[agent_id] = agent_required.get(agent_id, 0) + input_tokens + completion_cap
                agent_step_required[agent_id] = agent_step_required.get(agent_id, 0) + 1

            run_available = self.limits.max_tokens - self.tokens - self._reserved_tokens
            if run_required > run_available:
                return False, "token limit reached before model call"
            for agent_id, required in agent_required.items():
                available = self.limits.max_agent_tokens - self._agent_tokens.get(agent_id, 0)
                if required > available:
                    return False, f"agent token limit reached: {agent_id}"
                available_steps = self.limits.max_agent_steps - self._agent_steps.get(agent_id, 0)
                if agent_step_required[agent_id] > available_steps:
                    return False, f"agent step limit reached: {agent_id}"
            return True, None

    def finish_step(self, reservation: Reservation, *, output_payload: Any = None, actual_tokens: int | None = None) -> dict[str, int]:
        output_tokens = estimate_tokens(output_payload) if output_payload is not None else 0
        if actual_tokens is None:
            charged = reservation.input_tokens + output_tokens
        else:
            charged = max(0, int(actual_tokens))
        with self._lock:
            reserved = reservation.input_tokens + reservation.completion_cap
            self._reserved_tokens = max(0, self._reserved_tokens - reserved)
            agent_remaining = self.limits.max_agent_tokens - self._agent_tokens.get(reservation.agent_id, 0)
            run_remaining = self.limits.max_tokens - self.tokens
            if charged > min(agent_remaining, run_remaining):
                charged = max(0, min(agent_remaining, run_remaining))
                self.reason = self.reason or "token limit reached after model response"
            self.tokens += charged
            self._agent_tokens[reservation.agent_id] = self._agent_tokens.get(reservation.agent_id, 0) + charged
        return {"input_tokens_est": reservation.input_tokens, "output_tokens_est": output_tokens, "budget_tokens": charged}

    def fail_step(self, reservation: Reservation) -> None:
        with self._lock:
            self._reserved_tokens = max(
                0, self._reserved_tokens - reservation.input_tokens - reservation.completion_cap
            )
            self.tokens += reservation.input_tokens
            self._agent_tokens[reservation.agent_id] = (
                self._agent_tokens.get(reservation.agent_id, 0) + reservation.input_tokens
            )

    def snapshot(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.started
        return {
            "steps": self.steps,
            "tokens_budgeted": self.tokens,
            "elapsed_seconds": round(elapsed, 3),
            "limits": {
                "max_steps": self.limits.max_steps,
                "max_seconds": self.limits.max_seconds,
                "max_tokens": self.limits.max_tokens,
                "max_agent_steps": self.limits.max_agent_steps,
                "max_agent_seconds": self.limits.max_agent_seconds,
                "max_agent_tokens": self.limits.max_agent_tokens,
                "max_completion_tokens": self.limits.max_completion_tokens,
                "max_lead_completion_tokens": self.limits.max_lead_completion_tokens,
            },
            "agent_steps": dict(self._agent_steps),
            "agent_tokens": dict(self._agent_tokens),
            "limit_reason": self.reason,
        }
