"""Offline deterministic provider and OpenAI-compatible JSON adapter."""

from __future__ import annotations

import json
import ipaddress
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .budget import estimate_tokens


def _object_schema(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build an OpenAI strict-schema object with every property required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_STRING = {"type": "string"}
_CLAIM_SCHEMA = _object_schema({
    "text": {"type": "string", "description": "A concise factual claim supported by the cited evidence."},
    "evidence_ids": {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": 1,
        "description": "One or more non-empty evidence IDs present in the supplied evidence."},
})

ROLE_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "lead_planner": _object_schema({
        "contract": {
            **_object_schema({
                "required_facets": {
                    "type": "array",
                    "items": _STRING,
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Exactly three distinct evidence facets covering the task.",
                },
            }),
            "description": "The compact task coverage contract used by the specialists.",
        },
        "subtasks": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "description": "One focused evidence query for each facet, in facet-index order.",
            "items": _object_schema({
                "id": {"type": "string", "description": "Stable short identifier for this subtask."},
                "facet_index": {"type": "integer", "minimum": 0, "maximum": 2, "description": "Zero-based position of the facet this query supports."},
                "query": {"type": "string", "description": "Short search query grounded in the user's question."},
                "goal": {"type": "string", "description": "The evidence the specialist should find."},
            }),
        },
    }),
    "worker_causes": _object_schema({
        "summary": {"type": "string", "description": "A concise summary of this specialist's evidence findings."},
        "claims": {"type": "array", "items": _CLAIM_SCHEMA, "description": "Evidence-backed claims; use an empty array if none are supported."},
    }),
    "worker_outcomes": _object_schema({
        "summary": {"type": "string", "description": "A concise summary of this specialist's evidence findings."},
        "claims": {"type": "array", "items": _CLAIM_SCHEMA, "description": "Evidence-backed claims; use an empty array if none are supported."},
    }),
    "worker_risks": _object_schema({
        "summary": {"type": "string", "description": "A concise summary of this specialist's evidence findings."},
        "claims": {"type": "array", "items": _CLAIM_SCHEMA, "description": "Evidence-backed claims; use an empty array if none are supported."},
    }),
    "single_agent": _object_schema({
        "summary": {"type": "string", "description": "The answer text; follow any summary_first_line_prefix in the validation contract."},
        "claims": {"type": "array", "items": _CLAIM_SCHEMA, "description": "Evidence-backed claims; use an empty array if none are supported."},
    }),
    "lead_synthesizer": _object_schema({
        "summary": {"type": "string", "description": "The answer text; follow any summary_first_line_prefix in the validation contract."},
        "claims": {"type": "array", "items": _CLAIM_SCHEMA, "description": "Evidence-backed claims; use an empty array if none are supported."},
    }),
    "lead_reviser": _object_schema({
        "summary": {"type": "string", "description": "The revised answer; follow any summary_first_line_prefix in the validation contract."},
        "claims": {"type": "array", "items": _CLAIM_SCHEMA, "description": "Evidence-backed claims; use an empty array if none are supported."},
    }),
    "checker": _object_schema({
        "accepted": {"type": "boolean", "description": "True only when the draft meets the requested answer and evidence requirements."},
        "issues": {"type": "array", "items": _STRING, "description": "Specific unresolved problems; empty only when accepted."},
    }),
}


def _payload_evidence_ids(role: str, payload: dict[str, Any]) -> list[str]:
    """Get the evidence IDs this role may cite from its current structured request."""
    if role in {"worker_causes", "worker_outcomes", "worker_risks", "single_agent"}:
        evidence = payload.get("evidence", [])
        ids = [
            item.get("evidence_id")
            for item in evidence
            if isinstance(item, dict)
        ] if isinstance(evidence, list) else []
    elif role == "lead_synthesizer":
        evidence = payload.get("retrieved_evidence", [])
        ids = [
            item.get("evidence_id")
            for item in evidence
            if isinstance(item, dict)
        ] if isinstance(evidence, list) else []
    elif role == "lead_reviser":
        evidence_by_id = payload.get("evidence_by_id", {})
        ids = list(evidence_by_id) if isinstance(evidence_by_id, dict) else []
    else:
        return []
    return sorted({evidence_id for evidence_id in ids if isinstance(evidence_id, str) and evidence_id})


def role_response_format(
    role: str,
    payload: dict[str, Any] | None = None,
    *,
    prompt_variant: str = "d54",
) -> dict[str, Any]:
    """Return the strict JSON Schema response_format, scoped to request evidence when available."""
    from .prompts import WORKER_PROFILE_LABELS, normalize_prompt_variant

    prompt_variant = normalize_prompt_variant(prompt_variant)
    try:
        schema = deepcopy(ROLE_JSON_SCHEMAS[role])
    except KeyError as exc:
        raise ValueError(f"no JSON Schema is defined for model role {role!r}") from exc
    if prompt_variant == "d61" and role in WORKER_PROFILE_LABELS:
        # Keep specialist outputs short enough to reduce truncation and leave more
        # of the existing run-token budget for synthesis and review.
        schema["properties"]["summary"]["maxLength"] = 320
        claims_schema = schema["properties"]["claims"]
        claims_schema["maxItems"] = 2
        claims_schema["items"]["properties"]["text"]["maxLength"] = 240
        claims_schema["items"]["properties"]["evidence_ids"]["maxItems"] = 2
    if payload is not None and role in {
        "worker_causes", "worker_outcomes", "worker_risks", "single_agent", "lead_synthesizer", "lead_reviser",
    }:
        claims_schema = schema["properties"]["claims"]
        available_ids = _payload_evidence_ids(role, payload)
        if available_ids:
            claims_schema["items"]["properties"]["evidence_ids"]["items"]["enum"] = available_ids
        else:
            # An empty enum is rejected by some strict-schema providers. With no supplied
            # evidence, allow a summary but disallow claims that could not be cited.
            claims_schema["maxItems"] = 0
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"bytevox_{role}",
            "strict": True,
            "schema": schema,
        },
    }


@dataclass
class ModelResponse:
    value: dict[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def actual_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


class ProviderResponseParseError(ValueError):
    """A received provider completion that failed protocol parsing, with safe usage."""

    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class ProviderResponseOutcomeError(RuntimeError):
    """A received completion refused or filtered by the provider, with safe usage."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def _validated_usage_count(value: Any) -> int | None:
    """Return only provider token counts that are valid non-negative integers."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def validate_role_response(role: str, value: Any) -> dict[str, Any]:
    """Reject malformed structured responses before orchestration consumes them."""
    if not isinstance(value, dict):
        raise ValueError(f"{role} response must be a JSON object")

    def require_string(item: Any, path: str, *, nonempty: bool = True) -> None:
        if not isinstance(item, str) or (nonempty and not item.strip()):
            raise ValueError(f"{role} response {path} must be a non-empty string")

    if role == "lead_planner":
        contract = value.get("contract")
        facets = contract.get("required_facets") if isinstance(contract, dict) else None
        if not isinstance(facets, list) or len(facets) != 3:
            raise ValueError("lead_planner response contract.required_facets must contain three strings")
        for index, facet in enumerate(facets):
            require_string(facet, f"contract.required_facets[{index}]")
        subtasks = value.get("subtasks")
        if not isinstance(subtasks, list) or len(subtasks) < 3:
            raise ValueError("lead_planner response subtasks must contain at least three objects")
        for index, item in enumerate(subtasks[:3]):
            if not isinstance(item, dict):
                raise ValueError(f"lead_planner response subtasks[{index}] must be an object")
            if isinstance(item.get("facet_index"), bool) or not isinstance(item.get("facet_index"), int):
                raise ValueError(f"lead_planner response subtasks[{index}].facet_index must be an integer")
            require_string(item.get("query"), f"subtasks[{index}].query")
            require_string(item.get("goal"), f"subtasks[{index}].goal")
    elif role in {"worker_causes", "worker_outcomes", "worker_risks", "single_agent", "lead_synthesizer", "lead_reviser"}:
        require_string(value.get("summary"), "summary", nonempty=False)
        claims = value.get("claims")
        if not isinstance(claims, list):
            raise ValueError(f"{role} response claims must be an array")
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                raise ValueError(f"{role} response claims[{index}] must be an object")
            require_string(claim.get("text"), f"claims[{index}].text")
            evidence_ids = claim.get("evidence_ids")
            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise ValueError(f"{role} response claims[{index}].evidence_ids must be a non-empty array")
            for evidence_index, evidence_id in enumerate(evidence_ids):
                require_string(evidence_id, f"claims[{index}].evidence_ids[{evidence_index}]")
    elif role == "checker":
        if not isinstance(value.get("accepted"), bool):
            raise ValueError("checker response accepted must be a boolean")
        issues = value.get("issues")
        if not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues):
            raise ValueError("checker response issues must be an array of strings")
    return value


def _is_loopback(hostname: str) -> bool:
    host = hostname.rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_base_url(base_url: str) -> tuple[str, bool]:
    """Validate endpoint transport and return its normalized URL and remote flag."""
    if not isinstance(base_url, str) or any(character.isspace() or ord(character) < 32 for character in base_url):
        raise ValueError("model base URL must be a valid URL without whitespace or control characters")
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError("model base URL must use http or https and include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("model base URL must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("model base URL must not contain a query or fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("model base URL has an invalid port") from exc
    remote = not _is_loopback(host)
    if parsed.scheme == "http" and remote:
        raise ValueError("remote model endpoints must use HTTPS; HTTP is allowed only for localhost")
    return base_url.rstrip("/"), remote


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent endpoint redirects from weakening transport or forwarding keys elsewhere."""

    def __init__(self, hostname: str, port: int | None, remote: bool) -> None:
        super().__init__()
        self.hostname = hostname.lower().rstrip(".")
        self.port = port
        self.remote = remote

    def redirect_request(self, request: Any, response: Any, code: int, message: str, headers: Any, new_url: str) -> Any:
        parsed = urllib.parse.urlsplit(new_url)
        if (
            parsed.hostname is None
            or parsed.hostname.lower().rstrip(".") != self.hostname
            or parsed.port != self.port
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (self.remote and parsed.scheme != "https")
            or (not self.remote and parsed.scheme not in {"http", "https"})
        ):
            raise ValueError("model endpoint redirect changed origin or weakened transport")
        return super().redirect_request(request, response, code, message, headers, new_url)


def _normalize(word: str) -> str:
    word = word.lower()
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _words(text: str) -> set[str]:
    return {_normalize(word) for word in re.findall(r"[a-z0-9%'-]+", text.lower()) if len(word) > 2}


class OfflineProvider:
    """Deterministic extractive provider used for offline demos and harness evaluation.

    This is orchestration software, not a language model. It provides reproducible behavior so
    the limits, tools, recovery, checker, and trace can be inspected without an API or model.
    """

    name = "offline-deterministic"
    is_remote = False
    reports_usage = False

    def __init__(self, *, fail_agent: str | None = None) -> None:
        self.fail_agent = fail_agent

    @staticmethod
    def _bounded(value: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        """Keep offline outputs under the same completion ceiling passed to remote models."""
        value = dict(value)
        claims = list(value.get("claims", [])) if isinstance(value.get("claims"), list) else []
        if claims:
            while claims:
                summary = " ".join(str(claim.get("text", "")) for claim in claims)
                candidate = {**value, "summary": summary, "claims": claims}
                if estimate_tokens(candidate) <= max_tokens:
                    value = candidate
                    break
                claims.pop()
            else:
                value = {"summary": "", "claims": []}
        if estimate_tokens(value) > max_tokens:
            value = {"summary": "", "claims": []} if "claims" in value else {}
        return value

    def generate_json(
        self,
        role: str,
        payload: dict[str, Any],
        *,
        max_tokens: int,
        timeout: float,
    ) -> ModelResponse:
        del timeout
        if self.fail_agent == role:
            raise TimeoutError(f"simulated timeout for {role}")
        if role == "lead_planner":
            task = payload["task"]
            supplied = payload.get("validation_contract")
            facets = supplied.get("required_facets") if isinstance(supplied, dict) else None
            if not isinstance(facets, list) or len(facets) != 3 or not all(isinstance(item, str) and item.strip() for item in facets):
                facets = [
                    "Facts that directly answer the task question",
                    "Supporting evidence, measurements, or context relevant to the task",
                    "Material limitations, uncertainty, or unresolved parts of the task",
                ]
            facets = [re.sub(r"\s+", " ", item).strip()[:180] for item in facets]
            subtasks = [
                {
                    "id": f"facet-{index + 1}",
                    "facet_index": index,
                    "query": f"{task} {facet}",
                    "goal": f"Find evidence for: {facet}",
                }
                for index, facet in enumerate(facets)
            ]
            return ModelResponse(self._bounded({
                "contract": {"required_facets": facets},
                "subtasks": subtasks,
            }, max_tokens))
        if role in {"worker_causes", "worker_outcomes", "worker_risks", "single_agent"}:
            evidence = payload.get("evidence", [])
            focus = payload.get("focus", role)
            focus_words = _words(focus + " " + payload.get("task", ""))
            contract = payload.get("validation_contract", {})
            required_ids = set(contract.get("must_cover_evidence_ids", [])) if isinstance(contract, dict) else set()
            ranked = sorted(
                evidence,
                key=lambda item: (
                    0 if role == "single_agent" and item.get("evidence_id") in required_ids else 1,
                    -(
                        float(item.get("score", 0))
                        + len(focus_words & _words(item.get("text", "")))
                    ),
                    item.get("evidence_id", ""),
                ),
            )
            claims = []
            seen: set[str] = set()
            for item in ranked:
                evidence_id = item.get("evidence_id", "")
                if not evidence_id or evidence_id in seen:
                    continue
                seen.add(evidence_id)
                text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
                claims.append({"text": text, "evidence_ids": [evidence_id]})
                if len(claims) >= (6 if role == "single_agent" else 3):
                    break
            summary = " ".join(claim["text"] for claim in claims)
            return ModelResponse(self._bounded({"summary": summary, "claims": claims}, max_tokens))
        if role == "lead_synthesizer":
            claims: list[dict[str, Any]] = []
            seen: set[tuple[str, tuple[str, ...]]] = set()
            for specialist in payload.get("specialists", []):
                for claim in specialist.get("claims", []):
                    ids = tuple(sorted(set(claim.get("evidence_ids", []))))
                    text = re.sub(r"\s+", " ", str(claim.get("text", ""))).strip()
                    key = (text, ids)
                    if text and ids and key not in seen:
                        seen.add(key)
                        claims.append({"text": text, "evidence_ids": list(ids)})
            summary = " ".join(claim["text"] for claim in claims)
            return ModelResponse(self._bounded({"summary": summary, "claims": claims}, max_tokens))
        if role == "checker":
            evidence_by_id = payload.get("evidence_by_id", {})
            issues: list[str] = []
            for index, claim in enumerate(payload.get("draft", {}).get("claims", []), start=1):
                ids = claim.get("evidence_ids", [])
                if not ids:
                    issues.append(f"Claim {index} has no evidence ID.")
                    continue
                for evidence_id in ids:
                    evidence = evidence_by_id.get(evidence_id)
                    if not evidence:
                        issues.append(f"Claim {index} cites unknown evidence ID {evidence_id}.")
                    elif str(claim.get("text", "")).strip() not in evidence.get("text", ""):
                        issues.append(f"Claim {index} is not a source excerpt; review its support.")
            return ModelResponse(self._bounded({"accepted": not issues, "issues": issues}, max_tokens))
        if role == "lead_reviser":
            evidence_by_id = payload.get("evidence_by_id", {})
            claims = []
            for claim in payload.get("draft", {}).get("claims", []):
                ids = [evidence_id for evidence_id in claim.get("evidence_ids", []) if evidence_id in evidence_by_id]
                text = str(claim.get("text", "")).strip()
                if ids and text and any(text in evidence_by_id[evidence_id].get("text", "") for evidence_id in ids):
                    claims.append({"text": text, "evidence_ids": ids})
            used_ids = {evidence_id for claim in claims for evidence_id in claim["evidence_ids"]}
            contract = payload.get("validation_contract", {})
            required_ids = contract.get("must_cover_evidence_ids", []) if isinstance(contract, dict) else []
            required_terms = contract.get("must_cover_terms", []) if isinstance(contract, dict) else []
            for evidence_id in required_ids:
                evidence = evidence_by_id.get(evidence_id)
                if evidence and evidence_id not in used_ids:
                    claims.append({"text": str(evidence.get("text", "")).strip(), "evidence_ids": [evidence_id]})
                    used_ids.add(evidence_id)
            for term in required_terms:
                term_folded = str(term).casefold()
                already_covered = any(
                    term_folded in claim["text"].casefold()
                    and any(
                        term_folded in str(evidence_by_id.get(evidence_id, {}).get("text", "")).casefold()
                        for evidence_id in claim["evidence_ids"]
                    )
                    for claim in claims
                )
                if already_covered:
                    continue
                for evidence_id, evidence in evidence_by_id.items():
                    text = str(evidence.get("text", "")).strip()
                    if term_folded in text.casefold():
                        claims.append({"text": text, "evidence_ids": [evidence_id]})
                        used_ids.add(evidence_id)
                        break
            return ModelResponse(self._bounded({"summary": " ".join(claim["text"] for claim in claims), "claims": claims}, max_tokens))
        raise ValueError(f"Offline provider has no behavior for role {role}")


class OpenAICompatibleProvider:
    """Small stdlib adapter for local Ollama or another OpenAI-compatible endpoint."""

    name = "openai-compatible"
    reports_usage = True

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        response_format: str = "json_object",
    ) -> None:
        if response_format not in {"json_object", "json_schema"}:
            raise ValueError("response_format must be 'json_object' or 'json_schema'")
        self.base_url, self.is_remote = validate_base_url(base_url)
        self.model = model
        self.temperature = 0
        self.api_key = api_key
        self.response_format = response_format
        self.prompt_variant = "d54"
        parsed = urllib.parse.urlsplit(self.base_url)
        self._redirect_handler = _SameOriginRedirectHandler(parsed.hostname or "", parsed.port, self.is_remote)

    def configure_prompt_variant(self, variant: str) -> None:
        """Select a prompt/schema variant for a sequentially configured harness run."""
        from .prompts import normalize_prompt_variant

        variant = normalize_prompt_variant(variant)
        if variant == "d61" and self.response_format != "json_schema":
            raise ValueError("the d61 experimental prompt variant requires --response-format json_schema")
        self.prompt_variant = variant

    def build_request_envelope(
        self,
        role: str,
        payload: dict[str, Any],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build the key-free JSON body; omit max_tokens for separate budget reservation."""
        from .prompts import system_prompt

        if self.response_format == "json_schema":
            formatted_response = role_response_format(role, payload, prompt_variant=self.prompt_variant)
        else:
            formatted_response = {"type": "json_object"}
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max(1, max_tokens)
        body.update({
            "response_format": formatted_response,
            "messages": [
                {"role": "system", "content": system_prompt(role, variant=self.prompt_variant)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        })
        return body

    def generate_json(
        self,
        role: str,
        payload: dict[str, Any],
        *,
        max_tokens: int,
        timeout: float,
    ) -> ModelResponse:
        body = self.build_request_envelope(role, payload, max_tokens=max_tokens)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            opener = urllib.request.build_opener(self._redirect_handler)
            with opener.open(request, timeout=max(0.1, timeout)) as response:
                raw = response.read(2_000_000).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read(2_000).decode("utf-8", errors="replace")
            if self.api_key:
                detail = detail.replace(self.api_key, "<redacted>")
            raise RuntimeError(f"model endpoint returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise TimeoutError(f"model endpoint could not be reached: {exc.reason}") from exc
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"model endpoint returned an invalid response envelope at character {exc.pos}") from exc
        usage = envelope.get("usage", {}) if isinstance(envelope, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = _validated_usage_count(usage.get("prompt_tokens"))
        completion_tokens = _validated_usage_count(usage.get("completion_tokens"))
        try:
            value = parse_chat_completion(envelope)
        except ValueError as exc:
            raise ProviderResponseParseError(
                str(exc),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ) from exc
        except RuntimeError as exc:
            choice = envelope.get("choices", [{}])[0]
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            if isinstance(message, dict) and message.get("refusal"):
                category = "refusal"
            elif isinstance(choice, dict) and choice.get("finish_reason") == "content_filter":
                category = "content_filter"
            else:
                raise
            raise ProviderResponseOutcomeError(
                str(exc),
                category=category,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ) from exc
        return ModelResponse(
            value=value,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )


def parse_chat_completion(envelope: Any) -> dict[str, Any]:
    """Extract JSON object content and surface actionable protocol failures safely."""
    if not isinstance(envelope, dict):
        raise ValueError("model endpoint response must be a JSON object")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("model endpoint returned no completion choices")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("model endpoint completion is missing its message object")
    refusal = message.get("refusal")
    if refusal:
        raise RuntimeError(f"model refused the request: {str(refusal)[:240]}")
    tool_calls = message.get("tool_calls")
    if (isinstance(tool_calls, list) and tool_calls) or finish_reason == "tool_calls":
        count = len(tool_calls) if isinstance(tool_calls, list) else 1
        raise ValueError(f"model returned {count} unsupported tool call(s); this harness sends no callable tools")
    if message.get("function_call"):
        raise ValueError("model returned an unsupported legacy function call; this harness sends no callable tools")
    if finish_reason in {"length", "max_tokens"}:
        raise ValueError("model response was truncated at its completion-token limit")
    if finish_reason == "content_filter":
        raise RuntimeError("model response was stopped by the provider content filter")

    content = message.get("content")
    if isinstance(content, list):
        text_parts = [
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") in {"text", "output_text"}
            and isinstance(part.get("text"), str)
        ]
        content = "".join(text_parts)
    if not isinstance(content, str) or not content.strip():
        reason = f" (finish_reason={finish_reason})" if finish_reason else ""
        raise ValueError(f"model returned empty text content{reason}")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        reason = f", finish_reason={finish_reason}" if finish_reason else ""
        raise ValueError(f"model returned invalid JSON at character {exc.pos}{reason}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("model response JSON must be an object")
    return value


def provider_from_args(
    provider: str,
    base_url: str | None,
    model: str | None,
    api_key_env: str,
    response_format: str = "json_object",
) -> Any:
    if provider == "offline":
        return OfflineProvider()
    resolved_url = base_url or os.environ.get("BYTEVOX_BASE_URL", "http://127.0.0.1:11434/v1")
    resolved_model = model or os.environ.get("BYTEVOX_MODEL", "qwen2.5:7b")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    return OpenAICompatibleProvider(resolved_url, resolved_model, api_key, response_format=response_format)
