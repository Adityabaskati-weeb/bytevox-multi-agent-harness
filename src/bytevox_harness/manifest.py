"""Small, secret-safe run manifest helpers."""

from __future__ import annotations

import hashlib
import platform
import urllib.parse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


def _sha256_files(root: Path, candidates: list[Path]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for candidate in sorted(candidates):
        try:
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.stat().st_size > 128_000:
                continue
            relative = resolved.relative_to(root).as_posix()
            content = resolved.read_bytes()
        except OSError:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count


def corpus_fingerprint(corpus_dir: Path) -> tuple[str, int]:
    """Return the same stable content hash and file count used by run manifests."""
    root = corpus_dir.resolve(strict=True)
    candidates = [
        path for path in root.rglob("*")
        if path.suffix.lower() in {".md", ".txt"} and path.is_file()
    ]
    return _sha256_files(root, candidates)


def _safe_endpoint(provider: Any) -> str | None:
    base_url = getattr(provider, "base_url", None)
    if not isinstance(base_url, str):
        return None
    try:
        parsed = urllib.parse.urlsplit(base_url)
        host = parsed.hostname
        if not host:
            return "<invalid-endpoint>"
        try:
            parsed.port
        except ValueError:
            return "<invalid-endpoint>"
        locality = "loopback-endpoint" if not getattr(provider, "is_remote", False) else "remote-endpoint"
        # Fingerprint normalized origin/path without publishing any URL component or query value.
        canonical = "\0".join((
            parsed.scheme.lower(),
            (host or "").lower().rstrip("."),
            str(parsed.port or ""),
            parsed.path.rstrip("/"),
        ))
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{parsed.scheme.lower()}://<{locality}>#sha256:{fingerprint}"
    except ValueError:
        return "<invalid-endpoint>"


def build_run_manifest(
    *,
    provider: Any,
    corpus_dir: Path,
    limits: Any,
    mode: str,
    validation_contract: dict[str, Any] | None,
    max_cost_usd: float | None,
    input_rate: float | None,
    output_rate: float | None,
) -> dict[str, Any]:
    corpus_root = corpus_dir.resolve(strict=True)
    corpus_hash, corpus_file_count = corpus_fingerprint(corpus_root)
    prompt_file = Path(__file__).with_name("prompts.py")
    prompt_hash, _ = _sha256_files(prompt_file.parent, [prompt_file])
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "harness_version": __version__,
        "python_version": platform.python_version(),
        "provider": getattr(provider, "name", type(provider).__name__),
        "model": getattr(provider, "model", None),
        "response_format": getattr(provider, "response_format", None),
        "endpoint": _safe_endpoint(provider),
        "mode": mode,
        "temperature": getattr(provider, "temperature", None),
        "corpus_sha256": corpus_hash,
        "corpus_file_count": corpus_file_count,
        "prompt_sha256": prompt_hash,
        "harness_source_sha256": _sha256_files(
            Path(__file__).parent,
            list(Path(__file__).parent.rglob("*.py")),
        )[0],
        "limits": asdict(limits),
        "validation_contract": validation_contract,
        "estimated_remote_cost_cap_usd": max_cost_usd,
        "remote_rates_usd_per_1m_tokens": (
            {"input": input_rate, "output": output_rate}
            if input_rate is not None and output_rate is not None
            else None
        ),
    }
