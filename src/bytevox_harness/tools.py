"""The only tools available to agents: bounded, read-only corpus access."""

from __future__ import annotations

import io
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .budget import RunBudget
from .trace import TraceWriter

ROLE_TOOLS: dict[str, tuple[str, ...]] = {
    "lead_planner": ("list_documents", "search_documents"),
    "worker_causes": ("search_documents",),
    "worker_outcomes": ("list_documents", "search_documents"),
    "worker_risks": ("search_documents", "read_evidence"),
    "lead_synthesizer": (),
    "checker": ("read_evidence",),
    "single_agent": ("list_documents", "search_documents", "read_evidence"),
}

MAX_CORPUS_FILE_BYTES = 128_000
MAX_CORPUS_FILES = 10_000
MAX_CORPUS_ENTRIES = 50_000
MAX_CORPUS_TOTAL_BYTES = 8 * 1024 * 1024
MAX_CORPUS_SECTIONS = 10_000
MAX_CORPUS_INDEX_TOKENS = 250_000

INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override_language", re.compile(r"\b(system override|ignore (all )?instructions|developer message)\b", re.I)),
    ("secret_request", re.compile(r"\b(reveal|print|exfiltrate)\b.{0,40}\b(environment variables|api keys|secrets)\b", re.I)),
    ("unsafe_write_request", re.compile(r"\b(write|save|create)\b.{0,60}\b(outside|parent directory|approved folder)\b", re.I)),
)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    document: str
    section: str
    text: str
    score: float = 0.0
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "document": self.document,
            "section": self.section,
            "text": self.text,
            "score": self.score,
            "title": self.title,
        }


_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)
_LINE_BREAK_PATTERN = re.compile(r"\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]")
_STOP_WORDS = frozenset({
    "a", "about", "after", "again", "against", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "between", "both", "but", "by", "can", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
})


def _normalize_token(token: str) -> str:
    """Apply a small English suffix normalization without altering short names/acronyms."""
    if not token.isascii():
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 6 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 5 and token.endswith("es") and not token.endswith(("ses", "xes")):
        return token[:-2]
    if len(token) > 5 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Unicode-aware search tokens; keep short entity terms while dropping common stopwords."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    tokens: list[str] = []
    for raw in _WORD_PATTERN.findall(normalized):
        token = raw.strip("'’")
        if not token or token in _STOP_WORDS:
            continue
        tokens.append(_normalize_token(token))
    return tokens


def _iter_text_lines(text: str) -> Iterator[str]:
    """Yield splitlines-compatible lines without allocating a list of them."""
    start = 0
    for match in _LINE_BREAK_PATTERN.finditer(text):
        yield text[start:match.start()]
        start = match.end()
    if start < len(text):
        yield text[start:]


class CorpusTools:
    def __init__(self, root: Path, trace: TraceWriter, budget: RunBudget) -> None:
        self.root = root.resolve(strict=True)
        self.trace = trace
        self.budget = budget
        self._evidence = self._load_corpus()
        self._search_index, self._document_frequency, self._average_document_length = self._build_search_index()

    def _build_search_index(self) -> tuple[list[dict[str, Any]], Counter[str], float]:
        """Precompute weighted field term frequencies and document frequency for BM25."""
        index: list[dict[str, Any]] = []
        document_frequency: Counter[str] = Counter()
        weighted_lengths: list[float] = []
        index_token_count = 0

        def field_counter(text: str) -> Counter[str]:
            nonlocal index_token_count
            tokens = tokenize(text)
            index_token_count += len(tokens)
            if index_token_count > MAX_CORPUS_INDEX_TOKENS:
                raise ValueError(
                    f"Corpus BM25 index token limit of {MAX_CORPUS_INDEX_TOKENS:,} was exceeded under {self.root}"
                )
            return Counter(tokens)

        for item in self._evidence:
            fields = {
                "body": field_counter(item.text),
                "title": field_counter(item.title),
                "section": field_counter(item.section.replace("-", " ")),
                "document": field_counter(item.document.replace("-", " ")),
            }
            all_terms = set().union(*(set(counts) for counts in fields.values()))
            document_frequency.update(all_terms)
            weighted_length = (
                sum(fields["body"].values())
                + 3.0 * sum(fields["title"].values())
                + 1.5 * sum(fields["section"].values())
                + 0.5 * sum(fields["document"].values())
            )
            weighted_lengths.append(weighted_length)
            index.append({"fields": fields, "length": weighted_length})
        average_length = sum(weighted_lengths) / len(weighted_lengths) if weighted_lengths else 0.0
        return index, document_frequency, average_length

    def _rank(self, query: str) -> list[Evidence]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        count = len(self._evidence)
        k1 = 1.5
        b = 0.75
        ranked: list[Evidence] = []
        field_weights = {"body": 1.0, "title": 3.0, "section": 1.5, "document": 0.5}
        for item, indexed in zip(self._evidence, self._search_index):
            score = 0.0
            length_norm = (1.0 - b) + b * indexed["length"] / max(self._average_document_length, 1.0)
            for term in query_terms:
                df = self._document_frequency.get(term, 0)
                if not df:
                    continue
                inverse_document_frequency = math.log1p((count - df + 0.5) / (df + 0.5))
                weighted_tf = sum(
                    field_weights[field] * indexed["fields"][field].get(term, 0)
                    for field in field_weights
                )
                if weighted_tf:
                    score += inverse_document_frequency * (
                        weighted_tf * (k1 + 1.0) / (weighted_tf + k1 * length_norm)
                    )
            if score > 0:
                ranked.append(Evidence(
                    item.evidence_id,
                    item.document,
                    item.section,
                    item.text,
                    score,
                    title=item.title,
                ))
        ranked.sort(key=lambda item: (-item.score, item.evidence_id))
        return ranked

    def _iter_corpus_paths(self) -> Iterator[Path]:
        """Walk in sorted depth-first path order with bounded traversal state."""
        pending: list[tuple[Path, bool]] = [(self.root, True)]
        entries_seen = 0
        while pending:
            candidate, is_directory = pending.pop()
            if not is_directory:
                yield candidate
                continue

            children: list[tuple[Path, bool]] = []
            try:
                with os.scandir(candidate) as scanned:
                    for entry in scanned:
                        entries_seen += 1
                        if entries_seen > MAX_CORPUS_ENTRIES:
                            raise ValueError(
                                f"Corpus entry limit of {MAX_CORPUS_ENTRIES:,} was exceeded under {self.root}"
                            )
                        try:
                            child_is_directory = entry.is_dir(follow_symlinks=False)
                        except OSError as exc:
                            raise ValueError(f"Could not inspect corpus entry {entry.path}: {exc}") from exc
                        children.append((Path(entry.path), child_is_directory))
            except ValueError:
                raise
            except OSError as exc:
                raise ValueError(f"Could not traverse corpus directory {candidate}: {exc}") from exc

            # Sorting Path objects preserves the platform's Path ordering used by
            # the previous sorted(root.rglob('*')) traversal.
            children.sort(key=lambda child: child[0])
            pending.extend(reversed(children))

    def _load_corpus(self) -> list[Evidence]:
        found: list[Evidence] = []
        eligible_files = 0
        total_bytes = 0

        def append_evidence(item: Evidence) -> None:
            if len(found) >= MAX_CORPUS_SECTIONS:
                raise ValueError(
                    f"Corpus evidence section limit of {MAX_CORPUS_SECTIONS:,} was exceeded under {self.root}"
                )
            found.append(item)

        for candidate in self._iter_corpus_paths():
            if candidate.suffix.lower() not in {".md", ".txt"} or not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(self.root):
                continue
            size = resolved.stat().st_size
            if size > MAX_CORPUS_FILE_BYTES:
                continue
            eligible_files += 1
            if eligible_files > MAX_CORPUS_FILES:
                raise ValueError(
                    f"Corpus eligible .md/.txt file limit of {MAX_CORPUS_FILES:,} was exceeded under {self.root}"
                )
            remaining_bytes = MAX_CORPUS_TOTAL_BYTES - total_bytes
            if size > remaining_bytes:
                raise ValueError(
                    f"Aggregate corpus text limit of {MAX_CORPUS_TOTAL_BYTES:,} bytes would be exceeded under {self.root}"
                )
            relative = resolved.relative_to(self.root)
            try:
                read_limit = min(MAX_CORPUS_FILE_BYTES + 1, remaining_bytes + 1)
                with resolved.open("rb") as source:
                    raw_bytes = source.read(read_limit)
            except OSError as exc:
                raise ValueError(f"Could not read corpus file {relative}: {exc}") from exc
            if len(raw_bytes) > remaining_bytes:
                raise ValueError(
                    f"Aggregate corpus text limit of {MAX_CORPUS_TOTAL_BYTES:,} bytes was exceeded under {self.root}"
                )
            if len(raw_bytes) > MAX_CORPUS_FILE_BYTES:
                continue
            total_bytes += len(raw_bytes)
            try:
                raw = raw_bytes.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Corpus file {relative} is not valid UTF-8 (invalid byte sequence at offset {exc.start})"
                ) from exc
            current = "overview"
            document_title = ""
            section_text = io.StringIO()
            has_section_text = False
            for line in _iter_text_lines(raw):
                if line.startswith("## "):
                    text = section_text.getvalue()
                    if text:
                        append_evidence(Evidence(
                            f"{relative.stem}#{current}",
                            relative.as_posix(),
                            current,
                            text,
                            title=document_title,
                        ))
                    heading = line[3:].strip()
                    current = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") or "section"
                    section_text = io.StringIO()
                    has_section_text = False
                elif line.startswith("# "):
                    document_title = line[2:].strip()
                    continue
                else:
                    piece = line.strip()
                    if piece:
                        if has_section_text:
                            section_text.write(" ")
                        section_text.write(piece)
                        has_section_text = True
            text = section_text.getvalue()
            if text:
                append_evidence(Evidence(
                    f"{relative.stem}#{current}",
                    relative.as_posix(),
                    current,
                    text,
                    title=document_title,
                ))
        if not found:
            raise ValueError(f"No readable .md or .txt documents found under {self.root}")
        base_id_counts = Counter(item.evidence_id for item in found)
        duplicate_sections = Counter((item.document, item.section) for item in found)
        duplicate_occurrences: Counter[tuple[str, str]] = Counter()
        unique: list[Evidence] = []
        for item in found:
            if base_id_counts[item.evidence_id] == 1:
                unique.append(item)
                continue

            # Keep the legacy basename#slug format for unambiguous sections. When
            # basenames collide, include the relative document path; when the same
            # document has repeated normalized headings, add a stable occurrence.
            key = (item.document, item.section)
            duplicate_occurrences[key] += 1
            evidence_id = f"{item.document}#{item.section}"
            if duplicate_sections[key] > 1:
                evidence_id += f"~{duplicate_occurrences[key]}"
            unique.append(Evidence(
                evidence_id,
                item.document,
                item.section,
                item.text,
                item.score,
                item.title,
            ))
        return unique

    def _allowed(self, agent_id: str, tool: str) -> None:
        if tool not in ROLE_TOOLS.get(agent_id, ()):
            self.trace.emit("tool_rejected", agent_id=agent_id, tool=tool, reason="not in role allowlist")
            raise PermissionError(f"{agent_id} is not allowed to call {tool}")

    def _begin_tool(self, agent_id: str, tool: str, arguments: dict[str, Any]) -> None:
        self._allowed(agent_id, tool)
        try:
            reservation = self.budget.begin_step(agent_id)
        except Exception as exc:
            self.trace.emit("budget_stop", agent_id=agent_id, stage=f"tool:{tool}", reason=str(exc))
            raise
        self.trace.emit("tool_call", agent_id=agent_id, tool=tool, arguments=arguments)
        return reservation

    def _finish_tool(self, agent_id: str, tool: str, arguments: dict[str, Any], result: Any) -> Any:
        self.trace.emit("tool_result", agent_id=agent_id, tool=tool, arguments=arguments, result=result)
        source_ids: list[str] = []
        matched_pattern_names: list[str] = []
        if isinstance(result, list):
            for item in result:
                # Search each evidence record independently so an injected passage does
                # not make every neighboring result appear hostile in the trace.
                text = str(item.get("text", "")) if isinstance(item, dict) else str(item)
                patterns = [name for name, pattern in INJECTION_PATTERNS if pattern.search(text)]
                if patterns:
                    matched_pattern_names.extend(patterns)
                    if isinstance(item, dict) and item.get("evidence_id"):
                        source_ids.append(str(item["evidence_id"]))
        else:
            # Preserve detection behavior for non-list tool results.
            text = str(result)
            matched_pattern_names = [name for name, pattern in INJECTION_PATTERNS if pattern.search(text)]
            if matched_pattern_names and isinstance(result, dict) and result.get("evidence_id"):
                source_ids.append(str(result["evidence_id"]))

        patterns = list(dict.fromkeys(matched_pattern_names))
        source_ids = list(dict.fromkeys(source_ids))
        if patterns:
            self.trace.emit(
                "untrusted_content_detected",
                agent_id=agent_id,
                source_ids=source_ids,
                pattern_names=patterns,
                action="kept_as_data; no write, environment, shell, or network tool is available",
            )
        return result

    def list_documents(self, agent_id: str) -> list[str]:
        tool, args = "list_documents", {}
        self._begin_tool(agent_id, tool, args)
        result = sorted({item.document for item in self._evidence})
        return self._finish_tool(agent_id, tool, args, result)

    def search_documents(self, agent_id: str, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        tool = "search_documents"
        query = str(query)[:300]
        limit = max(1, min(int(limit), 15))
        args = {"query": query, "limit": limit}
        self._begin_tool(agent_id, tool, args)
        hits = self._rank(query)
        result = [item.to_dict() for item in hits[:limit]]
        return self._finish_tool(agent_id, tool, args, result)

    def read_evidence(self, agent_id: str, evidence_id: str) -> dict[str, Any] | None:
        tool, args = "read_evidence", {"evidence_id": str(evidence_id)}
        self._begin_tool(agent_id, tool, args)
        result = next((item.to_dict() for item in self._evidence if item.evidence_id == args["evidence_id"]), None)
        return self._finish_tool(agent_id, tool, args, result)

    def all_evidence_for_benchmark(self) -> list[dict[str, Any]]:
        """Internal evaluator view; this is not exposed as an agent tool."""
        return [item.to_dict() for item in self._evidence]
