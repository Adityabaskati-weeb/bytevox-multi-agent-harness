"""Local-only HotpotQA dev-distractor loading and scorer-only labels."""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .manifest import corpus_fingerprint

HOTPOTQA_VERSION = "hotpot_dev_distractor_v1"
HOTPOTQA_SPLIT = "dev distractor"
HOTPOTQA_SOURCE_URL = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HOTPOTQA_HOME_URL = "https://hotpotqa.github.io/"
HOTPOTQA_EVALUATOR_URL = "https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py"
HOTPOTQA_LICENSE = "CC BY-SA 4.0"
HOTPOTQA_MAX_FILE_BYTES = 128_000
_PUNCTUATION = set(string.punctuation)


@dataclass(frozen=True)
class HotpotQAScoringRecord:
    """Reference labels held for evaluation after the harness run."""

    answer: str
    supporting_facts: tuple[tuple[str, int], ...]
    supporting_evidence_ids: tuple[str, ...]
    evidence_id_to_fact: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True)
class HotpotQATask:
    """Agent-visible case data separated from its evaluator-only record."""

    task_id: str
    dataset_id: str
    question: str
    corpus_dir: Path
    corpus_sha256: str
    scoring: HotpotQAScoringRecord


@dataclass(frozen=True)
class HotpotQADataset:
    """A deterministic sample and the provenance needed to reproduce it."""

    tasks: tuple[HotpotQATask, ...]
    provenance: dict[str, Any]


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"HotpotQA record {label} must be a non-empty string")
    return value.strip()


def _context_items(value: Any) -> list[tuple[str, list[str]]]:
    """Read the official list schema and the equivalent columnar JSON form."""
    if isinstance(value, dict):
        titles = value.get("title")
        sentences = value.get("sentences")
        if not isinstance(titles, list) or not isinstance(sentences, list) or len(titles) != len(sentences):
            raise ValueError("HotpotQA context object must have equally sized title and sentences arrays")
        pairs = list(zip(titles, sentences))
    elif isinstance(value, list):
        pairs = value
    else:
        raise ValueError("HotpotQA context must be an array of [title, sentences] pairs")

    normalized: list[tuple[str, list[str]]] = []
    titles_seen: set[str] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"HotpotQA context[{index}] must contain a title and sentence array")
        title = _required_text(pair[0], f"context[{index}].title")
        if title in titles_seen:
            raise ValueError(f"HotpotQA context contains duplicate title {title!r}")
        titles_seen.add(title)
        sentences = pair[1]
        if not isinstance(sentences, list) or not sentences:
            raise ValueError(f"HotpotQA context[{index}].sentences must be a non-empty array")
        normalized_sentences = [
            _required_text(sentence, f"context[{index}].sentences[{sentence_index}]")
            for sentence_index, sentence in enumerate(sentences)
        ]
        normalized.append((title, normalized_sentences))
    if not normalized:
        raise ValueError("HotpotQA context must contain at least one paragraph")
    return normalized


def _supporting_facts(value: Any) -> tuple[tuple[str, int], ...]:
    if isinstance(value, dict):
        titles = value.get("title")
        sentence_ids = value.get("sent_id")
        if not isinstance(titles, list) or not isinstance(sentence_ids, list) or len(titles) != len(sentence_ids):
            raise ValueError("HotpotQA supporting_facts object must have matching title and sent_id arrays")
        pairs = list(zip(titles, sentence_ids))
    elif isinstance(value, list):
        pairs = value
    else:
        raise ValueError("HotpotQA supporting_facts must be an array of [title, sent_id] pairs")

    result: list[tuple[str, int]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"HotpotQA supporting_facts[{index}] must contain title and sentence id")
        title = _required_text(pair[0], f"supporting_facts[{index}].title")
        sentence_id = pair[1]
        if isinstance(sentence_id, bool) or not isinstance(sentence_id, int) or sentence_id < 0:
            raise ValueError(f"HotpotQA supporting_facts[{index}].sent_id must be a non-negative integer")
        result.append((title, sentence_id))
    if not result:
        raise ValueError("HotpotQA record must contain at least one supporting fact")
    return tuple(dict.fromkeys(result))


def _title_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")[:48]
    return slug or "document"


def _sentence_evidence_id(paragraph_index: int, sentence_index: int, title: str) -> str:
    stem = f"doc-{paragraph_index:02d}-s-{sentence_index:03d}-{_title_slug(title)}"
    return f"{stem}#sentence"


def _public_source_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source_url must be a non-empty public HTTPS URL")
    source_url = value.strip()
    parsed = urlsplit(source_url)
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("source_url must be a public HTTPS URL without credentials, query, or fragment")
    return source_url


def load_hotpotqa_dev_distractor(
    dataset_path: Path,
    *,
    corpora_root: Path,
    sample_size: int = 3,
    source_url: str = HOTPOTQA_SOURCE_URL,
    sample_seed: int | None = None,
    excluded_ids: set[str] | None = None,
) -> HotpotQADataset:
    """Load a caller-provided dataset file; this function performs no network I/O."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    source_url = _public_source_url(source_url)
    source = dataset_path.expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("HotpotQA dataset path must point to a local JSON file")
    if source.stat().st_size > 80_000_000:
        raise ValueError("HotpotQA dataset JSON exceeds the 80 MB local input limit")
    raw = source.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        records = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("HotpotQA dataset path must contain valid UTF-8 JSON") from exc
    if not isinstance(records, list) or not records:
        raise ValueError("HotpotQA dataset JSON must contain a non-empty array of records")

    if sample_seed is not None and (isinstance(sample_seed, bool) or not isinstance(sample_seed, int)):
        raise ValueError("sample_seed must be an integer or None")
    exclusion = set(excluded_ids or ())
    if any(not isinstance(dataset_id, str) or not dataset_id.strip() for dataset_id in exclusion):
        raise ValueError("excluded_ids must contain non-empty strings")
    eligible_indices = [
        index
        for index, record in enumerate(records)
        if not isinstance(record, dict)
        or not isinstance(record.get("_id"), str)
        or record.get("_id") not in exclusion
    ]
    if len(eligible_indices) < sample_size:
        raise ValueError(
            f"requested sample size {sample_size} exceeds {len(eligible_indices)} records left after exclusions"
        )
    selected_indices = _select_sample_indices(
        records,
        sample_size,
        seed=sample_seed,
        eligible_indices=eligible_indices,
    )
    selected = [records[index] for index in selected_indices]
    root = corpora_root.expanduser().resolve()
    dataset_root = root / source_sha256[:16]
    tasks: list[HotpotQATask] = []
    dataset_ids: set[str] = set()
    for sample_index, record in enumerate(selected, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"HotpotQA record {sample_index} must be an object")
        dataset_id = _required_text(record.get("_id"), f"record[{sample_index}]._id")
        if dataset_id in dataset_ids:
            raise ValueError(f"HotpotQA sample contains duplicate _id {dataset_id!r}")
        dataset_ids.add(dataset_id)
        question = _required_text(record.get("question"), f"record[{sample_index}].question")
        answer = _required_text(record.get("answer"), f"record[{sample_index}].answer")
        contexts = _context_items(record.get("context"))
        gold_facts = _supporting_facts(record.get("supporting_facts"))

        title_to_position = {title: index for index, (title, _) in enumerate(contexts)}
        evidence_id_to_fact: list[tuple[str, str, int]] = []
        evidence_by_fact: dict[tuple[str, int], str] = {}
        generated_files: list[tuple[str, bytes]] = []
        for paragraph_index, (title, sentences) in enumerate(contexts):
            for sentence_index, sentence in enumerate(sentences):
                evidence_id = _sentence_evidence_id(paragraph_index, sentence_index, title)
                body = " ".join(sentence.splitlines()).strip()
                # Keep Markdown headings in source text from changing the evidence boundary.
                if body.startswith("#"):
                    body = "\\" + body
                markdown = f"# {title.replace(chr(10), ' ').replace(chr(13), ' ')}\n\n## sentence\n{body}\n"
                encoded = markdown.encode("utf-8")
                if len(encoded) > HOTPOTQA_MAX_FILE_BYTES:
                    raise ValueError(
                        f"HotpotQA sentence {title!r}/{sentence_index} exceeds the harness 128 KB evidence-file limit"
                    )
                filename = evidence_id.split("#", 1)[0] + ".md"
                generated_files.append((filename, encoded))
                evidence_id_to_fact.append((evidence_id, title, sentence_index))
                evidence_by_fact[(title, sentence_index)] = evidence_id

        for title, sentence_index in gold_facts:
            paragraph_index = title_to_position.get(title)
            if paragraph_index is None:
                raise ValueError(f"HotpotQA support fact references missing context title {title!r}")
            if sentence_index >= len(contexts[paragraph_index][1]):
                raise ValueError(f"HotpotQA support fact references missing sentence {title!r}/{sentence_index}")

        gold_evidence_ids = tuple(evidence_by_fact[fact] for fact in gold_facts)
        expected_corpus_digest = hashlib.sha256()
        for filename, content in sorted(generated_files):
            expected_corpus_digest.update(filename.encode("utf-8"))
            expected_corpus_digest.update(b"\0")
            expected_corpus_digest.update(content)
            expected_corpus_digest.update(b"\0")
        expected_corpus_sha256 = expected_corpus_digest.hexdigest()
        record_key = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:16]
        content_key = expected_corpus_sha256[:16]
        case_dir = dataset_root / f"case-{record_key}-{content_key}"
        case_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in generated_files:
            (case_dir / filename).write_bytes(content)
        corpus_sha256, file_count = corpus_fingerprint(case_dir)
        if corpus_sha256 != expected_corpus_sha256 or file_count != len(generated_files):
            raise RuntimeError(f"HotpotQA corpus directory contains stale or missing evidence for {dataset_id!r}")
        tasks.append(HotpotQATask(
            task_id=f"hotpotqa-{sample_index:04d}",
            dataset_id=dataset_id,
            question=question,
            corpus_dir=case_dir,
            corpus_sha256=corpus_sha256,
            scoring=HotpotQAScoringRecord(
                answer=answer,
                supporting_facts=gold_facts,
                supporting_evidence_ids=gold_evidence_ids,
                evidence_id_to_fact=tuple(evidence_id_to_fact),
            ),
        ))

    if not tasks:
        raise ValueError("HotpotQA sample contains no tasks")
    corpus_digest = hashlib.sha256()
    for task in tasks:
        corpus_digest.update(task.task_id.encode("utf-8"))
        corpus_digest.update(b"\0")
        corpus_digest.update(task.corpus_sha256.encode("ascii"))
        corpus_digest.update(b"\0")
    exclusion_digest = hashlib.sha256("\n".join(sorted(exclusion)).encode("utf-8")).hexdigest()
    exclusion_rule = (
        "exclude supplied prior IDs, then "
        if exclusion
        else "without prior-ID exclusions, "
    )
    provenance = {
        "dataset": "HotpotQA",
        "dataset_version": HOTPOTQA_VERSION,
        "split": HOTPOTQA_SPLIT,
        "source_url": source_url,
        "source_homepage": HOTPOTQA_HOME_URL,
        "official_evaluator_url": HOTPOTQA_EVALUATOR_URL,
        "license": HOTPOTQA_LICENSE,
        "local_source_file": source.name,
        "source_sha256": source_sha256,
        "requested_sample_size": sample_size,
        "sample_seed": sample_seed,
        "excluded_sample_id_count": len(exclusion),
        "excluded_sample_ids_sha256": exclusion_digest,
        "eligible_record_count": len(eligible_indices),
        "sample_count": len(tasks),
        "sample_ids": [task.dataset_id for task in tasks],
        "sample_types": [str(records[index].get("type", "unknown")) for index in selected_indices],
        "sample_source_indices": selected_indices,
        "sample_selection_rule": (
            f"{exclusion_rule}select first comparison and first bridge, then fill remaining slots in source order"
            if sample_seed is None
            else f"{exclusion_rule}use proportional stratified allocation by record type using Hamilton's largest remainder; "
                 "sample within each type using the recorded seed, then report original source order"
        ),
        "corpus_sha256": corpus_digest.hexdigest(),
        "corpus_sha256_by_task": {task.task_id: task.corpus_sha256 for task in tasks},
    }
    return HotpotQADataset(tuple(tasks), provenance)


def _select_sample_indices(
    records: list[Any],
    sample_size: int,
    *,
    seed: int | None = None,
    eligible_indices: list[int] | None = None,
) -> list[int]:
    candidates = eligible_indices if eligible_indices is not None else list(range(len(records)))
    if seed is not None:
        groups: dict[str, list[int]] = {}
        for index in candidates:
            record = records[index]
            label = str(record.get("type", "unknown")) if isinstance(record, dict) else "unknown"
            groups.setdefault(label, []).append(index)
        target_size = min(sample_size, len(candidates))
        exact = {label: target_size * len(indices) / len(candidates) for label, indices in groups.items()}
        allocation = {label: int(value) for label, value in exact.items()}
        remaining = target_size - sum(allocation.values())
        rng = random.Random(seed)
        ties = sorted(groups)
        rng.shuffle(ties)
        tie_order = {label: index for index, label in enumerate(ties)}
        for label in sorted(groups, key=lambda item: (-(exact[item] - allocation[item]), tie_order[item]))[:remaining]:
            allocation[label] += 1
        selected: list[int] = []
        for label in sorted(groups):
            take = allocation[label]
            if take:
                selected.extend(rng.sample(groups[label], take))
        return sorted(selected)

    if sample_size == 1:
        return [candidates[0]]
    selected: list[int] = []
    for example_type in ("comparison", "bridge"):
        match = None
        for index in candidates:
            record = records[index]
            if isinstance(record, dict) and record.get("type") == example_type:
                match = index
                break
        if match is not None and match not in selected and len(selected) < sample_size:
            selected.append(match)
    for index in candidates:
        if index not in selected:
            selected.append(index)
        if len(selected) >= sample_size:
            break
    return selected


def normalize_hotpotqa_answer(text: str) -> str:
    """Match the normalization used by the official HotpotQA answer scorer."""
    lowered = str(text).lower()
    no_punctuation = "".join(character for character in lowered if character not in _PUNCTUATION)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def _answer_f1(prediction: str, gold: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_hotpotqa_answer(prediction)
    normalized_gold = normalize_hotpotqa_answer(gold)
    special = {"yes", "no", "noanswer"}
    if normalized_prediction in special or normalized_gold in special:
        if normalized_prediction == normalized_gold:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0
    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    matched = sum(common.values())
    if not matched:
        return 0.0, 0.0, 0.0
    precision = matched / len(prediction_tokens)
    recall = matched / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def score_hotpotqa(
    answer: dict[str, Any],
    scoring: HotpotQAScoringRecord,
) -> dict[str, Any]:
    """Score an already completed harness answer; reference labels are never inputs."""
    predicted_answer, answer_parse_success = extract_hotpotqa_answer(str(answer.get("summary", "")))
    normalized_prediction = normalize_hotpotqa_answer(predicted_answer)
    normalized_gold = normalize_hotpotqa_answer(scoring.answer)
    answer_em = float(normalized_prediction == normalized_gold)
    answer_f1, answer_precision, answer_recall = _answer_f1(predicted_answer, scoring.answer)

    id_to_fact = {
        evidence_id: (title, sentence_index)
        for evidence_id, title, sentence_index in scoring.evidence_id_to_fact
    }
    predicted_facts: set[tuple[str, int] | tuple[str, str]] = set()
    invalid_citations = 0
    claims = answer.get("claims", [])
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            evidence_ids = claim.get("evidence_ids", [])
            if not isinstance(evidence_ids, list):
                continue
            for evidence_id in evidence_ids:
                if not isinstance(evidence_id, str):
                    continue
                fact = id_to_fact.get(evidence_id)
                if fact is not None:
                    predicted_facts.add(fact)
                else:
                    invalid_citations += 1
                    # Keep an invalid citation as a false positive, as in the official scorer.
                    predicted_facts.add(("__invalid_citation__", evidence_id))

    gold_facts = set(scoring.supporting_facts)
    matched = len(predicted_facts & gold_facts)
    false_positive = len(predicted_facts - gold_facts)
    false_negative = len(gold_facts - predicted_facts)
    support_precision = matched / (matched + false_positive) if matched + false_positive else 0.0
    support_recall = matched / (matched + false_negative) if matched + false_negative else 0.0
    support_f1 = (
        2 * support_precision * support_recall / (support_precision + support_recall)
        if support_precision + support_recall
        else 0.0
    )
    support_em = float(predicted_facts == gold_facts)
    joint_precision = answer_precision * support_precision
    joint_recall = answer_recall * support_recall
    joint_f1 = (
        2 * joint_precision * joint_recall / (joint_precision + joint_recall)
        if joint_precision + joint_recall
        else 0.0
    )
    joint_em = answer_em * support_em
    return {
        "predicted_short_answer": predicted_answer,
        "answer_parse_success": answer_parse_success,
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "answer_precision": answer_precision,
        "answer_recall": answer_recall,
        "supporting_fact_em": support_em,
        "supporting_fact_precision": support_precision,
        "supporting_fact_recall": support_recall,
        "supporting_fact_f1": support_f1,
        "joint_em": joint_em,
        "joint_f1": joint_f1,
        "gold_supporting_fact_count": len(gold_facts),
        "predicted_supporting_fact_count": len(predicted_facts),
        "matched_supporting_fact_count": matched,
        "invalid_citation_count": invalid_citations,
    }


def score_retrieval_support_recall(
    trace_path: Path,
    scoring: HotpotQAScoringRecord,
) -> dict[str, Any]:
    """Score how many gold support sentences were retrieved, using only a closed run trace."""
    retrieved_ids: set[str] = set()
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "tool_result" or event.get("tool") != "search_documents":
            continue
        result = event.get("result")
        if not isinstance(result, list):
            continue
        retrieved_ids.update(
            item["evidence_id"]
            for item in result
            if isinstance(item, dict) and isinstance(item.get("evidence_id"), str)
        )
    gold_ids = set(scoring.supporting_evidence_ids)
    found_gold_ids = retrieved_ids & gold_ids
    return {
        "retrieval_support_recall": len(found_gold_ids) / len(gold_ids) if gold_ids else 0.0,
        "retrieved_supporting_fact_count": len(found_gold_ids),
        "retrieved_evidence_count": len(retrieved_ids),
    }


def extract_hotpotqa_answer(summary: str) -> tuple[str, bool]:
    """Read the label-free first-line answer contract used by the public pilot."""
    first_line = str(summary).strip().splitlines()[0] if str(summary).strip() else ""
    match = re.match(r"^\s*answer\s*:\s*(.*?)\s*$", first_line, flags=re.IGNORECASE)
    if not match or not match.group(1).strip():
        return "", False
    return match.group(1).strip(), True
