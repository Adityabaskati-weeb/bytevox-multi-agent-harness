"""Three repeatable evidence tasks and a narrow, transparent fixture score."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .budget import Limits
from .harness import Harness, RunResult, make_validation_contract
from .hotpotqa import HotpotQADataset, HotpotQATask, score_hotpotqa, score_retrieval_support_recall
from .providers import OfflineProvider


HOTPOTQA_BUDGET_POLICY = "equal-run-envelope-v1"


def _hotpotqa_mode_limits(run_limits: Limits) -> dict[str, Limits]:
    """Give both HotpotQA arms the same run envelope while retaining multi role caps."""
    return {
        "multi": run_limits,
        "single": replace(
            run_limits,
            max_agent_steps=run_limits.max_steps,
            max_agent_seconds=run_limits.max_seconds,
            max_agent_tokens=run_limits.max_tokens,
            max_completion_tokens=run_limits.max_lead_completion_tokens,
        ),
    }


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    question: str
    gold_evidence_ids: tuple[str, ...]
    gold_terms: tuple[str, ...]
    validation_contract: tuple[str, str, str]


SAMPLE_TASKS = (
    BenchmarkTask(
        "delay-causes",
        "What caused most North Harbor pilot delays, and which operational changes were tried to address them?",
        ("operations#delay-causes", "operations#interventions"),
        ("38 percent", "27 percent", "all-door boarding", "four-minute"),
        (
            "The main documented causes of North Harbor pilot delays, including quantitative attribution where reported",
            "Operational changes tried or proposed to address those causes",
            "Evidence about how the changes relate to delay patterns or results, including caveats",
        ),
    ),
    BenchmarkTask(
        "pilot-results",
        "How did the pilot's on-time performance change, what other result was measured, and what caveat limits the conclusion?",
        ("operations#pilot", "operations#results"),
        ("82 percent", "88 percent", "22 seconds", "not randomized"),
        (
            "The pilot's before-and-after on-time performance, including reported percentages",
            "Other measured pilot results",
            "Caveats that limit causal or general conclusions",
        ),
    ),
    BenchmarkTask(
        "incident-and-risk",
        "What happened in the ramp incident, what was changed afterward, and what accessibility risk remains open?",
        ("incident_review#ramp-incident", "incident_review#follow-up", "risk_register#remaining-risk"),
        ("no one was injured", "checklist", "wheelchair users", "46 responses"),
        (
            "What happened during the ramp incident and its documented impact",
            "Changes made after the incident",
            "Accessibility risks that remain open and their supporting evidence",
        ),
    ),
)

HOT_POTQA_FACETS = (
    "Answer the question using the supplied document sentences.",
    "Support the answer with the sentence evidence IDs that are relevant to it.",
    "State uncertainty or missing information when the provided evidence is insufficient.",
)


def evaluate(
    result: RunResult,
    task: BenchmarkTask,
    valid_evidence_ids: set[str] | dict[str, dict[str, Any]],
    *,
    repetition: int = 1,
) -> dict[str, Any]:
    try:
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    claims = result.answer.get("claims", [])
    used_ids: list[str] = []
    evidence_by_id = valid_evidence_ids if isinstance(valid_evidence_ids, dict) else {}
    valid_ids = set(evidence_by_id) if evidence_by_id else set(valid_evidence_ids)
    supported_claims: list[tuple[str, list[str]]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_text = str(claim.get("text", ""))
        ids = claim.get("evidence_ids", [])
        if isinstance(ids, list):
            used_ids.extend(str(evidence_id) for evidence_id in ids)
            if any(str(evidence_id) in valid_ids for evidence_id in ids):
                supported_claims.append((claim_text, [str(evidence_id) for evidence_id in ids]))
    used_unique = set(used_ids)
    covered = set(task.gold_evidence_ids) & used_unique
    evidence_coverage = len(covered) / len(task.gold_evidence_ids) if task.gold_evidence_ids else 1.0
    citation_id_validity = (
        sum(evidence_id in valid_ids for evidence_id in used_ids) / len(used_ids)
        if used_ids
        else 0.0
    )
    def term_is_cited(term: str) -> bool:
        needle = term.casefold()
        for claim_text, evidence_ids in supported_claims:
            if needle not in claim_text.casefold():
                continue
            if evidence_by_id:
                if any(
                    needle in str(evidence_by_id.get(evidence_id, {}).get("text", "")).casefold()
                    for evidence_id in evidence_ids
                ):
                    return True
            elif any(evidence_id in valid_ids for evidence_id in evidence_ids):
                return True
        return False

    keypoint_coverage = (
        sum(term_is_cited(term) for term in task.gold_terms) / len(task.gold_terms)
        if task.gold_terms
        else 1.0
    )
    fixture_score = 100 * (0.50 * evidence_coverage + 0.25 * citation_id_validity + 0.25 * keypoint_coverage)
    matched_terms = [term for term in task.gold_terms if term_is_cited(term)]
    row = {
        "task_id": task.task_id,
        "mode": result.mode,
        "provider": result.provider,
        "model": result.model,
        "prompt_variant": manifest.get("prompt_variant", "d54"),
        "fixture_score_scope": (
            "offline mechanics only; not model quality"
            if result.provider == "offline-deterministic"
            else "narrow fixture-task comparison; not general model quality"
        ),
        "run_id": result.run_id,
        "repetition": repetition,
        "status": result.status,
        "partial": result.partial,
        "fixture_score": round(fixture_score, 1),
        "evidence_coverage": round(evidence_coverage, 3),
        "citation_id_validity": round(citation_id_validity, 3),
        "keypoint_coverage": round(keypoint_coverage, 3),
        "expected_evidence_ids": sorted(task.gold_evidence_ids),
        "covered_evidence_ids": sorted(covered),
        "cited_evidence_ids": used_ids,
        "invalid_evidence_ids": sorted(used_unique - valid_ids),
        "matched_key_terms": matched_terms,
        "missing_key_terms": [term for term in task.gold_terms if term not in matched_terms],
        "wall_time_seconds": result.budget["elapsed_seconds"],
        "api_cost_usd": result.cost_usd,
        "cost_budget": result.cost_budget,
        "provider_tokens": result.provider_tokens,
        "provider_input_tokens": result.provider_input_tokens,
        "provider_output_tokens": result.provider_output_tokens,
        "validation_contract": result.validation_contract,
        "manifest": result.manifest_path.name,
        "trace": result.trace_path.name,
        "report": result.report_path.name,
    }
    _append_benchmark_scoring(result.report_path, row)
    return row


def _append_benchmark_scoring(report_path: Path, row: dict[str, Any]) -> None:
    """Put the rubric breakdown next to the answer and source citations it scores."""
    if not report_path.is_file():
        return
    cited_ids = ", ".join(f"`{item}`" for item in row["cited_evidence_ids"]) or "none"
    expected_ids = ", ".join(f"`{item}`" for item in row["expected_evidence_ids"]) or "none"
    covered_ids = ", ".join(f"`{item}`" for item in row["covered_evidence_ids"]) or "none"
    invalid_ids = ", ".join(f"`{item}`" for item in row["invalid_evidence_ids"]) or "none"
    matched_terms = ", ".join(f"`{item}`" for item in row["matched_key_terms"]) or "none"
    missing_terms = ", ".join(f"`{item}`" for item in row["missing_key_terms"]) or "none"
    api_cost = "unavailable" if row["api_cost_usd"] is None else f"${row['api_cost_usd']:.8f}"
    provider_tokens = row["provider_tokens"] if row["provider_tokens"] is not None else "not reported"
    lines = [
        "",
        "## Benchmark scoring",
        "",
        f"- Task/repetition: {row['task_id']} / {row['repetition']}",
        f"- Prompt variant: {row['prompt_variant']}",
        f"- Fixture score: {row['fixture_score']:.1f} / 100",
        f"- Score scope: {row['fixture_score_scope']}",
        f"- Evidence coverage (50%): {len(row['covered_evidence_ids'])} / {len(row['expected_evidence_ids'])}; expected: {expected_ids}; covered: {covered_ids}",
        f"- Citation ID validity (25%): {row['citation_id_validity']:.0%}; cited IDs, including repeats: {cited_ids}; invalid IDs: {invalid_ids}",
        f"- Keypoint coverage (25%): {row['keypoint_coverage']:.0%}; matched terms: {matched_terms}; missing terms: {missing_terms}",
        "- The keypoint component checks case-insensitive term presence in a claim and in at least one valid cited source section; summary-only terms do not count.",
        f"- Wall time: {row['wall_time_seconds']:.3f} seconds; API cost: {api_cost}; provider tokens: {provider_tokens}",
        f"- Trace: `{row['trace']}`",
        f"- Manifest: `{row['manifest']}`",
        "",
    ]
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _median_range(values: list[float], digits: int) -> str:
    if not values:
        return "unavailable"
    return (
        f"{statistics.median(values):.{digits}f} "
        f"[{min(values):.{digits}f}–{max(values):.{digits}f}]"
    )


def _mean_delta(values: list[float], digits: int) -> str:
    return f"{statistics.mean(values):+.{digits}f}" if values else "unavailable"


def _d61_paired_summary(rows: list[dict[str, Any]]) -> list[str]:
    arms: dict[tuple[str, str], dict[tuple[str, int], dict[str, Any]]] = {
        ("multi", "d54"): {},
        ("multi", "d61"): {},
        ("single", "d54"): {},
    }
    for row in rows:
        arm = (str(row.get("mode")), str(row.get("prompt_variant", "d54")))
        if arm in arms:
            key = (str(row["task_id"]), int(row["repetition"]))
            arms[arm][key] = row

    contrasts = (
        ("D61 multi − D54 multi", ("multi", "d61"), ("multi", "d54")),
        ("D54 multi − D54 single", ("multi", "d54"), ("single", "d54")),
    )
    lines = [
        "",
        "## Paired D61 comparisons",
        "",
        "Rows are paired by task ID and repetition. Deltas are the arithmetic mean of within-pair differences; partial runs remain in fixture, completion, and time summaries.",
        "",
        "| Contrast | Matched blocks (n) | Fixture score Δ / 100 | Completion Δ (percentage points) | Wall-time Δ (seconds) | Provider-token Δ (usable n; unknown pairs) | Known API-cost Δ (USD; usable n; unknown pairs) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, first_arm, second_arm in contrasts:
        first = arms[first_arm]
        second = arms[second_arm]
        keys = sorted(first.keys() & second.keys())
        pairs = [(first[key], second[key]) for key in keys]
        score_deltas = [float(left["fixture_score"]) - float(right["fixture_score"]) for left, right in pairs]
        completion_deltas = [
            100 * (int(left["status"] == "completed") - int(right["status"] == "completed"))
            for left, right in pairs
        ]
        time_deltas = [float(left["wall_time_seconds"]) - float(right["wall_time_seconds"]) for left, right in pairs]
        known_token_deltas = [
            float(left["provider_tokens"]) - float(right["provider_tokens"])
            for left, right in pairs
            if left["provider_tokens"] is not None and right["provider_tokens"] is not None
        ]
        unknown_token_pairs = len(pairs) - len(known_token_deltas)
        known_cost_deltas = [
            float(left["api_cost_usd"]) - float(right["api_cost_usd"])
            for left, right in pairs
            if left["api_cost_usd"] is not None and right["api_cost_usd"] is not None
        ]
        unknown_cost_pairs = len(pairs) - len(known_cost_deltas)
        token_delta = _mean_delta(known_token_deltas, 1)
        cost_delta = _mean_delta(known_cost_deltas, 8)
        lines.append(
            f"| {label} | {len(pairs)} | {_mean_delta(score_deltas, 2)} | "
            f"{_mean_delta(completion_deltas, 1)} pp | {_mean_delta(time_deltas, 3)} | "
            f"{token_delta} (n={len(known_token_deltas)}; {unknown_token_pairs} unknown) | "
            f"{cost_delta} (n={len(known_cost_deltas)}; {unknown_cost_pairs} unknown) |"
        )
    lines.extend([
        "",
        "Positive score/completion deltas favor the first arm; positive time/token/cost deltas mean the first arm took longer, used more reported tokens, or cost more. Token and cost deltas use only pairs where both values were reported; excluded pairs are counted above. These paired summaries are descriptive and make no significance claim. Fixture scores do not measure semantic answer quality; use a separately documented semantic review for answer-quality observations.",
        "",
    ])
    return lines


def write_comparison(
    rows: list[dict[str, Any]],
    runs_dir: Path,
    *,
    repeats: int = 1,
    d61_experiment: bool = False,
) -> tuple[Path, Path, Path]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = runs_dir / "benchmark.json"
    csv_path = runs_dir / "benchmark.csv"
    md_path = runs_dir / "benchmark.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    providers = sorted({str(row.get("provider", "unknown")) for row in rows})
    models = sorted({str(row.get("model") or "not applicable") for row in rows})
    prompt_variants = sorted({str(row.get("prompt_variant", "d54")) for row in rows})
    offline = providers == ["offline-deterministic"]
    title = (
        "Model-backed D54/D61 three-arm pilot"
        if d61_experiment
        else ("Offline mechanics benchmark" if offline else "Model-backed paired benchmark")
    )
    task_count = len({str(row.get("task_id", "")) for row in rows})
    blocks_per_arm = task_count * repeats
    lines = [
        f"# {title}",
        "",
        f"**Provider:** {', '.join(providers) or 'unknown'}  ",
        f"**Model:** {', '.join(models) or 'not applicable'}  ",
        f"**Prompt variant:** {', '.join(prompt_variants) or 'd54'}",
        "",
        (
            f"**Design:** D54 multi, D61 multi, and D54 single; {blocks_per_arm} task/repetition blocks per arm, "
            f"{len(rows)} scheduled run rows total. Arm order rotates across repeats."
            if d61_experiment
            else f"**Trials:** {len(rows)} run rows across {repeats} repetition(s); multi/single order alternates for each task."
        ),
        *(
            [
                "The per-arm n is descriptive only: it covers three fixed fixture tasks, with partial runs retained.",
                "Fixture scores measure expected evidence-ID coverage, citation-ID validity, and key-term presence; they do not judge semantic correctness or model quality. See the separate AI-judge review for exploratory semantic ratings; one AI-judge pass is not human ground truth.",
            ]
            if d61_experiment
            else []
        ),
        "",
        "Both modes use the same task, runtime validation contract, corpus, and read-only corpus-tool family. The evaluator applies the same gold-label rubric after each run; role-specific tool allowlists differ by mode and role.",
        "Offline scores describe deterministic harness mechanics, not language-model quality." if offline else "Model-backed fixture scores are a narrow task comparison, not a general measure of model quality.",
        "Each row links its score to the run's JSONL trace and Markdown report; the report includes the exact evidence IDs and key terms behind the score.",
        "These rows use post-run gold-label scoring (D22). Older run artifacts in this folder may predate D22; see `README.md` and do not use those legacy artifacts as evaluation evidence.",
        "",
        "| Task | Repetition | Mode | Prompt variant | Provider / model | Fixture score / 100 | Evidence coverage | Citation ID validity | Keypoint coverage | Wall time (s) | API cost | Provider tokens | Estimated USD cap | Estimated committed | Status | Trace | Report | Manifest |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        cost = "unavailable" if row["api_cost_usd"] is None else f"${row['api_cost_usd']:.4f}"
        cost_budget = row.get("cost_budget") or {}
        cap = (
            f"${cost_budget['estimated_cap_usd']:.4f}"
            if cost_budget.get("enforced")
            else ("not applied" if cost_budget else "not configured")
        )
        committed = f"${cost_budget['estimated_committed_usd']:.4f}" if cost_budget.get("enforced") else "n/a"
        provider_tokens = row["provider_tokens"] if row["provider_tokens"] is not None else "not reported"
        lines.append(
            f"| {row['task_id']} | {row['repetition']} | {row['mode']} | {row.get('prompt_variant', 'd54')} | {row.get('provider', 'unknown')} / {row.get('model') or 'not applicable'} | {row['fixture_score']:.1f} | "
            f"{row['evidence_coverage']:.0%} | {row['citation_id_validity']:.0%} | "
            f"{row['keypoint_coverage']:.0%} | {row['wall_time_seconds']:.3f} | {cost} | {provider_tokens} | {cap} | {committed} | "
            f"{row['status']} | [{row['trace']}](./{row['trace']}) | [{row['report']}](./{row['report']}) | [{row['manifest']}](./{row['manifest']}) |"
        )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["task_id"], row["mode"], row.get("prompt_variant", "d54")), []).append(row)
    lines.extend([
        "",
        "## Repeated-run summary",
        "",
        "Completed counts are shown against the requested repetition count. Score and wall-time median/range include every trial, including partial runs; API cost includes known values only.",
        "",
        "| Task | Mode | Prompt variant | Completed | Fixture score median [range] | Wall time median [range] (s) | API cost median [range] USD (known) |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for (task_id, mode, prompt_variant), group in grouped.items():
        completed = sum(row["status"] == "completed" for row in group)
        scores = [float(row["fixture_score"]) for row in group]
        wall_times = [float(row["wall_time_seconds"]) for row in group]
        known_costs = [
            float(row["api_cost_usd"])
            for row in group
            if row["api_cost_usd"] is not None
        ]
        cost_summary = _median_range(known_costs, 8)
        if known_costs:
            cost_summary += f" ({len(known_costs)}/{len(group)} known)"
        lines.append(
            f"| {task_id} | {mode} | {prompt_variant} | {completed}/{repeats} | {_median_range(scores, 1)} | "
            f"{_median_range(wall_times, 3)} | {cost_summary} |"
        )
    if d61_experiment:
        lines.extend(_d61_paired_summary(rows))
    lines.extend([
        "",
        "The fixture score gives 50% to coverage of expected evidence IDs, 25% to citation ID validity (whether each cited ID exists in the corpus), and 25% to expected terms present in a claim and its valid cited source. ID validity does not check whether cited text entails each claim, and term matching does not judge meaning. Each linked report lists the expected, covered, cited, and invalid IDs plus matched and missing terms.",
        "Wall time is measured by the harness. API cost excludes local CPU/GPU use: localhost API cost is $0; remote cost is calculated only when provider usage and per-token rates are available, and is otherwise unavailable.",
        "The estimated USD cap is a token-based pre-request guard, not a guaranteed billing limit; usage above an estimate marks that run partial, and parallel requests already in flight may still finish.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, md_path


def run_comparison(
    *,
    corpus_dir: Path,
    runs_dir: Path,
    provider: Any | None = None,
    max_cost_usd: float | None = None,
    repeats: int = 1,
    d61_experiment: bool = False,
    prompt_variant: str = "d54",
) -> list[dict[str, Any]]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    rows: list[dict[str, Any]] = []
    configured_provider = provider or OfflineProvider()
    if d61_experiment and (
        not callable(getattr(configured_provider, "configure_prompt_variant", None))
        or getattr(configured_provider, "response_format", None) != "json_schema"
    ):
        raise ValueError("the D61 three-arm experiment requires the OpenAI-compatible provider with json_schema response format")
    if d61_experiment and prompt_variant != "d54":
        raise ValueError("the D61 experiment selects its own prompt variants")
    experiment_arms = (("multi", "d54"), ("multi", "d61"), ("single", "d54"))
    for task in SAMPLE_TASKS:
        # Gold labels stay in this evaluator process and are used only after the run.
        # Only task facets enter the harness contract, prompts, manifest, and trace.
        validation_contract = make_validation_contract(task.validation_contract)
        for repetition in range(repeats):
            if d61_experiment:
                offset = repetition % len(experiment_arms)
                arms = experiment_arms[offset:] + experiment_arms[:offset]
            else:
                modes = ("multi", "single") if repetition % 2 == 0 else ("single", "multi")
                arms = tuple((mode, prompt_variant) for mode in modes)
            for mode, prompt_variant in arms:
                harness = Harness(
                    corpus_dir=corpus_dir,
                    runs_dir=runs_dir,
                    provider=configured_provider,
                    mode=mode,
                    limits=Limits(),
                    validation_contract=validation_contract,
                    max_cost_usd=max_cost_usd,
                    prompt_variant=prompt_variant,
                )
                result = harness.run(task.question, task_id=task.task_id)
                evidence_by_id = {
                    item["evidence_id"]: item for item in harness.tools.all_evidence_for_benchmark()
                }
                rows.append(evaluate(result, task, evidence_by_id, repetition=repetition + 1))
    write_comparison(rows, runs_dir, repeats=repeats, d61_experiment=d61_experiment)
    return rows


def _hotpotqa_summary(task: HotpotQATask, result: RunResult, metrics: dict[str, Any]) -> None:
    report_path = result.report_path
    if not report_path.is_file():
        return
    api_cost = "unavailable" if result.cost_usd is None else f"${result.cost_usd:.8f}"
    provider_tokens = result.provider_tokens if result.provider_tokens is not None else "not reported"
    limits = result.budget["limits"]
    with report_path.open("a", encoding="utf-8") as report:
        report.write("\n".join([
            "",
            "## HotpotQA pilot scoring",
            "",
            f"- Case: {task.task_id}",
            f"- Answer parse success: {metrics['answer_parse_success']}",
            f"- Answer EM / F1: {metrics['answer_em']:.3f} / {metrics['answer_f1']:.3f}",
            "- Supporting-fact precision / recall / F1: "
            f"{metrics['supporting_fact_precision']:.3f} / {metrics['supporting_fact_recall']:.3f} / {metrics['supporting_fact_f1']:.3f}",
            f"- Retrieval support recall: {metrics['retrieval_support_recall']:.3f} ({metrics['retrieved_supporting_fact_count']} supporting sentences retrieved; {metrics['retrieved_evidence_count']} unique evidence items)",
            f"- Joint EM / F1: {metrics['joint_em']:.3f} / {metrics['joint_f1']:.3f}",
            f"- Wall time: {result.budget['elapsed_seconds']:.3f} seconds; provider tokens: {provider_tokens}; API cost: {api_cost}",
            "- Effective limits: "
            f"run {limits['max_steps']} steps / {limits['max_seconds']:g} seconds / {limits['max_tokens']} tokens; "
            f"per agent {limits['max_agent_steps']} steps / {limits['max_agent_seconds']:g} seconds / {limits['max_agent_tokens']} tokens.",
            "- Per-request completion caps: "
            f"max_completion_tokens={limits['max_completion_tokens']}; "
            f"max_lead_completion_tokens={limits['max_lead_completion_tokens']}.",
            "- Labels were scored after the harness completed; this section is not part of the run trace or manifest.",
            "",
        ]))


def run_hotpotqa_comparison(
    *,
    dataset: HotpotQADataset,
    runs_dir: Path,
    provider: Any | None = None,
    max_cost_usd: float | None = None,
    max_aggregate_cost_usd: float | None = None,
    repeats: int = 1,
    limits: Limits | None = None,
    prompt_variant: str = "d54",
) -> list[dict[str, Any]]:
    """Run a label-isolated paired pilot and write separate HotpotQA artifacts."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not dataset.tasks:
        raise ValueError("HotpotQA dataset sample must contain at least one task")
    rows: list[dict[str, Any]] = []
    configured_provider = provider or OfflineProvider()
    planned_run_count = len(dataset.tasks) * repeats * 2
    if max_aggregate_cost_usd is not None:
        if not isinstance(max_aggregate_cost_usd, (int, float)) or not math.isfinite(max_aggregate_cost_usd) or max_aggregate_cost_usd < 0:
            raise ValueError("max_aggregate_cost_usd must be a finite, non-negative amount")
        if max_cost_usd is None or not math.isfinite(max_cost_usd) or max_cost_usd < 0:
            raise ValueError("aggregate USD guard requires a finite, non-negative per-run USD cap")
        if not getattr(configured_provider, "is_remote", False) or max_cost_usd is None:
            raise ValueError("aggregate USD guard requires a remote provider and a per-run USD cap")
        planned_cap = planned_run_count * max_cost_usd
        if planned_cap > max_aggregate_cost_usd + 1e-12:
            raise ValueError(
                f"planned estimated run caps total ${planned_cap:.8f}, exceeding aggregate guard ${max_aggregate_cost_usd:.8f}; no runs started"
            )
    run_limits = limits or Limits()
    mode_limits = _hotpotqa_mode_limits(run_limits)
    validation_contract = make_validation_contract(
        HOT_POTQA_FACETS,
        summary_first_line_prefix="Answer: ",
        summary_first_line_max_words=8,
    )
    for task in dataset.tasks:
        # Keep the search question clean; answer formatting lives in the trusted runtime contract.
        agent_task = task.question
        for repetition in range(repeats):
            modes = ("multi", "single") if repetition % 2 == 0 else ("single", "multi")
            for mode in modes:
                harness = Harness(
                    corpus_dir=task.corpus_dir,
                    runs_dir=runs_dir,
                    provider=configured_provider,
                    mode=mode,
                    limits=mode_limits[mode],
                    validation_contract=validation_contract,
                    max_cost_usd=max_cost_usd,
                    prompt_variant=prompt_variant,
                )
                result = harness.run(agent_task, task_id=task.task_id)
                # The evaluator is called only after Harness.run has closed the trace and returned.
                metrics = score_hotpotqa(result.answer, task.scoring)
                metrics.update(score_retrieval_support_recall(result.trace_path, task.scoring))
                manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
                if manifest.get("corpus_sha256") != task.corpus_sha256:
                    raise RuntimeError(f"corpus hash changed during HotpotQA run {task.task_id}")
                row = {
                    "dataset": "HotpotQA",
                    "dataset_version": dataset.provenance["dataset_version"],
                    "split": dataset.provenance["split"],
                    "task_id": task.task_id,
                    "dataset_id": task.dataset_id,
                    "repetition": repetition + 1,
                    "mode": result.mode,
                    "provider": result.provider,
                    "model": result.model,
                    "response_format": getattr(configured_provider, "response_format", None),
                    "prompt_variant": manifest.get("prompt_variant", prompt_variant),
                    "status": result.status,
                    "partial": result.partial,
                    **metrics,
                    "wall_time_seconds": result.budget["elapsed_seconds"],
                    "provider_tokens": result.provider_tokens,
                    "provider_input_tokens": result.provider_input_tokens,
                    "provider_output_tokens": result.provider_output_tokens,
                    "api_cost_usd": result.cost_usd,
                    "cost_budget": result.cost_budget,
                    "corpus_sha256": task.corpus_sha256,
                    "run_id": result.run_id,
                    "trace": result.trace_path.name,
                    "report": result.report_path.name,
                    "manifest": result.manifest_path.name,
                    "score_scope": "small public-data pilot; not generalizable",
                    "budget_policy": HOTPOTQA_BUDGET_POLICY,
                    "effective_limits": json.dumps(manifest["limits"], sort_keys=True, separators=(",", ":")),
                }
                _hotpotqa_summary(task, result, metrics)
                rows.append(row)
    report_provenance = dict(dataset.provenance)
    report_provenance.update({
        "planned_harness_run_count": planned_run_count,
        "per_run_estimated_cost_cap_usd": max_cost_usd,
        "planned_aggregate_estimated_cap_usd": planned_run_count * max_cost_usd if max_cost_usd is not None else None,
        "aggregate_estimated_cost_guard_usd": max_aggregate_cost_usd,
        "prompt_variants": sorted({str(row.get("prompt_variant", "d54")) for row in rows}),
        "budget_policy": HOTPOTQA_BUDGET_POLICY,
        "effective_mode_limits": {
            mode: asdict(mode_limits[mode]) for mode in ("multi", "single")
        },
    })
    write_hotpotqa_comparison(rows, report_provenance, runs_dir, repeats=repeats)
    return rows


def _hotpotqa_aggregate(rows: list[dict[str, Any]], mode: str, repeats: int) -> dict[str, Any]:
    group = [row for row in rows if row["mode"] == mode]
    summary: dict[str, Any] = {
        "mode": mode,
        "runs": len(group),
        "completed_runs": sum(row["status"] == "completed" for row in group),
        "requested_repeats": repeats,
    }
    for key in (
        "answer_em", "answer_f1", "supporting_fact_precision", "supporting_fact_recall",
        "supporting_fact_f1", "supporting_fact_em", "joint_em", "joint_f1", "retrieval_support_recall",
    ):
        values = [float(row[key]) for row in group]
        summary[f"{key}_mean"] = round(statistics.mean(values), 4) if values else None
    latencies = [float(row["wall_time_seconds"]) for row in group]
    summary["wall_time_median_seconds"] = round(statistics.median(latencies), 4) if latencies else None
    summary["wall_time_min_seconds"] = round(min(latencies), 4) if latencies else None
    summary["wall_time_max_seconds"] = round(max(latencies), 4) if latencies else None
    known_tokens = [int(row["provider_tokens"]) for row in group if row["provider_tokens"] is not None]
    summary["provider_tokens_known_total"] = sum(known_tokens) if known_tokens else None
    summary["provider_tokens_known_runs"] = len(known_tokens)
    known_costs = [float(row["api_cost_usd"]) for row in group if row["api_cost_usd"] is not None]
    summary["api_cost_known_total_usd"] = round(sum(known_costs), 8) if known_costs else None
    summary["api_cost_known_runs"] = len(known_costs)
    return summary


def _paired_hierarchical_bootstrap(
    rows: list[dict[str, Any]],
    *,
    iterations: int = 10_000,
    seed: int = 20_261_016,
) -> dict[str, Any]:
    """Resample task clusters, then matched repetitions, for multi-minus-single deltas."""
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    metrics = ("joint_f1", "answer_f1", "supporting_fact_f1")
    pairs: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for row in rows:
        task_id = str(row["dataset_id"])
        repetition = int(row["repetition"])
        mode = str(row["mode"])
        if mode not in {"multi", "single"}:
            raise ValueError(f"unexpected mode in paired bootstrap: {mode}")
        by_mode = pairs.setdefault(task_id, {}).setdefault(repetition, {})
        if mode in by_mode:
            raise ValueError(f"duplicate paired bootstrap row for {task_id}/{repetition}/{mode}")
        by_mode[mode] = row

    task_ids = sorted(pairs)
    if not task_ids:
        raise ValueError("paired bootstrap requires at least one task")
    repetitions_by_task: dict[str, list[int]] = {}
    for task_id in task_ids:
        complete = sorted(
            repetition
            for repetition, by_mode in pairs[task_id].items()
            if "multi" in by_mode and "single" in by_mode
        )
        if not complete or len(complete) != len(pairs[task_id]):
            raise ValueError(f"paired bootstrap requires matched multi/single rows for every repeat of {task_id}")
        repetitions_by_task[task_id] = complete
    repeat_counts = {len(values) for values in repetitions_by_task.values()}
    if len(repeat_counts) != 1:
        raise ValueError("paired bootstrap requires the same repeat count for every task")

    paired_deltas = {
        metric: {
            task_id: [
                float(pairs[task_id][repetition]["multi"][metric])
                - float(pairs[task_id][repetition]["single"][metric])
                for repetition in repetitions_by_task[task_id]
            ]
            for task_id in task_ids
        }
        for metric in metrics
    }
    rng = random.Random(seed)
    bootstrap_values: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(iterations):
        sampled_deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
        for _task_draw in task_ids:
            task_id = rng.choice(task_ids)
            available_repeats = repetitions_by_task[task_id]
            for _repeat_draw in available_repeats:
                repetition_index = rng.randrange(len(available_repeats))
                for metric in metrics:
                    sampled_deltas[metric].append(paired_deltas[metric][task_id][repetition_index])
        for metric in metrics:
            bootstrap_values[metric].append(statistics.fmean(sampled_deltas[metric]))

    def percentile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    metric_results = {}
    for metric in metrics:
        all_deltas = [value for task_id in task_ids for value in paired_deltas[metric][task_id]]
        values = bootstrap_values[metric]
        metric_results[metric] = {
            "mean_multi_minus_single": statistics.fmean(all_deltas),
            "ci95_percentile_low": percentile(values, 0.025),
            "ci95_percentile_high": percentile(values, 0.975),
            "paired_trial_count": len(all_deltas),
        }
    return {
        "method": "two-level paired cluster bootstrap",
        "task_resampling_unit": "dataset_id (sampled with replacement)",
        "repeat_resampling_unit": "repetition index within each sampled task (paired modes resampled together)",
        "task_count": len(task_ids),
        "repeats_per_task": next(iter(repeat_counts)),
        "paired_trial_count": sum(len(values) for values in repetitions_by_task.values()),
        "iterations": iterations,
        "seed": seed,
        "confidence_level": 0.95,
        "metrics": metric_results,
    }


def write_hotpotqa_comparison(
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    runs_dir: Path,
    *,
    repeats: int,
) -> tuple[Path, Path, Path]:
    """Write public-data pilot output without replacing fixture benchmark files."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = runs_dir / "hotpotqa-benchmark.json"
    csv_path = runs_dir / "hotpotqa-benchmark.csv"
    md_path = runs_dir / "hotpotqa-benchmark.md"
    summaries = {
        mode: _hotpotqa_aggregate(rows, mode, repeats)
        for mode in ("multi", "single")
    }
    paired_uncertainty = _paired_hierarchical_bootstrap(rows)
    payload = {
        "metadata": {
            **provenance,
            "requested_repeats": repeats,
            "run_count": len(rows),
            "scope": "small public-data pilot; not generalizable",
            "paired_uncertainty": paired_uncertainty,
            "answer_metric_protocol": "official HotpotQA normalization, answer EM/F1 and joint formulas; short answer parsed from the first line of the JSON summary field",
            "support_metric_protocol": "official support-fact set precision/recall/F1 and exact match over cited sentence IDs",
            "retrieval_metric_protocol": "post-run trace scorer: unique evidence IDs returned by search_documents intersected with gold supporting sentence IDs, divided by the number of gold support sentences",
        },
        "summary": summaries,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        csv_rows = [
            {
                **row,
                "source_url": provenance.get("source_url"),
                "source_sha256": provenance.get("source_sha256"),
                "local_source_file": provenance.get("local_source_file"),
            }
            for row in rows
        ]
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# HotpotQA public-data pilot comparison",
        "",
        "**Scope:** small, deterministic pilot sample from the public dev distractor split; results are not generalizable and may reflect model pretraining exposure.",
        f"**Source:** [{provenance['source_url']}]({provenance['source_url']}); [HotpotQA homepage]({provenance['source_homepage']}); local file `{provenance['local_source_file']}`; SHA-256 `{provenance['source_sha256']}`.",
        f"**Version / split:** `{provenance['dataset_version']}` / `{provenance['split']}`; license `{provenance['license']}`.",
        f"**Sample:** {provenance['sample_count']} of {provenance['requested_sample_size']} requested; IDs: {', '.join(provenance['sample_ids'])}; source indices: {provenance['sample_source_indices']}; rule: {provenance['sample_selection_rule']}.",
        f"**Corpus hash:** `{provenance['corpus_sha256']}` (aggregate); each run row and manifest includes its case corpus SHA-256.",
        f"**Prompt variant:** {', '.join(provenance.get('prompt_variants', ['d54']))}.",
        f"**Official metrics reference:** [{provenance['official_evaluator_url']}]({provenance['official_evaluator_url']}).",
        f"**Trials:** {len(rows)} runs across {repeats} repetition(s); the multi/single order alternates by task and repetition.",
        "**Budget policy:** `equal-run-envelope-v1`; multi-agent retains the configured per-agent caps, while the single-agent baseline's per-agent step/time/token caps are raised to the matching shared run caps. Its per-request completion cap is raised to match the multi-agent lead cap. Both modes retain the same run-level limits.",
        "",
        "The runner reads only the local JSON file supplied on the command line; it does not download data. Gold answers and supporting-fact labels are held by the scorer and evaluated after each `Harness.run()`. Both modes receive the same raw question, corpus, and generic three-facet contract. The trusted contract requires the first line of `summary` to begin `Answer: ` and contain only the concise answer; the question sent to retrieval has no formatting boilerplate. Citation support and retrieval support recall are scored separately from the closed post-run trace.",
        "",
        "## Mode summary",
        "",
        "Quality metrics are arithmetic means over all trials, including partial runs. Latency is median [minimum–maximum]. Token totals and API cost include known runs only; local CPU/GPU time is not priced.",
        "",
        "| Mode | Completed | Answer EM | Answer F1 | Support P / R / F1 | Retrieval support recall | Joint EM / F1 | Wall time median [range] (s) | Provider tokens known total | API cost known total (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("multi", "single"):
        item = summaries[mode]
        lines.append(
            f"| {mode} | {item['completed_runs']}/{item['runs']} | {item['answer_em_mean']:.3f} | {item['answer_f1_mean']:.3f} | "
            f"{item['supporting_fact_precision_mean']:.3f} / {item['supporting_fact_recall_mean']:.3f} / {item['supporting_fact_f1_mean']:.3f} | "
            f"{item['retrieval_support_recall_mean']:.3f} | "
            f"{item['joint_em_mean']:.3f} / {item['joint_f1_mean']:.3f} | "
            f"{item['wall_time_median_seconds']:.3f} [{item['wall_time_min_seconds']:.3f}–{item['wall_time_max_seconds']:.3f}] | "
            f"{item['provider_tokens_known_total'] if item['provider_tokens_known_total'] is not None else 'unavailable'} | "
            f"{item['api_cost_known_total_usd'] if item['api_cost_known_total_usd'] is not None else 'unavailable'} |"
        )
    lines.extend([
        "",
        "## Effective mode limits",
        "",
        "Every run manifest records its effective limits; each linked run report repeats the run and per-agent caps. The table below shows the configured multi-agent limits and the equal-run-envelope single-agent limits.",
        "",
        "| Mode | Run steps / seconds / tokens | Per-agent steps / seconds / tokens | Completion tokens (general / lead) |",
        "|---|---|---|---:|",
    ])
    effective_mode_limits = provenance.get("effective_mode_limits", {})
    for mode in ("multi", "single"):
        effective = effective_mode_limits.get(mode)
        if not isinstance(effective, dict):
            continue
        lines.append(
            f"| {mode} | {effective['max_steps']} / {effective['max_seconds']:g} / {effective['max_tokens']} | "
            f"{effective['max_agent_steps']} / {effective['max_agent_seconds']:g} / {effective['max_agent_tokens']} | "
            f"{effective['max_completion_tokens']} / {effective['max_lead_completion_tokens']} |"
        )
    lines.extend([
        "",
        "## Paired uncertainty analysis",
        "",
        f"The comparison resamples {paired_uncertainty['task_count']} task IDs with replacement, then resamples each selected task's {paired_uncertainty['repeats_per_task']} repetition indices with replacement; multi/single values stay paired. The analysis uses {paired_uncertainty['iterations']:,} draws (seed `{paired_uncertainty['seed']}`), includes partial runs, and reports percentile 95% intervals. These intervals describe this small public sample and run configuration; they do not establish causal effects or general performance.",
        "",
        "| Metric | Mean multi − single | 95% paired bootstrap interval | Paired trials |",
        "|---|---:|---:|---:|",
    ])
    for metric in ("joint_f1", "answer_f1", "supporting_fact_f1"):
        item = paired_uncertainty["metrics"][metric]
        lines.append(
            f"| {metric} | {item['mean_multi_minus_single']:+.4f} | "
            f"[{item['ci95_percentile_low']:+.4f}, {item['ci95_percentile_high']:+.4f}] | "
            f"{item['paired_trial_count']} |"
        )
    lines.extend([
        "",
        "## Run rows",
        "",
        "| Case | Repetition | Mode | Prompt variant | Provider / model / format | Answer EM / F1 | Support P / R / F1 | Retrieval support recall | Joint EM / F1 | Wall time (s) | Tokens | API cost (USD) | Status | Trace | Report | Manifest |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ])
    for row in rows:
        token_text = row["provider_tokens"] if row["provider_tokens"] is not None else "unavailable"
        cost_text = "unavailable" if row["api_cost_usd"] is None else f"{row['api_cost_usd']:.8f}"
        lines.append(
            f"| {row['dataset_id']} | {row['repetition']} | {row['mode']} | {row.get('prompt_variant', 'd54')} | {row['provider']} / {row['model'] or 'not reported'} / {row['response_format'] or 'provider default'} | "
            f"{row['answer_em']:.3f} / {row['answer_f1']:.3f} | "
            f"{row['supporting_fact_precision']:.3f} / {row['supporting_fact_recall']:.3f} / {row['supporting_fact_f1']:.3f} | "
            f"{row['retrieval_support_recall']:.3f} | "
            f"{row['joint_em']:.3f} / {row['joint_f1']:.3f} | {row['wall_time_seconds']:.3f} | {token_text} | {cost_text} | "
            f"{row['status']} | [{row['trace']}](./{row['trace']}) | [{row['report']}](./{row['report']}) | [{row['manifest']}](./{row['manifest']}) |"
        )
    lines.extend([
        "",
        "A missing `Answer:` first line produces an answer parse failure and zero answer EM/F1; inspect the linked report. The support scorer maps each cited sentence evidence ID back to its source `(title, sent_id)` and counts unknown citation IDs as false positives. Answer EM/F1 and support P/R/F1 follow the official HotpotQA normalization and set-metric conventions; joint EM/F1 combines answer and support scores. This compact pilot is evidence about this sample, model, and harness run configuration only.",
        "",
        "Offline-provider rows exercise parsing, scoring, artifact generation, and label isolation only; they are not language-model quality evidence. Remote calls still require the existing interactive approval gate. A per-run USD cap still requires both token-rate variables.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, md_path
