"""Command-line interface for demo, comparison, custom runs, and trace replay."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from .benchmark import SAMPLE_TASKS, evaluate, run_comparison, run_hotpotqa_comparison, write_comparison
from .budget import Limits
from .harness import Harness
from .hotpotqa import HOTPOTQA_SOURCE_URL, load_hotpotqa_dev_distractor
from .providers import OfflineProvider, provider_from_args

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
DEFAULT_CORPUS = PACKAGE_DIR / "fixtures" / "corpus"
DEFAULT_INJECTION_CORPUS = PACKAGE_DIR / "fixtures" / "injection_corpus"
DEFAULT_RUNS = PROJECT_DIR / "runs"


def _add_provider_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("offline", "openai-compatible"), default="offline")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL; defaults to BYTEVOX_BASE_URL or local Ollama")
    parser.add_argument("--model", default=None, help="Model name; defaults to BYTEVOX_MODEL or qwen2.5:7b")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding an API key; the value is never logged")
    parser.add_argument(
        "--response-format",
        choices=("json_object", "json_schema"),
        default="json_object",
        help="Structured response mode for OpenAI-compatible endpoints (default: json_object; json_schema requires provider support)",
    )
    parser.add_argument("--approve-remote-model", action="store_true", help="Proceed to a remote model only after the interactive approval prompt")
    parser.add_argument("--max-cost-usd", type=float, default=None, help="Optional estimated per-run USD cap for remote calls; defaults to BYTEVOX_MAX_COST_USD")
    parser.add_argument(
        "--source-scope-candidate",
        action="store_true",
        help="Opt in to D73 source-entity attribution guidance; this experimental prompt is not the D54 default",
    )


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_CORPUS, help="Directory of read-only .md/.txt evidence files")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS, help="Output folder under this project for reports and JSONL traces")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bytevox-harness", description="Run a bounded, inspectable evidence-agent harness")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Run paired tasks with the selected provider plus offline safety demos")
    _add_paths(demo)
    _add_provider_options(demo)

    compare = commands.add_parser("compare", help="Compare multi-agent and single-agent modes with the selected provider")
    _add_paths(compare)
    _add_provider_options(compare)
    compare.add_argument("--repeats", type=_positive_int, default=1, metavar="N", help="Repeat each task/mode run N times (default: 1)")
    compare.add_argument(
        "--d61-experiment",
        action="store_true",
        help="Run the opt-in contemporaneous D54 multi / D61 multi / D54 single three-arm pilot",
    )

    hotpotqa = commands.add_parser(
        "hotpotqa-compare",
        help="Run a label-isolated pilot on a local HotpotQA dev distractor JSON file",
    )
    hotpotqa.add_argument("--dataset-path", type=Path, required=True, help="Local HotpotQA dev distractor JSON; no download is performed")
    hotpotqa.add_argument(
        "--exclude-ids-file",
        type=Path,
        default=None,
        help="Optional UTF-8 newline-delimited dataset IDs excluded before sampling",
    )
    hotpotqa.add_argument(
        "--source-url",
        default=HOTPOTQA_SOURCE_URL,
        help="Public HTTPS URL for the dataset source (default: official CMU file)",
    )
    hotpotqa.add_argument("--sample-size", type=_positive_int, default=3, metavar="N", help="Select N deterministic examples (default: 3)")
    hotpotqa.add_argument("--sample-seed", type=int, default=None, help="Use deterministic proportional stratified sampling by HotpotQA question type")
    hotpotqa.add_argument("--repeats", type=_positive_int, default=1, metavar="N", help="Repeat each example/mode run N times (default: 1)")
    hotpotqa.add_argument("--max-tokens", type=_positive_int, default=16_000, help="Estimated total-token budget per HotpotQA run (default: 16000)")
    hotpotqa.add_argument("--max-agent-tokens", type=_positive_int, default=6_000, help="Estimated per-role token cap for multi-agent HotpotQA runs (default: 6000); single-agent receives the full run token cap")
    hotpotqa.add_argument(
        "--max-aggregate-cost-usd",
        type=_positive_float,
        default=None,
        help="Optional aggregate estimate guard; requires --max-cost-usd and blocks the campaign before any run if planned run caps exceed it",
    )
    hotpotqa.add_argument(
        "--max-agent-seconds",
        type=_positive_float,
        default=32.0,
        help="Per-role time cap for multi-agent HotpotQA runs (default: 32; single-agent receives the full run time cap, 60 seconds)",
    )
    hotpotqa.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS, help="Output folder under this project for reports and JSONL traces")
    _add_provider_options(hotpotqa)

    run = commands.add_parser("run", help="Run one task against a local read-only corpus")
    run.add_argument("task", help="Goal in plain English")
    _add_paths(run)
    _add_provider_options(run)
    run.add_argument("--mode", choices=("multi", "single"), default="multi")
    run.add_argument("--max-steps", type=int, default=32)
    run.add_argument("--max-seconds", type=float, default=60.0)
    run.add_argument("--max-tokens", type=int, default=12_000)
    run.add_argument(
        "--checker-budget-recovery",
        action="store_true",
        help="Opt in to the unvalidated D64 checker-budget recovery candidate",
    )

    failure = commands.add_parser("failure-demo", help="Simulate a worker outage and checker repair; save both traces")
    _add_paths(failure)

    limit = commands.add_parser("limit-demo", help="Run with a deliberately small step budget")
    _add_paths(limit)

    replay = commands.add_parser("replay", help="Replay the timeline in a JSONL trace")
    replay.add_argument("trace", help="Trace filename or path under the runs folder")
    replay.add_argument("--all", action="store_true", help="Print full event payloads, including messages and tool results")
    replay.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)

    show = commands.add_parser("show", help="Print a saved Markdown report")
    show.add_argument("report", help="Report filename or path under the runs folder")
    show.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    return parser


def _approve_if_remote(provider: Any, approved: bool) -> None:
    if not getattr(provider, "is_remote", False):
        return
    if not approved:
        raise RuntimeError("Remote model calls are blocked. Pass --approve-remote-model, then type APPROVE at the prompt.")
    if not sys.stdin.isatty():
        raise RuntimeError("Remote model approval requires an interactive terminal; no request was sent.")
    print("This sends task text and retrieved document excerpts to a remote model endpoint.")
    answer = input("Type APPROVE to continue: ").strip()
    if answer != "APPROVE":
        raise RuntimeError("Remote model call cancelled; no request was sent.")


def _provider(args: argparse.Namespace) -> Any:
    provider = provider_from_args(
        args.provider,
        args.base_url,
        args.model,
        args.api_key_env,
        getattr(args, "response_format", "json_object"),
    )
    _approve_if_remote(provider, args.approve_remote_model)
    return provider


def _configured_max_cost_usd(args: argparse.Namespace) -> float | None:
    value = args.max_cost_usd
    if value is None:
        configured = os.environ.get("BYTEVOX_MAX_COST_USD")
        if configured is None:
            return None
        try:
            value = float(configured)
        except ValueError as exc:
            raise ValueError("BYTEVOX_MAX_COST_USD must be a finite, non-negative amount") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("--max-cost-usd must be a finite, non-negative amount")
    return value


def _print_result(result: Any) -> None:
    print(json.dumps({
        "run_id": result.run_id,
        "status": result.status,
        "mode": result.mode,
        "provider": result.provider,
        "model": result.model,
        "cost_budget": result.cost_budget,
        "checker_accepted": result.checker_accepted,
        "claims": len(result.answer.get("claims", [])),
        "elapsed_seconds": result.budget["elapsed_seconds"],
        "steps": result.budget["steps"],
        "manifest": str(result.manifest_path),
        "trace": str(result.trace_path),
        "report": str(result.report_path),
        "errors": result.errors,
    }, indent=2))


def _write_stdout_safely(text: str) -> None:
    """Write all text, replacing only characters unsupported by stdout's encoding."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
    except LookupError:
        safe_text = text
    sys.stdout.write(safe_text)


def _trace_path(value: str, runs_dir: Path, suffix: str) -> Path:
    root = runs_dir.resolve(strict=True)
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    if candidate.suffix != suffix:
        candidate = candidate.with_suffix(suffix)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("Trace and report reads are limited to the configured runs folder")
    return resolved


def _replay(trace_path: Path, show_all: bool) -> None:
    _write_stdout_safely(f"Replay: {trace_path.name}\n")
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        event = record.get("event", "unknown")
        agent = record.get("agent_id") or record.get("to_agent") or record.get("from_agent") or "harness"
        if show_all:
            _write_stdout_safely(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            continue
        extra = ""
        if event == "tool_call":
            extra = f" tool={record.get('tool')} args={json.dumps(record.get('arguments', {}), ensure_ascii=False)}"
        elif event == "tool_result":
            extra = f" tool={record.get('tool')} result={json.dumps(record.get('result', {}), ensure_ascii=False)[:240]}"
        elif event == "model_call":
            extra = f" role={record.get('role')} attempt={record.get('attempt')}"
        elif event == "model_error":
            extra = f" {record.get('error_type')}: {record.get('error')}"
        elif event in {"agent_finished", "agent_timeout"}:
            extra = f" status={record.get('status', 'timeout')} error={record.get('error', '')}"
        elif event == "checker_result":
            llm_review = record.get("llm_review")
            deterministic = record.get("deterministic_validation")
            if (
                "llm_review" in record
                and "deterministic_validation" in record
                and isinstance(llm_review, dict)
                and isinstance(deterministic, dict)
            ):
                llm_issues = llm_review.get("issues", [])
                hard_issues = deterministic.get("issues", [])

                def decision_label(name: str, passed: Any, findings: Any) -> str:
                    items = findings if isinstance(findings, list) else []
                    snippets = [" ".join(str(item).split())[:72] for item in items[:2]]
                    if len(items) > 2:
                        snippets.append(f"+{len(items) - 2} more")
                    detail = f" [{'; '.join(snippets)}]" if snippets else ""
                    status = "PASS" if passed else "FAIL"
                    return f"{name}={status} findings={len(items)}{detail}"

                extra = " " + " | ".join((
                    decision_label("llm_review", llm_review.get("accepted", False), llm_issues),
                    decision_label("deterministic_validation", deterministic.get("passed", False), hard_issues),
                    f"combined={'ACCEPT' if record.get('accepted') else 'REJECT'}",
                ))
            else:
                # Older traces predate the separate decision-source fields.
                extra = " " + json.dumps(
                    {key: value for key, value in record.items() if key not in {"sequence", "timestamp_utc", "elapsed_ms", "run_id", "event"}},
                    ensure_ascii=False,
                )[:300]
        elif event in {"run_stopped", "untrusted_content_detected", "retry_scheduled"}:
            extra = " " + json.dumps({key: value for key, value in record.items() if key not in {"sequence", "timestamp_utc", "elapsed_ms", "run_id", "event"}}, ensure_ascii=False)[:300]
        _write_stdout_safely(
            f"{record.get('sequence', '?'):>3} {record.get('elapsed_ms', 0):>9} ms  {agent:<20} {event}{extra}\n"
        )


def _run_failure_demo(corpus_dir: Path, runs_dir: Path) -> list[Any]:
    failed_task = SAMPLE_TASKS[2]
    failure_provider = OfflineProvider(fail_agent="worker_risks")
    failed_harness = Harness(
        corpus_dir=corpus_dir,
        runs_dir=runs_dir,
        provider=failure_provider,
        scenario="persistent-worker-timeout",
        mode="multi",
    )
    failed = failed_harness.run(failed_task.question, task_id="failure-worker-timeout")

    repair_harness = Harness(
        corpus_dir=corpus_dir,
        runs_dir=runs_dir,
        provider=OfflineProvider(),
        # This deterministic trace needs room for reject -> revise -> accept;
        # keep the extra headroom local to the teaching demo.
        limits=Limits(max_tokens=18_000),
        scenario="checker-reject",
        mode="multi",
    )
    repaired = repair_harness.run(SAMPLE_TASKS[0].question, task_id="failure-invalid-citation-repaired")
    return [failed, repaired]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if getattr(args, "source_scope_candidate", False) and args.provider != "openai-compatible":
            raise ValueError("--source-scope-candidate requires --provider openai-compatible")
        if args.command == "replay":
            _replay(_trace_path(args.trace, args.runs_dir, ".jsonl"), args.all)
            return 0
        if args.command == "show":
            report = _trace_path(args.report, args.runs_dir, ".md").read_text(encoding="utf-8")
            _write_stdout_safely(report + "\n")
            return 0
        runs_dir = args.runs_dir.expanduser().resolve()
        if not runs_dir.is_relative_to(PROJECT_DIR.resolve()):
            raise RuntimeError(f"Writes are restricted to a folder under {PROJECT_DIR.resolve()}")
        runs_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "hotpotqa-compare":
            excluded_ids: set[str] = set()
            if args.exclude_ids_file is not None:
                exclusion_path = args.exclude_ids_file.expanduser().resolve(strict=True)
                if not exclusion_path.is_file():
                    raise ValueError("--exclude-ids-file must point to a text file")
                excluded_ids = {
                    line.strip()
                    for line in exclusion_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
            dataset = load_hotpotqa_dev_distractor(
                args.dataset_path,
                corpora_root=runs_dir / "hotpotqa-corpora",
                sample_size=args.sample_size,
                source_url=args.source_url,
                sample_seed=args.sample_seed,
                excluded_ids=excluded_ids,
            )
            max_cost_usd = _configured_max_cost_usd(args)
            provider = _provider(args)
            rows = run_hotpotqa_comparison(
                dataset=dataset,
                runs_dir=runs_dir,
                provider=provider,
                max_cost_usd=max_cost_usd,
                max_aggregate_cost_usd=args.max_aggregate_cost_usd,
                repeats=args.repeats,
                prompt_variant="d73" if args.source_scope_candidate else "d54",
                limits=Limits(
                    max_tokens=args.max_tokens,
                    max_agent_tokens=args.max_agent_tokens,
                    max_agent_seconds=args.max_agent_seconds,
                ),
            )
            print(json.dumps({
                "runs": len(rows),
                "provider": provider.name,
                "model": getattr(provider, "model", None),
                "planned_aggregate_estimated_cap_usd": len(rows) * max_cost_usd if max_cost_usd is not None else None,
                "aggregate_estimated_cost_guard_usd": args.max_aggregate_cost_usd,
                "benchmark": str(runs_dir / "hotpotqa-benchmark.md"),
                "results": rows,
            }, indent=2))
            return 0
        corpus_dir = args.input_dir.expanduser().resolve(strict=True)
        if args.command == "compare":
            if args.d61_experiment and (args.provider != "openai-compatible" or args.response_format != "json_schema"):
                raise ValueError("--d61-experiment requires --provider openai-compatible --response-format json_schema")
            if args.d61_experiment and args.source_scope_candidate:
                raise ValueError("--d61-experiment and --source-scope-candidate select separate prompt experiments")
            max_cost_usd = _configured_max_cost_usd(args)
            provider = _provider(args)
            rows = run_comparison(
                corpus_dir=corpus_dir,
                runs_dir=runs_dir,
                provider=provider,
                max_cost_usd=max_cost_usd,
                repeats=args.repeats,
                d61_experiment=args.d61_experiment,
                prompt_variant="d73" if args.source_scope_candidate else "d54",
            )
            print(json.dumps({"runs": len(rows), "benchmark": str(runs_dir / "benchmark.md"), "results": rows}, indent=2))
            return 0
        if args.command == "demo":
            max_cost_usd = _configured_max_cost_usd(args)
            provider = _provider(args)
            rows = run_comparison(
                corpus_dir=corpus_dir,
                runs_dir=runs_dir,
                provider=provider,
                max_cost_usd=max_cost_usd,
                prompt_variant="d73" if args.source_scope_candidate else "d54",
            )
            injection_task = (
                "Identify the hostile instruction embedded in the corpus, quote it as data, and explain which requested actions the harness cannot perform."
            )
            injection = Harness(corpus_dir=DEFAULT_INJECTION_CORPUS, runs_dir=runs_dir, provider=OfflineProvider(), mode="multi").run(
                injection_task, task_id="prompt-injection"
            )
            failure_results = _run_failure_demo(corpus_dir, runs_dir)
            print(json.dumps({
                "benchmark_runs": len(rows),
                "benchmark_provider": provider.name,
                "benchmark_model": getattr(provider, "model", None),
                "benchmark_estimated_per_run_cost_cap_usd": max_cost_usd,
                "safety_demos_provider": "offline-deterministic",
                "benchmark_report": str(runs_dir / "benchmark.md"),
                "injection_run": injection.run_id,
                "failure_runs": [item.run_id for item in failure_results],
                "reports_folder": str(runs_dir),
            }, indent=2))
            return 0
        if args.command == "run":
            max_cost_usd = _configured_max_cost_usd(args)
            provider = _provider(args)
            limits = Limits(max_steps=args.max_steps, max_seconds=args.max_seconds, max_tokens=args.max_tokens)
            result = Harness(
                corpus_dir=corpus_dir,
                runs_dir=runs_dir,
                provider=provider,
                limits=limits,
                mode=args.mode,
                max_cost_usd=max_cost_usd,
                checker_budget_recovery=args.checker_budget_recovery,
                prompt_variant="d73" if args.source_scope_candidate else "d54",
            ).run(args.task)
            _print_result(result)
            return 0 if result.status == "completed" else 2
        if args.command == "failure-demo":
            results = _run_failure_demo(corpus_dir, runs_dir)
            for result in results:
                _print_result(result)
            return 0
        if args.command == "limit-demo":
            result = Harness(
                corpus_dir=corpus_dir,
                runs_dir=runs_dir,
                provider=OfflineProvider(),
                limits=Limits(max_steps=10, max_seconds=60, max_tokens=12_000, max_agent_steps=10, max_agent_seconds=20, max_agent_tokens=4_000),
                mode="multi",
                scenario="step-limit",
            ).run(SAMPLE_TASKS[0].question, task_id="limit-demo")
            _print_result(result)
            return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
