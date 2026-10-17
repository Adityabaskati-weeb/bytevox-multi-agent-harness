"""Render a short video from captured ByteVox CLI output and saved traces.

This produces a terminal-style replay, not a desktop screen recording. It uses
saved D95 and D96 current-source model runs plus historical and offline failure
evidence. No model or network requests are made by this renderer.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "runs" / "demo-recording"
OUTPUT = Path(__file__).resolve().parent / "bytevox-demo-replay.mp4"
D95_MULTI_REPORT = ROOT / "runs" / "current-source-e6184eb-3x2-20261017" / "20261017T125550Z-909c1192.md"
D95_MULTI_TRACE = ROOT / "runs" / "current-source-e6184eb-3x2-20261017" / "20261017T125550Z-909c1192.jsonl"
D95_MULTI_MANIFEST = ROOT / "runs" / "current-source-e6184eb-3x2-20261017" / "20261017T125550Z-909c1192.manifest.json"
D95_BENCHMARK = ROOT / "runs" / "current-source-e6184eb-3x2-20261017" / "hotpotqa-benchmark.json"
D96_INJECTION_TRACE = ROOT / "runs" / "d96-injection-current-20261017" / "20261017T133247Z-dfb9aeaf.jsonl"
D96_INJECTION_MANIFEST = ROOT / "runs" / "d96-injection-current-20261017" / "20261017T133247Z-dfb9aeaf.manifest.json"
D74_TRACE = ROOT / "runs" / "d74-unwarned-injection-smoke" / "model-run" / "20261017T034335Z-504d8d6f.jsonl"
WIDTH, HEIGHT = 1920, 1080
FPS = 30
VISIBLE_LINES = 19
LINE_HEIGHT = 34


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def read_json_objects(path: Path) -> list[dict[str, Any]]:
    text = read_text(path)
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        value, offset = decoder.raw_decode(text, offset)
        if isinstance(value, dict):
            objects.append(value)
    return objects


def choose_font(*names: str, size: int) -> ImageFont.FreeTypeFont:
    font_root = Path("C:/Windows/Fonts")
    for name in names:
        path = font_root / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def normalize_display(text: str) -> str:
    return text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")


def wrap_lines(lines: list[str], width: int = 128) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        line = normalize_display(line.expandtabs(2)).rstrip()
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(
            line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=True,
            subsequent_indent="    ",
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""])
    return wrapped


def selected_current_source_injection_lines() -> list[str]:
    events = read_json_objects(D96_INJECTION_TRACE)
    started = next(event for event in events if event.get("event") == "run_started")
    manifest = json.loads(read_text(D96_INJECTION_MANIFEST))
    detections = [event for event in events if event.get("event") == "untrusted_content_detected"]
    tool_calls = [event for event in events if event.get("event") == "tool_call"]
    tool_counts = Counter(event.get("tool", "unknown") for event in tool_calls)
    allowlists = started["tool_policy"]["role_allowlists"]
    allowed_tools = {tool for tools in allowlists.values() for tool in tools}
    non_allowlisted = sorted({tool for tool in tool_counts if tool not in allowed_tools})
    checker = next(event for event in events if event.get("event") == "checker_result")
    finished = next(event for event in events if event.get("event") == "run_finished")["result"]
    source_ids = sorted({source_id for event in detections for source_id in event.get("source_ids", [])})
    tool_summary = ", ".join(f"{tool} x{count}" for tool, count in sorted(tool_counts.items()))
    status = "none" if not non_allowlisted else ", ".join(non_allowlisted)
    return [
        "D96 current-source configured-model trace · task: summarize security-relevant document content",
        f"Run: {started['run_id']} · model={started['model']} · prompt=D54",
        f"Source SHA-256: {manifest['harness_source_sha256']}",
        f"untrusted_content_detected: {len(detections)} events · source IDs: {', '.join(source_ids)}",
        f"Allowlisted corpus-tool calls: {len(tool_calls)} ({tool_summary})",
        f"Unavailable/non-allowlisted tool calls: {status}",
        f"checker_result accepted={str(checker['accepted']).lower()}",
        f"run_finished status={finished['status']} partial={str(finished['partial']).lower()} checker_accepted={str(finished['checker_accepted']).lower()}",
        f"Elapsed: {finished['budget']['elapsed_seconds']:.3f}s · provider tokens: {finished['provider_tokens']} · token-derived cost: ${finished['cost_usd']:.7f}",
        "This one observation stayed on task; it is not a guarantee of injection resistance.",
    ]


def selected_checker_fix_lines() -> list[str]:
    events = read_json_objects(D74_TRACE)
    detections = [event for event in events if event.get("event") == "untrusted_content_detected"]
    checker = next(event for event in events if event.get("event") == "checker_result")
    revision = next(event for event in events if event.get("event") == "revision_requested")
    stop = next(event for event in events if event.get("event") == "budget_stop")
    ended = next(event for event in events if event.get("event") == "run_stopped")
    return [
        "D74 was a neutral risk-summary task; the caller did not ask to find an injection paragraph.",
        "untrusted_content_detected: " + ", ".join(dict.fromkeys(event["agent_id"] for event in detections)),
        f"checker_result accepted={str(checker['accepted']).lower()} -> revision_requested to_agent={revision.get('to_agent', 'unknown')}",
        f"budget_stop agent={stop.get('agent_id')}: {stop.get('reason')}",
        f"run_stopped partial={str(ended.get('partial')).lower()}; answer remains unverified.",
        "This is one incomplete observation, not a general injection-robustness result.",
    ]


def selected_failure_lines() -> list[str]:
    runs = read_json_objects(ARTIFACTS / "failure" / "console.txt")
    lines = ["Captured output from: python -m bytevox_harness failure-demo"]
    for item in runs:
        lines.extend([
            "{",
            f'  "run_id": "{item.get("run_id")}",',
            f'  "status": "{item.get("status")}",',
            f'  "provider": "{item.get("provider")}",',
            f'  "checker_accepted": {str(item.get("checker_accepted")).lower()},',
            f'  "steps": {item.get("steps")},',
            f'  "errors": {json.dumps(item.get("errors", []), ensure_ascii=False)}',
            "}",
        ])
    replay = read_text(ARTIFACTS / "failure" / "repaired-replay.txt").splitlines()
    key_events = [
        line for line in replay
        if "model_error" in line or "checker_result" in line or "revision_requested" in line or "run_finished" in line
    ]
    lines.extend(["", "Checker repair trace (selected events):", *key_events[-9:]])
    return lines


def selected_d95_trace_lines() -> list[str]:
    events = read_json_objects(D95_MULTI_TRACE)
    manifest = json.loads(read_text(D95_MULTI_MANIFEST))
    started = next(event for event in events if event.get("event") == "run_started")
    plan = next(event for event in events if event.get("event") == "message_received" and event.get("from_agent") == "lead_planner")
    worker_names = sorted({
        event.get("agent_id") for event in events
        if event.get("event") == "tool_call" and str(event.get("agent_id", "")).startswith("worker_")
    })
    handoffs = [
        event for event in events
        if event.get("event") == "message_received" and event.get("to_agent") == "lead_synthesizer"
    ]
    checker = next(event for event in events if event.get("event") == "checker_result")
    finished = next(event for event in events if event.get("event") == "run_finished")["result"]
    limits = started["limits"]
    allowlists = started["tool_policy"]["role_allowlists"]
    return [
        f"Trace: {started['run_id']} · mode={started['mode']} · prompt=D54 · source=e6184eb · model={started['model']}",
        f"Source SHA-256: {manifest['harness_source_sha256']}",
        f"Limits: {limits['max_tokens']} run tokens · {limits['max_agent_tokens']} per agent · {limits['max_agent_seconds']} s per agent · {limits['max_seconds']} s overall",
        f"Planner handoff: {plan['from_agent']} -> {plan['to_agent']} ({plan['message']['kind']}; 3 task facets)",
        "Parallel specialist tool work: " + ", ".join(worker_names),
        "Least-privilege tools: causes=" + "/".join(allowlists["worker_causes"]) + "; outcomes=" + "/".join(allowlists["worker_outcomes"]) + "; risks=" + "/".join(allowlists["worker_risks"]) + "; checker=" + "/".join(allowlists["checker"]),
        f"Specialist result handoffs to lead_synthesizer: {len(handoffs)}",
        f"checker_result accepted={str(checker['accepted']).lower()} issues={len(checker.get('issues', []))}",
        f"run_finished status={finished['status']} checker_accepted={str(finished['checker_accepted']).lower()}",
        f"Answer: {finished['answer']['summary'].removeprefix('Answer: ')}",
        f"Elapsed: {finished['budget']['elapsed_seconds']:.3f} s · provider tokens: {finished['provider_tokens']} · known token-derived cost: ${finished['cost_usd']:.7f}",
        "One saved successful configured-model run; not a general quality evaluation.",
    ]


def selected_d95_comparison_lines() -> list[str]:
    benchmark = json.loads(read_text(D95_BENCHMARK))
    multi = benchmark["summary"]["multi"]
    single = benchmark["summary"]["single"]
    paired = benchmark["metadata"]["paired_uncertainty"]["metrics"]
    joint = paired["joint_f1"]
    support = paired["supporting_fact_f1"]
    total_tokens = multi["provider_tokens_known_total"] + single["provider_tokens_known_total"]
    total_cost = multi["api_cost_known_total_usd"] + single["api_cost_known_total_usd"]
    speed_ratio = multi["wall_time_median_seconds"] / single["wall_time_median_seconds"]
    cost_ratio = multi["api_cost_known_total_usd"] / single["api_cost_known_total_usd"]
    return [
        "D95: current e6184eb source after D92; prompt hash still D54.",
        "3 public HotpotQA dev tasks x 1 repeat per mode; labels stayed scorer-only.",
        "Shared run envelope: 32 steps / 60 sec / 12,000 tokens for each mode.",
        f"Multi: {multi['completed_runs']}/3 complete; joint F1 {multi['joint_f1_mean']:.3f}; support F1 {multi['supporting_fact_f1_mean']:.3f}; median {multi['wall_time_median_seconds']:.3f}s.",
        f"Single: {single['completed_runs']}/3 complete; joint F1 {single['joint_f1_mean']:.3f}; support F1 {single['supporting_fact_f1_mean']:.3f}; median {single['wall_time_median_seconds']:.3f}s.",
        f"Known token-derived cost: multi ${multi['api_cost_known_total_usd']:.7f}; single ${single['api_cost_known_total_usd']:.7f}.",
        f"Joint-F1 delta {joint['mean_multi_minus_single']:+.3f}; paired 95% interval [{joint['ci95_percentile_low']:+.3f}, {joint['ci95_percentile_high']:+.3f}].",
        f"Support-F1 delta {support['mean_multi_minus_single']:+.3f}; interval [{support['ci95_percentile_low']:+.3f}, {support['ci95_percentile_high']:+.3f}] includes zero.",
        f"Multi was {speed_ratio:.2f}x slower and {cost_ratio:.2f}x higher known cost; all six runs reported usage.",
        f"Known total: {total_tokens:,} provider tokens; ${total_cost:.7f} vs $0.06 estimated guard.",
        "No joint-F1 gain is measured; three tasks and one repeat cannot show superiority or generalize.",
        "One answer has source support but scores 0 against a semantically broader public reference.",
        "Baseline gets full run envelope; multi workers use 1,024-token calls and leads 1,536. Topology differs.",
        "D91/D90 are historical; D91 predates D92 and one multi run stopped before checker.",
        "This small descriptive pilot does not establish a causal effect or equivalence.",
    ]


def selected_limit_lines() -> list[str]:
    runs = read_json_objects(ARTIFACTS / "limit" / "console.txt")
    item = runs[-1]
    replay = read_text(ARTIFACTS / "limit" / "limit-replay.txt").splitlines()
    key_events = [line for line in replay if "budget_stop" in line or "run_stopped" in line or "run_finished" in line]
    return [
        "Captured output from: python -m bytevox_harness limit-demo",
        "{",
        f'  "run_id": "{item.get("run_id")}",',
        f'  "status": "{item.get("status")}",',
        f'  "steps": {item.get("steps")}  // configured maximum: 10',
        f'  "errors": {json.dumps(item.get("errors", []), ensure_ascii=False)}',
        "}",
        "",
        "Trace confirms the stop and preserves the partial run:",
        *key_events,
    ]


def build_scenes() -> list[dict[str, Any]]:
    normal_report = read_text(D95_MULTI_REPORT).splitlines()
    normal_replay = selected_d95_trace_lines()
    current_injection_lines = selected_current_source_injection_lines()
    failure_lines = selected_failure_lines()
    final_comparison_lines = selected_d95_comparison_lines()
    limit_lines = selected_limit_lines()

    return [
        {
            "title": "ByteVox multi-agent harness",
            "caption": "D95 and D96 current-source model runs + D74 postmortem; offline failure and limit examples.",
            "command": "Saved model artifacts only · no request is made during rendering",
            "lines": [
                "No model request is made while rendering this video.",
                "",
                "Walkthrough:",
                "  1. Inspect D95: a saved configured-model run from current source commit e6184eb",
                "  2. Inspect one D96 current-source security-summary run over the injection fixture",
                "  3. Review D74's neutral-task partial run; it remains unverified",
                "  4. Demonstrate offline failure recovery, a limit, and the D95 measured trade-off",
            ],
            "seconds": 10,
        },
        {
            "title": "01  D95 current-source configured-model answer",
            "caption": "Saved run from e6184eb after D92 fixture isolation; checker accepted this run.",
            "command": "python -m bytevox_harness show 20261017T125550Z-909c1192 --runs-dir runs/current-source-e6184eb-3x2-20261017",
            "lines": normal_report,
            "seconds": 30,
        },
        {
            "title": "02  Follow the D95 handoffs and review",
            "caption": "Current-source e6184eb trace: lead plan, parallel specialists, synthesis, checker, limits, and usage.",
            "command": "python -m bytevox_harness replay 20261017T125550Z-909c1192 --runs-dir runs/current-source-e6184eb-3x2-20261017",
            "lines": normal_replay,
            "seconds": 35,
        },
        {
            "title": "03  D96 current-source model injection observation",
            "caption": "One completed model run stayed on the security-summary task and used only allowlisted corpus tools.",
            "command": "Saved D96 report, JSONL trace, and manifest · no request is made during rendering",
            "lines": current_injection_lines,
            "seconds": 22,
        },
        {
            "title": "04  D74 neutral-task partial postmortem",
            "caption": "The checker rejected an unsupported claim; budget stopped revision, leaving a partial run.",
            "command": "Saved D74 trace · neutral risk-summary task",
            "lines": selected_checker_fix_lines(),
            "seconds": 20,
        },
        {
            "title": "05  Recover from a specialist failure",
            "caption": "Deterministic offline demo: one worker times out; another run rejects an invalid citation, revises, then passes.",
            "command": "python -m bytevox_harness failure-demo --runs-dir runs/demo-recording/failure",
            "lines": failure_lines,
            "seconds": 37,
        },
        {
            "title": "06  Enforce the run step limit",
            "caption": "Deterministic offline demo: the harness stops at step 10 and keeps a partial result and trace.",
            "command": "python -m bytevox_harness limit-demo --runs-dir runs/demo-recording/limit",
            "lines": limit_lines,
            "seconds": 21,
        },
        {
            "title": "07  D95 current-source comparison",
            "caption": "Three public tasks x one repeat; a descriptive result, not a quality win or general conclusion.",
            "command": "D95 aggregate report · D91 and D90 remain historical",
            "lines": final_comparison_lines,
            "seconds": 40,
        },
    ]


def render_frame(
    scene: dict[str, Any],
    chapter_index: int,
    chapter_count: int,
    elapsed: int,
    total_seconds: int,
    fonts: tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont],
) -> Image.Image:
    title_font, ui_font, mono_font = fonts
    image = Image.new("RGB", (WIDTH, HEIGHT), "#090f1d")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 10), fill="#4f8cff")
    draw.text((70, 36), scene["title"], fill="#f2f6ff", font=title_font)
    draw.text((72, 96), scene["caption"], fill="#aab8ce", font=ui_font)
    draw.text((1590, 44), f"{chapter_index:02d} / {chapter_count:02d}", fill="#7d93b3", font=ui_font)

    panel = (62, 175, WIDTH - 62, 1000)
    draw.rounded_rectangle(panel, radius=16, fill="#101827", outline="#26364d", width=2)
    draw.rounded_rectangle((62, 175, WIDTH - 62, 226), radius=16, fill="#182438")
    draw.rectangle((62, 209, WIDTH - 62, 226), fill="#182438")
    for x, color in ((91, "#ff6b6b"), (118, "#f4bf4f"), (145, "#42c98a")):
        draw.ellipse((x - 7, 195 - 7, x + 7, 195 + 7), fill=color)
    draw.text((176, 182), "bytevox-harness  ·  captured CLI replay", fill="#b7c7df", font=ui_font)

    command = normalize_display(scene["command"])
    command_lines = textwrap.wrap(command, width=128, break_long_words=True, break_on_hyphens=False)
    y = 244
    for line in command_lines[:2]:
        draw.text((88, y), f"PS> {line}", fill="#7ee787", font=mono_font)
        y += LINE_HEIGHT
    y += 10

    body = wrap_lines(scene["lines"])
    if len(body) <= VISIBLE_LINES:
        start = 0
    else:
        last_start = len(body) - VISIBLE_LINES
        start = round(last_start * elapsed / max(1, scene["seconds"] - 1))
    visible = body[start:start + VISIBLE_LINES]
    for line in visible:
        color = "#e4ebf5"
        if line.startswith(("Replay:", "## ", "**Status:", "**Mode:", "**Provider:", "**Model:")):
            color = "#9dc1ff"
        draw.text((88, y), line, fill=color, font=mono_font)
        y += LINE_HEIGHT

    draw.text((72, 1020), "Prepared from real CLI output and saved UTF-8 traces · no model call during rendering", fill="#72839e", font=ui_font)
    bar_x, bar_y, bar_w, bar_h = 1420, 1032, 425, 8
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=4, fill="#1d2a3e")
    progress = min(1.0, (sum(s["seconds"] for s in SCENES[:chapter_index - 1]) + elapsed + 1) / total_seconds)
    draw.rounded_rectangle((bar_x, bar_y, bar_x + int(bar_w * progress), bar_y + bar_h), radius=4, fill="#4f8cff")
    return image


SCENES: list[dict[str, Any]] = []


def main() -> None:
    global SCENES
    SCENES = build_scenes()
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required to create and validate the MP4")
    fonts = (
        choose_font("segoeuib.ttf", "segoeui.ttf", size=36),
        choose_font("segoeui.ttf", "arial.ttf", size=21),
        choose_font("consola.ttf", "lucon.ttf", size=22),
    )
    total_seconds = sum(scene["seconds"] for scene in SCENES)
    if total_seconds > 240:
        raise RuntimeError(f"Demo is {total_seconds} seconds; maximum is 240 seconds")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(prefix=".bytevox-demo-candidate-", suffix=".mp4", dir=OUTPUT.parent, delete=False) as temp_output:
        candidate = Path(temp_output.name)
    try:
        with tempfile.TemporaryDirectory(prefix="bytevox-demo-frames-") as temp:
            frame_dir = Path(temp)
            frame_index = 0
            for chapter_index, scene in enumerate(SCENES, start=1):
                for elapsed in range(scene["seconds"]):
                    frame = render_frame(scene, chapter_index, len(SCENES), elapsed, total_seconds, fonts)
                    frame.save(frame_dir / f"frame-{frame_index:05d}.png", optimize=True)
                    frame_index += 1
            subprocess.run([
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", "1", "-start_number", "0",
                "-i", str(frame_dir / "frame-%05d.png"),
                "-r", str(FPS), "-c:v", "libx264", "-crf", "22",
                "-pix_fmt", "yuv420p", str(candidate),
            ], check=True)

        probe = subprocess.run([
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(candidate),
        ], check=True, capture_output=True, text=True)
        duration = float(probe.stdout.strip())
        if duration > 240 or abs(duration - total_seconds) > 1 / FPS:
            raise RuntimeError(f"Candidate duration {duration:.3f}s does not match the {total_seconds}s script or exceeds 240s")

        # Decode every video frame before atomically replacing the previous deliverable.
        subprocess.run([
            ffmpeg, "-v", "error", "-i", str(candidate), "-f", "null", "-",
        ], check=True, capture_output=True, text=True)
        candidate.replace(OUTPUT)
        print(f"Validated and created {OUTPUT} ({duration:.3f}s, terminal-style replay)")
    finally:
        candidate.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
