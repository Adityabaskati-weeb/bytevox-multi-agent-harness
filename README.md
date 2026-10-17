# ByteVox Harness

A Python CLI harness for answering questions from local text documents. A lead plans the work, three specialists gather evidence in parallel, a lead synthesizes an answer, and a checker reviews its claims. Runs save reports, manifests, and inspectable JSONL traces. The default is deterministic and offline; the sample North Harbor corpus is fictional.

**[Watch the 3:35 demo replay](submission/bytevox-demo-replay.mp4)** · [Walkthrough script](DEMO_SCRIPT.md). The video is assembled from saved command output and traces, not a live screen recording. It shows current-source D95 and D96 model runs, D74's partial/unverified run, failure handling, a run limit, and the measured trade-off. D77 and D91 are preserved as historical evidence from before D92's fixture-isolation change.

## Run it

Requires Python 3.11+; no runtime dependencies are needed. From the project root in PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "src"
python -m bytevox_harness demo
```

The demo uses the offline provider. Other useful commands:

```powershell
python -m bytevox_harness compare
python -m bytevox_harness run "Summarize the open accessibility risks"
python -m bytevox_harness failure-demo
python -m bytevox_harness limit-demo
python -m bytevox_harness show <run-id>
python -m bytevox_harness replay <run-id> --all
```

Runs write artifacts under `runs/`. `show` and `replay` inspect saved runs without rerunning agents. Use a local UTF-8 `.md` or `.txt` corpus with `--input-dir` for a new task.

### Configured model

An OpenAI-compatible endpoint can use `OPENAI_API_KEY` (default), `BYTEVOX_BASE_URL` and `BYTEVOX_MODEL`. The base URL defaults to local Ollama; the example below sets illustrative rates, which you should replace with the chosen model's rates.

```powershell
$env:OPENAI_API_KEY = "<your-api-key>"
$env:BYTEVOX_BASE_URL = "https://api.example.com/v1"
$env:BYTEVOX_MODEL = "<provider-model-id>"
$env:BYTEVOX_INPUT_USD_PER_1M = "0.10"
$env:BYTEVOX_OUTPUT_USD_PER_1M = "0.40"
python -m bytevox_harness run "Summarize the main claims" `
  --provider openai-compatible `
  --input-dir "C:\path\to\your\corpus" `
  --approve-remote-model `
  --max-cost-usd 0.05
```

Remote use requires typing `APPROVE`. The USD guard estimates cost before requests using an approximate character-based token count and the requested completion cap; it is not a provider billing limit. `--api-key-env` selects another key variable. See the completed [D91 comparison protocol and results](runs/current-source-d90-3x2-20261017/equal-budget-followup.md) and the frozen [D90 plan](runs/current-source-d90-3x2-20261017/campaign-plan.md).

## How it works

```text
Goal → lead planner → three parallel specialists → lead synthesizer → checker
              └──── structured handoffs and shared evidence ─────┘
                                           reject → lead reviser → checker (once)
                                      run events → JSONL trace
```

| Specialist | Responsibility | Allowed corpus tools |
|---|---|---|
| `worker_causes` | Find direct evidence | Search |
| `worker_outcomes` | Find independent corroboration | List and search |
| `worker_risks` | Find gaps and caveats | Search and read evidence |

The lead plans task-specific facets. Specialists have distinct instructions and tool allowlists. The single-agent baseline has the union of the same three corpus tools; the multi-agent run splits those tools across roles. A bounded shared evidence pool is merged for synthesis and checker review, and its handoffs are visible in the trace. The harness enforces run and per-agent step, time, and token limits, retries once with backoff, and limits checker revision to one loop. When a worker or later stage fails, usable work is retained and the run is labelled partial. Checker acceptance also requires an empty issue list and passing deterministic checks.

Agents only receive fixed, read-only corpus tools. They cannot write files, run shell commands, or browse the network. Corpus text and tool results are treated as untrusted data. Remote model requests require explicit CLI consent and typed approval. Reports include run status and elapsed time; JSONL traces record timestamps, agent handoffs, model inputs and results, tool calls and results, token usage, budget events, and checker decisions. Use `show` for the report or `replay` for the event timeline.

## Latest model-backed comparison: D95 current source (`e6184eb`, post-D92)

D95 ran the current `e6184eb` source on the same three public HotpotQA dev-distractor tasks as D91, with one repeat per mode. It used the same Hugging Face Router model, D54 prompt hash, question, task corpora, generic validation contract, and equal run-level limits. The dataset's gold answers and supporting-fact labels stayed scorer-only. The model prompt hash matches D91; the source hash is different and is recorded in each manifest.

| Mode | Completed | Answer F1 | Support F1 | Joint F1 | Median seconds | Known token-derived cost |
|---|---:|---:|---:|---:|---:|---:|
| Multi-agent | 3/3 | 0.667 | 1.000 | 0.667 | 19.800 | $0.0056391 |
| Single-agent | 3/3 | 0.667 | 0.889 | 0.667 | 4.095 | $0.0013652 |

The paired joint-F1 difference was 0.0000 (95% descriptive interval [0.0000, 0.0000]); support-F1 difference was +0.1111 ([0.0000, +0.3333]), which includes zero. Multi-agent was 4.84× slower by median time and 4.13× higher by known token-derived cost. The six runs reported 40,781 provider tokens and $0.0070043 estimated token cost in total, within the $0.06 aggregate preflight guard. Each run stayed below the $0.01 estimate cap. These rate-based costs are not an invoice or provider billing limit.

All six D95 runs completed; D91 had one partial multi-agent run. This difference across one repeat per task does not show that the current source or D92 caused a completion improvement. One answer receives zero official answer F1 despite support in its cited source because the public reference is semantically broader; this limits interpretation. With three public development tasks and one repeat, the sample cannot establish superiority, equivalence, or generalization; the data may overlap model pretraining. The [D95 reproduction package](runs/current-source-e6184eb-3x2-20261017/reproducibility/README.md) includes only the three selected HotpotQA records and instructions to rerun both modes; its context regenerates the same per-task corpora, while reference labels remain scorer-only.

Both modes shared a 32-step, 60-second, 12,000-token run envelope. The baseline's per-agent caps were raised to the run caps; its response cap was 1,536 tokens. Multi-agent roles used per-agent caps of 24 steps, 20 seconds, and 4,000 tokens; worker calls use a 1,024-token completion cap and lead calls use 1,536. Role topology and per-request budgets therefore still differ. No prompt or runtime default changed. The video shows completed multi-agent run `20261017T125550Z-909c1192`; its [report](runs/current-source-e6184eb-3x2-20261017/20261017T125550Z-909c1192.md), [trace](runs/current-source-e6184eb-3x2-20261017/20261017T125550Z-909c1192.jsonl), and [manifest](runs/current-source-e6184eb-3x2-20261017/20261017T125550Z-909c1192.manifest.json) record the displayed answer, handoffs, checker result, source hash, and usage. See the [D95 campaign record](runs/current-source-e6184eb-3x2-20261017/campaign-record.md), [aggregate report](runs/current-source-e6184eb-3x2-20261017/hotpotqa-benchmark.md), [CSV](runs/current-source-e6184eb-3x2-20261017/hotpotqa-benchmark.csv), and [JSON](runs/current-source-e6184eb-3x2-20261017/hotpotqa-benchmark.json).

### Historical model-backed comparison: D91 (pre-D92 D54 snapshot)

D91 is a model-backed, equal-run-envelope follow-up on three public HotpotQA development tasks with one repeat per mode. Both modes used the same Hugging Face model, question, task corpus, generic validation contract, and shared run-level limits; the scorer kept answer and supporting-fact labels outside model input. D91 used the D54 evaluation source snapshot from D90, before D92's offline injection-fixture isolation. It is historical evidence and not a result on current source; the D95 follow-up above supplies the later current-source run. It remains a small descriptive pilot, not a general or causal test of multi-agent quality.

| Mode | Completed | Answer F1 | Support F1 | Joint F1 | Median seconds | Known token-derived cost |
|---|---:|---:|---:|---:|---:|---:|
| Multi-agent | 2/3 | 0.667 | 1.000 | 0.667 | 20.563 | $0.0061378 |
| Single-agent | 3/3 | 0.667 | 0.889 | 0.667 | 4.576 | $0.0012950 |

The paired joint-F1 difference was 0.0000 (95% descriptive interval [0.0000, 0.0000]); support-F1 difference was +0.1111 ([0.0000, +0.3333]), which includes no difference. Multi-agent was 4.49× slower by median time and 4.74× higher by known token-derived cost. All six runs reported provider token usage: 42,315 tokens and $0.0074328 estimated token cost in total, within the $0.06 aggregate preflight guard. These costs use provider-reported tokens and listed rates, not an invoice.

Both modes shared a 32-step, 60-second, 12,000-token run envelope. The baseline's per-agent caps were raised to the run caps; its response cap was raised to 1,536 tokens to match multi-agent lead calls. Role topology still differs: multi-agent worker calls use a 1,024-token completion cap, while the single-agent call cap is 1,536. One multi-agent run stopped at 11,811/12,000 provider-reported tokens before synthesis and checker; its partial answer is unverified. On one task, the gold answer is the generic phrase “international football competition,” while the cited source supports the more specific “FIFA Women's World Cup”; official normalization scores that supported answer as zero. Scores include the partial run.

With only three public dev tasks and one repeat, this pilot cannot establish superiority, equivalence, or generalization; the data may overlap model pretraining. The results show no joint-F1 gain in this sample and higher measured latency and cost for multi-agent. D90 is the earlier two-repeat pilot and remains historical because its effective per-mode budgets were asymmetric. No runtime default changed. See the [D91 campaign record](runs/current-source-d90-3x2-20261017/campaign-record.md), [D91 aggregate report](runs/current-source-d90-3x2-20261017/equal-budget-followup/hotpotqa-benchmark.md), [completed protocol](runs/current-source-d90-3x2-20261017/equal-budget-followup.md), and [experiment history](EXPERIMENT_HISTORY.md).

## Safety and failure evidence

The [demo walkthrough](DEMO_SCRIPT.md) shows a completed run and trace, a configured-model D96 security-summary run, D74's partial/unverified run, offline failure handling, and a step-limit stop. In D96, the checker accepted the answer; the trace records five detections attributed to the fixture source and only allowlisted corpus-tool calls, with no unavailable tool calls. This one observation shows the run stayed on task; it does not establish general injection resistance. See D96's [report](runs/d96-injection-current-20261017/20261017T133247Z-dfb9aeaf.md), [trace](runs/d96-injection-current-20261017/20261017T133247Z-dfb9aeaf.jsonl), and [manifest](runs/d96-injection-current-20261017/20261017T133247Z-dfb9aeaf.manifest.json). [D58](runs/d58-injection-live-20261016/20261016T161101Z-2926a876.jsonl) explicitly asked agents to identify a hostile document instruction and completed; [D74](runs/d74-unwarned-injection-smoke/model-run/20261017T034335Z-504d8d6f.jsonl) used a risk-summary task, where the checker rejected the draft before a budget stop prevented revision. The current default corpus excludes the hostile paragraph; the offline fixture remains available for trace-attribution checks. A synchronous custom provider that ignores its timeout cannot be forcibly stopped by Python, so final artifact writing can wait for it to return.

## Decisions, scope, and AI use

| Decision | Why it fits | Trade-off |
|---|---|---|
| Use a local text corpus and deterministic offline default. | Reproducible without credentials; local tools also support configured models. | Requires a corpus and has no live web retrieval. |
| Use a lead and three distinct, least-privilege specialists. | Parallel coverage with inspectable, structured handoffs. | More calls add latency and cost; tools are fixed. |
| Bound retries and checker revisions; keep partial results. | Failures stop cleanly and leave a reviewable trace. | A budget stop can leave an answer unchecked. |
| Use CLI reports and JSONL replay; require approval for remote requests. | Reviewable without a separate UI; remote model use requires explicit approval. | Terminal interface; estimated USD cap is not provider-enforced. |

AI assistants helped with architecture, implementation, and evaluation review. One AI judge's prose said three single-agent outputs lacked a substantive answer while its rating table recorded four; comparing packet rows with benchmark records exposed the mismatch, and the report was corrected. AI reviewer scores are exploratory, not human ground truth.

### Assignment scope and time guidance

The 5.5-hour cap is assignment guidance for prioritizing a small, inspectable deliverable; it is not a measured completion time.

| Assignment area | Shipped scope | Scope cut |
|---|---|---|
| Document analysis | Local read-only corpus, offline-default CLI, and saved run artifacts | Web retrieval and a separate web UI |
| Agent workflow | Bounded fixed-role coordination with checker review and inspectable traces | Persistent memory and dynamic tool invocation |

Elapsed hands-on time was not tracked or verified, so this project does not claim completion within 5.5 hours or report an actual-hours figure.

## More detail

- [Execution flow](flow.md): file and function paths through a run.
- [Decision log](decision.md): implementation choices, evidence, and limits.
- [Experiment history](EXPERIMENT_HISTORY.md): detailed prior results and methods.
- [Run artifact index](runs/README.md): trace, report, and manifest provenance.
- [Demo walkthrough](DEMO_SCRIPT.md): commands and narration for the recording.
