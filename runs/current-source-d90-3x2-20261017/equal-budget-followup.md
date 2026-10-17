# D91 equal-run-envelope HotpotQA follow-up (completed)

## Purpose and scope

D91 is a completed, post-D90 comparison of the D54 multi-agent harness and its same-build single-agent baseline on three public HotpotQA dev-distractor tasks, one repeat per mode (six runs). The two modes received the same task, corpus, provider/model, generic validation contract, and shared run-level limits. Gold answers and supporting-fact labels remained scorer-only. The comparison used the same evaluation source snapshot as D90; a later offline injection-fixture isolation change was not included and no model run was repeated after it. This is a small descriptive pilot, not a general or causal quality test. D90's historical measurements remain unchanged; D90 used asymmetric effective budgets and is not the equal-run-envelope result.

## Effective budget policy

The benchmark policy is `equal-run-envelope-v1`. Both modes had 32 run steps, 60 seconds, and 12,000 total tokens. Multi-agent kept its configured per-agent caps; the single-agent baseline's per-agent caps were raised to the run caps, and its `max_completion_tokens` was raised to the multi-agent lead cap. The call topology still differs: multi-agent worker requests retain the 1,024-token general completion cap, while single-agent requests can use 1,536 tokens. The policy equalizes the maximum shared run envelope and the lead-response ceiling; it does not make prompts, call counts, or role topology identical.

| Mode | Run steps / seconds / tokens | Per-agent steps / seconds / tokens | Completion tokens (general / lead) |
|---|---|---|---|
| Multi-agent | 32 / 60 / 12,000 | 24 / 20 / 4,000 | 1,024 / 1,536 |
| Single-agent | 32 / 60 / 12,000 | 32 / 60 / 12,000 | 1,536 / 1,536 |

## Results

| Mode | Completed | Answer F1 | Support F1 | Joint F1 | Median seconds | Known provider tokens | Known token-derived cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multi-agent | 2/3 | 0.667 | 1.000 | 0.667 | 20.563 | 32,998 | $0.0061378 |
| Single-agent | 3/3 | 0.667 | 0.889 | 0.667 | 4.576 | 9,317 | $0.0012950 |

The paired multi-minus-single joint-F1 difference was 0.0000 (95% descriptive paired-bootstrap interval [0.0000, 0.0000]); support-F1 difference was +0.1111 ([0.0000, +0.3333]). The support interval includes zero. Multi-agent median latency was 4.49× higher and known token-derived cost was 4.74× higher. All six runs had known provider usage, totaling 42,315 provider-reported tokens and $0.0074328 estimated token cost. The $0.06 aggregate preflight guard is a rate-based estimate, not an invoice or provider billing ceiling.

Metrics include partial outputs. The first multi-agent run used 11,811 of 12,000 provider-reported tokens and stopped before synthesis and checker; its saved answer is partial and unverified. For that task, the reference answer is the generic phrase “international football competition,” while the corpus supports “FIFA Women's World Cup.” Official answer normalization assigns the specific supported answer zero answer F1. The same reference mismatch limits interpretation of this three-task score.

This sample cannot establish multi-agent superiority, equivalence, or generalization; it uses a public dev split that may overlap model pretraining. It shows a joint-F1 tie in these three task IDs alongside higher measured latency and token-derived cost for multi-agent. No runtime default was changed.

## Exact command used

The run used the local dataset and D90 exclusion list, with labels withheld from model input. The credential was read from the `HF_TOKEN` environment variable; its value was not printed or written to artifacts. The CLI's typed `APPROVE` prompt was answered before requests began.

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONUTF8 = "1"
$env:BYTEVOX_INPUT_USD_PER_1M = "0.10"
$env:BYTEVOX_OUTPUT_USD_PER_1M = "0.40"
python -m bytevox_harness hotpotqa-compare `
  --dataset-path runs/public-data/hotpot_dev_distractor_v1.json `
  --exclude-ids-file runs/current-source-d90-3x2-20261017/excluded-dataset-ids.txt `
  --sample-size 3 --sample-seed 20261017 --repeats 1 `
  --max-tokens 12000 --max-agent-tokens 4000 --max-agent-seconds 20 `
  --provider openai-compatible --base-url https://router.huggingface.co/v1 `
  --model openai/gpt-oss-120b:nscale --api-key-env HF_TOKEN `
  --response-format json_schema --approve-remote-model `
  --max-cost-usd 0.01 --max-aggregate-cost-usd 0.06 `
  --runs-dir runs/current-source-d90-3x2-20261017/equal-budget-followup
```

The provider was the Hugging Face Inference Providers Router and model `openai/gpt-oss-120b:nscale`; the run used `json_schema` responses. The six runs reported $0.0074328 token-derived cost against the $0.06 estimated aggregate guard. Per-run and aggregate guards are estimates from token rates, not provider billing caps.

## Artifacts

- [Aggregate benchmark report](equal-budget-followup/hotpotqa-benchmark.md), [CSV](equal-budget-followup/hotpotqa-benchmark.csv), and [JSON](equal-budget-followup/hotpotqa-benchmark.json).
- Partial multi-agent run: [report](equal-budget-followup/20261017T115928Z-3d7856f0.md), [trace](equal-budget-followup/20261017T115928Z-3d7856f0.jsonl), and [manifest](equal-budget-followup/20261017T115928Z-3d7856f0.manifest.json).
- The aggregate report links the reports, traces, and manifests for all six runs.
- D90 remains documented in the [historical D90 campaign record](campaign-record.md) and its [frozen plan](campaign-plan.md); D91 does not rewrite its scores.
