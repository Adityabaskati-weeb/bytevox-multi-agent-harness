# D95 current-source HotpotQA pilot

## Scope and provenance

D95 is a current-source, three-task public HotpotQA dev-distractor comparison, with one repeat in each mode (six harness runs). It ran from source commit `e6184eb8ea09668e3b9c366d28a8ef773b401444`, after D92's offline injection-fixture isolation change. Both arms used prompt variant `d54`; the prompt SHA-256 is `6992422a52882a05c9c214fa7aaf45abae6e60994195fb34ff897c19d8721717`, unchanged from D91. The three selected dataset IDs are `5a886886554299206df2b258`, `5a78eea555429970f5fffddf`, and `5a8463d05542990548d0b2a5`, matching D91. This is a small descriptive pilot, not a general or causal test.

The local dataset file is `runs/public-data/hotpot_dev_distractor_v1.json`, SHA-256 `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`; the D91 exclusion file contains 440 IDs (raw-file SHA-256 `77057576ab6a3c9e30386fa27cac7140ac3db58b7b78391c68096be80e4f4256`). The six selected corpora have aggregate SHA-256 `59553494ad297e0631023c3efbb3f8329413a03e89cbea728166932978540ccc`. Gold answers and supporting-fact labels remained in the evaluator and were scored only after each `Harness.run()`; the per-run manifests and traces contain no scoring fields or API-key values. The full source dataset, generated corpora, and exclusion list remain local and ignored by Git. The [D95 reproduction package](reproducibility/README.md) includes only the three selected public records; the current loader regenerates their exact corpus hashes, and the existing scorer-isolation regression verifies that reference labels do not enter agent payloads.

**Source provenance:** The full local file came from the [Hugging Face HotpotQA mirror](https://huggingface.co/datasets/namlh2004/hotpotqa), SHA-256 `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`. Those bytes differ from the canonical [CMU dev-distractor download](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json). D95's command omitted `--source-url`, so the original aggregate metadata incorrectly inherited the CLI's CMU default; the Markdown, JSON, and CSV provenance now identify the actual mirror. The hash and benchmark scores are unchanged.

## Configuration and exact command

The provider was the Hugging Face Inference Providers Router through the OpenAI-compatible API, model `openai/gpt-oss-120b:nscale`, with `json_schema` responses. Rates were $0.10/M input tokens and $0.40/M output tokens. The credential came from `HF_TOKEN`; its value was not printed or saved. The CLI's interactive `APPROVE` prompt was answered before any model requests.

Both modes shared a 32-step, 60-second, 12,000-token run envelope. Multi-agent retained per-agent limits of 24 steps, 20 seconds, and 4,000 tokens, with 1,024-token worker and 1,536-token lead completion caps. The single-agent baseline received per-agent limits equal to the full run envelope and a 1,536-token completion cap. Call topology and per-request budgets therefore still differ.

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
  --runs-dir runs/current-source-e6184eb-3x2-20261017
```

## Results and interpretation

All six runs completed with provider usage reported. Quality metrics are means over the three paired tasks; time is median.

| Mode | Completed | Answer F1 | Support F1 | Joint F1 | Median seconds | Known tokens | Known token-derived cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multi-agent | 3/3 | 0.667 | 1.000 | 0.667 | 19.800 | 31,290 | $0.0056391 |
| Single-agent | 3/3 | 0.667 | 0.889 | 0.667 | 4.095 | 9,491 | $0.0013652 |

The paired joint-F1 difference was 0.0000 (95% descriptive interval [0.0000, 0.0000]); support-F1 difference was +0.1111 ([0.0000, +0.3333]), which includes zero. Multi-agent was 4.84× slower by median time and 4.13× higher by known token-derived cost. Total known usage was 40,781 provider tokens and $0.0070043, below the $0.06 aggregate preflight estimate; the $0.01 per-run and $0.06 aggregate guards are rate-based estimates, not provider billing limits or invoices.

This run had six completed artifacts; D91 had one partial multi-agent run. That difference across one repeat per task does not show that D92 or the current source caused a completion improvement. The same task IDs were reused, and the prompt hash is unchanged. One answer receives zero official answer F1 despite support in its cited source because the public reference is semantically broader; this limits interpretation. With three public development tasks and one repeat, the sample does not establish superiority, equivalence, or generalization. No prompt or runtime default was changed. D91 remains a historical pre-D92 result, not overwritten by D95.

## Artifacts

- [Aggregate report](hotpotqa-benchmark.md), [CSV](hotpotqa-benchmark.csv), and [JSON](hotpotqa-benchmark.json).
- Six per-run reports, JSONL traces, and manifests are linked in the aggregate report.
- Only this record and the scoring/evidence artifacts are eligible for the curated Git package. The raw dataset, exclusion list, and generated task corpora remain local.
