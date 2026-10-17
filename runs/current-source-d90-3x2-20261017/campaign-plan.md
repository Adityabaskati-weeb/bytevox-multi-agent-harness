# D90 current-source HotpotQA comparison

## Frozen question

Does the current ByteVox source produce different answer and support quality from its same-build single-agent baseline on three new public multi-hop document tasks, and what latency and estimated token-cost overhead does multi-agent add?

This is a small, descriptive model-backed pilot. It is not a test of D87's one-slot synthesis-pool idea, not a causal estimate of the D87 change, and not evidence of general superiority. No runtime default or prompt will be promoted based on this sample.

## Frozen protocol

- **Source:** current harness SHA-256 `c97f4ae3b0b7fef582655ad7f04f27ca3a8aa3974f99aaca07e1155836311c5a`; current `prompts.py` SHA-256 `6992422a52882a05c9c214fa7aaf45abae6e60994195fb34ff897c19d8721717`.
- **Dataset:** local HotpotQA `hotpot_dev_distractor_v1`, dev distractor split, CC BY-SA 4.0. Source SHA-256 `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`. It is a public development set and may overlap model pretraining.
- **Sampling:** 3 new task IDs, seed `20261017`, stratified by question type. The exclusion file adds the three D86 task IDs to the 437 IDs already excluded in D86. It contains 440 IDs; SHA-256 `77057576ab6a3c9e30386fa27cac7140ac3db58b7b78391c68096be80e4f4256`.
- **Selected tasks/corpora:**
  - `hotpotqa-0001`, dataset ID `5a886886554299206df2b258` (bridge), corpus SHA-256 `ce645aa1817776e71317b97215054b8d223d95c8c1e463d44c086c33c460b2ff`.
  - `hotpotqa-0002`, dataset ID `5a78eea555429970f5fffddf` (comparison), corpus SHA-256 `a2a2a987e98e1e6eee9c78a2d2d1fec29b52333ac2b6835038394d71708f4f6c`.
  - `hotpotqa-0003`, dataset ID `5a8463d05542990548d0b2a5` (bridge), corpus SHA-256 `f1df16f451f9ac7be4966fb20b9cf31a4984aa14ee4a42601c9ea83e2aac484f`.
- **Arms:** current multi-agent mode vs current same-build single-agent mode. Both receive the same raw question, task corpus, provider, response format, and limits. The dataset loader keeps answer and supporting-fact labels in scorer-only records; the model request and validation contract receive no gold labels.
- **Model:** Hugging Face Inference Providers Router, `openai/gpt-oss-120b:nscale`, JSON Schema responses, temperature 0. The current Hugging Face catalog lists Nscale rates of $0.10/M input tokens and $0.40/M output tokens; see [supported models](https://huggingface.co/inference/models?model=openai%2Fgpt-oss-120b) and [pricing](https://huggingface.co/docs/inference-providers/pricing), checked 2026-10-17.
- **Repeats/order:** 2 repeats per task and mode (12 harness runs). Mode order alternates by repetition within each task.
- **Limits:** 12,000 estimated run tokens; 4,000 per agent; 20 seconds per agent; 60 seconds per run. Estimated cost guard: $0.01 per run and $0.12 aggregate. These guards use the harness's character-based token estimate and current listed token rates; they are not provider billing ceilings.
- **Primary metric:** official HotpotQA joint F1. Also report answer EM/F1, supporting-fact precision/recall/F1, retrieval support recall, completion/checker status, latency, provider-reported tokens, and known token-derived cost.
- **Analysis:** show each task/repeat pair and aggregate descriptives. At three task clusters, uncertainty is too large to establish a general or causal quality effect. Treat unknown token usage as unknown, not zero.

## Exact command

From the project root in PowerShell, after checking model request approval:

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONUTF8 = "1"
$env:BYTEVOX_INPUT_USD_PER_1M = "0.10"
$env:BYTEVOX_OUTPUT_USD_PER_1M = "0.40"
python -m bytevox_harness hotpotqa-compare `
  --dataset-path runs/public-data/hotpot_dev_distractor_v1.json `
  --exclude-ids-file runs/current-source-d90-3x2-20261017/excluded-dataset-ids.txt `
  --sample-size 3 --sample-seed 20261017 --repeats 2 `
  --max-tokens 12000 --max-agent-tokens 4000 --max-agent-seconds 20 `
  --provider openai-compatible --base-url https://router.huggingface.co/v1 `
  --model openai/gpt-oss-120b:nscale --api-key-env HF_TOKEN `
  --response-format json_schema --approve-remote-model `
  --max-cost-usd 0.01 --max-aggregate-cost-usd 0.12 `
  --runs-dir runs/current-source-d90-3x2-20261017
```

Before any remote request, the CLI requires typing `APPROVE`. Do not include the credential value in logs or artifacts.

## Interpretation rule

Report the exact measured result, including failures and unknown usage. Keep the current default regardless of this pilot: a positive three-task result would justify a larger, predeclared follow-up, not a superiority claim; a negative or mixed result would keep the multi-agent quality question unresolved or unfavorable for this sample. Do not present the D87 offline retrieval recall gain as answer-quality evidence.
