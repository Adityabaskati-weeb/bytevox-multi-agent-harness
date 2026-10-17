# D77 unfamiliar task pilot

## Question

Can the unchanged D54 harness handle a new plain-English question over the existing local North Harbor corpus under normal `run()` limits, and how does one multi-agent run compare with the single-agent baseline?

## Frozen protocol

- Task: “How did the North Harbor pilot change on-time performance, and what limitation does the report give on attributing that change to its interventions?”
- Corpus: `src/bytevox_harness/fixtures/corpus` (same read-only corpus for both modes).
- Model: Hugging Face Inference Providers route `openai/gpt-oss-120b:nscale`, JSON Schema response format.
- Conditions: D54 prompt version, one multi-agent run and one single-agent run, same task, corpus, model, response format, and normal `run()` limits (12,000 run tokens; 4,000 per agent; 20 seconds per agent; 60 seconds overall). Each mode uses its own role-specific prompt text by design; “same D54” identifies the shared prompt version, not identical prompts across roles or modes.
- Cost guard: estimated USD 0.01 per run; maximum planned guard total USD 0.02. This is an estimator guard, not a provider billing limit.
- Gold facts are scorer-only and are not sent in the task, validation contract, or agent prompts.
- One run per mode is a functional demonstration only. It cannot establish a quality difference or generalization.

## Scoring, applied after both outputs are saved

1. Answer completeness: state the baseline and final on-time rates, and that the report says the pilot was not randomized.
2. Citation support: each factual claim is cited to evidence that directly supports it; expected source is `operations#results`.
3. Run behavior: record completion/partial status, final checker decision where applicable, elapsed time, tokens, and known token-derived cost.

The reference answer is that on-time performance rose from 82% baseline to 88% during the final four weeks, while the non-randomized pilot does not attribute the improvement to the interventions alone. These labels are used only after runs for scorer review.

## Promotion rule

Do not change prompts, defaults, tools, limits, or the official benchmark from this pilot. If either run is partial, record the failure; if both complete, use the pair as one fresh-task functionality example and do not claim multi-agent quality superiority. D54 remains the active default.
