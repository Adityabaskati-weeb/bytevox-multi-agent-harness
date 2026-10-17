# D86 current shipped-role HotpotQA pilot

## Preregistered question
Does the exact current D54 implementation, including distinct specialist instructions, produce different answer/support quality from the same-build single-agent baseline on a small fresh sample of multi-hop public-document tasks, and what time and estimated-cost overhead does it add?

This is a descriptive, small-sample comparison—not a claim that multi-agent is generally better.

## Frozen protocol
- Dataset: local HotpotQA `hotpot_dev_distractor_v1.json`, dev distractor split; CC BY-SA 4.0. SHA-256: `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`; local mirror attribution: https://huggingface.co/datasets/namlh2004/hotpotqa.
- Selection: 3 examples, seed `20261017`, proportional stratification by question type; 2 bridge and 1 comparison. Exclude 437 IDs from prior D56/D57/D70 experiments. Exclusion file SHA-256: `f191d11d95d83c5092c82b9b2d58ad5ac9dc751c88962d0f4e0b9d792c02261f`; normalized ID-set hash in loader provenance: `61e6faa1ad4721234813fa737d4bf4db4eb170d564390ddde58f547c4616bb8b`.
- Selected IDs: `5a85ab905542994c784ddb35`, `5a7a2be25542996a35c17118`, `5ae736cb5542991bbc9761c2`. The frozen [sample record](sample-records.json) contains only IDs, task text, task types, and corpus hashes; it omits answers and support labels.
- Compared modes: current multi-agent D54 vs same-build single-agent baseline. Both receive the same raw question, evidence corpus, generic contract, model, response format, and limits. No gold answer/evidence labels enter model input or validation.
- Current source SHA-256: `180000d46fd2b3e88eb714a9b86e703899f9315e4c9e20e44a8fd43cdfe427cb`; current prompt SHA-256: `6992422a52882a05c9c214fa7aaf45abae6e60994195fb34ff897c19d8721717`. This prompt fingerprint matches D77 and differs from the older D56/D57/D70/D71 prompt snapshots.
- Model route: Hugging Face Inference Providers Router, `openai/gpt-oss-120b:nscale`; `json_schema`; temperature 0. Rates: $0.10/M input and $0.40/M output tokens as listed by Hugging Face on 2026-10-17. Credential variable: `HF_TOKEN`; do not record its value.
- Repeats: 2 per task/mode, alternating mode order across repeats. Planned total: 3 tasks × 2 repeats × 2 modes = 12 harness runs.
- Limits: 12,000 estimated run tokens; 4,000 per agent; 20 seconds per agent; 60 seconds per run. Estimated cost guard: $0.01 per run, $0.12 total. These token-based guards are not billing ceilings.

## Scoring and interpretation
- Primary quality measure: official HotpotQA joint F1, with answer F1 and support-fact F1 reported separately.
- Operational measures: completion/partial status, checker outcome, median latency, provider-reported tokens, and known/unknown token-derived cost.
- Compare each matched task and repeat. Treat intervals and aggregate differences as descriptive: only 3 task clusters cannot establish a reliable quality win.
- After outputs are saved, two independent AI reviewers will receive anonymized answer/source packets with arm identities hidden. Their claim-support and completeness ratings are secondary qualitative evidence, not human ratings or ground truth.
- Public dev questions can overlap pretraining. Report that limitation.
- Decision rule: do not change prompts, tools, limits, or active defaults based on this pilot. Record all partials and report the result even if it disfavors multi-agent.



