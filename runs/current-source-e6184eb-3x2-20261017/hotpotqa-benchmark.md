# HotpotQA public-data pilot comparison

**Scope:** small, deterministic pilot sample from the public dev distractor split; results are not generalizable and may reflect model pretraining exposure.
**Source:** [Hugging Face HotpotQA mirror](https://huggingface.co/datasets/namlh2004/hotpotqa); [official HotpotQA homepage](https://hotpotqa.github.io/); local file `hotpot_dev_distractor_v1.json`; SHA-256 `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`.
**Source note:** This full-file hash identifies the Hugging Face mirror bytes. They differ from the canonical [CMU dev-distractor download](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json); the two URLs must not be treated as the same file checksum.
**Version / split:** `hotpot_dev_distractor_v1` / `dev distractor`; license `CC BY-SA 4.0`.
**Sample:** 3 of 3 requested; IDs: 5a886886554299206df2b258, 5a78eea555429970f5fffddf, 5a8463d05542990548d0b2a5; source indices: [321, 2117, 4724]; rule: exclude supplied prior IDs, then use proportional stratified allocation by record type using Hamilton's largest remainder; sample within each type using the recorded seed, then report original source order.
**Corpus hash:** `59553494ad297e0631023c3efbb3f8329413a03e89cbea728166932978540ccc` (aggregate); each run row and manifest includes its case corpus SHA-256.
**Prompt variant:** d54.
**Official metrics reference:** [https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).
**Trials:** 6 runs across 1 repetition(s); the multi/single order alternates by task and repetition.
**Budget policy:** `equal-run-envelope-v1`; multi-agent retains the configured per-agent caps, while the single-agent baseline's per-agent step/time/token caps are raised to the matching shared run caps. Its per-request completion cap is raised to match the multi-agent lead cap. Both modes retain the same run-level limits.

The runner reads only the local JSON file supplied on the command line; it does not download data. Gold answers and supporting-fact labels are held by the scorer and evaluated after each `Harness.run()`. Both modes receive the same raw question, corpus, and generic three-facet contract. The trusted contract requires the first line of `summary` to begin `Answer: ` and contain only the concise answer; the question sent to retrieval has no formatting boilerplate. Citation support and retrieval support recall are scored separately from the closed post-run trace.

## Mode summary

Quality metrics are arithmetic means over all trials, including partial runs. Latency is median [minimum–maximum]. Token totals and API cost include known runs only; local CPU/GPU time is not priced.

| Mode | Completed | Answer EM | Answer F1 | Support P / R / F1 | Retrieval support recall | Joint EM / F1 | Wall time median [range] (s) | Provider tokens known total | API cost known total (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi | 3/3 | 0.667 | 0.667 | 1.000 / 1.000 / 1.000 | 1.000 | 0.667 / 0.667 | 19.800 [17.789–21.459] | 31290 | 0.0056391 |
| single | 3/3 | 0.667 | 0.667 | 1.000 / 0.833 / 0.889 | 0.833 | 0.667 / 0.667 | 4.095 [3.690–6.344] | 9491 | 0.0013652 |

## Effective mode limits

Every run manifest records its effective limits; each linked run report repeats the run and per-agent caps. The table below shows the configured multi-agent limits and the equal-run-envelope single-agent limits.

| Mode | Run steps / seconds / tokens | Per-agent steps / seconds / tokens | Completion tokens (general / lead) |
|---|---|---|---:|
| multi | 32 / 60 / 12000 | 24 / 20 / 4000 | 1024 / 1536 |
| single | 32 / 60 / 12000 | 32 / 60 / 12000 | 1536 / 1536 |

## Paired uncertainty analysis

The comparison resamples 3 task IDs with replacement, then resamples each selected task's 1 repetition indices with replacement; multi/single values stay paired. The analysis uses 10,000 draws (seed `20261016`), includes partial runs, and reports percentile 95% intervals. These intervals describe this small public sample and run configuration; they do not establish causal effects or general performance.

| Metric | Mean multi − single | 95% paired bootstrap interval | Paired trials |
|---|---:|---:|---:|
| joint_f1 | +0.0000 | [+0.0000, +0.0000] | 3 |
| answer_f1 | +0.0000 | [+0.0000, +0.0000] | 3 |
| supporting_fact_f1 | +0.1111 | [+0.0000, +0.3333] | 3 |

## Run rows

| Case | Repetition | Mode | Prompt variant | Provider / model / format | Answer EM / F1 | Support P / R / F1 | Retrieval support recall | Joint EM / F1 | Wall time (s) | Tokens | API cost (USD) | Status | Trace | Report | Manifest |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 5a886886554299206df2b258 | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 1.000 / 1.000 / 1.000 | 1.000 | 0.000 / 0.000 | 21.459 | 11682 | 0.00214170 | completed | [20261017T125524Z-7d11adaa.jsonl](./20261017T125524Z-7d11adaa.jsonl) | [20261017T125524Z-7d11adaa.md](./20261017T125524Z-7d11adaa.md) | [20261017T125524Z-7d11adaa.manifest.json](./20261017T125524Z-7d11adaa.manifest.json) |
| 5a886886554299206df2b258 | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 1.000 / 0.500 / 0.667 | 0.500 | 0.000 / 0.000 | 4.095 | 3221 | 0.00042890 | completed | [20261017T125546Z-95e0cca4.jsonl](./20261017T125546Z-95e0cca4.jsonl) | [20261017T125546Z-95e0cca4.md](./20261017T125546Z-95e0cca4.md) | [20261017T125546Z-95e0cca4.manifest.json](./20261017T125546Z-95e0cca4.manifest.json) |
| 5a78eea555429970f5fffddf | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 17.789 | 9912 | 0.00171690 | completed | [20261017T125550Z-909c1192.jsonl](./20261017T125550Z-909c1192.jsonl) | [20261017T125550Z-909c1192.md](./20261017T125550Z-909c1192.md) | [20261017T125550Z-909c1192.manifest.json](./20261017T125550Z-909c1192.manifest.json) |
| 5a78eea555429970f5fffddf | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 6.344 | 3364 | 0.00054280 | completed | [20261017T125608Z-4fcd276d.jsonl](./20261017T125608Z-4fcd276d.jsonl) | [20261017T125608Z-4fcd276d.md](./20261017T125608Z-4fcd276d.md) | [20261017T125608Z-4fcd276d.manifest.json](./20261017T125608Z-4fcd276d.manifest.json) |
| 5a8463d05542990548d0b2a5 | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 19.800 | 9696 | 0.00178050 | completed | [20261017T125614Z-d23c44fb.jsonl](./20261017T125614Z-d23c44fb.jsonl) | [20261017T125614Z-d23c44fb.md](./20261017T125614Z-d23c44fb.md) | [20261017T125614Z-d23c44fb.manifest.json](./20261017T125614Z-d23c44fb.manifest.json) |
| 5a8463d05542990548d0b2a5 | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 3.690 | 2906 | 0.00039350 | completed | [20261017T125634Z-9e1e3d86.jsonl](./20261017T125634Z-9e1e3d86.jsonl) | [20261017T125634Z-9e1e3d86.md](./20261017T125634Z-9e1e3d86.md) | [20261017T125634Z-9e1e3d86.manifest.json](./20261017T125634Z-9e1e3d86.manifest.json) |

A missing `Answer:` first line produces an answer parse failure and zero answer EM/F1; inspect the linked report. The support scorer maps each cited sentence evidence ID back to its source `(title, sent_id)` and counts unknown citation IDs as false positives. Answer EM/F1 and support P/R/F1 follow the official HotpotQA normalization and set-metric conventions; joint EM/F1 combines answer and support scores. This compact pilot is evidence about this sample, model, and harness run configuration only.

Offline-provider rows exercise parsing, scoring, artifact generation, and label isolation only; they are not language-model quality evidence. Remote calls still require the existing interactive approval gate. A per-run USD cap still requires both token-rate variables.
