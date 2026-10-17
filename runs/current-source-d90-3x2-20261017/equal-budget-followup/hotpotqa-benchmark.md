# HotpotQA public-data pilot comparison

**Scope:** small, deterministic pilot sample from the public dev distractor split; results are not generalizable and may reflect model pretraining exposure.
**Source:** [https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json); [HotpotQA homepage](https://hotpotqa.github.io/); local file `hotpot_dev_distractor_v1.json`; SHA-256 `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`.
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
| multi | 2/3 | 0.667 | 0.667 | 1.000 / 1.000 / 1.000 | 1.000 | 0.667 / 0.667 | 20.563 [18.951–20.781] | 32998 | 0.0061378 |
| single | 3/3 | 0.667 | 0.667 | 1.000 / 0.833 / 0.889 | 0.833 | 0.667 / 0.667 | 4.576 [3.136–5.218] | 9317 | 0.001295 |

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
| 5a886886554299206df2b258 | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 1.000 / 1.000 / 1.000 | 1.000 | 0.000 / 0.000 | 20.563 | 11811 | 0.00210060 | partial | [20261017T115928Z-3d7856f0.jsonl](./20261017T115928Z-3d7856f0.jsonl) | [20261017T115928Z-3d7856f0.md](./20261017T115928Z-3d7856f0.md) | [20261017T115928Z-3d7856f0.manifest.json](./20261017T115928Z-3d7856f0.manifest.json) |
| 5a886886554299206df2b258 | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 1.000 / 0.500 / 0.667 | 0.500 | 0.000 / 0.000 | 4.576 | 3294 | 0.00045750 | completed | [20261017T115948Z-02c0b768.jsonl](./20261017T115948Z-02c0b768.jsonl) | [20261017T115948Z-02c0b768.md](./20261017T115948Z-02c0b768.md) | [20261017T115948Z-02c0b768.manifest.json](./20261017T115948Z-02c0b768.manifest.json) |
| 5a78eea555429970f5fffddf | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 20.781 | 10805 | 0.00203480 | completed | [20261017T115953Z-aee81bcb.jsonl](./20261017T115953Z-aee81bcb.jsonl) | [20261017T115953Z-aee81bcb.md](./20261017T115953Z-aee81bcb.md) | [20261017T115953Z-aee81bcb.manifest.json](./20261017T115953Z-aee81bcb.manifest.json) |
| 5a78eea555429970f5fffddf | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 5.218 | 3223 | 0.00048640 | completed | [20261017T120014Z-6e61c3c3.jsonl](./20261017T120014Z-6e61c3c3.jsonl) | [20261017T120014Z-6e61c3c3.md](./20261017T120014Z-6e61c3c3.md) | [20261017T120014Z-6e61c3c3.manifest.json](./20261017T120014Z-6e61c3c3.manifest.json) |
| 5a8463d05542990548d0b2a5 | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 18.951 | 10382 | 0.00200240 | completed | [20261017T120019Z-9a0c68dd.jsonl](./20261017T120019Z-9a0c68dd.jsonl) | [20261017T120019Z-9a0c68dd.md](./20261017T120019Z-9a0c68dd.md) | [20261017T120019Z-9a0c68dd.manifest.json](./20261017T120019Z-9a0c68dd.manifest.json) |
| 5a8463d05542990548d0b2a5 | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 3.136 | 2800 | 0.00035110 | completed | [20261017T120038Z-5b8f951e.jsonl](./20261017T120038Z-5b8f951e.jsonl) | [20261017T120038Z-5b8f951e.md](./20261017T120038Z-5b8f951e.md) | [20261017T120038Z-5b8f951e.manifest.json](./20261017T120038Z-5b8f951e.manifest.json) |

A missing `Answer:` first line produces an answer parse failure and zero answer EM/F1; inspect the linked report. The support scorer maps each cited sentence evidence ID back to its source `(title, sent_id)` and counts unknown citation IDs as false positives. Answer EM/F1 and support P/R/F1 follow the official HotpotQA normalization and set-metric conventions; joint EM/F1 combines answer and support scores. This compact pilot is evidence about this sample, model, and harness run configuration only.

Offline-provider rows exercise parsing, scoring, artifact generation, and label isolation only; they are not language-model quality evidence. Remote calls still require the existing interactive approval gate. A per-run USD cap still requires both token-rate variables.
