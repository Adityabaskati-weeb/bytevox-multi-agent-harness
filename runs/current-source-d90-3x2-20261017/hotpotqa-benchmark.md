# HotpotQA public-data pilot comparison

**Scope:** small, deterministic pilot sample from the public dev distractor split; results are not generalizable and may reflect model pretraining exposure.
**Source:** [https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json); [HotpotQA homepage](https://hotpotqa.github.io/); local file `hotpot_dev_distractor_v1.json`; SHA-256 `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`.
**Version / split:** `hotpot_dev_distractor_v1` / `dev distractor`; license `CC BY-SA 4.0`.
**Sample:** 3 of 3 requested; IDs: 5a886886554299206df2b258, 5a78eea555429970f5fffddf, 5a8463d05542990548d0b2a5; source indices: [321, 2117, 4724]; rule: exclude supplied prior IDs, then use proportional stratified allocation by record type using Hamilton's largest remainder; sample within each type using the recorded seed, then report original source order.
**Corpus hash:** `59553494ad297e0631023c3efbb3f8329413a03e89cbea728166932978540ccc` (aggregate); each run row and manifest includes its case corpus SHA-256.
**Prompt variant:** d54.
**Official metrics reference:** [https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).
**Trials:** 12 runs across 2 repetition(s); the multi/single order alternates by task and repetition.

The runner reads only the local JSON file supplied on the command line; it does not download data. Gold answers and supporting-fact labels are held by the scorer and evaluated after each `Harness.run()`. Both modes receive the same raw question, corpus, and generic three-facet contract. The trusted contract requires the first line of `summary` to begin `Answer: ` and contain only the concise answer; the question sent to retrieval has no formatting boilerplate. Citation support and retrieval support recall are scored separately from the closed post-run trace.

## Mode summary

Quality metrics are arithmetic means over all trials, including partial runs. Latency is median [minimum–maximum]. Token totals and API cost include known runs only; local CPU/GPU time is not priced.

| Mode | Completed | Answer EM | Answer F1 | Support P / R / F1 | Retrieval support recall | Joint EM / F1 | Wall time median [range] (s) | Provider tokens known total | API cost known total (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi | 5/6 | 0.667 | 0.667 | 0.944 / 1.000 / 0.967 | 1.000 | 0.667 / 0.667 | 20.340 [18.470–22.852] | 63526 | 0.0119644 |
| single | 6/6 | 0.667 | 0.667 | 1.000 / 0.833 / 0.889 | 0.833 | 0.667 / 0.667 | 4.113 [3.037–5.313] | 18471 | 0.0025248 |

## Paired uncertainty analysis

The comparison resamples 3 task IDs with replacement, then resamples each selected task's 2 repetition indices with replacement; multi/single values stay paired. The analysis uses 10,000 draws (seed `20261016`), includes partial runs, and reports percentile 95% intervals. These intervals describe this small public sample and run configuration; they do not establish causal effects or general performance.

| Metric | Mean multi − single | 95% paired bootstrap interval | Paired trials |
|---|---:|---:|---:|
| joint_f1 | +0.0000 | [+0.0000, +0.0000] | 6 |
| answer_f1 | +0.0000 | [+0.0000, +0.0000] | 6 |
| supporting_fact_f1 | +0.0778 | [+0.0000, +0.2333] | 6 |

## Run rows

| Case | Repetition | Mode | Prompt variant | Provider / model / format | Answer EM / F1 | Support P / R / F1 | Retrieval support recall | Joint EM / F1 | Wall time (s) | Tokens | API cost (USD) | Status | Trace | Report | Manifest |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 5a886886554299206df2b258 | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 0.667 / 1.000 / 0.800 | 1.000 | 0.000 / 0.000 | 18.470 | 10111 | 0.00204700 | partial | [20261017T112741Z-ee9e2c91.jsonl](./20261017T112741Z-ee9e2c91.jsonl) | [20261017T112741Z-ee9e2c91.md](./20261017T112741Z-ee9e2c91.md) | [20261017T112741Z-ee9e2c91.manifest.json](./20261017T112741Z-ee9e2c91.manifest.json) |
| 5a886886554299206df2b258 | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 1.000 / 0.500 / 0.667 | 0.500 | 0.000 / 0.000 | 3.037 | 3115 | 0.00038590 | completed | [20261017T112800Z-540af114.jsonl](./20261017T112800Z-540af114.jsonl) | [20261017T112800Z-540af114.md](./20261017T112800Z-540af114.md) | [20261017T112800Z-540af114.manifest.json](./20261017T112800Z-540af114.manifest.json) |
| 5a886886554299206df2b258 | 2 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 1.000 / 0.500 / 0.667 | 0.500 | 0.000 / 0.000 | 4.292 | 3268 | 0.00044710 | completed | [20261017T112803Z-3521d72e.jsonl](./20261017T112803Z-3521d72e.jsonl) | [20261017T112803Z-3521d72e.md](./20261017T112803Z-3521d72e.md) | [20261017T112803Z-3521d72e.manifest.json](./20261017T112803Z-3521d72e.manifest.json) |
| 5a886886554299206df2b258 | 2 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 1.000 / 1.000 / 1.000 | 1.000 | 0.000 / 0.000 | 20.297 | 11526 | 0.00206850 | completed | [20261017T112807Z-ab92a815.jsonl](./20261017T112807Z-ab92a815.jsonl) | [20261017T112807Z-ab92a815.md](./20261017T112807Z-ab92a815.md) | [20261017T112807Z-ab92a815.manifest.json](./20261017T112807Z-ab92a815.manifest.json) |
| 5a78eea555429970f5fffddf | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 22.852 | 10857 | 0.00204120 | completed | [20261017T112827Z-3f9297a9.jsonl](./20261017T112827Z-3f9297a9.jsonl) | [20261017T112827Z-3f9297a9.md](./20261017T112827Z-3f9297a9.md) | [20261017T112827Z-3f9297a9.manifest.json](./20261017T112827Z-3f9297a9.manifest.json) |
| 5a78eea555429970f5fffddf | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 5.313 | 3217 | 0.00048400 | completed | [20261017T112850Z-d5d4397e.jsonl](./20261017T112850Z-d5d4397e.jsonl) | [20261017T112850Z-d5d4397e.md](./20261017T112850Z-d5d4397e.md) | [20261017T112850Z-d5d4397e.manifest.json](./20261017T112850Z-d5d4397e.manifest.json) |
| 5a78eea555429970f5fffddf | 2 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 4.325 | 3109 | 0.00044080 | completed | [20261017T112856Z-a97f5549.jsonl](./20261017T112856Z-a97f5549.jsonl) | [20261017T112856Z-a97f5549.md](./20261017T112856Z-a97f5549.md) | [20261017T112856Z-a97f5549.manifest.json](./20261017T112856Z-a97f5549.manifest.json) |
| 5a78eea555429970f5fffddf | 2 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 20.384 | 10594 | 0.00194560 | completed | [20261017T112900Z-2d307271.jsonl](./20261017T112900Z-2d307271.jsonl) | [20261017T112900Z-2d307271.md](./20261017T112900Z-2d307271.md) | [20261017T112900Z-2d307271.manifest.json](./20261017T112900Z-2d307271.manifest.json) |
| 5a8463d05542990548d0b2a5 | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 19.346 | 9732 | 0.00176280 | completed | [20261017T112920Z-74a58a55.jsonl](./20261017T112920Z-74a58a55.jsonl) | [20261017T112920Z-74a58a55.md](./20261017T112920Z-74a58a55.md) | [20261017T112920Z-74a58a55.manifest.json](./20261017T112920Z-74a58a55.manifest.json) |
| 5a8463d05542990548d0b2a5 | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 3.934 | 2932 | 0.00040390 | completed | [20261017T112940Z-7dd85725.jsonl](./20261017T112940Z-7dd85725.jsonl) | [20261017T112940Z-7dd85725.md](./20261017T112940Z-7dd85725.md) | [20261017T112940Z-7dd85725.manifest.json](./20261017T112940Z-7dd85725.manifest.json) |
| 5a8463d05542990548d0b2a5 | 2 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 3.385 | 2830 | 0.00036310 | completed | [20261017T112944Z-a6ea68f5.jsonl](./20261017T112944Z-a6ea68f5.jsonl) | [20261017T112944Z-a6ea68f5.md](./20261017T112944Z-a6ea68f5.md) | [20261017T112944Z-a6ea68f5.manifest.json](./20261017T112944Z-a6ea68f5.manifest.json) |
| 5a8463d05542990548d0b2a5 | 2 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 22.668 | 10706 | 0.00209930 | completed | [20261017T112947Z-e4831899.jsonl](./20261017T112947Z-e4831899.jsonl) | [20261017T112947Z-e4831899.md](./20261017T112947Z-e4831899.md) | [20261017T112947Z-e4831899.manifest.json](./20261017T112947Z-e4831899.manifest.json) |

A missing `Answer:` first line produces an answer parse failure and zero answer EM/F1; inspect the linked report. The support scorer maps each cited sentence evidence ID back to its source `(title, sent_id)` and counts unknown citation IDs as false positives. Answer EM/F1 and support P/R/F1 follow the official HotpotQA normalization and set-metric conventions; joint EM/F1 combines answer and support scores. This compact pilot is evidence about this sample, model, and harness run configuration only.

Offline-provider rows exercise parsing, scoring, artifact generation, and label isolation only; they are not language-model quality evidence. Remote calls still require the existing interactive approval gate. A per-run USD cap still requires both token-rate variables.
