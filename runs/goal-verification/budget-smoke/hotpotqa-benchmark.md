# HotpotQA public-data pilot comparison

**Scope:** small, deterministic pilot sample from the public dev distractor split; results are not generalizable and may reflect model pretraining exposure.
**Source:** [https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json); [HotpotQA homepage](https://hotpotqa.github.io/); local file `hotpot_dev_distractor_v1.json`; SHA-256 `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`.
**Version / split:** `hotpot_dev_distractor_v1` / `dev distractor`; license `CC BY-SA 4.0`.
**Sample:** 1 of 1 requested; IDs: 5a7a2be25542996a35c17118; source indices: [2112]; rule: exclude supplied prior IDs, then select first comparison and first bridge, then fill remaining slots in source order.
**Corpus hash:** `dd59f7045cd27b59b0f0c42b711b19f66c460944a7b2ce9916286d9640932ec3` (aggregate); each run row and manifest includes its case corpus SHA-256.
**Prompt variant:** d54.
**Official metrics reference:** [https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).
**Trials:** 2 runs across 1 repetition(s); the multi/single order alternates by task and repetition.

The runner reads only the local JSON file supplied on the command line; it does not download data. Gold answers and supporting-fact labels are held by the scorer and evaluated after each `Harness.run()`. Both modes receive the same raw question, corpus, and generic three-facet contract. The trusted contract requires the first line of `summary` to begin `Answer: ` and contain only the concise answer; the question sent to retrieval has no formatting boilerplate. Citation support and retrieval support recall are scored separately from the closed post-run trace.

## Mode summary

Quality metrics are arithmetic means over all trials, including partial runs. Latency is median [minimum–maximum]. Token totals and API cost include known runs only; local CPU/GPU time is not priced.

| Mode | Completed | Answer EM | Answer F1 | Support P / R / F1 | Retrieval support recall | Joint EM / F1 | Wall time median [range] (s) | Provider tokens known total | API cost known total (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi | 1/1 | 0.000 | 0.000 | 0.333 / 0.333 / 0.333 | 1.000 | 0.000 / 0.000 | 32.103 [32.103–32.103] | 11430 | 0.0022341 |
| single | 1/1 | 0.000 | 0.000 | 1.000 / 0.667 / 0.800 | 1.000 | 0.000 / 0.000 | 5.880 [5.880–5.880] | 3305 | 0.000524 |

## Paired uncertainty analysis

The comparison resamples 1 task IDs with replacement, then resamples each selected task's 1 repetition indices with replacement; multi/single values stay paired. The analysis uses 10,000 draws (seed `20261016`), includes partial runs, and reports percentile 95% intervals. These intervals describe this small public sample and run configuration; they do not establish causal effects or general performance.

| Metric | Mean multi − single | 95% paired bootstrap interval | Paired trials |
|---|---:|---:|---:|
| joint_f1 | +0.0000 | [+0.0000, +0.0000] | 1 |
| answer_f1 | +0.0000 | [+0.0000, +0.0000] | 1 |
| supporting_fact_f1 | -0.4667 | [-0.4667, -0.4667] | 1 |

## Run rows

| Case | Repetition | Mode | Prompt variant | Provider / model / format | Answer EM / F1 | Support P / R / F1 | Retrieval support recall | Joint EM / F1 | Wall time (s) | Tokens | API cost (USD) | Status | Trace | Report | Manifest |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 5a7a2be25542996a35c17118 | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 0.333 / 0.333 / 0.333 | 1.000 | 0.000 / 0.000 | 32.103 | 11430 | 0.00223410 | completed | [20261017T105905Z-364f0ad6.jsonl](./20261017T105905Z-364f0ad6.jsonl) | [20261017T105905Z-364f0ad6.md](./20261017T105905Z-364f0ad6.md) | [20261017T105905Z-364f0ad6.manifest.json](./20261017T105905Z-364f0ad6.manifest.json) |
| 5a7a2be25542996a35c17118 | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 1.000 / 0.667 / 0.800 | 1.000 | 0.000 / 0.000 | 5.880 | 3305 | 0.00052400 | completed | [20261017T105937Z-da38938b.jsonl](./20261017T105937Z-da38938b.jsonl) | [20261017T105937Z-da38938b.md](./20261017T105937Z-da38938b.md) | [20261017T105937Z-da38938b.manifest.json](./20261017T105937Z-da38938b.manifest.json) |

A missing `Answer:` first line produces an answer parse failure and zero answer EM/F1; inspect the linked report. The support scorer maps each cited sentence evidence ID back to its source `(title, sent_id)` and counts unknown citation IDs as false positives. Answer EM/F1 and support P/R/F1 follow the official HotpotQA normalization and set-metric conventions; joint EM/F1 combines answer and support scores. This compact pilot is evidence about this sample, model, and harness run configuration only.

Offline-provider rows exercise parsing, scoring, artifact generation, and label isolation only; they are not language-model quality evidence. Remote calls still require the existing interactive approval gate. A per-run USD cap still requires both token-rate variables.
