# HotpotQA public-data pilot comparison

**Scope:** small, deterministic pilot sample from the public dev distractor split; results are not generalizable and may reflect model pretraining exposure.
**Source:** [https://huggingface.co/datasets/namlh2004/hotpotqa](https://huggingface.co/datasets/namlh2004/hotpotqa); [HotpotQA homepage](https://hotpotqa.github.io/); local file `hotpot_dev_distractor_v1.json`; SHA-256 `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`.
**Version / split:** `hotpot_dev_distractor_v1` / `dev distractor`; license `CC BY-SA 4.0`.
**Sample:** 3 of 3 requested; IDs: 5a85ab905542994c784ddb35, 5a7a2be25542996a35c17118, 5ae736cb5542991bbc9761c2; source indices: [319, 2112, 4722]; rule: exclude supplied prior IDs, then use proportional stratified allocation by record type using Hamilton's largest remainder; sample within each type using the recorded seed, then report original source order.
**Corpus hash:** `9ad5b2b85360ad50084254ce8b87a40e0631631870f58822dce677733cf84caf` (aggregate); each run row and manifest includes its case corpus SHA-256.
**Prompt variant:** d54.
**Official metrics reference:** [https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).
**Trials:** 12 runs across 2 repetition(s); the multi/single order alternates by task and repetition.

The runner reads only the local JSON file supplied on the command line; it does not download data. Gold answers and supporting-fact labels are held by the scorer and evaluated after each `Harness.run()`. Both modes receive the same raw question, corpus, and generic three-facet contract. The trusted contract requires the first line of `summary` to begin `Answer: ` and contain only the concise answer; the question sent to retrieval has no formatting boilerplate. Citation support and retrieval support recall are scored separately from the closed post-run trace.

## Mode summary

Quality metrics are arithmetic means over all trials, including partial runs. Latency is median [minimum–maximum]. Token totals and API cost include known runs only; local CPU/GPU time is not priced.

| Mode | Completed | Answer EM | Answer F1 | Support P / R / F1 | Retrieval support recall | Joint EM / F1 | Wall time median [range] (s) | Provider tokens known total | API cost known total (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi | 5/6 | 0.500 | 0.556 | 0.833 / 0.778 / 0.800 | 0.889 | 0.500 / 0.556 | 22.419 [18.640–25.164] | 68814 | 0.0126885 |
| single | 6/6 | 0.500 | 0.611 | 1.000 / 0.889 / 0.933 | 1.000 | 0.333 / 0.578 | 5.695 [4.277–7.850] | 20729 | 0.003197 |

## Paired uncertainty analysis

The comparison resamples 3 task IDs with replacement, then resamples each selected task's 2 repetition indices with replacement; multi/single values stay paired. The analysis uses 10,000 draws (seed `20261016`), includes partial runs, and reports percentile 95% intervals. These intervals describe this small public sample and run configuration; they do not establish causal effects or general performance.

| Metric | Mean multi − single | 95% paired bootstrap interval | Paired trials |
|---|---:|---:|---:|
| joint_f1 | -0.0222 | [-0.4000, +0.3333] | 6 |
| answer_f1 | -0.0556 | [-0.5000, +0.3333] | 6 |
| supporting_fact_f1 | -0.1333 | [-0.4000, +0.0000] | 6 |

## Run rows

| Case | Repetition | Mode | Prompt variant | Provider / model / format | Answer EM / F1 | Support P / R / F1 | Retrieval support recall | Joint EM / F1 | Wall time (s) | Tokens | API cost (USD) | Status | Trace | Report | Manifest |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 5a85ab905542994c784ddb35 | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 22.255 | 11340 | 0.00211680 | completed | [20261017T051315Z-95f5e12c.jsonl](./20261017T051315Z-95f5e12c.jsonl) | [20261017T051315Z-95f5e12c.md](./20261017T051315Z-95f5e12c.md) | [20261017T051315Z-95f5e12c.manifest.json](./20261017T051315Z-95f5e12c.manifest.json) |
| 5a85ab905542994c784ddb35 | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 5.346 | 3315 | 0.00049560 | completed | [20261017T051337Z-c88c68b7.jsonl](./20261017T051337Z-c88c68b7.jsonl) | [20261017T051337Z-c88c68b7.md](./20261017T051337Z-c88c68b7.md) | [20261017T051337Z-c88c68b7.manifest.json](./20261017T051337Z-c88c68b7.manifest.json) |
| 5a85ab905542994c784ddb35 | 2 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 4.277 | 3174 | 0.00043920 | completed | [20261017T051342Z-0d1ad3bc.jsonl](./20261017T051342Z-0d1ad3bc.jsonl) | [20261017T051342Z-0d1ad3bc.md](./20261017T051342Z-0d1ad3bc.md) | [20261017T051342Z-0d1ad3bc.manifest.json](./20261017T051342Z-0d1ad3bc.manifest.json) |
| 5a85ab905542994c784ddb35 | 2 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 24.306 | 11461 | 0.00218410 | completed | [20261017T051346Z-0a0648ff.jsonl](./20261017T051346Z-0a0648ff.jsonl) | [20261017T051346Z-0a0648ff.md](./20261017T051346Z-0a0648ff.md) | [20261017T051346Z-0a0648ff.manifest.json](./20261017T051346Z-0a0648ff.manifest.json) |
| 5a7a2be25542996a35c17118 | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 0.500 / 0.333 / 0.400 | 0.667 | 0.000 / 0.000 | 22.584 | 11286 | 0.00214260 | completed | [20261017T051411Z-92e2e1ca.jsonl](./20261017T051411Z-92e2e1ca.jsonl) | [20261017T051411Z-92e2e1ca.md](./20261017T051411Z-92e2e1ca.md) | [20261017T051411Z-92e2e1ca.manifest.json](./20261017T051411Z-92e2e1ca.manifest.json) |
| 5a7a2be25542996a35c17118 | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 1.000 / 0.667 / 0.800 | 1.000 | 0.000 / 0.000 | 6.160 | 3339 | 0.00053790 | completed | [20261017T051433Z-969f991b.jsonl](./20261017T051433Z-969f991b.jsonl) | [20261017T051433Z-969f991b.md](./20261017T051433Z-969f991b.md) | [20261017T051433Z-969f991b.manifest.json](./20261017T051433Z-969f991b.manifest.json) |
| 5a7a2be25542996a35c17118 | 2 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 0.667 / 0.800 | 1.000 | 0.000 / 0.800 | 7.850 | 3554 | 0.00062390 | completed | [20261017T051440Z-656d053b.jsonl](./20261017T051440Z-656d053b.jsonl) | [20261017T051440Z-656d053b.md](./20261017T051440Z-656d053b.md) | [20261017T051440Z-656d053b.manifest.json](./20261017T051440Z-656d053b.manifest.json) |
| 5a7a2be25542996a35c17118 | 2 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.000 | 0.500 / 0.333 / 0.400 | 0.667 | 0.000 / 0.000 | 25.164 | 11822 | 0.00227960 | partial | [20261017T051447Z-125adfc2.jsonl](./20261017T051447Z-125adfc2.jsonl) | [20261017T051447Z-125adfc2.md](./20261017T051447Z-125adfc2.md) | [20261017T051447Z-125adfc2.manifest.json](./20261017T051447Z-125adfc2.manifest.json) |
| 5ae736cb5542991bbc9761c2 | 1 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.333 | 1.000 / 1.000 / 1.000 | 1.000 | 0.000 / 0.333 | 18.640 | 11305 | 0.00195610 | completed | [20261017T051513Z-4143c342.jsonl](./20261017T051513Z-4143c342.jsonl) | [20261017T051513Z-4143c342.md](./20261017T051513Z-4143c342.md) | [20261017T051513Z-4143c342.manifest.json](./20261017T051513Z-4143c342.manifest.json) |
| 5ae736cb5542991bbc9761c2 | 1 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.333 | 1.000 / 1.000 / 1.000 | 1.000 | 0.000 / 0.333 | 5.620 | 3649 | 0.00054040 | completed | [20261017T051531Z-86d32232.jsonl](./20261017T051531Z-86d32232.jsonl) | [20261017T051531Z-86d32232.md](./20261017T051531Z-86d32232.md) | [20261017T051531Z-86d32232.manifest.json](./20261017T051531Z-86d32232.manifest.json) |
| 5ae736cb5542991bbc9761c2 | 2 | single | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 0.000 / 0.333 | 1.000 / 1.000 / 1.000 | 1.000 | 0.000 / 0.333 | 5.770 | 3698 | 0.00056000 | completed | [20261017T051537Z-7f1e0636.jsonl](./20261017T051537Z-7f1e0636.jsonl) | [20261017T051537Z-7f1e0636.md](./20261017T051537Z-7f1e0636.md) | [20261017T051537Z-7f1e0636.manifest.json](./20261017T051537Z-7f1e0636.manifest.json) |
| 5ae736cb5542991bbc9761c2 | 2 | multi | d54 | openai-compatible / openai/gpt-oss-120b:nscale / json_schema | 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 20.719 | 11600 | 0.00200930 | completed | [20261017T051543Z-3e031ef1.jsonl](./20261017T051543Z-3e031ef1.jsonl) | [20261017T051543Z-3e031ef1.md](./20261017T051543Z-3e031ef1.md) | [20261017T051543Z-3e031ef1.manifest.json](./20261017T051543Z-3e031ef1.manifest.json) |

A missing `Answer:` first line produces an answer parse failure and zero answer EM/F1; inspect the linked report. The support scorer maps each cited sentence evidence ID back to its source `(title, sent_id)` and counts unknown citation IDs as false positives. Answer EM/F1 and support P/R/F1 follow the official HotpotQA normalization and set-metric conventions; joint EM/F1 combines answer and support scores. This compact pilot is evidence about this sample, model, and harness run configuration only.

Offline-provider rows exercise parsing, scoring, artifact generation, and label isolation only; they are not language-model quality evidence. Remote calls still require the existing interactive approval gate. A per-run USD cap still requires both token-rate variables.
