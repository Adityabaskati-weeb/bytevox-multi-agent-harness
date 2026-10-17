# D86 current-role HotpotQA pilot: campaign record

## Question and frozen design

D86 asked whether the exact current D54 implementation and its distinct specialist prompts improve answers over the same-build single-agent baseline on a small sample, and what time and estimated API-cost overhead they add. The protocol was frozen in [campaign-plan.md](campaign-plan.md) before model runs. It used three fresh HotpotQA dev distractor questions, two repeats, the same model/corpus/task/limits in both modes, alternating mode order, and scorer-only labels. Gold answers and supporting facts were not sent to the model or used in the validation contract.

- Source revision fingerprint: `180000d46fd2b3e88eb714a9b86e703899f9315e4c9e20e44a8fd43cdfe427cb`.
- Current role-prompt fingerprint: `6992422a52882a05c9c214fa7aaf45abae6e60994195fb34ff897c19d8721717`.
- Model route: Hugging Face Inference Providers, `openai/gpt-oss-120b:nscale`; 12 remote runs, all captured with reports, JSONL traces, and manifests.
- Sample IDs and scorer-isolation record: [sample-records.json](sample-records.json). Run-level results: [hotpotqa-benchmark.md](hotpotqa-benchmark.md), [CSV](hotpotqa-benchmark.csv), and [JSON](hotpotqa-benchmark.json).

## Measured results

| Measure | Multi-agent | Single-agent | Multi minus single / ratio |
|---|---:|---:|---:|
| Completed | 5/6 | 6/6 | One multi partial |
| Answer F1 | 0.556 | 0.611 | -0.056 |
| Supporting-fact F1 | 0.800 | 0.933 | -0.133 |
| Joint F1 | 0.556 | 0.578 | -0.022; paired 95% interval [-0.400, +0.333] |
| Median elapsed time | 22.419 s | 5.695 s | 3.94x slower |
| Provider tokens | 68,814 | 20,729 | 3.32x |
| Estimated API cost | $0.0126885 | $0.003197 | 3.97x |

The interval resamples only three task clusters and is very wide. These descriptive results do not establish a general quality difference. In the one partial multi-agent run, checker attempt 1 reached provider I/O, used 880 input and 566 output tokens, and was truncated at the 566-token completion cap. Retry 2 was blocked by the run token budget before provider I/O. No `checker_result` completed (`checker_accepted=null`); the saved answer remains partial and unverified, and the benchmark retains it under its declared scoring rule. Token-derived cost is an estimate, not an invoice or billing ceiling.

## Blind qualitative review

After the twelve outputs were saved, two independent AI reviewers received the corrected [anonymized packet](blind-review-packet.md), containing task questions, outputs, claims, and every cited source passage. They did not receive the modes, metrics, run IDs, benchmark labels, or key, and they were asked not to browse. The A/B mapping and per-run information were opened only after both ratings were returned. The `blind-review-key.json` mapping is retained in the local experiment archive and omitted from the curated submission package. An earlier packet omitted passages for multi-citation claims; those first ratings were discarded and are not used here.

Each reviewer rated correctness, completeness, citation support, and relevance/clarity from 0 to 2 for six outputs per arm (maximum 48 points per arm):

| AI reviewer | Multi-agent | Single-agent | Pairwise judgment |
|---|---:|---:|---|
| C | 48/48 | 41/48 | Multi favored on 2 pairs; 4 ties |
| D | 48/48 | 43/48 | Multi favored on 1 pair; 5 ties |

Both reviewers favored the multi-agent output on the second repeat of the band-membership task. The partial multi-agent output abstained because the cited passages did not provide complete historical member counts; the single-agent output confidently chose Del Amitri using only its original lineup as evidence. This was a reviewer preference, not checker validation: no checker decision completed, and the answer remains unverified. Reviewer C also marked a single-agent present-tense claim about The Saturdays as unsupported by the quoted past-tense passage; reviewer D did not. All other pairs were ties.

These are AI judgments about a six-pair, evidence-only packet, not human ratings or ground truth. They provide a small qualitative signal about grounding and safe abstention; they do not override the official metric, demonstrate a robust multi-agent quality win, or establish that the additional latency and cost are worthwhile.

## Decision

Keep D54 as the active implementation. Do not promote new prompts, limits, or defaults based on this pilot. The official joint-F1 result is slightly lower for multi-agent, the interval is uninformative at this sample size, one multi run was partial, and multi-agent took about 3.9 times the median time and estimated cost. The blind review suggests that an evidence-aware abstention can be preferable to an unsupported answer on an individual task, but the study is too small to estimate the frequency or value of that behavior.

The pilot strengthens the evidence that the current roles execute and produce inspectable outputs. It does not show that multi-agent is generally better than single-agent or earns its average overhead. Public dev questions may overlap model pretraining, and all three examples are from one task family.
