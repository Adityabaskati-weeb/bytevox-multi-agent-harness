# Run artifact provenance

## D91 current-source equal-run-envelope HotpotQA pilot

D91 is the current model-backed comparison: three public HotpotQA dev tasks × one repeat per mode, with paired multi-agent and same-build single-agent runs. Both received the same task corpus and generic contract; labels stayed in the scorer. Both modes shared a 32-step / 60-second / 12,000-token run envelope. Multi-agent completed 2/3 versus 3/3, with joint F1 0.667 in both modes and support F1 1.000 versus 0.889. Median time was 20.563 versus 4.576 seconds; known token-derived cost was $0.0061378 versus $0.0012950. Paired joint-F1 difference was 0.0000; support-F1 difference was +0.1111 with an interval that includes zero.

One multi-agent run stopped at 11,811/12,000 tokens before synthesis and checker. A source-supported “FIFA Women's World Cup” answer scored zero against the generic HotpotQA reference “international football competition.” All six runs had known token usage; $0.0074328 token-derived cost remained within the $0.06 estimated guard. With only three public dev tasks and one repeat, D91 does not establish superiority, equivalence, or generalization. Role topology and per-call completion caps still differ. See the [completed protocol and caveats](current-source-d90-3x2-20261017/equal-budget-followup.md), [aggregate benchmark](current-source-d90-3x2-20261017/equal-budget-followup/hotpotqa-benchmark.md), and per-run reports, traces, and manifests linked by that report.

## D90 historical current-source HotpotQA pilot

D90 is the earlier comparison: three new public HotpotQA dev tasks × two repeats, with paired current multi-agent and same-build single-agent runs. The agents received the same task corpus and generic contract; the scorer held answer and support labels. Multi-agent and single-agent joint F1 were both 0.667; supporting-fact F1 was 0.967 vs 0.889. Multi-agent completed 5/6 vs 6/6, took 20.340 s median vs 4.113 s, and had $0.0119644 known token-derived cost vs $0.0025248. A paired interval over three task clusters does not establish a quality difference. One multi-agent run hit the run-token limit before synthesis and checker. One reference answer also appears semantically narrower than the answer supported by the cited source; see the campaign record for the item-level analysis.

Although D90 used the same configured limit values, single-agent calls shared one 24-step / 4,000-token / 20-second budget bucket and 1,024-token response cap; multi-agent roles could use the 32-step / 12,000-token / 60-second run envelope and 1,536-token lead calls. D90 is therefore historical evidence under unequal effective budgets, not an equal-resource architecture comparison. Two exploratory AI reviewers, given a packet with mode and gold labels masked but completion/checker state visible, preferred the multi-agent output in 5/6 pairs. They preferred the single-agent output for the partial, repetitive multi-agent answer; other preferences mostly tracked checker/completion state. This is not human ground truth. No runtime change was promoted. See the [frozen plan](current-source-d90-3x2-20261017/campaign-plan.md), [campaign record](current-source-d90-3x2-20261017/campaign-record.md), [benchmark](current-source-d90-3x2-20261017/hotpotqa-benchmark.md), [review packet](current-source-d90-3x2-20261017/blind-review-packet.md), and the per-run reports, traces, and manifests in that directory. The raw corpus and mapping key are excluded from the curated package.

## D77 new task on the existing North Harbor corpus

D77 ran one new plain-English question in multi-agent and single-agent modes over the same three fictional local documents, same model, D54 prompt version, corpus hash, response format, and normal `run()` limits. The role-specific prompt texts differ across roles and modes by design; “same D54” means the same shared prompt version. Both runs completed and answered that on-time performance rose from 82% to 88%, with the report cautioning that its non-randomized design cannot attribute the change to the interventions alone. Two blinded AI reviewers scored both outputs 8/8 after independent review; their ratings found no material difference. This is one descriptive pair on the existing corpus, not a new-corpus or quality-superiority result.

| Mode | Status | Checker | Elapsed | Provider tokens | Reported/token-derived cost |
|---|---|---|---:|---:|---:|
| Multi-agent | Completed | Accepted | 21.495 s | 10,079 | $0.00193010 |
| Single-agent | Completed | Not applicable | 5.254 s | 2,225 | $0.00038660 |

Multi-agent took 4.09× as long and cost 4.99× as much in this pair. No prompt or default was promoted; D54 remains active and official HotpotQA figures are unchanged. See the [campaign record](d77-unfamiliar-document-pilot/campaign-record.md), [frozen plan](d77-unfamiliar-document-pilot/campaign-plan.md), [blind-review packet](d77-unfamiliar-document-pilot/blind-review-packet.md), multi-agent [report](20261017T035806Z-9f955e04.md), [trace](20261017T035806Z-9f955e04.jsonl), [manifest](20261017T035806Z-9f955e04.manifest.json), and single-agent [report](20261017T035843Z-48b3136c.md), [trace](20261017T035843Z-48b3136c.jsonl), [manifest](20261017T035843Z-48b3136c.manifest.json).

## D73 source-scope diagnostic (completed; not promoted)

The completed D73 pilot compares the opt-in source-scope prompt with D54 on the same three HotpotQA IDs and corpora used by D70/D71: one repeat per task and mode (12 total runs). D73 did not improve mean answer or joint F1 in either mode, and its single-agent Girl 6 answer retained an unqualified company count. The qualitative source-scope check is exploratory and no formal blind D73 rating was performed. D73 is not promoted; D54 remains the default. See the campaign record (local archive, omitted from curated package: `runs/d73-source-scope-pilot/campaign-record.md`), protocol and decision (local archive, omitted from curated package: `runs/d73-source-scope-pilot/campaign-plan.md`), D54 benchmark (local archive, omitted from curated package: `runs/d73-source-scope-pilot/control-d54/hotpotqa-benchmark.md`), and D73 benchmark (local archive, omitted from curated package: `runs/d73-source-scope-pilot/candidate-d73/hotpotqa-benchmark.md`).

| Variant | Mode | Completed | Answer F1 | Support F1 | Joint F1 | Median seconds | Known cost USD |
|---|---|---:|---:|---:|---:|---:|---:|
| D54 | Multi-agent | 2/3 | 0.833 | 0.833 | 0.667 | 21.700 | 0.0057717 |
| D73 | Multi-agent | 2/3 | 0.750 | 0.800 | 0.595 | 23.171 | 0.0062527 |
| D54 | Single-agent | 3/3 | 0.933 | 0.833 | 0.767 | 5.010 | 0.0014067 |
| D73 | Single-agent | 3/3 | 0.767 | 0.889 | 0.667 | 4.254 | 0.0013617 |

All rows use the same model and normal run limits. The one-repeat, three-question public dev sample is a diagnostic replication, not a holdout or general quality estimate; the full metric rows and official scorer outputs remain in the linked benchmark reports.

## Neutral-task injection smoke (D74/D75; partial runs)

A task about the North Harbor risk register did not mention injection, but retrieval returned the hostile test paragraph. D74's trace records untrusted-content detection for the planner and two specialists, then `checker_result accepted=false`; the revision request hit the token budget before the reviser model call. Its saved report predates the observability correction and incorrectly says the checker was not run. D75 ended partial because the remaining budget could not provide the required 1,024-token completion cap before a checker result; this is not evidence of D64 recovery success. D76's 16,000-token trace also records `checker_result accepted=false`, followed by a truncated lead-reviser response and a retry stopped by the token budget. Its saved report predates the same correction. These are individual incomplete observations, not checker-accepted results or quality gains. See the [D74 report](d74-unwarned-injection-smoke/model-run/20261017T034335Z-504d8d6f.md) and [trace](d74-unwarned-injection-smoke/model-run/20261017T034335Z-504d8d6f.jsonl), the [D75 report](d75-recovery-unwarned-injection/model-run/20261017T034429Z-0edcd727.md) and [trace](d75-recovery-unwarned-injection/model-run/20261017T034429Z-0edcd727.jsonl), and the [D76 report](d76-unwarned-injection-16k/model-run/20261017T034611Z-5c508b3e.md) and [trace](d76-unwarned-injection-16k/model-run/20261017T034611Z-5c508b3e.jsonl).

## Latest post-D66 normal-limit smoke (D71)

D71 is a one-repeat-per-mode budget-sensitivity smoke check on the same three HotpotQA IDs and corpora as D70, using `openai/gpt-oss-120b:nscale` through the Hugging Face Inference Providers Router with `json_schema`. It explicitly supplied the normal `run()` limits to `hotpotqa-compare`: 12,000 estimated run tokens, 4,000 per-agent tokens, 20 seconds per agent, and 60 seconds overall. All six runs completed with known provider usage. Mean answer/support/joint F1 was 0.833/0.833/0.667 for multi-agent and 0.867/0.889/0.767 for single-agent. The paired joint-F1 difference was −0.1000 (95% descriptive interval [−0.3000, 0.0000], n=3), showing no quality win. Known costs were $0.005548 and $0.0013496 ($0.0068976 total); the $0.01 per-run cap and $0.06 aggregate guard are estimates, not billing ceilings. D71 reuses D70's exact public IDs and corpora, so it is not an independent sample or holdout; the dev data may overlap model pretraining. The post-run campaign record (local archive, omitted from curated package: `runs/post-d66-normal-run-limits-3x1-20261017/campaign-record.md`), benchmark report (local archive, omitted from curated package: `runs/post-d66-normal-run-limits-3x1-20261017/hotpotqa-benchmark.md`), and traces, reports, and manifests are in that folder.

D70 is a separate two-repeat-per-mode pilot using the `hotpotqa-compare` defaults (16,000 run tokens, 6,000 per-agent tokens, 32 seconds per agent, 60 seconds overall). It shares D71's sample and corpora and is likewise not a strict holdout. See the D70 campaign plan (local archive, omitted from curated package: `runs/post-d66-hotpotqa-3x2-20261017/campaign-plan.md`) and benchmark report (local archive, omitted from curated package: `runs/post-d66-hotpotqa-3x2-20261017/hotpotqa-benchmark.md`).

## Earlier post-D66 model-backed smoke (D70)

D70 is the earlier model-backed evaluation of the active source after D66 aligned the request body with budget estimation. It uses three public HotpotQA task IDs × two repeats in current D54 multi-agent and single-agent modes (12 harness runs). All 12 completed, all usage was reported, and no run exceeded its $0.01 estimated cap. The paired joint-F1 difference was −0.0111 (95% descriptive interval [−0.1556, +0.1333]); this small sample does not establish a quality gain. Multi-agent median latency was 19.799 s versus 5.107 s for single-agent, with known token-derived cost of $0.0108058 versus $0.0027738. The run's total known cost was $0.0135796; the $0.12 aggregate guard is only a preflight estimate. See the campaign plan (local archive, omitted from curated package: `runs/post-d66-hotpotqa-3x2-20261017/campaign-plan.md`), benchmark report (local archive, omitted from curated package: `runs/post-d66-hotpotqa-3x2-20261017/hotpotqa-benchmark.md`), and per-run traces, reports, and manifests in that folder.

The three IDs were absent from 38 earlier scored benchmark JSON files and 767 earlier generated HotpotQA case corpora, but occur in pre-existing public candidate-pool files. The sample is not a strict held-out set and may overlap model pretraining. Gold labels remain scorer-only. Treat this as a current-source smoke pilot, not a general quality estimate.

## Latest broad public-data evaluation (D56)

D56 is the broadest public-data evaluation: 110 unique HotpotQA question/task clusters × three repeats across three arms. Each arm has 330 rows, and each comparison has n=330 paired run-pairs matched by task ID and repeat; these are not 330 unique questions. There are 990 planned run slots total. The campaign summary (local archive, omitted from curated package: `runs/hotpotqa-d56-evidence-chain-ab/campaign-summary.md`) records the full results and paired uncertainty; the campaign record (local archive, omitted from curated package: `runs/hotpotqa-d56-evidence-chain-ab/campaign-record.md`) documents the plan and provenance.

| Group | Completed | Answer F1 | Support F1 | Joint F1 | Median seconds | Known cost subtotal | Known / unknown usage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen D54 multi-agent control | 300/330 | 0.6739 | 0.7488 | 0.5563 | 21.736 | $0.609889 | 305 / 25 |
| D56 candidate multi-agent | 298/330 | 0.6841 | 0.7452 | 0.5700 | 21.886 | $0.612703 | 302 / 28 |
| Matched D54 single-agent | 310/330 | 0.6416 | 0.7465 | 0.5358 | 5.105 | $0.148368 | 309 / 21 |

The candidate-minus-control multi-agent joint-F1 difference was +0.0137 (95% interval [−0.0191, +0.0462]); the candidate-minus-single difference was +0.0342 ([−0.0227, +0.0942]). Both intervals include zero. The candidate completed fewer runs than the control and was not promoted. Multi-agent runs were about four times slower and had higher known token-derived cost. These costs are incomplete estimates, not invoices; 74 of 990 rows had unknown usage. The campaign covers one task family and does not establish a general or causal multi-agent advantage.

A post-hoc supplementary comparison of the active D54 multi-agent control against the matched D54 single-agent baseline gives joint-F1 difference +0.02047394 (95% hierarchical-bootstrap interval [−0.03849046, +0.08167185]; 330 pairs, 110 tasks; 10,000 draws, seed `20261022`). The interval includes zero and this is not a preregistered promotion gate. D54 multi-agent median latency was 21.7365 s versus 5.1055 s (4.26×), with a paired mean difference of +17.0302 s. Paired known-cost difference was +$0.00151807 per run for 302/330 pairs with both costs reported. See the summary (local archive, omitted from curated package: `runs/hotpotqa-d56-evidence-chain-ab/campaign-summary.md`) and campaign record (local archive, omitted from curated package: `runs/hotpotqa-d56-evidence-chain-ab/campaign-record.md`); no prompt or routing rule was promoted.

## Latest local-fixture model comparison (D61)

D61 is the latest local-fixture comparison: three fixed tasks × three repeats in three arms (27 scheduled harness runs). See the benchmark report (local archive, omitted from curated package: `runs/d61-three-arm-3x3/benchmark.md`) and AI-rating report (local archive, omitted from curated package: `runs/d61-three-arm-3x3/ai-rating-results.md`).

| Group | Completed | Mean fixture score / 100 |
|---|---:|---:|
| D54 multi-agent control | 6/9 | 75.0 |
| D61 concise-specialist candidate | 4/9 | 74.3 |
| D54 single-agent baseline | 5/9 | 38.2 |

The fixture score measures evidence-ID coverage, citation-ID validity, and key-term presence; it does not judge entailment. One exploratory AI judge rated the outputs for factuality, source support, and completeness/usefulness. The packet included partial-run notices, masking may have been imperfect, and there was no human ground truth. D61 did not improve on D54 and remains unpromoted. The three-task sample cannot establish general quality or statistical significance.

## Offline mechanics benchmark (D22 label isolation)

The top-level offline benchmark report (local archive, omitted from curated package: `runs/benchmark.md`), CSV (local archive, omitted from curated package: `runs/benchmark.csv`), and JSON (local archive, omitted from curated package: `runs/benchmark.json`) contain 30 run rows: three tasks × five repeats × two modes. The evaluator applies gold labels after each run; those references are not present in the runtime contracts. This checks deterministic harness mechanics, not model answer quality. Older artifacts in this folder may predate D22. Do not use an artifact as label-isolated evidence if its manifest or trace has non-empty `validation_contract.must_cover_evidence_ids` or `validation_contract.must_cover_terms` fields.

## Earlier public-data evaluations

The earlier D52 v8 comparison (local archive, omitted from curated package: `runs/hotpotqa-hf-gpt-oss-120b-nscale-v8-3rep-30case/hotpotqa-benchmark.md`) used 30 new HotpotQA IDs × three paired repeats per mode (180 scheduled runs). Multi/single completed 77/90 and 88/90; joint F1 was 0.5260/0.4482 with paired delta +0.0778 (95% two-level task/repeat bootstrap interval [−0.0077, +0.1711]). The sample was small and does not establish causal or general performance.

The D51 label-isolated prompt comparison (local archive, omitted from curated package: `runs/prompt-answer-slot-ab/d50-ab-summary.md`) evaluated ten new IDs. Candidate/control completed 18/20 and 19/20; paired multi-agent joint-F1 delta was +0.0136 with an interval crossing zero. Gold labels were scorer-only. D51 is historical and has been superseded by D56 as the latest public-data comparison.

## Historical pre-D22 runs

Older comparison artifacts are preserved for history. The contaminated files identified in this workspace include run IDs from `20261015T152348Z` through `20261015T153445Z`; the timestamp range alone does not identify each affected file. Do not use any run as label-isolated evaluation evidence if its manifest or trace has a non-empty `validation_contract.must_cover_evidence_ids` or `validation_contract.must_cover_terms`. Ordinary corpus results can naturally contain the same words or evidence IDs, so inspect the contract fields rather than searching the entire trace text.
