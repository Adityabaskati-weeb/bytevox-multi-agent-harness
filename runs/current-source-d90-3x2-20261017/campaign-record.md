# D90 historical current-source HotpotQA pilot: campaign record

D90 remains preserved as originally measured and is no longer the current comparison. The later [D91 equal-run-envelope follow-up](equal-budget-followup.md) ran three task pairs with one repeat per mode; its aggregate report is [here](equal-budget-followup/hotpotqa-benchmark.md). D90's historical result and effective-budget caveat are retained below.

## Decision

Keep the current D54 prompt and runtime defaults. This pilot found no joint-F1 difference between multi-agent and single-agent modes, while multi-agent was slower, more costly, and completed one fewer run. The small sample and a reference-answer ambiguity prevent a reliable quality conclusion. D90 does not justify a prompt, routing, budget, or synthesis-policy change.

## Method

D90 ran three new tasks from the public HotpotQA dev distractor set, with two repeats per task and mode (12 total runs). The multi-agent and single-agent arms used the same current source, D54 prompt version, provider/model, per-task corpus, raw question, generic contract, response format, and limits. Gold answers and supporting-fact labels remained in scorer-only records and were evaluated after each harness run.

The frozen plan records the source, prompt, dataset and corpus hashes, exact sample IDs, selection method, provider rate assumptions, command, budgets, and analysis rule. The run caps were 12,000 estimated tokens total, 4,000 per agent, 20 seconds per agent, 60 seconds per run, $0.01 estimated per run, and $0.12 estimated aggregate. Rates were $0.10/M input tokens and $0.40/M output tokens. The aggregate preflight estimate is not a billing ceiling. The local HotpotQA dev split is public development data and may overlap model pretraining.

Although both modes received the same configured limit values, their effective aggregate step/token/time and per-call completion budgets differed. The single-agent baseline reused one `single_agent` budget identity, so its 24-step, 4,000-token, and 20-second per-agent limits applied across all calls; its standard completion cap was 1,024 tokens. Multi-agent roles used separate per-agent buckets and could use the shared 32-step, 12,000-token, and 60-second run caps; lead planner/synthesizer calls had a 1,536-token completion cap. D90 is not an equal-total-resource comparison; its recorded results describe the modes under these effective limits and do not isolate architecture at equal budgets.

A [completed D91 equal-run-envelope follow-up](equal-budget-followup.md) reused these three task IDs for one repeat per mode and capped both modes at the same 12,000-token / 60-second / 32-step run envelope. It raised the single-agent per-agent caps and its response cap to the multi-agent lead cap. D91's completion, metrics, partial-run caveat, and artifact links are documented separately; the D90 results below are unchanged.

## Results

| Measure | Multi-agent | Single-agent |
|---|---:|---:|
| Completed | 5/6 | 6/6 |
| Answer EM / F1 | 0.667 / 0.667 | 0.667 / 0.667 |
| Supporting-fact precision / recall / F1 | 0.944 / 1.000 / 0.967 | 1.000 / 0.833 / 0.889 |
| Retrieval supporting-fact recall | 1.000 | 0.833 |
| Joint EM / F1 | 0.667 / 0.667 | 0.667 / 0.667 |
| Median wall time | 20.340 s | 4.113 s |
| Known provider tokens | 63,526 | 18,471 |
| Known token-derived API cost | $0.0119644 | $0.0025248 |

Across six paired trials and three task clusters, mean multi-minus-single joint-F1 difference was 0.0000 (95% task/repeat bootstrap interval [0.0000, 0.0000]); answer-F1 difference was also 0.0000. Supporting-fact F1 difference was +0.0778 (95% interval [0.0000, +0.2333]). The degenerate joint-F1 interval reflects this particular sample: two tasks were answered perfectly by both modes and the remaining task scored zero under the official answer reference. It is not evidence of equivalence or a general null effect.

Multi-agent was 4.95× slower by median wall time and 4.74× higher by known token-derived cost. Total known cost was $0.0144892, below the $0.12 estimated aggregate guard. All 12 model calls had known provider usage. Costs are calculated from listed rates and provider token counts, not provider invoices.

## Item-level interpretation and partial run

For task `hotpotqa-0001`, the HotpotQA gold answer is the generic phrase “international football competition.” Both single-agent repetitions answer “FIFA Women's World Cup” and cite the supplied Liu Ailing passage, which explicitly says she played those editions of that tournament. Official token-normalized answer F1 gives those semantically specific answers zero. The scorer was left unchanged; the result is reported as scored, with the reference mismatch documented as a limitation of this item.

The first multi-agent repetition for that task is partial. It used 10,111 of the 12,000 estimated run tokens in 12 steps over 18.470 seconds. The token-budget preflight blocked the lead synthesizer before model I/O, so no checker ran. Its preserved specialist draft is repetitive and does not begin with the required `Answer:` line, giving it a parse failure and zero answer score. This is a real budget/completion weakness, not a quality win. The remaining 1,889-token arithmetic difference did not guarantee that a synthesis or full checker request could fit.

## Exploratory blinded AI review

After the six output pairs were frozen, two independent AI reviewers read the same packet. The packet hid mode names, gold answers, and benchmark metrics, but showed output text, citations and evidence, plus completion/checker status. Both reviewers preferred multi-agent in five of six pairs and preferred single-agent for the partial, repetitive multi-agent output. They found factual answers largely comparable; preferences in the other pairs mostly followed visible checker/completion status. Reversing presentation order did not change either reviewer's preferences.

These are exploratory AI judgments, not human ratings or ground truth. Because run status was visible and multiple preference decisions followed checker status, the 5/6 count should not be read as a blinded factual-quality win. The answer/citation comments and the mode mapping are in the review packet and its local-only key; the key is excluded from the curated package.

## Cost and budget review

No model run exceeded its $0.01 estimated per-run guard or the $0.12 aggregate preflight guard. The partial run was stopped by the run token budget, not its time or USD guard. A post-run code-path review found that reserving an arbitrary fixed token amount for synthesis/checker is not supported by the input-size variability, and the existing opt-in D64 recovery has not been shown to fit this case. D89 is one different task at a 16,000-token cap and does not establish that raising the default improves quality or resolves D90's partial run.

Keep the 12,000-token normal default and D64 disabled. A future 16,000-token pilot may be useful as an operational completion diagnostic if it holds all other limits and USD guards constant, but it must not be presented as quality evidence without paired repetitions on the same tasks.

## Conclusion

D90 supports an honest trade-off statement: on this small HotpotQA sample, multi-agent found more supporting facts, but joint answer quality tied the single-agent baseline, one multi-agent run remained partial, and multi-agent cost about 4.74× more and took about 4.95× longer. This is insufficient evidence that multi-agent is worth the overhead for this workload. It does not establish that multi-agent cannot help on other tasks.

## Later D91 equal-run-envelope follow-up

D91 used the same three public task IDs and one repeat per mode, with both modes sharing 32 run steps, 60 seconds, and 12,000 total tokens. The single-agent baseline received the full run-level per-agent allowance and a 1,536-token completion cap; multi-agent workers retained 1,024-token caps and lead calls used 1,536. Thus the shared run envelope was equal, while role topology and per-call worker caps still differed.

| Measure | Multi-agent | Single-agent |
|---|---:|---:|
| Completed | 2/3 | 3/3 |
| Answer F1 / support F1 / joint F1 | 0.667 / 1.000 / 0.667 | 0.667 / 0.889 / 0.667 |
| Median wall time | 20.563 s | 4.576 s |
| Known provider tokens | 32,998 | 9,317 |
| Known token-derived cost | $0.0061378 | $0.0012950 |

Joint-F1 difference was 0.0000 (95% descriptive interval [0.0000, 0.0000]); support-F1 difference was +0.1111 ([0.0000, +0.3333]). Multi-agent was 4.49× slower and 4.74× more costly by known token-derived usage. The first multi-agent run stopped after 11,811/12,000 provider-reported tokens before synthesis and checker, leaving a partial, unverified answer. For that task, the specific source-supported “FIFA Women's World Cup” answer scores zero against HotpotQA's generic “international football competition” reference. All six runs had known usage, totaling $0.0074328 estimated token cost under the $0.06 aggregate estimate guard. With only three public dev tasks and one repeat, this does not establish superiority, equivalence, or generalization; public data may overlap pretraining. The complete results, exact command, and artifact links are in the [D91 follow-up record](equal-budget-followup.md) and [aggregate report](equal-budget-followup/hotpotqa-benchmark.md). D90's scores and artifacts above are unchanged.

## Artifacts

- [Frozen campaign plan](campaign-plan.md)
- [Scored benchmark](hotpotqa-benchmark.md)
- [Machine-readable benchmark JSON](hotpotqa-benchmark.json)
- [Machine-readable benchmark CSV](hotpotqa-benchmark.csv)
- [Mode-masked review packet](blind-review-packet.md)
- The scorer-excluded corpus payload and reviewer mode-mapping key are kept local and excluded from the curated package.
