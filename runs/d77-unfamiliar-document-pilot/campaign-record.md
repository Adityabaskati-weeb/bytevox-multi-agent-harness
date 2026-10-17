# D77 unfamiliar task comparison — campaign record

Here, “D77” is the campaign identifier. The separate D77 entry in the project [decision log](../../decision.md) refers to preserving checker decisions on partial runs; the shared number does not mean these are the same record.

## Result

D77 asked a new plain-English question over the existing fictional North Harbor corpus: “How did the North Harbor pilot change on-time performance, and what limitation does the report give on attributing that change to its interventions?” This is a new task on the same three local documents, not a new or unseen corpus. Both runs used the same corpus SHA-256 (`65148018862af53820517893bdd3e66456c1fe2c045942e14fa711595f0a6c3a`), model route, prompt variant D54, and normal `run()` limits.

| Mode | Run | Status | Checker | Elapsed | Provider tokens | Reported/token-derived cost |
|---|---|---|---|---:|---:|---:|
| Multi-agent | [`20261017T035806Z-9f955e04`](../20261017T035806Z-9f955e04.md) | Completed | Accepted | 21.495 s | 10,079 | $0.00193010 |
| Single-agent | [`20261017T035843Z-48b3136c`](../20261017T035843Z-48b3136c.md) | Completed | Not applicable in this mode | 5.254 s | 2,225 | $0.00038660 |

Both answered that on-time performance rose from 82% at baseline to 88% in the final four weeks and that the pilot was not randomized, so the report does not attribute the change to the interventions alone. Both cite `operations#results`; the multi-agent answer also lists the interventions from `operations#interventions`.

The multi-agent run took 4.09× as long and had 4.99× the reported token-derived cost of the single-agent run. These are single-run measurements; cost is not an invoice or a provider billing guarantee.

## Blinded AI review

Two AI reviewers scored anonymized outputs A and B independently using the four-criterion, 0–2-per-criterion rubric in the [review packet](blind-review-packet.md). After both ratings were frozen, the mapping was opened: A was multi-agent and B was single-agent. Both reviewers gave each output 8/8 and reported no material difference. These are AI review scores, not human ground truth; the single paired task cannot establish a quality advantage.

## Decision and limits

**No promotion. D54 remains the default.** D77 is a descriptive functionality and cost/time example for a new task over the existing fictional corpus. It does not change the official HotpotQA results, establish generalization to a new corpus, or show multi-agent quality superiority. The paired AI ratings found no difference on this one answer, while single-agent was substantially faster and cheaper.

The frozen [campaign plan](campaign-plan.md), answer [review packet](blind-review-packet.md), both saved reports, traces, and manifests provide the audit trail:

- Multi-agent: [report](../20261017T035806Z-9f955e04.md) · [trace](../20261017T035806Z-9f955e04.jsonl) · [manifest](../20261017T035806Z-9f955e04.manifest.json)
- Single-agent: [report](../20261017T035843Z-48b3136c.md) · [trace](../20261017T035843Z-48b3136c.jsonl) · [manifest](../20261017T035843Z-48b3136c.manifest.json)
