# D87 offline retrieval and synthesis diagnostic

This record replays six frozen D86 multi-agent traces and their saved local corpora. It makes no model or network calls and is not an answer-quality benchmark. Exact evidence IDs, selected pools, estimate values, and run links are in [offline-diagnostic.json](offline-diagnostic.json).

## Fingerprints and source runs

- D86 frozen pre-change harness source: `180000d46fd2b3e88eb714a9b86e703899f9315e4c9e20e44a8fd43cdfe427cb`.
- D87 current harness source (12 Python files): `21b4ee12cc67fb5510c6cf85561c78d3b0241028a3cfd248a3122269d7b682ba`.
- Current `prompts.py`: `6992422a52882a05c9c214fa7aaf45abae6e60994195fb34ff897c19d8721717`.
- HotpotQA `hotpot_dev_distractor_v1`, public dev distractor data SHA-256: `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`.

| Task | Type | Corpus SHA-256 | Frozen D86 multi-agent source runs |
|---|---|---|---|
| `hotpotqa-0001` | bridge | `9e481356f1e6490f82e865d3acc396aa20f9640296cf5ece7622fb9c44213bb5` | [051315](../current-role-d54-3x2-20261017/20261017T051315Z-95f5e12c.jsonl), [051346](../current-role-d54-3x2-20261017/20261017T051346Z-0a0648ff.jsonl) |
| `hotpotqa-0002` | comparison | `e9671d9c2efc6358aff219d0e26183ba000bcf542277d0bb023ece0e3c5084a8` | [051411](../current-role-d54-3x2-20261017/20261017T051411Z-92e2e1ca.jsonl), [051447](../current-role-d54-3x2-20261017/20261017T051447Z-125adfc2.jsonl) |
| `hotpotqa-0003` | bridge | `999e269dc7817aeadd67a8a7b6f8daff1e7ad7be623658a676ce6aef812ec989` | [051513](../current-role-d54-3x2-20261017/20261017T051513Z-4143c342.jsonl), [051543](../current-role-d54-3x2-20261017/20261017T051543Z-3e031ef1.jsonl) |

## Retrieval replay

The metric is the union of `search_documents` evidence IDs from all three specialists; it excludes planner seed retrieval and reread evidence. With each specialist at five results, gold support recall was 12/14 (85.7%). Replaying only the corroborator at eight results raised the union to 14/14 (100%): both comparison-task repeats moved from 2/3 to 3/3, while the other four repeats were already full. Corroborator-query recall alone was 1/3→2/3 on the comparison task. These are saved-corpus retrieval results, not answer or citation-quality results.

Request sizes below use the same OpenAI-compatible request-envelope builder and `estimate_tokens()` character/4 estimator for the saved five-result/full-inventory request and proposed eight-result/compacted request.

| Frozen run | Five/full | Eight/compact raw | Raw delta | Gate selection | Selected delta |
|---|---:|---:|---:|---|---:|
| `20261017T051315Z-95f5e12c` | 1,818 | 1,740 | −78 | eight/compact | −78 |
| `20261017T051346Z-0a0648ff` | 1,821 | 1,744 | −77 | eight/compact | −77 |
| `20261017T051411Z-92e2e1ca` | 1,593 | 1,627 | +34 | five/full | 0 |
| `20261017T051447Z-125adfc2` | 1,637 | 1,672 | +35 | five/full | 0 |
| `20261017T051513Z-4143c342` | 1,843 | 1,759 | −84 | eight/compact | −84 |
| `20261017T051543Z-3e031ef1` | 1,816 | 1,726 | −90 | eight/compact | −90 |

The raw expanded estimate deltas sum to −260; the gate-selected request deltas sum to −329. An earlier rough estimate gave +34 for both comparison-task requests; recomputing each serialized envelope with the same builder and estimator gives +34 and +35, so both correctly select the five-result/full-inventory payload. Estimates are harness budgeting values, not provider tokenization, billed cost, or latency.

## Synthesis projection and rejected design

The frozen D86 worker claims were combined with replayed eight-result corroborator evidence and passed through the unchanged eight-record synthesis selector. The newly retrieved gold fact `doc-06-s-002-del-amitri#sentence` was absent from both comparison-task pools. D87-current projected synthesis estimates versus D86 were `0, 0, 0, 0, +10, +10`; exact per-run values and selected IDs are in the JSON. The last two increases are present in this offline projection and are not attributed to a quality or latency effect.

A design-only one-slot simulation promoted the deepest new corroborator result by replacing the last unclaimed, non-planner-seed item when the exact synthesis estimate did not rise versus D87-current. Simulated gold coverage rose from 12/14 to 14/14; request-estimate deltas versus D87-current were `−5, −10, −12, −11, 0, 0`. The two bridge-task swaps would replace `doc-06-s-004-the-voice-of-ireland-series-4#sentence` with non-gold `doc-03-s-002-headlines#sentence`. The comparison-task swaps would replace `doc-02-s-000-the-silencers-band#sentence` with `doc-06-s-002-del-amitri#sentence`. The selected bridge records are not gold facts, but this does not show they are useless to answer generation. The rule was rejected: rank-based replacement can displace arbitrary relevant evidence, and gold coverage on three tasks cannot establish the swaps are safe. Runtime synthesis selection is unchanged. D87 remains an offline retrieval-only diagnostic; there is no model-backed answer-quality result.
