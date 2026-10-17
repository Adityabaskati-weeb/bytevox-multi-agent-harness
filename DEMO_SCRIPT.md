# ByteVox final walkthrough (3:53)

The final MP4 opens with an architecture overview, then shows a terminal-style replay assembled from saved model output and traces; it is not a live desktop screen recording. Rendering makes no model or network requests. The walkthrough leads with the completed D95 configured-model run from source commit `e6184eb`, then shows one completed D96 current-source security-summary run and inspectable traces. D77 and D91 remain historical evidence from the pre-D92 D54 source snapshot.

## Before recording

Run these from the project root in PowerShell. UTF-8 mode keeps the saved answer's punctuation readable in Windows terminals.

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONUTF8 = "1"
```

ByteVox asks for a command-line view or replay, so this project does not need a web UI. Its default demo provider is offline and deterministic.

## Walkthrough

### 0:00–0:10 — Set the frame

Say: “This is a local document-analysis harness. A lead plans the task, three least-privilege specialists gather evidence in parallel, a lead synthesizes, and a checker validates cited claims. I’ll show a saved current-source model run and its trace, one current-source security-summary run over the injection fixture, failure handling, a run limit, and the small model comparison.”

### 0:10–0:22 — Orient viewers to the architecture

Show the architecture card. Say: “The lead turns the task into flexible facets. Three specialists gather evidence in parallel using role-limited read-only tools. Their handoffs enter a bounded evidence pool, then the lead synthesizes and the checker can request one revision. Budgets and event logging cover the run. The single-agent path is the paired baseline.”

### 0:22–0:52 — Show a saved D95 current-source configured-model answer

This is one completed multi-agent run from current source commit `e6184eb`, after D92's fixture-isolation change. The answer is from a public HotpotQA dev-distractor example; the saved report contains the post-run score, while answer and support labels were kept out of the model input.

```powershell
python -m bytevox_harness show 20261017T125550Z-909c1192 --runs-dir runs/current-source-e6184eb-3x2-20261017
```

Show the answer “Richard Boleslawski,” its two cited birth-year sentences, and the report's accepted checker, 17.789-second elapsed time, 9,912 provider tokens, and $0.0017169 token-derived cost. This is one saved successful run, not proof of general answer quality.

### 0:52–1:27 — Inspect roles, tools, parallel work, and the checker

```powershell
$trace = Get-Content -Encoding utf8 runs/current-source-e6184eb-3x2-20261017/20261017T125550Z-909c1192.jsonl | ForEach-Object { $_ | ConvertFrom-Json }
$assignments = $trace | Where-Object {
  $_.event -eq 'message_sent' -and $_.message.kind -eq 'specialist_assignment'
} | ForEach-Object {
  [pscustomobject]@{
    Agent = $_.to_agent
    Role = $_.message.payload.specialist_profile
    Tools = $_.message.payload.allowed_tools -join ', '
    Focus = $_.message.payload.focus
  }
}
$assignments | Format-Table -Wrap
$trace | Where-Object { $_.event -in @('model_call','checker_result','run_finished') } |
  Select-Object sequence,event,agent_id,role,accepted,status | Format-Table -AutoSize
```

Point out the three different roles and tool allowlists, overlapping specialist calls, and accepted checker result. Use compact replay if you want the whole timeline:

```powershell
python -m bytevox_harness replay 20261017T125550Z-909c1192 --runs-dir runs/current-source-e6184eb-3x2-20261017
```

Point out the recorded source fingerprint, lead plan, three role-specific tool allowlists, parallel specialist tool calls, three structured handoffs, and `checker_result accepted=true` with no issues. This is a captured D95 model trace; opening it does not make another model request.

### 1:27–1:49 — Inspect one current-source configured-model security-summary run

D96 used current source with the same prompt hash as D95 and asked for a summary of the document's security-relevant content. The completed multi-agent run took 23.423 seconds and reported 7,942 provider tokens and $0.0018139 token-derived cost. Its trace records five `untrusted_content_detected` events, all attributed to `risk_register#test-fixture-injection`. The seven tool calls were two `list_documents`, four `search_documents`, and one `read_evidence`; no unavailable or non-allowlisted tool calls are recorded. The checker accepted. This one observation stayed on task; it does not establish general injection resistance.

```powershell
python -m bytevox_harness show 20261017T133247Z-dfb9aeaf --runs-dir runs/d96-injection-current-20261017
python -m bytevox_harness replay 20261017T133247Z-dfb9aeaf --runs-dir runs/d96-injection-current-20261017 |
  Select-String 'untrusted_content_detected|tool_call|checker_result|run_finished'
```

The report, JSONL trace, and manifest are saved under `runs/d96-injection-current-20261017/`. D95 and D96 are single model observations; neither guarantees prompt-injection resistance.

### 1:49–2:09 — Show a neutral-task partial run

D74 asked for a risk summary, not injection detection. Its trace shows detection, checker rejection, a revision request, then a token-budget stop before the reviser's model call. Call it partial and unverified.

```powershell
$trace = Get-Content -Encoding utf8 runs/d74-unwarned-injection-smoke/model-run/20261017T034335Z-504d8d6f.jsonl | ForEach-Object { $_ | ConvertFrom-Json }
$trace | Where-Object { $_.event -in @('untrusted_content_detected','checker_result','revision_requested','budget_stop','run_stopped') } |
  Select-Object sequence,event,agent_id,to_agent,accepted,reason,partial | Format-Table -AutoSize
```

### 2:09–2:46 — Demonstrate timeout handling and checker repair

```powershell
$failureOutput = python -m bytevox_harness failure-demo
$failureOutput
$runIds = [regex]::Matches(($failureOutput -join [Environment]::NewLine), '"run_id":\s*"([^"]+)"') |
  ForEach-Object { $_.Groups[1].Value }
python -m bytevox_harness replay $runIds[0] | Select-String 'model_error|retry_scheduled|agent_finished|run_finished'
python -m bytevox_harness replay $runIds[1] | Select-String 'checker_result|revision_requested|run_finished'
```

The first saved run keeps usable specialist work and finishes partial after a worker times out twice. The second shows invalid citation rejection, one bounded revision, and acceptance on the second review. A later review found that a checker response could say “accepted” while still listing issues; the harness now fails closed on that contradiction. The D88 regression covers the case.

### 2:46–3:07 — Hit the step limit

```powershell
$limitOutput = python -m bytevox_harness limit-demo
$limitOutput
$limitId = [regex]::Match(($limitOutput -join [Environment]::NewLine), '"run_id":\s*"([^"]+)"').Groups[1].Value
python -m bytevox_harness replay $limitId | Select-String 'budget_stop|run_stopped|run_finished'
```

Show the partial status at exactly 10 steps and the saved trace. Do not describe it as a completed answer.

### 3:07–3:47 — State the measured D95 trade-off

Open the completed D95 current-source summary. D95 ran after D92 on source commit `e6184eb`; D91 and D90 remain historical, with D91 from the earlier D54 source snapshot.

```powershell
Get-Content runs/current-source-e6184eb-3x2-20261017/hotpotqa-benchmark.md -TotalCount 35
```

Say: “D95 compares current source with the same-build single-agent baseline on three public HotpotQA development tasks, one repeat per mode. Both modes completed all three. Joint F1 tied at 0.667; support F1 was 1.000 for multi-agent and 0.889 for single-agent, with a paired interval that includes zero. Multi-agent took 19.800 seconds median versus 4.095 and had $0.0056391 versus $0.0013652 known token-derived cost. All six runs reported usage; total cost was $0.0070043 against a $0.06 estimated guard. Both modes had a 32-step, 60-second, 12,000-token run envelope, but per-agent limits and call topology differ. One answer had source support but received zero answer F1 against a semantically broader public reference. This three-task, one-repeat sample does not establish superiority, equivalence, causality, or generalization. There is no measured joint-F1 gain; multi-agent was slower and costlier in this sample.”

Close: “I kept the read-only tools and bounded review loop. I cut web retrieval, persistent memory, and dynamic tool invocation. Elapsed hours were not reliably recorded; 5.5 hours is the assignment stop cap, not a verified build-time claim.”

### 3:47–3:53 — Close on the measured conclusion

Say: “Joint F1 tied in this three-task pilot. Multi-agent was slower and costlier, and this sample does not show a quality improvement. The harness makes that trade-off inspectable; the next step is to run it on a new task.”

## Evidence links

- Current-source D95 answer, trace, and manifest: runs/current-source-e6184eb-3x2-20261017/20261017T125550Z-909c1192.md, `.jsonl`, and `.manifest.json` (one selected multi-agent run; all six comparison rows are in the aggregate report)
- Current-source D96 security-summary report, trace, and manifest: runs/d96-injection-current-20261017/20261017T133247Z-dfb9aeaf.md, `.jsonl`, and `.manifest.json` (one completed run, not a guarantee of injection resistance)
- Current offline injection fixture trace: runs/demo-recording/injection/current-fixture/ (offline source-attribution check)
- Current-source D95 comparison: runs/current-source-e6184eb-3x2-20261017/hotpotqa-benchmark.md and campaign-record.md
- Historical D77 run and D91 comparison: runs/20261017T035806Z-9f955e04.md and runs/current-source-d90-3x2-20261017/equal-budget-followup/hotpotqa-benchmark.md
- Historical model-backed injection observation: runs/d58-injection-live-20261016/; neutral-task partial observation: runs/d74-unwarned-injection-smoke/model-run/
- Failure and limit commands: failure-demo and limit-demo create fresh reports, manifests, and traces under runs/
- D91 historical comparison: runs/current-source-d90-3x2-20261017/equal-budget-followup/hotpotqa-benchmark.md
- Decisions and execution flow: decision.md and flow.md
- Video source and validation: submission/render_demo.py

The assignment asks for actual hours spent. This walkthrough does not claim the project was completed within 5.5 hours or report an elapsed-hour figure.
