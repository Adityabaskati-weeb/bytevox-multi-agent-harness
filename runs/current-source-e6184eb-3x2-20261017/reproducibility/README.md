# Reproduce D95's fixed three-task sample

This package contains only the three HotpotQA records used by D95, including their original questions, contexts, and reference labels. The record values are unchanged; the file is a compact subset of the public `hotpot_dev_distractor_v1` dev-distractor split. It replaces the 61 MB source download and 440-ID exclusion list for reproducing these exact tasks. It does not recreate the original full-dataset sampling process.

The bundled file SHA-256 is `dd9696c5f1e37f99e38114a49dd8c2bbfba6e688aec4cba422edc1c030b9d36b`. The source file used for D95 was a Hugging Face mirror download with SHA-256 `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`; the records were checked against that file. The D95 run command omitted `--source-url`, so its original generated metadata inherited the CLI's CMU default; the archived Markdown, JSON, and CSV provenance now identify the actual mirror. For a new run over this packaged subset, the command below records the mirror as the source from which the records were obtained.

## Attribution and license

HotpotQA is by Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Cite Yang et al., “HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering,” EMNLP 2018, [ACL Anthology D18-1259](https://aclanthology.org/D18-1259/). The [official HotpotQA homepage](https://hotpotqa.github.io/) identifies the dataset's license as [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/); the [official dataset repository](https://github.com/hotpotqa/hotpot) states the same license. This three-record subset and the corpus text derived from it are distributed under CC BY-SA 4.0 with that attribution.

The records came from the public [Hugging Face HotpotQA mirror](https://huggingface.co/datasets/namlh2004/hotpotqa). The [official CMU dev-distractor file](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json) is the canonical dataset download. D95's full input file hash matches the mirror copy recorded in the experiment history, not the CMU file checksum.

## Run the same comparison

From the repository root, set `HF_TOKEN` in the environment. This command runs one `multi` and one `single` trial for each bundled record with the D95 model, response format, prompt variant, run envelope, and cost guards. The harness asks for explicit `APPROVE` before making remote model requests.

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "src"
$env:BYTEVOX_INPUT_USD_PER_1M = "0.10"
$env:BYTEVOX_OUTPUT_USD_PER_1M = "0.40"

python -m bytevox_harness hotpotqa-compare `
  --dataset-path runs/current-source-e6184eb-3x2-20261017/reproducibility/hotpotqa-dev-distractor-v1-d95-sample.json `
  --source-url https://huggingface.co/datasets/namlh2004/hotpotqa `
  --sample-size 3 --sample-seed 20261017 --repeats 1 `
  --max-tokens 12000 --max-agent-tokens 4000 --max-agent-seconds 20 `
  --provider openai-compatible --base-url https://router.huggingface.co/v1 `
  --model openai/gpt-oss-120b:nscale --api-key-env HF_TOKEN `
  --response-format json_schema --approve-remote-model `
  --max-cost-usd 0.01 --max-aggregate-cost-usd 0.06 `
  --runs-dir runs/d95-reproduction
```

The three-record file is already the selected sample. The seed is retained in the command for recorded sampling provenance; because every bundled record is selected, its output order stays the same. The loader rebuilds the sentence-level corpus and records the resulting IDs and hashes in the new benchmark artifacts. Expected task IDs and corpus hashes are:

| HotpotQA ID | Task ID | Corpus SHA-256 |
|---|---|---|
| `5a886886554299206df2b258` | `hotpotqa-0001` | `ce645aa1817776e71317b97215054b8d223d95c8c1e463d44c086c33c460b2ff` |
| `5a78eea555429970f5fffddf` | `hotpotqa-0002` | `a2a2a987e98e1e6eee9c78a2d2d1fec29b52333ac2b6835038394d71708f4f6c` |
| `5a8463d05542990548d0b2a5` | `hotpotqa-0003` | `f1df16f451f9ac7be4966fb20b9cf31a4984aa14ee4a42601c9ea83e2aac484f` |

The answer and supporting-fact fields are present only so the post-run scorer can evaluate each result. The harness passes each question and generated corpus evidence to the agents, then scores only after the run closes; the existing `test_hotpotqa_gold_answer_stays_scorer_only_and_artifacts_are_separate` regression checks this boundary. New runs may differ from D95 because model service behavior can change; this package makes the inputs reproducible, not remote outputs.
