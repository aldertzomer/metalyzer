# Local LLM backend validation

This is the historical Qwen validation report. The default `--method llm`
backend is now Ministral 3; its supplied test-server results are in the main
README and `benchmark_smoke/local_ministral_evaluation/`.

This report records validation of the implementation in `metalyzer.py`,
including the completed default-model GPU benchmark.

## Environment

- Windows, Intel Core i5-14500, approximately 15.7 GiB physical RAM.
- PyCharm interpreter: `.venv/Scripts/python.exe`, Python 3.14.7.
- Installed packages: PyTorch 2.14.0, Transformers 5.17.0, pandas 3.0.5.
- CUDA unavailable, zero CUDA devices; `nvidia-smi` unavailable.
- No local `taxonomy/` dump. Mixed and taxonomy-only routes were tested with fixtures/mocks.
- No packages added. Both Conda environment specifications are unchanged.
- Before the LLM check, only about 2.2 GiB physical and 5.3 GiB virtual memory
  were available, insufficient headroom for the default 4B model's roughly
  8 GB of weights plus runtime overhead.

## Unit tests

Command:

```text
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Final test output:

```text
----------------------------------------------------------------------
Ran 30 tests in 2.035s

OK
```

All 11 existing taxonomy tests and 19 new LLM tests passed. The new tests mock
the tokenizer/model module boundary and use only tiny real tensors. No model
download occurs during unit tests. Existing assertions were not weakened.

Coverage includes default NLI selection; unchanged NLI pipeline/scoring;
exclusive/lazy loading; empty and taxonomy-only inputs; mixed row ordering;
NA score columns; quoted, JSON, normalized and unknown labels; rejection of
prose and ambiguous normalized labels; sanitized evidence; ignored LLM
threshold; CLI validation; left padding, prompt slicing and inference mode;
device/revision/dtype handling; natural metadata with NA omission; and
deterministic year/country output.

AST comparisons against `8b8d4b6:metalyzer.py` confirmed these functions are
unchanged: `build_record`, `is_empty_like`, `yy_to_yyyy`, `year_from_value`,
`extract_year_from_row`, `_ascii_fold`, `normalize_country`, and
`extract_country_from_row`. `taxonomy.py`, `metalyzer_mistral.py`, both source
files, both environment files and `tests/test_taxonomy.py` were not changed.

## Real NLI smoke test

Ran on CPU with the existing cached DeBERTa weights, offline:

```bash
python metalyzer.py --metadata benchmark.tsv --sources sources.tsv \
  --out benchmark_smoke/nli_backend_20.tsv --id-col run_accession \
  --method nli --device -1 --batch-size 2 --min-score 0.2 --limit 20
```

The validation process set `HF_HUB_CACHE` to the existing
`benchmark_smoke/hf_cache/hub` and `HF_HUB_OFFLINE=1`; production defaults are
unchanged. Batch size 2 accommodated this host's limited memory.

- Exactly 20 rows; all `source_method=nli`.
- All 16 candidate score columns are numeric; no LLM loaded or downloaded.
- All 20 labels and year/country values match the saved benchmark's first 20 rows.
- Zero unknown predictions.
- The external process monitor observed the Windows venv launcher rather than
  the inference child, so its peak-memory reading is not used.

## Full default-model GPU validation

The requested full GPU benchmark completed with the default
`Qwen/Qwen3-4B-Instruct-2507` model and local taxonomy:

```bash
python metalyzer.py --metadata benchmark.tsv --sources sources.tsv \
  --out benchmark_llm.tsv --id-col run_accession --taxonomy-dir taxonomy \
  --method llm --llm-model Qwen/Qwen3-4B-Instruct-2507 \
  --device 0 --llm-batch-size 10
```

`benchmark_llm.tsv` contains all 1,320 records and has a one-to-one accession
match with `benchmark_true_labels.tsv`. Evaluation with
`benchmark_smoke/evaluate_local_backends.py` produced:

- 1,223 correct calls: **92.7% overall accuracy**.
- 255 `host_tax_id` calls and 1,065 LLM calls.
- 104 `unknown` predictions and 10 invalid LLM outputs.
- 1,216 non-unknown assignments (92.1% coverage), with 93.3% precision among
  assigned records.
- All 16 source-score columns are `NA` for LLM and taxonomy rows, all output
  labels are canonical labels or `unknown`, and year/country values are present.

The invalid outputs are traceable rather than silently mapped: three
`environmental`, two `duck`, two `environmental water`, and one each for
`duck: duck (poultry host Anas spp. including duck meat or poultry`,
`pheasant: chicken (poultry host Gallus gallus including chicken meat or`, and
`food`. The full per-source metrics and count/row-percentage confusion matrices
are saved under `benchmark_smoke/local_llm_evaluation/` and presented in the
README.

This resolves the pending default-model functional and full-benchmark hardware
validation. Peak GPU memory was not recorded by the supplied run, so this report
does not claim a measured GPU-memory requirement or guaranteed 12-GB fit.

## Smaller CPU LLM smoke test

The requested default-model command with `--device 0` exited before loading
weights with:

```text
RuntimeError: CUDA was requested but is unavailable; use --device -1 for CPU.
```

The explicitly selected smaller CPU model then completed:

```bash
python benchmark_smoke/measure_local_backend.py \
  --metadata benchmark.tsv --sources sources.tsv \
  --out benchmark_smoke/llm_1_7b_10.tsv --id-col run_accession \
  --method llm --llm-model Qwen/Qwen3-1.7B \
  --device -1 --llm-batch-size 1 --limit 10
```

This wrapper calls the normal `metalyzer.main()` and reads native Windows
process memory counters at exit. It does not alter inference. Transformers
downloaded weights/tokenizer into the standard Hugging Face cache without an
API key. `enable_thinking=False` was passed through the chat template.

```text
Source classification:
  taxonomy host_tax_id: 0
  LLM:                  10
  LLM -> unknown:        1
  invalid LLM outputs:  1
Wrote benchmark_smoke/llm_1_7b_10.tsv (n=10)
{"elapsed_seconds": 411.6763482999995, "peak_working_set_bytes": 4619579392, "peak_commit_bytes": 5260189696}
```

- Exactly 10 rows, in original accession order; all `source_method=llm`.
- All 16 score columns are literal `NA`; no implicit `unknown` score column.
- All predictions are canonical labels or `unknown`; year/country populated.
- 9/10 correct on this **cat-only smoke subset**, not a full benchmark accuracy.
- Invalid output: accession `SRR6627731`, response `kitten`; stored as
  `best_hit=unknown`, `source_evidence=invalid_llm_output=kitten`.
- Peak working set: **4.30 GiB**. Peak commit: **4.90 GiB**.
- Duration: **411.7 seconds**, including first-use download/loading.
- These memory measurements apply only to Qwen3-1.7B on this host; they do not
  establish 12-GB compatibility for the default Qwen3-4B-Instruct-2507.
- No prompt tuning was performed after observing the output.

## Benchmark comparison

Accession joins were checked for uniqueness and coverage. The saved DeBERTa
and Mistral predictions were re-evaluated, not regenerated.

| Saved prediction file | Rows | Correct | Accuracy | Unknown | Taxonomy calls | LLM calls |
|---|---:|---:|---:|---:|---:|---:|
| benchmark_classified.tsv | 1320 | 1144 | 86.7% | 136 | 255 | 0 |
| benchmark_mistral.tsv | 1320 | 1267 | 96.0% | 69 | Not recorded | API backend |
| benchmark_llm.tsv (Qwen3-4B GPU) | 1320 | 1223 | 92.7% | 104 | 255 | 1065 |
| llm_1_7b_10.tsv (CPU smoke subset only) | 10 | 9 | 90.0% | 1 | 0 | 10 |

Per-class recall/precision, counts, and row-percentage confusion matrices are
saved in `benchmark_smoke/local_llm_evaluation/`. The 10-row CPU result is not
directly comparable to the full 1320-row results.

## Files changed or created in this task

Implementation and documentation:

- `metalyzer.py`
- `README.md`
- `tests/test_llm.py`

Validation artifacts (existing unrelated `benchmark_smoke/` files retained;
the two raw NLI log files below remain local and are not included in the commit):

- `benchmark_smoke/measure_local_backend.py`
- `benchmark_smoke/evaluate_local_backends.py`
- `benchmark_smoke/LOCAL_LLM_VALIDATION.md` (this report)
- `benchmark_smoke/nli_backend_20.tsv`
- `benchmark_smoke/nli_backend_20.log`
- `benchmark_smoke/nli_backend_20.stderr.log`
- `benchmark_smoke/llm_1_7b_10.tsv`
- `benchmark_llm.tsv`
- `benchmark_smoke/local_llm_evaluation/summary.json`
- `benchmark_smoke/local_llm_evaluation/benchmark_classified_per_class.tsv`
- `benchmark_smoke/local_llm_evaluation/benchmark_classified_confusion_counts.tsv`
- `benchmark_smoke/local_llm_evaluation/benchmark_classified_confusion_percent.tsv`
- `benchmark_smoke/local_llm_evaluation/benchmark_mistral_per_class.tsv`
- `benchmark_smoke/local_llm_evaluation/benchmark_mistral_confusion_counts.tsv`
- `benchmark_smoke/local_llm_evaluation/benchmark_mistral_confusion_percent.tsv`
- `benchmark_smoke/local_llm_evaluation/llm_1_7b_10_per_class.tsv`
- `benchmark_smoke/local_llm_evaluation/llm_1_7b_10_confusion_counts.tsv`
- `benchmark_smoke/local_llm_evaluation/llm_1_7b_10_confusion_percent.tsv`
- `benchmark_smoke/local_llm_evaluation/benchmark_llm_per_class.tsv`
- `benchmark_smoke/local_llm_evaluation/benchmark_llm_confusion_counts.tsv`
- `benchmark_smoke/local_llm_evaluation/benchmark_llm_confusion_percent.tsv`

The normal Hugging Face cache also now contains Qwen3-1.7B outside the repository.
