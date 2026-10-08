# Local LLM backend validation

This is an archived Qwen validation report from the earlier source list with
separate waterbird and wildbird categories. The default `--method llm` backend
is now Ministral 3. The old full-benchmark output and its derived matrices have
been replaced; current NLI and local Ministral results are in the main README.

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
`extract_country_from_row`. `taxonomy.py`, `sources.tsv`, both environment files
and `tests/test_taxonomy.py` were not changed.

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
  --out historical_qwen_llm.tsv --id-col run_accession --taxonomy-dir taxonomy \
  --method llm --llm-model Qwen/Qwen3-4B-Instruct-2507 \
  --device 0 --llm-batch-size 10
```

The historical output contained all 1,320 records. At the time, accession
matching and evaluation produced:

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
`food`. The historical full-benchmark metrics and matrices have been retired
because they use the previous taxonomy and source labels. The command above
shows the original model settings; reproducing these numbers would require the
earlier source list and true labels.

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

The historical 10-row Qwen CPU smoke result remains in
`benchmark_smoke/llm_1_7b_10.tsv`. It is a cat-only subset under the older
labels, so it is not directly comparable to the current full benchmark.

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
- `benchmark_smoke/local_llm_evaluation/llm_1_7b_10_per_class.tsv`
- `benchmark_smoke/local_llm_evaluation/llm_1_7b_10_confusion_counts.tsv`
- `benchmark_smoke/local_llm_evaluation/llm_1_7b_10_confusion_percent.tsv`

The current `benchmark_llm.tsv`, evaluation summary, and NLI/LLM/API matrices
replace the historical full-benchmark files named here in the original report.

The normal Hugging Face cache also now contains Qwen3-1.7B outside the repository.
