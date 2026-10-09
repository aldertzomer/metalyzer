#!/usr/bin/env python3
"""Metalyzer command-line options and explicit pipeline orchestration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modules import combine, country, date, deterministic_source, input, llm, mistral, nli, runlog, verification
from modules.contracts import MetadataBatch, SourceResult, SourceVocabulary
from modules.version import __version__


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", action="version", version=f"metalyzer {__version__}")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--metadata", help="Input metadata TSV")
    mode.add_argument("--biosample", nargs="+", help="Local BioSample accessions (requested order)")
    mode.add_argument("--biosample-file", type=Path, help="One BioSample accession per line")
    ap.add_argument("--biosample-path", type=Path, default=Path('/mnt/sdd1/data/biosample'),
                    help="Local BioSample Parquet file or dataset directory")
    ap.add_argument("--sources", help="Sources TSV (labels)")
    ap.add_argument("--out", help="Output TSV")
    ap.add_argument("--id-col", help="ID column name in metadata")
    ap.add_argument("--taxonomy-dir", help="Local NCBI nodes.dmp, names.dmp, merged.dmp directory")
    ap.add_argument("--download-taxonomy", metavar="DIR", help="Download NCBI taxonomy into DIR and exit")

    ap.add_argument("--device", type=int, default=0, help="GPU index (default: 0), or -1 for CPU")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--method", choices=("nli", "llm", "mistral"), default="nli")
    ap.add_argument("--llm-model", default=llm.DEFAULT_LLM_MODEL)
    ap.add_argument("--llm-revision", default=None, help="Optional HF model/tokenizer revision")
    ap.add_argument("--llm-batch-size", type=int, default=1)
    ap.add_argument("--llm-max-new-tokens", type=int, default=16)
    ap.add_argument("--llm-min-score", type=float, default=0.75,
                    help="Minimum local LLM generated-label score; lower scores become unknown (default: 0.75)")
    ap.add_argument("--api-key-file", help="Mistral API key file (only read for unresolved Mistral rows)")
    ap.add_argument("--mistral-model", default=mistral.DEFAULT_MISTRAL_MODEL)
    ap.add_argument("--mistral-concurrency", type=int, default=8)
    ap.add_argument("--mistral-retries", type=int, default=5)
    ap.add_argument("--mistral-timeout", type=float, default=60.0, help="Timeout per API attempt in seconds")
    ap.add_argument("--mistral-random-seed", type=int, default=12345)
    ap.add_argument("--mistral-max-tokens", type=int, default=32)
    ap.add_argument("--mistral-progress-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="Classify only the first N metadata rows")
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.2,
        help="Minimum top source score for best_hit; lower scores become unknown (default: 0.2)",
    )
    ap.add_argument("--max-value-chars", type=int, default=300)
    ap.add_argument("--max-record-chars", type=int, default=2000)

    ap.add_argument("--skip-date", action="store_true", help="Disable year extraction (omit year column)")
    ap.add_argument("--skip-country", action="store_true", help="Disable country normalization (omit country column)")
    ap.add_argument("--disable-verify-source", action="store_true",
                    help="Skip binary NLI verification; write NA verification scores")

    args = ap.parse_args(argv)
    if args.download_taxonomy:
        return args
    if not (args.metadata or args.biosample or args.biosample_file):
        ap.error("Exactly one of --metadata, --biosample, --biosample-file is required")
    if args.biosample or args.biosample_file:
        args.id_col = "accession"
    for option in ("sources", "out", "id_col"):
        if not getattr(args, option):
            ap.error(f"--{option.replace('_', '-')} is required for analysis")
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    if args.device < -1:
        ap.error("--device must be >= -1")
    for option in ("llm_batch_size", "llm_max_new_tokens", "limit", "max_value_chars", "max_record_chars"):
        value = getattr(args, option)
        if value is not None and value < 1:
            ap.error(f"--{option.replace('_', '-')} must be >= 1")
    if not 0.0 <= args.min_score <= 1.0:
        ap.error("--min-score must be between 0 and 1")
    if not 0.0 <= args.llm_min_score <= 1.0:
        ap.error("--llm-min-score must be between 0 and 1")
    try:
        args.mistral_config = mistral.MistralConfig(
            api_key_file=args.api_key_file, model=args.mistral_model,
            concurrency=args.mistral_concurrency, retries=args.mistral_retries,
            timeout=args.mistral_timeout, random_seed=args.mistral_random_seed,
            max_tokens=args.mistral_max_tokens, progress_every=args.mistral_progress_every,
        )
    except ValueError as exc:
        ap.error(str(exc))
    return args


def classify_sources(batch: MetadataBatch, sources: SourceVocabulary, *,
                     method: str = "nli", taxonomy=None,
                     anchor_sources=None,
                     nli_config: nli.NLIConfig = nli.NLIConfig(),
                     llm_config: llm.LLMConfig = llm.LLMConfig(),
                     mistral_config: mistral.MistralConfig = mistral.MistralConfig(),
                     nli_model: nli.NLIModel | None = None) -> SourceResult:
    """Run deterministic source resolution, then the selected fallback stage."""
    if method not in {"nli", "llm", "mistral"}:
        raise ValueError("method must be nli, llm or mistral")
    deterministic = deterministic_source.run(batch, sources, taxonomy, anchor_sources=anchor_sources)
    pending = combine.unresolved(batch, sources, deterministic)
    if method == "llm":
        runlog.emit(f"LLM model: {llm_config.model}")
        runlog.emit(f"LLM minimum generation score: {llm_config.min_score}")
        runlog.emit("--min-score applies only to NLI classification; --llm-min-score filters local LLM calls using source_llm_score, which is not a calibrated probability of correctness.")
    elif method == "mistral":
        runlog.emit(f"Mistral model: {mistral_config.model}")
        runlog.emit("--min-score is not applicable to Mistral mode: the API does not produce calibrated candidate scores.")
    results = [deterministic]
    if len(pending):
        stage, config = {"nli": (nli.run, nli_config), "llm": (llm.run, llm_config),
                         "mistral": (mistral.run, mistral_config)}[method]
        results.append(stage(pending, sources, config, model=nli_model)
                       if method == "nli" else stage(pending, sources, config))
    result = combine.sources(batch, sources, *results)
    combine.report(result, method)
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.download_taxonomy:
        deterministic_source.download_taxonomy(args.download_taxonomy)
        print(f"Downloaded taxonomy to {args.download_taxonomy}")
        return
    command = list(sys.argv[1:] if argv is None else argv)
    with runlog.analysis_log(args.out) as log_path:
        try:
            runlog.emit(f"Metalyzer {__version__} started")
            if args.metadata:
                batch, sources = input.load(
                    args.metadata, args.sources, args.id_col, limit=args.limit,
                    max_value_chars=args.max_value_chars, max_record_chars=args.max_record_chars,
                )
            else:
                from modules.biosample import load_biosamples
                accessions = args.biosample
                if args.biosample_file:
                    accessions = [line.strip() for line in args.biosample_file.read_text().splitlines() if line.strip()]
                metadata = load_biosamples(accessions, args.biosample_path)
                if args.limit is not None:
                    metadata = metadata.iloc[:args.limit]
                sources = input.load_sources(args.sources, 'accession')
                batch = input.prepare_batch(metadata, id_col='accession',
                                            max_value_chars=args.max_value_chars,
                                            max_record_chars=args.max_record_chars)
            runlog.provenance(args, command, batch)
            taxonomy = deterministic_source.load_taxonomy(args.taxonomy_dir, fallback_method=args.method)
            anchors = (deterministic_source.configured_anchors(sources, taxonomy, report=True)
                       if taxonomy is not None else None)
            nli_config = nli.NLIConfig(device=args.device, batch_size=args.batch_size, min_score=args.min_score)
            nli_model = nli.NLIModel(nli_config)
            source_result = classify_sources(
                batch, sources, method=args.method, taxonomy=taxonomy, anchor_sources=anchors,
                nli_config=nli_config, nli_model=nli_model,
                llm_config=llm.LLMConfig(model=args.llm_model, revision=args.llm_revision,
                                         device=args.device, batch_size=args.llm_batch_size,
                                         max_new_tokens=args.llm_max_new_tokens,
                                         min_score=args.llm_min_score),
                mistral_config=args.mistral_config,
            )
            verification_result = verification.run(
                batch, sources, source_result, nli_config,
                disabled=args.disable_verify_source, model=nli_model,
            )
            fields = []
            if not args.skip_date:
                fields.append(date.run(batch))
            if not args.skip_country:
                fields.append(country.run(batch))
            output = combine.run(batch, sources, source_result, *fields, verification=verification_result)
            combine.write(output, args.out)
            table = source_result.table
            selected = table[table.source_method == args.method]
            invalid_prefix = {"llm": "invalid_llm_output=", "mistral": "invalid_mistral_output="}.get(args.method)
            llm_totals = ""
            if args.method == "llm":
                rejected = selected.source_evidence.str.startswith("llm_low_score=").sum()
                retained = (selected.best_hit != "unknown").sum()
                llm_totals = (f"  LLM assignments before confidence filtering: {retained + rejected}\n"
                              f"  LLM low-score calls converted to unknown: {rejected}\n"
                              f"  LLM assignments retained after confidence filtering: {retained}\n")
            runlog.emit("Run totals:\n"
                        f"  total rows: {len(output)}\n"
                        f"  deterministic taxonomy assignments: {(table.source_method == 'host_tax_id').sum()}\n"
                        f"  {args.method.upper()} assignments: {len(selected)}\n"
                        f"{llm_totals}"
                        f"  unknown calls: {(table.best_hit == 'unknown').sum()}\n"
                        f"  invalid model outputs: {selected.source_evidence.str.startswith(invalid_prefix).sum() if invalid_prefix else 0}\n"
                        f"  NLI verification scores: {verification_result.values.notna().sum()}\n"
                        f"  output path: {args.out}\n"
                        f"  log path: {log_path}")
            runlog.emit(f"Wrote {args.out} (n={len(output)})")
        except Exception:
            runlog.LOGGER.exception("Metalyzer analysis failed")
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1) from None
