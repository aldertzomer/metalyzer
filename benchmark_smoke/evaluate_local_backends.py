"""Evaluate saved predictions by accession without changing any classifier.

Example (run after complete NLI, local LLM, and Mistral API benchmarks):
  python benchmark_smoke/evaluate_local_backends.py

The three saved full-benchmark files are evaluated by default. Pass an explicit
--prediction to evaluate an additional backend under the current labels.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
try:
    from .benchmark_evaluation import align_predictions
except ImportError:
    from benchmark_evaluation import align_predictions


def evaluate(truth, path, column, directory, allow_subset, accession_mapping=None, excluded_runs=None):
    predictions = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    _, aligned = align_predictions(truth.to_dict('records'), predictions.to_dict('records'),
                                   accession_mapping, allow_subset=allow_subset)
    predictions = pd.DataFrame(aligned.values())
    merged = truth.merge(predictions, on="run_accession", validate="one_to_one")
    original_records = len(merged)
    if excluded_runs is not None:
        merged = merged[~merged.run_accession.isin(excluded_runs)].copy()
    labels = sorted(set(truth.final_species) | set(merged[column]))
    matrix = pd.crosstab(merged.final_species, merged[column]).reindex(index=labels, columns=labels, fill_value=0)
    support, called = matrix.sum(axis=1), matrix.sum(axis=0)
    correct = pd.Series([matrix.loc[label, label] for label in labels], index=labels)
    per_class = pd.DataFrame({"true_rows": support, "called_rows": called, "correct": correct,
                              "recall": correct / support.replace(0, float("nan")),
                              "precision": correct / called.replace(0, float("nan"))})
    stem = Path(path).stem
    matrix.to_csv(directory / f"{stem}_confusion_counts.tsv", sep="\t", index_label="true_source")
    (100 * matrix.div(support.replace(0, float("nan")), axis=0)).to_csv(
        directory / f"{stem}_confusion_percent.tsv", sep="\t", index_label="true_source", na_rep="NA")
    per_class.to_csv(directory / f"{stem}_per_class.tsv", sep="\t", index_label="source", na_rep="NA")
    methods = merged.source_method.value_counts().to_dict() if "source_method" in merged else None
    assigned = merged[column] != 'unknown'
    correct_assigned = int(((merged[column] == merged.final_species) & assigned).sum())
    result = {
        "file": str(path), "records": len(merged), "correct": int(correct.sum()),
        "overall_accuracy": float(correct.sum() / len(merged)) if len(merged) else None,
        "evaluation": "metadata-version-adjusted" if excluded_runs is not None else "original",
        "original_records": original_records,
        "excluded_records": original_records - len(merged),
        "unknown_predictions": int((merged[column] == "unknown").sum()),
        "assigned_predictions": int(assigned.sum()),
        "correct_assigned": correct_assigned,
        "precision_assigned": correct_assigned / int(assigned.sum()) if assigned.any() else None,
        "invalid_llm_outputs": int(merged.source_evidence.str.startswith("invalid_llm_output=").sum())
        if "source_evidence" in merged else None,
        "numeric_taxonomy_calls": methods.get("host_tax_id", 0) if methods is not None else None,
        "host_name_calls": methods.get("host_name", 0) if methods is not None else None,
        "taxonomy_calls": sum(methods.get(route, 0) for route in ("host_tax_id", "host_name")) if methods is not None else None,
        "llm_calls": methods.get("llm", 0) if methods is not None else None,
        "source_methods": methods,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", default="benchmark_true_labels.tsv")
    parser.add_argument("--prediction", nargs=2, action="append", default=[], metavar=("FILE", "COLUMN"))
    parser.add_argument("--out-dir", default="benchmark_smoke/local_llm_evaluation")
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--only-predictions", action="store_true", help="Evaluate only explicit --prediction files")
    parser.add_argument("--accession-mapping", type=Path, help="Run-to-BioSample mapping TSV")
    parser.add_argument("--conflicts", type=Path, help="Validate and separately exclude the 44 documented CAGED pairs")
    parser.add_argument("--benchmark-metadata", type=Path, default=Path('benchmark.tsv'))
    args = parser.parse_args()
    directory = Path(args.out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(args.truth, sep="\t", dtype=str, keep_default_na=False)
    sources = ([] if args.only_predictions else [("benchmark_nli.tsv", "best_hit"), ("benchmark_llm.tsv", "best_hit"),
               ("benchmark_mistral_api.tsv", "best_hit")]) + args.prediction
    mapping = (pd.read_csv(args.accession_mapping, sep='\t', dtype=str, keep_default_na=False).to_dict('records')
               if args.accession_mapping else None)
    if args.conflicts:
        try:
            from .metadata_version_evaluation import validated_conflict_runs
        except ImportError:
            from metadata_version_evaluation import validated_conflict_runs
        metadata = pd.read_csv(args.benchmark_metadata, sep='\t', dtype=str, keep_default_na=False).to_dict('records')
        mapping = mapping or [dict(run_accession=r['run_accession'], accession=r['sample_accession']) for r in metadata]
        conflicts = pd.read_csv(args.conflicts, sep='\t', dtype=str, keep_default_na=False).to_dict('records')
        excluded = validated_conflict_runs(truth.to_dict('records'), metadata, mapping, conflicts)
        results = []
        for label, exclusions in [('original', None), ('metadata_version_adjusted', excluded)]:
            destination = directory / label
            destination.mkdir(exist_ok=True)
            group = [evaluate(truth, Path(path), col, destination, args.allow_subset, mapping, exclusions)
                     for path, col in sources]
            (destination / 'summary.json').write_text(json.dumps(group, indent=2) + '\n')
            pd.DataFrame(group).drop(columns='source_methods').to_csv(destination / 'summary.tsv', sep='\t', index=False)
            results.extend(group)
    else:
        results = [evaluate(truth, Path(path), col, directory, args.allow_subset, mapping) for path, col in sources]
    report = json.dumps(results, indent=2)
    (directory / "summary.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
