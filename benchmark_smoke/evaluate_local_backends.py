"""Evaluate saved predictions by accession without changing any classifier.

Example (run after a complete LLM benchmark):
  python benchmark_smoke/evaluate_local_backends.py --prediction benchmark_llm.tsv best_hit

Existing DeBERTa and Mistral benchmark files are always evaluated as baselines.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def evaluate(truth, path, column, directory, allow_subset):
    predictions = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if predictions.run_accession.duplicated().any() or truth.run_accession.duplicated().any():
        raise ValueError("Duplicate accession in truth or predictions")
    predicted_ids, true_ids = set(predictions.run_accession), set(truth.run_accession)
    if not predicted_ids <= true_ids or (not allow_subset and predicted_ids != true_ids):
        raise ValueError(f"Accession mismatch: {path}; use --allow-subset only for smoke tests")
    merged = truth.merge(predictions, on="run_accession", validate="one_to_one")
    labels = sorted(set(truth.final_species) | set(predictions[column]))
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
    methods = predictions.source_method.value_counts().to_dict() if "source_method" in predictions else None
    result = {
        "file": str(path), "records": len(merged), "correct": int(correct.sum()),
        "overall_accuracy": float(correct.sum() / len(merged)) if len(merged) else None,
        "unknown_predictions": int((predictions[column] == "unknown").sum()),
        "invalid_llm_outputs": int(predictions.source_evidence.str.startswith("invalid_llm_output=").sum())
        if "source_evidence" in predictions else None,
        "taxonomy_calls": methods.get("host_tax_id", 0) if methods is not None else None,
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
    args = parser.parse_args()
    directory = Path(args.out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(args.truth, sep="\t", dtype=str, keep_default_na=False)
    sources = [("benchmark_classified.tsv", "best_hit"), ("benchmark_mistral.tsv", "source"), *args.prediction]
    results = [evaluate(truth, Path(path), col, directory, args.allow_subset) for path, col in sources]
    report = json.dumps(results, indent=2)
    (directory / "summary.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
