"""Evaluate the synchronized 1,275-run benchmark with strict run-ID alignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from benchmark_smoke.benchmark_evaluation import align_predictions
from benchmark_smoke.evaluate_local_backends import evaluate
from benchmark_smoke.evaluate_noisy_fields import paired_statistics
from modules import deterministic_source, input as metadata_input
from render_benchmark_matrices import LABELS, render_svg

DEFAULT_WORK = ROOT / "benchmark_smoke/current_benchmark"


def publish_tables(work: Path, summaries: list[dict], pairs: list[dict], statuses: list[dict]) -> None:
    """Refresh the two marked README sections using only completed measurements."""
    scores = {(row["input"], row["method"], row["filter"]): row for row in summaries}
    state = {(row["input"], row["method"], row["filter"]): row for row in statuses}
    lines = [
        "| Input | Method | Filter | Accuracy | Assigned precision | Unknown | Taxonomy assignments |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for mode in ("tsv", "biosample"):
        for method in ("nli", "llm", "mistral"):
            for condition in ("filtered", "unfiltered"):
                row = scores.get((mode, method, condition))
                label = f"| {'ATB TSV' if mode == 'tsv' else 'BioSample'} | " \
                        f"{'Ministral' if method == 'llm' else 'Mistral API' if method == 'mistral' else 'NLI'} | " \
                        f"{'On' if condition == 'filtered' else 'Off'} |"
                if row is None:
                    reason = state.get((mode, method, condition), {}).get("reason", "not run")
                    lines.append(f"{label} Incomplete ({reason}) | — | — | — |")
                else:
                    lines.append(
                        f"{label} {row['overall_accuracy']:.1%} | {row['precision_assigned']:.1%} | "
                        f"{row['unknown_predictions']} | {row['taxonomy_calls']} |"
                    )
    table = "\n".join(lines)
    paired_lines = [
        "| Input | Method | Δ accuracy | Δ assigned precision | Wrong → correct | Correct → wrong | Median model-text reduction | Δ runtime |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        if pair["scope"] != "all_records":
            continue
        paired_lines.append(
            f"| {'ATB TSV' if pair['input'] == 'tsv' else 'BioSample'} | "
            f"{'Ministral' if pair['method'] == 'llm' else 'Mistral API' if pair['method'] == 'mistral' else 'NLI'} | "
            f"{pair['accuracy_difference']:+.1%} | {pair['assigned_precision_difference_full']:+.1%} | "
            f"{pair['wrong_to_correct']} | {pair['correct_to_wrong']} | "
            f"{-pair['model_median_char_difference']:.0f} chars | "
            f"{pair['total_seconds_difference']:+.1f} s |"
        )

    def replace_marked(path: Path, marker: str, content: str) -> None:
        opening = f"<!-- BEGIN {marker} -->"
        closing = f"<!-- END {marker} -->"
        text = path.read_text()
        if opening not in text:
            legacy = f"<!-- {marker} -->"
            if text.count(legacy) != 1:
                raise ValueError(f"Missing unique README marker: {path}: {marker}")
            text = text.replace(legacy, opening + "\n" + closing)
        before, rest = text.split(opening, 1)
        _, after = rest.split(closing, 1)
        path.write_text(before + opening + "\n\n" + content + "\n\n" + closing + after)

    replace_marked(ROOT / "README.md", "CURRENT_BENCHMARK_TABLE", table)
    replace_marked(work / "README.md", "CURRENT_REPORT_TABLE", table)
    comparison = "\n".join(paired_lines)
    if len(paired_lines) == 2:
        comparison += "\n\nPaired comparisons are pending completion of both filtering conditions."
    api_rows = [row for row in summaries if row["method"] == "mistral"]
    if api_rows:
        calls = sum(row["api_requests"] for row in api_rows)
        retries = sum(row["api_retries"] for row in api_rows)
        estimates = [row["api_estimated_usd_at_small4_standard_rate"] for row in api_rows]
        comparison += f"\n\nThe completed API runs made {calls:,} HTTP requests, including {retries} retries."
        if all(value is not None for value in estimates):
            comparison += (f" Reported token usage gives an estimated US${sum(estimates):.2f} "
                           "at the published Small 4 standard rates; this is not a billing statement.")
    replace_marked(work / "README.md", "CURRENT_REPORT_ANALYSIS", comparison)


def logged_seconds(log: str, marker: str) -> float | None:
    found = re.search(re.escape(marker) + r": ([0-9.]+)", log)
    return float(found.group(1)) if found else None


def collect(work: Path) -> None:
    import numpy as np

    work = work.resolve()
    manifest = json.loads((work / "manifest.json").read_text())
    truth = metadata_input.read_tsv(work / "inputs/benchmark_true_labels_1275.tsv")
    mapping = metadata_input.read_tsv(work / "inputs/run_to_biosample.tsv").to_dict("records")
    if len(truth) != 1275 or len(mapping) != 1275:
        raise ValueError("Current evaluation requires exactly 1,275 truth and mapping rows")
    if set(truth.run_accession) != {row["run_accession"] for row in mapping}:
        raise ValueError("Truth and accession mapping do not cover the same runs")

    input_frames = {
        "tsv": metadata_input.read_tsv(work / "inputs/benchmark_atb_1275.tsv"),
        "biosample": pd.read_pickle(work / "input_cache/biosample.pkl"),
    }
    sequence = [row["run_accession"] for row in mapping]
    if input_frames["tsv"].run_accession.tolist() != sequence:
        raise ValueError("ATB input order differs from the run mapping")
    if input_frames["biosample"].accession.tolist() != [row["accession"] for row in mapping]:
        raise ValueError("BioSample input order differs from the run mapping")
    texts = {}
    for mode, frame in input_frames.items():
        id_col = "run_accession" if mode == "tsv" else "accession"
        for condition in ("unfiltered", "filtered"):
            batch = metadata_input.prepare_batch(
                frame, id_col=id_col, exclude_noisy_fields=(condition == "filtered")
            )
            texts[(mode, condition)] = dict(zip(sequence, batch.records))

    summaries, paired_summaries, pathways, examples, statuses = [], [], [], [], []
    evaluation_dir = work / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    for mode in ("tsv", "biosample"):
        for method in ("nli", "llm", "mistral"):
            compared = {}
            for condition in ("unfiltered", "filtered"):
                key = f"{mode}_{method}_{condition}"
                state = manifest["runs"].get(key, {"status": "incomplete", "reason": "not run"})
                statuses.append({"input": mode, "method": method, "filter": condition,
                                 "status": state["status"], "reason": state.get("reason", "")})
                if state["status"] != "complete":
                    continue
                prediction_path = work / "results" / condition / f"{mode}_{method}.tsv"
                predictions = metadata_input.read_tsv(prediction_path)
                _, aligned = align_predictions(
                    truth.to_dict("records"), predictions.to_dict("records"), mapping
                )
                merged = truth.merge(pd.DataFrame(aligned.values()), on="run_accession",
                                     validate="one_to_one").set_index("run_accession").loc[sequence]
                if len(merged) != 1275 or merged.index.has_duplicates:
                    raise ValueError(f"Prediction coverage changed: {key}")
                compared[condition] = merged
                audit = json.loads(Path(str(prediction_path) + ".filtering.json").read_text())
                log = Path(str(prediction_path) + ".log").read_text()
                match = re.search(r"Mistral API usage: (\{[^\n]+\})", log)
                usage = json.loads(match.group(1)) if match else {}
                model_calls = int(merged.source_method.eq(method).sum())
                if method == "mistral" and usage.get("completed") != model_calls:
                    raise ValueError(f"API request accounting differs for {key}")
                out = evaluation_dir / mode / method / condition
                out.mkdir(parents=True, exist_ok=True)
                score = evaluate(truth, prediction_path, "best_hit", out, False, mapping)
                score.update({
                    "input": mode, "method": method, "filter": condition,
                    "result_file": str(prediction_path.relative_to(ROOT)),
                    "source_model_calls": model_calls,
                    "nli_verification_calls": int(merged.nli_verification_score.ne("NA").sum()),
                    "total_seconds": state["elapsed_seconds"],
                    "source_model_seconds": logged_seconds(
                        log, "Source model stage seconds (including weight loading)"),
                    "nli_verification_seconds": logged_seconds(
                        log, "NLI verification stage seconds (including any weight loading)"),
                    "observed_device_peak_used_mib": state.get("observed_device_peak_used_mib"),
                    "all_median_chars": audit["all_records"]["after_length_median"],
                    "all_mean_chars": audit["all_records"]["after_length_mean"],
                    "all_truncated": audit["all_records"]["after_truncated"],
                    "model_median_chars": audit["source_model_records"]["after_length_median"],
                    "model_mean_chars": audit["source_model_records"]["after_length_mean"],
                    "model_truncated": audit["source_model_records"]["after_truncated"],
                    "model_token_median": audit["nli_token_exposure_source_model"].get("premise_token_median"),
                    "model_token_mean": audit["nli_token_exposure_source_model"].get("premise_token_mean"),
                    "model_token_max": audit["nli_token_exposure_source_model"].get("premise_token_max"),
                    "model_nli_limit_exposed": audit["nli_token_exposure_source_model"].get(
                        "records_any_candidate_exceeds_limit"),
                    "administrative_technical_omissions": audit["all_records"]["administrative_technical"],
                    "redundant_original_omissions": audit["all_records"]["redundant_original"],
                    "api_requests": usage.get("requests"),
                    "api_retries": usage.get("retries"),
                    "api_prompt_tokens": usage.get("prompt_tokens"),
                    "api_completion_tokens": usage.get("completion_tokens"),
                    "api_total_tokens": usage.get("total_tokens"),
                    "api_errors": 0 if method == "mistral" else None,
                    "api_estimated_usd_at_small4_standard_rate": (
                        (usage["prompt_tokens"] * 0.15 + usage["completion_tokens"] * 0.60) / 1_000_000
                        if method == "mistral" and usage.get("usage_available") else None
                    ),
                })
                summaries.append(score)
                matrix = pd.crosstab(
                    merged.final_species, merged.best_hit
                ).reindex(index=LABELS, columns=LABELS, fill_value=0)
                per_class_path = out / f"{prediction_path.stem}_per_class.tsv"
                per_class = metadata_input.read_tsv(per_class_path).set_index("source").reindex(LABELS)
                per_class["one_vs_rest_accuracy"] = [
                    (1275 - matrix.loc[label].sum() - matrix[label].sum()
                     + 2 * matrix.loc[label, label]) / 1275 for label in LABELS
                ]
                per_class.to_csv(per_class_path, sep="\t", index_label="source", na_rep="NA")
                figure_dir = work / "figures"
                figure_dir.mkdir(exist_ok=True)
                totals = matrix.sum(axis=1).to_dict()
                cells = {label: matrix.loc[label].to_dict() for label in LABELS}
                for percentages, suffix in ((False, "counts"), (True, "percent")):
                    render_svg(
                        figure_dir / f"{mode}_{method}_{condition}_{suffix}.svg",
                        f"{mode.upper()} {method.upper()} · {condition} (n=1275)",
                        totals, cells, percentages,
                    )
                for route, group in merged.groupby("source_method"):
                    assigned = group.best_hit.ne("unknown")
                    correct = group.best_hit.eq(group.final_species)
                    pathways.append({
                        "input": mode, "method": method, "filter": condition,
                        "pathway": route, "records": len(group),
                        "accuracy": float(correct.mean()),
                        "assigned_precision": float(correct[assigned].mean()) if assigned.any() else None,
                    })
            if len(compared) != 2:
                continue
            before, after = compared["unfiltered"], compared["filtered"].loc[sequence]
            if before.columns.tolist() != after.columns.tolist():
                raise ValueError(f"Output schema changed between filter conditions: {mode}/{method}")
            deterministic = before.source_method.isin(deterministic_source.DETERMINISTIC_METHODS)
            if not deterministic.equals(after.source_method.isin(deterministic_source.DETERMINISTIC_METHODS)):
                raise ValueError("Deterministic coverage changed")
            for column in ("best_hit", "source_method", "source_evidence", "source_llm_score"):
                if not before.loc[deterministic, column].equals(after.loc[deterministic, column]):
                    raise ValueError(f"Deterministic {column} changed for {mode}/{method}")
            for column in ("year", "country", *input_frames[mode].columns):
                if column in before and not before[column].equals(after[column]):
                    raise ValueError(f"Raw metadata or extracted {column} changed for {mode}/{method}")
            paired = pd.DataFrame({
                "run_accession": sequence,
                "accession": [row["accession"] for row in mapping],
                "true_source": before.final_species.to_numpy(),
                "unfiltered_best_hit": before.best_hit.to_numpy(),
                "filtered_best_hit": after.best_hit.to_numpy(),
                "unfiltered_source_method": before.source_method.to_numpy(),
                "filtered_source_method": after.source_method.to_numpy(),
                "unfiltered_correct": before.best_hit.eq(before.final_species).to_numpy(),
                "filtered_correct": after.best_hit.eq(after.final_species).to_numpy(),
                "unfiltered_llm_score": before.source_llm_score.to_numpy(),
                "filtered_llm_score": after.source_llm_score.to_numpy(),
                "unfiltered_verification": before.nli_verification_score.to_numpy(),
                "filtered_verification": after.nli_verification_score.to_numpy(),
                "model_classified": (~deterministic).to_numpy(),
                "changed_prediction": before.best_hit.ne(after.best_hit).to_numpy(),
            })
            paired.to_csv(evaluation_dir / f"paired_{mode}_{method}.tsv", sep="\t", index=False)
            paired[paired.unfiltered_correct & ~paired.filtered_correct].to_csv(
                evaluation_dir / f"new_errors_{mode}_{method}.tsv", sep="\t", index=False
            )
            before_score = next(row for row in summaries if row["input"] == mode and
                                row["method"] == method and row["filter"] == "unfiltered")
            after_score = next(row for row in summaries if row["input"] == mode and
                               row["method"] == method and row["filter"] == "filtered")
            for scope, group in (("all_records", paired),
                                 ("source_model_records", paired[paired.model_classified])):
                stats = paired_statistics(group)
                paired_summaries.append({
                    "input": mode, "method": method, "scope": scope,
                    **stats,
                    "correct_to_correct": int((group.unfiltered_correct & group.filtered_correct).sum()),
                    "wrong_to_wrong": int((~group.unfiltered_correct & ~group.filtered_correct).sum()),
                    "assigned_precision_difference_full": (
                        after_score["precision_assigned"] - before_score["precision_assigned"]),
                    "model_median_char_difference": (
                        after_score["model_median_chars"] - before_score["model_median_chars"]),
                    "total_seconds_difference": (
                        after_score["total_seconds"] - before_score["total_seconds"]),
                })
            cases = {
                "wrong_to_correct": ~paired.unfiltered_correct & paired.filtered_correct,
                "correct_to_wrong": paired.unfiltered_correct & ~paired.filtered_correct,
                "unknown_to_assigned": paired.unfiltered_best_hit.eq("unknown") &
                                       paired.filtered_best_hit.ne("unknown"),
                "assigned_to_unknown": paired.unfiltered_best_hit.ne("unknown") &
                                       paired.filtered_best_hit.eq("unknown"),
            }
            for case, mask in cases.items():
                for row in paired[mask].head(2).to_dict("records"):
                    run = row["run_accession"]
                    examples.append({
                        "input": mode, "method": method, "case": case, **row,
                        "unfiltered_model_text": texts[(mode, "unfiltered")][run],
                        "filtered_model_text": texts[(mode, "filtered")][run],
                    })
    pd.DataFrame(summaries).drop(columns=["source_methods"], errors="ignore").to_csv(
        evaluation_dir / "summary.tsv", sep="\t", index=False, na_rep="NA")
    (evaluation_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    for name, values in [
        ("paired_summary.tsv", paired_summaries),
        ("pathway_metrics.tsv", pathways),
        ("outcomes.tsv", statuses),
        ("error_examples.tsv", examples),
    ]:
        pd.DataFrame(values).to_csv(evaluation_dir / name, sep="\t", index=False, na_rep="NA")
    publish_tables(work, summaries, paired_summaries, statuses)
    print(pd.DataFrame(statuses).to_string(index=False))
    if summaries:
        print(pd.DataFrame(summaries)[["input", "method", "filter", "overall_accuracy",
                                       "precision_assigned"]].to_string(index=False))
    print("Completed pairs passed run alignment, taxonomy, metadata, date/country and schema checks.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    collect(args.work_dir)


if __name__ == "__main__":
    main()
