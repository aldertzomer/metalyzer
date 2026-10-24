"""Recompute the country-only audit against the current 1,275-run reference."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from modules import country
from modules.input import prepare_batch, read_tsv


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs"
OUT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    truth_path = INPUTS / "benchmark_true_labels_1275.tsv"
    mapping_path = INPUTS / "run_to_biosample.tsv"
    atb_path = INPUTS / "benchmark_atb_1275.tsv"
    cache_path = ROOT / "input_cache/biosample.pkl"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Required cached BioSample input is missing: {cache_path}")

    truth = read_tsv(truth_path)
    mapping = read_tsv(mapping_path)
    atb = read_tsv(atb_path)
    biosample = pd.read_pickle(cache_path)
    if len(truth) != 1275 or len(mapping) != 1275 or len(atb) != 1275 or len(biosample) != 1275:
        raise ValueError("All country benchmark inputs must contain exactly 1,275 rows")
    if truth.run_accession.duplicated().any() or mapping.run_accession.duplicated().any():
        raise ValueError("Duplicate sequencing-run accession in reference or mapping")
    if set(truth.run_accession) != set(mapping.run_accession):
        raise ValueError("Reference and mapping run sets differ")
    if atb.run_accession.tolist() != mapping.run_accession.tolist():
        raise ValueError("ATB input and mapping run order differ")
    if biosample.accession.tolist() != mapping.accession.tolist():
        raise ValueError("BioSample cache and mapping accession order differ")

    reference = truth.set_index("run_accession").loc[
        mapping.run_accession, "final_country"
    ].to_numpy()
    scored = reference != "unknown"
    summary = {
        "reference_runs": 1275,
        "known_reference_runs": int(scored.sum()),
        "unscored_unknown_reference_runs": int((~scored).sum()),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (truth_path, mapping_path, atb_path, cache_path)
        },
        "runner_sha256": sha256(Path(__file__)),
        "results": {},
    }
    for name, frame, id_col in (
        ("atb", atb, "run_accession"),
        ("biosample", biosample, "accession"),
    ):
        predictions = country.run(prepare_batch(frame, id_col=id_col)).values.to_numpy()
        if len(predictions) != len(mapping):
            raise ValueError(f"Country prediction count differs for {name}")
        matches = predictions == reference
        result = mapping.copy()
        result["reference_country"] = reference
        result["predicted_country"] = predictions
        result["scored"] = scored
        result["correct"] = pd.Series(matches).where(scored, pd.NA).astype("boolean")
        for col in frame.columns:
            if country._field_rank(col) is not None:
                result[f"raw_{col}"] = frame[col].to_numpy()
        result.to_csv(OUT / f"{name}_countries.tsv", sep="\t", index=False)
        errors = result.loc[scored & ~matches]
        errors.to_csv(OUT / f"{name}_mismatches.tsv", sep="\t", index=False)
        correct = int((scored & matches).sum())
        summary["results"][name] = {
            "correct": correct,
            "denominator": int(scored.sum()),
            "accuracy": correct / int(scored.sum()),
            "mismatches": len(errors),
            "predicted_unknown_scored": int((scored & (predictions == "unknown")).sum()),
            "predicted_unknown_all": int((predictions == "unknown").sum()),
        }
        print(f"{name}: {correct}/{scored.sum()} correct ({correct / scored.sum():.1%}); "
              f"{len(errors)} scored mismatches")
        print(errors.groupby(["reference_country", "predicted_country"]).size()
              .sort_values(ascending=False).head(15))
        print()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
