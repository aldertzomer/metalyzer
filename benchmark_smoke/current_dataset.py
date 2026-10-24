"""Derive the current benchmark population from adjudicated run IDs.

The historical ATB TSV and accession request file are never edited.  Reference
labels are read only; predictions are never used to make or alter labels.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pycountry

from modules.input import read_tsv

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = 1275
DUCK_CONFLICT = "ERR2197811"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _runs(frame: pd.DataFrame, label: str) -> list[str]:
    if "run_accession" not in frame:
        raise ValueError(f"{label} lacks run_accession")
    runs = frame.run_accession.astype(str).tolist()
    if any(not run.strip() for run in runs) or len(runs) != len(set(runs)):
        raise ValueError(f"{label} has blank or duplicate run_accession values")
    return runs


def _write_derived(frame: pd.DataFrame, path: Path) -> None:
    value = frame.to_csv(sep="\t", index=False)
    if path.exists() and path.read_text() != value:
        raise ValueError(f"Derived benchmark input changed: {path}; use a new output directory")
    path.write_text(value)


def build_dataset(
    destination: Path,
    *,
    metadata_path: Path = ROOT / "benchmark.tsv",
    accessions_path: Path = ROOT / "benchmark_accessions.tsv",
    truth_path: Path = ROOT / "benchmark_true_labels.tsv",
    conflicts_path: Path = ROOT / "benchmark_smoke/biosample_diagnostics/goat_host_conflicts.tsv",
    expected_rows: int = EXPECTED_ROWS,
    expected_excluded: set[str] | None = None,
) -> dict:
    """Validate and write synchronized, ordered inputs plus provenance."""
    metadata_path, accessions_path, truth_path, conflicts_path = map(
        Path, (metadata_path, accessions_path, truth_path, conflicts_path)
    )
    metadata = read_tsv(metadata_path)
    requests = read_tsv(accessions_path)
    truth = read_tsv(truth_path)
    if "final_country" not in truth.columns:
        raise ValueError("Reference labels lack final_country")
    invalid_countries = sorted({
        country for country in truth.final_country
        if country != "unknown"
        and (len(country) != 2 or not country.isupper()
             or pycountry.countries.get(alpha_2=country) is None)
    })
    if invalid_countries:
        raise ValueError(
            "Reference final_country must use ISO 3166-1 alpha-2 or 'unknown': "
            + ", ".join(invalid_countries)
        )
    metadata_runs = _runs(metadata, "ATB metadata")
    request_runs = _runs(requests, "accession requests")
    truth_runs = _runs(truth, "reference labels")
    if len(truth_runs) != expected_rows:
        raise ValueError(f"Expected {expected_rows} reference runs; found {len(truth_runs)}")
    if "sample_accession" not in metadata:
        raise ValueError("ATB metadata lacks sample_accession")
    if set(request_runs) != set(metadata_runs):
        raise ValueError("ATB metadata and accession request run sets differ")
    missing = set(truth_runs) - set(metadata_runs)
    if missing:
        raise ValueError(f"Reference runs missing from ATB metadata: {sorted(missing)[:10]}")

    excluded = set(metadata_runs) - set(truth_runs)
    if expected_excluded is None:
        conflicts = read_tsv(conflicts_path)
        caged = set(_runs(conflicts, "CAGED conflicts"))
        if len(caged) != 44 or any(conflicts.final_species != "goat"):
            raise ValueError("Expected the 44 documented historical goat conflicts")
        expected_excluded = caged | {DUCK_CONFLICT}
        if len(expected_excluded) != 45:
            raise ValueError("Expected 45 distinct disputed runs")
        indexed = metadata.set_index("run_accession")
        for row in conflicts.itertuples():
            if indexed.loc[row.run_accession, "sample_accession"] != row.accession:
                raise ValueError(f"CAGED run/BioSample mapping changed: {row.run_accession}")
    if excluded != expected_excluded:
        extra = sorted(excluded - expected_excluded)
        retained_disputes = sorted(expected_excluded - excluded)
        raise ValueError(f"Excluded run set differs: extra={extra[:10]}, retained_disputes={retained_disputes[:10]}")

    selected_metadata = metadata[metadata.run_accession.isin(truth_runs)].copy()
    selected_requests = requests[requests.run_accession.isin(truth_runs)].copy()
    if set(selected_metadata.run_accession) != set(truth_runs) or set(selected_requests.run_accession) != set(truth_runs):
        raise ValueError("Synchronized benchmark inputs do not cover every reference run")
    mapping = selected_metadata[["run_accession", "sample_accession"]].rename(
        columns={"sample_accession": "accession"}
    )
    if mapping.accession.eq("").any():
        raise ValueError("A retained run lacks a BioSample accession")
    if selected_requests.run_accession.tolist() != selected_metadata.run_accession.tolist():
        raise ValueError("Accession request and ATB metadata order differs")
    duplicate_accessions = {name: count for name, count in Counter(mapping.accession).items() if count > 1}
    if expected_rows == EXPECTED_ROWS and duplicate_accessions != {"SAMN08169257": 2}:
        raise ValueError(f"Unexpected repeated BioSample accession: {duplicate_accessions}")

    excluded_rows = metadata[metadata.run_accession.isin(excluded)][
        ["run_accession", "sample_accession"]
    ].rename(columns={"sample_accession": "accession"})
    excluded_rows["reason"] = excluded_rows.run_accession.map(
        lambda run: "historical duck/current chicken host conflict"
        if run == DUCK_CONFLICT else "historical CAGED goat/current chicken host conflict"
    )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "atb": destination / "benchmark_atb_1275.tsv",
        "accessions": destination / "benchmark_accessions_1275.tsv",
        "truth": destination / "benchmark_true_labels_1275.tsv",
        "mapping": destination / "run_to_biosample.tsv",
        "excluded": destination / "excluded_accessions.tsv",
    }
    for key, frame in [
        ("atb", selected_metadata), ("accessions", selected_requests),
        ("truth", truth), ("mapping", mapping), ("excluded", excluded_rows),
    ]:
        _write_derived(frame, paths[key])
    request_text = "".join(f"{accession}\n" for accession in mapping.accession)
    request_file = destination / "biosample_accessions.txt"
    if request_file.exists() and request_file.read_text() != request_text:
        raise ValueError("Derived BioSample request sequence changed")
    request_file.write_text(request_text)
    paths["biosample_requests"] = request_file
    manifest = {
        "reference": ("benchmark_true_labels.tsv, source labels adjudicated at "
                      "c01e1389b89aefaf748b6ae79a71ca3dcc08d45c; "
                      "country codes ISO 3166-1 alpha-2; 23 unsupported and 7 "
                      "conflicted country references set to unknown; ERR946747 set "
                      "to GB on 2026-10-24 (see country_audit review TSV files)"),
        "country_standard": "ISO 3166-1 alpha-2; missing: unknown",
        "historical_metadata": "AllTheBacteria ENA export, 2025-05-06",
        "source_sha256": {name: digest(path) for name, path in [
            ("benchmark.tsv", metadata_path), ("benchmark_accessions.tsv", accessions_path),
            ("benchmark_true_labels.tsv", truth_path), ("goat_host_conflicts.tsv", conflicts_path),
        ]},
        "derived_sha256": {name: digest(path) for name, path in paths.items()},
        "included_runs": len(truth_runs),
        "excluded_runs": len(excluded),
        "excluded_caged_goat_chicken": len(excluded - {DUCK_CONFLICT}),
        "excluded_duck_chicken": int(DUCK_CONFLICT in excluded),
        "duplicate_biosample_accessions": duplicate_accessions,
        "order": "Original ATB and accession-request relative order; truth joined by run_accession",
        "labels_from_predictions": False,
    }
    manifest_path = destination / "dataset_manifest.json"
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != text:
        raise ValueError("Dataset provenance changed; use a new output directory")
    manifest_path.write_text(text)
    return manifest
