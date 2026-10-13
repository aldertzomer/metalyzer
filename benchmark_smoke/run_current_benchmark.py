"""Authoritative 1,275-run benchmark: prepare, local GPU, then approved API.

Examples:
  python benchmark_smoke/run_current_benchmark.py --phase prepare
  python benchmark_smoke/run_current_benchmark.py --phase local
  python benchmark_smoke/run_current_benchmark.py --phase api --confirm-paid-api

Earlier benchmark scripts and outputs remain historical snapshots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_smoke.current_dataset import build_dataset, digest
from modules import biosample, deterministic_source, input as metadata_input, llm, mistral, nli

DEFAULT_WORK = ROOT / "benchmark_smoke/current_benchmark"
DEFAULT_TAXONOMY = Path("/home/zomer007/metalyzer/taxonomy")
CONFIG = {
    "nli_model": nli.DEFAULT_NLI_MODEL,
    "nli_min_score": 0.2,
    "nli_batch_size": 64,
    "llm_model": llm.DEFAULT_LLM_MODEL,
    "llm_batch_size": 1,
    "llm_min_score": 0.75,
    "llm_max_new_tokens": 16,
    "mistral_model": "mistral-small-latest",
    "mistral_seed": 12345,
    "mistral_concurrency": 8,
    "mistral_retries": 5,
    "mistral_timeout_seconds": 60,
    "mistral_max_tokens": 32,
    "max_value_chars": 300,
    "max_record_chars": 2000,
    "host_aliases": True,
    "other_animal_taxonomy": True,
    "nli_verification": True,
    "date_country": True,
    "filter_profile": "variant_c_v1",
    "biosample_store": str(biosample.DEFAULT_STORE),
}


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def taxonomy_hash(path: Path) -> str:
    from hashlib import sha256
    digest_state = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest_state.update(block)
    return digest_state.hexdigest()


def signature(work: Path, taxonomy_dir: Path) -> dict:
    inputs = work / "inputs"
    source_files = [
        ROOT / "metalyzer.py", ROOT / "benchmark_smoke/current_dataset.py",
        ROOT / "benchmark_smoke/run_current_benchmark.py",
        *sorted((ROOT / "modules").glob("*.py")),
    ]
    return {
        "input_sha256": {p.name: digest(p) for p in sorted(inputs.iterdir()) if p.is_file()},
        "sources_sha256": digest(ROOT / "sources.tsv"),
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_files},
        "taxonomy": {name: {**fingerprint(taxonomy_dir / name),
                             "sha256": taxonomy_hash(taxonomy_dir / name)}
                     for name in deterministic_source.FILES},
        "biosample_data": [fingerprint(p) for p in biosample.store_files(biosample.DEFAULT_STORE)],
        "biosample_index": fingerprint(biosample.index_path(biosample.DEFAULT_STORE)),
        "settings": CONFIG,
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
        "mistralai_version": importlib.metadata.version("mistralai"),
    }


def requests(work: Path) -> list[str]:
    return (work / "inputs/biosample_accessions.txt").read_text().splitlines()


def checked_cache_hash(cache: Path, manifest: dict) -> str:
    """Reject a changed selected BioSample snapshot before reusing any result."""
    actual = digest(cache)
    recorded = manifest.get("selected_biosample_cache_sha256")
    if recorded is not None and recorded != actual:
        raise ValueError("Selected BioSample input hash changed; use a new work directory")
    return actual


def prepare(work: Path, taxonomy_dir: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    for relative in ("inputs", "results/filtered", "results/unfiltered",
                     "evaluation", "figures", "input_cache"):
        (work / relative).mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(work / "inputs")
    sig = signature(work, taxonomy_dir)
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != sig:
            raise ValueError("Benchmark input, code, taxonomy, model or configuration changed; use a new work directory")
    else:
        manifest = {"signature": sig, "dataset_manifest": str(work / "inputs/dataset_manifest.json"),
                    "created": datetime.now(timezone.utc).isoformat(), "runs": {}}
    manifest["dataset_rows"] = dataset["included_runs"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    cache = work / "input_cache/biosample.pkl"
    if not cache.exists():
        print("Reading the 1,275 selected BioSample rows from local indexed Parquet", flush=True)
        biosample.load_biosamples(requests(work), biosample.DEFAULT_STORE).to_pickle(cache)
    cache_sha256 = checked_cache_hash(cache, manifest)
    import pandas as pd
    frame = pd.read_pickle(cache)
    if frame.accession.tolist() != requests(work):
        raise ValueError("Cached BioSample rows differ from the current request sequence")
    manifest["selected_biosample_cache_sha256"] = cache_sha256
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def deterministic_audit(work: Path, taxonomy_dir: Path, manifest: dict) -> dict:
    """Count exact paid calls with local taxonomy, without loading a model."""
    import pandas as pd
    taxonomy = deterministic_source.NCBITaxonomy(taxonomy_dir)
    populations = {
        "tsv": metadata_input.read_tsv(work / "inputs/benchmark_atb_1275.tsv"),
        "biosample": pd.read_pickle(work / "input_cache/biosample.pkl"),
    }
    counts = {}
    for mode, frame in populations.items():
        id_col = "run_accession" if mode == "tsv" else "accession"
        batch = metadata_input.prepare_batch(frame, id_col=id_col)
        sources = metadata_input.load_sources(ROOT / "sources.tsv", id_col)
        anchors = deterministic_source.configured_anchors(sources, taxonomy)
        result = deterministic_source.run(batch, sources, taxonomy, anchor_sources=anchors,
                                          use_host_aliases=True, other_animal=True)
        route_counts = result.table.source_method.value_counts().to_dict()
        counts[mode] = {
            "records": len(frame), "taxonomy_by_pathway": route_counts,
            "deterministic": len(result.table),
            "source_model_calls_per_condition": len(frame) - len(result.table),
        }
    counts["paid_api_calls_if_all_four_runs"] = 2 * sum(
        counts[mode]["source_model_calls_per_condition"] for mode in populations
    )
    manifest["deterministic_audit"] = counts
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (work / "api_plan.json").write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
    return counts


def options(mode: str, method: str, condition: str, work: Path,
            taxonomy_dir: Path, api_key_file: Path) -> list[str]:
    output = work / "results" / condition / f"{mode}_{method}.tsv"
    values = (["--metadata", str(work / "inputs/benchmark_atb_1275.tsv"),
               "--id-col", "run_accession"] if mode == "tsv" else
              ["--biosample-file", str(work / "inputs/biosample_accessions.txt")])
    values += [
        "--sources", str(ROOT / "sources.tsv"), "--out", str(output),
        "--method", method, "--taxonomy-dir", str(taxonomy_dir),
        "--other-animal-taxonomy", "--device", "0", "--batch-size", "64",
        "--min-score", "0.2", "--llm-batch-size", "1", "--llm-min-score", "0.75",
        "--max-value-chars", "300", "--max-record-chars", "2000",
    ]
    if condition == "unfiltered":
        values += ["--disable-noisy-field-exclusion"]
    if method == "mistral":
        values += [
            "--api-key-file", str(api_key_file), "--mistral-model",
            "mistral-small-latest", "--mistral-random-seed", "12345",
        ]
    return values


def run_child(args) -> None:
    import metalyzer
    mode, method, condition = args.child_key.split("_")
    command = options(mode, method, condition, args.work_dir, args.taxonomy_dir, args.api_key_file)
    if mode == "tsv":
        metalyzer.main(command)
        return
    import pandas as pd
    frame = pd.read_pickle(args.work_dir / "input_cache/biosample.pkl")
    if frame.accession.tolist() != requests(args.work_dir):
        raise ValueError("Cached BioSample accession order changed")

    def selected(accessions, path):
        if [str(value).strip() for value in accessions] != frame.accession.tolist():
            raise ValueError("Requested BioSamples differ from the benchmark cache")
        from modules.runlog import emit
        emit(f"Input mode: BioSample; selected local Parquet records: {len(frame)}")
        emit(f"BioSample store: {biosample.DEFAULT_STORE}")
        emit(f"BioSample accession index: {biosample.index_path(biosample.DEFAULT_STORE)}")
        return frame.copy(deep=True)

    with patch.object(biosample, "load_biosamples", side_effect=selected):
        metalyzer.main(command)


def device_used_mib() -> int | None:
    try:
        value = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=5, check=True,
        ).stdout.splitlines()[0].strip()
        return int(value)
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def run_phase(args, manifest: dict) -> None:
    import pandas as pd
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU benchmarks were not started")
    manifest["gpu"] = {"name": torch.cuda.get_device_name(0),
                       "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory}
    work = args.work_dir
    manifest_path = work / "manifest.json"
    modes = ("tsv", "biosample")
    methods = ("nli", "llm") if args.phase == "local" else ("mistral",)
    if args.phase == "api":
        if not args.confirm_paid_api:
            raise ValueError("Paid API phase requires --confirm-paid-api after user approval")
        try:
            mistral.read_api_key(args.api_key_file)
            import mistralai.client  # noqa: F401
        except (OSError, ValueError, ImportError) as exc:
            for mode in modes:
                for condition in ("unfiltered", "filtered"):
                    manifest["runs"][f"{mode}_mistral_{condition}"] = {
                        "status": "incomplete", "reason": f"API credentials or SDK unavailable: {type(exc).__name__}"
                    }
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            return
    for mode in modes:
        for method in methods:
            for condition in ("unfiltered", "filtered"):
                key = f"{mode}_{method}_{condition}"
                output = work / "results" / condition / f"{mode}_{method}.tsv"
                state = manifest["runs"].get(key, {})
                if state.get("status") == "complete":
                    if not output.exists() or digest(output) != state.get("output_sha256"):
                        raise ValueError(f"Completed run artifact changed or disappeared: {key}")
                    print(f"Skipping completed {key}", flush=True)
                    continue
                if output.exists():
                    raise FileExistsError(f"Unregistered result already exists: {output}")
                command = [sys.executable, str(Path(__file__).resolve()), "--child-key", key,
                           "--work-dir", str(work), "--taxonomy-dir", str(args.taxonomy_dir),
                           "--api-key-file", str(args.api_key_file)]
                print(f"Starting {key}", flush=True)
                started = time.monotonic()
                peak = device_used_mib()
                log_path = Path(str(output) + ".console.log")
                env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                with log_path.open("w") as handle:
                    process = subprocess.Popen(command, cwd=ROOT, env=env,
                                               stdout=handle, stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        value = device_used_mib()
                        if value is not None:
                            peak = max(peak or 0, value)
                        time.sleep(2)
                elapsed = time.monotonic() - started
                status = "complete" if process.returncode == 0 else "incomplete"
                result = {
                    "status": status, "returncode": process.returncode,
                    "elapsed_seconds": elapsed, "observed_device_peak_used_mib": peak,
                    "command": [sys.executable, str(ROOT / "metalyzer.py"),
                                *options(mode, method, condition, work, args.taxonomy_dir, args.api_key_file)],
                    "finished": datetime.now(timezone.utc).isoformat(),
                }
                if status == "complete":
                    table = pd.read_csv(output, sep="\t", dtype=str, keep_default_na=False)
                    expected = (metadata_input.read_tsv(work / "inputs/benchmark_atb_1275.tsv").run_accession.tolist()
                                if mode == "tsv" else requests(work))
                    if len(table) != 1275 or table["run_accession" if mode == "tsv" else "accession"].tolist() != expected:
                        raise ValueError(f"Completed output has wrong run/BioSample order: {key}")
                    for field in ("best_hit", "source_method", "source_evidence", "source_llm_score",
                                  "nli_verification_score", "year", "country"):
                        if field not in table:
                            raise ValueError(f"Completed output lacks {field}: {key}")
                    result["output_sha256"] = digest(output)
                manifest["runs"][key] = result
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                print(f"{key}: {status}; {elapsed:.1f}s", flush=True)
                if status != "complete":
                    raise RuntimeError(f"Run {key} failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "local", "api"), default="prepare")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--taxonomy-dir", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--api-key-file", type=Path, default=Path("/home/zomer007/mistral.key"))
    parser.add_argument("--confirm-paid-api", action="store_true",
                        help="Acknowledge user approval for fresh paid API requests")
    parser.add_argument("--child-key", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    args.taxonomy_dir = args.taxonomy_dir.resolve()
    if args.child_key:
        run_child(args)
        return
    manifest = prepare(args.work_dir, args.taxonomy_dir)
    audit = deterministic_audit(args.work_dir, args.taxonomy_dir, manifest)
    print(f"Current benchmark: 1,275 runs; API source calls if all four paid runs: "
          f"{audit['paid_api_calls_if_all_four_runs']}", flush=True)
    if args.phase != "prepare":
        run_phase(args, manifest)


if __name__ == "__main__":
    main()
