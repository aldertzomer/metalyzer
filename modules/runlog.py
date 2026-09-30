"""Console and file logging for a single command-line analysis run."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import logging
import platform
from pathlib import Path
import subprocess
import sys
import warnings

from . import deterministic_source
from .version import __version__


LOGGER = logging.getLogger("metalyzer")


def emit(message: str) -> None:
    """Use the run logger in CLI mode and preserve console output for stage callers."""
    if LOGGER.handlers:
        LOGGER.info(message)
    else:
        print(message, flush=True)


@contextmanager
def analysis_log(output_path: str | Path):
    """Attach one console and one file handler; restore warning behavior afterward."""
    log_path = Path(f"{output_path}.log")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    screen_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    screen_handler.setFormatter(formatter)
    old_handlers, old_level, old_propagate = LOGGER.handlers[:], LOGGER.level, LOGGER.propagate
    old_showwarning = warnings.showwarning
    LOGGER.handlers = [screen_handler, file_handler]
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    def showwarning(message, category, filename, lineno, file=None, line=None):
        # The existing warning display remains intact, including catch_warnings
        # used by callers; only the file gets an additional copy.
        file_handler.handle(logging.LogRecord(
            LOGGER.name, logging.WARNING, filename, lineno,
            warnings.formatwarning(message, category, filename, lineno, line).rstrip(),
            (), None,
        ))
        old_showwarning(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = showwarning
    try:
        yield log_path
    finally:
        warnings.showwarning = old_showwarning
        LOGGER.handlers = old_handlers
        LOGGER.setLevel(old_level)
        LOGGER.propagate = old_propagate
        screen_handler.close()
        file_handler.close()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True, timeout=3,
        )
        return result.stdout.strip() or "unavailable"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(args, argv: list[str], batch) -> None:
    """Record reproducibility details without reading model weights or API keys."""
    from . import nli
    emit("Run provenance:")
    values = [
        ("Metalyzer version", __version__),
        ("Git commit", _git_commit()),
        ("Command line", subprocess.list2cmdline([
            sys.executable, str(Path(__file__).resolve().parents[1] / "metalyzer.py"), *argv,
        ])),
        ("Python version", sys.version.split()[0]),
        ("Platform", platform.platform()),
        ("Source method", args.method),
        ("NLI model", nli.DEFAULT_NLI_MODEL),
    ]
    if args.method == "llm":
        values.append(("Local LLM model", args.llm_model))
    if args.method == "mistral":
        values.append(("Mistral API model", args.mistral_model))
    values.extend([
        ("Device", args.device),
        ("NLI batch size", args.batch_size),
    ])
    if args.method == "llm":
        values.append(("LLM batch size", args.llm_batch_size))
    values.extend([
        ("Min score (NLI only)", args.min_score),
        ("Max value chars", args.max_value_chars),
        ("Max record chars", args.max_record_chars),
        ("Metadata path", str(Path(args.metadata).resolve())),
        ("Sources path", str(Path(args.sources).resolve())),
        ("Sources SHA256", _sha256(args.sources)),
        ("Taxonomy directory", str(Path(args.taxonomy_dir).resolve()) if args.taxonomy_dir else "none"),
        ("Metadata rows", len(batch)),
        ("Metadata columns", len(batch.metadata.columns)),
    ])
    for key, value in values:
        emit(f"  {key}: {value}")
    if args.taxonomy_dir:
        for name in deterministic_source.FILES:
            path = Path(args.taxonomy_dir) / name
            try:
                stat = path.stat()
            except OSError:
                emit(f"  Taxonomy file {name}: missing/unreadable ({path.resolve()})")
            else:
                modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
                emit(f"  Taxonomy file {name}: {path.resolve()}; {stat.st_size} bytes; modified {modified}")
