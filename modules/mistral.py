"""Mistral API source stage with bounded concurrency and optional SDK loading.

run(batch, sources, config) -> SourceResult; run_async exposes the same contract
to async callers. Only supplied rows are sent to the API. Unknown is always an
allowed answer, and API errors never become source predictions.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .contracts import MetadataBatch, SourceResult, SourceVocabulary, empty_source_table
from .generative import allowed_source_names, invalid_response_evidence, system_prompt

DEFAULT_MISTRAL_MODEL = "mistral-small-latest"


@dataclass(frozen=True)
class MistralConfig:
    api_key_file: str | Path | None = None
    model: str = DEFAULT_MISTRAL_MODEL
    concurrency: int = 8
    retries: int = 5
    timeout: float = 60.0
    random_seed: int = 12345
    max_tokens: int = 32
    progress_every: int = 10

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("Mistral model must not be empty")
        if self.concurrency < 1 or self.max_tokens < 1 or self.progress_every < 1:
            raise ValueError("Mistral concurrency, max_tokens and progress_every must be positive")
        if self.retries < 0:
            raise ValueError("Mistral retries must be >= 0")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Mistral timeout must be finite and > 0")


def read_api_key(path: str | Path | None) -> str:
    """Read a raw key or MISTRAL_API_KEY=... assignment, without logging it."""
    if path is None:
        raise ValueError("--api-key-file is required when unresolved rows use --method mistral")
    key_path = Path(path).expanduser()
    key = key_path.read_text(encoding="utf-8").strip()
    if key.startswith("MISTRAL_API_KEY="):
        key = key.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise ValueError(f"API key file is empty: {key_path}")
    if any(character.isspace() for character in key):
        raise ValueError("API key file should contain only one API key.")
    return key


def create_client(config: MistralConfig):
    """Load the optional SDK only when a nonempty batch needs API inference."""
    api_key = read_api_key(config.api_key_file)
    try:
        from mistralai.client import Mistral
    except ImportError as exc:
        raise ImportError("Mistral mode requires the optional mistralai v2 SDK; see README.md.") from exc
    return Mistral(api_key=api_key)


def make_response_format(source_names) -> dict:
    """Constrain JSON output to the canonical vocabulary plus implicit unknown."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "source_classification",
            "schema_definition": {
                "type": "object",
                "properties": {"source": {"type": "string", "enum": allowed_source_names(source_names)}},
                "required": ["source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def parse_response(content, source_names) -> tuple[str, str]:
    """Accept the requested JSON object; malformed answers are evidenced unknowns."""
    try:
        parsed = json.loads(content) if isinstance(content, str) else None
    except ValueError:
        parsed = None
    if (isinstance(parsed, dict) and set(parsed) == {"source"}
            and isinstance(parsed["source"], str)
            and parsed["source"] in allowed_source_names(source_names)):
        return parsed["source"], ""
    return "unknown", invalid_response_evidence(content, "invalid_mistral_output")


async def classify_one(client, record: str, row_key: int, sources: SourceVocabulary,
                       config: MistralConfig, prompt: str, response_format: dict) -> tuple[str, str]:
    """Bound each request and retry only timeouts and transient HTTP statuses."""
    for attempt in range(config.retries + 1):
        try:
            response = await asyncio.wait_for(client.chat.complete_async(
                model=config.model,
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": f"Classify the source of this sample.\n\nMETADATA\n{record}"}],
                response_format=response_format, temperature=0,
                random_seed=config.random_seed, max_tokens=config.max_tokens,
                # Avoid multiplying our retry budget by the SDK's retries.
                retries=None,
            ), timeout=config.timeout)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = isinstance(exc, TimeoutError) or status in {408, 429, 500, 502, 503, 504}
            if retryable and attempt < config.retries:
                await asyncio.sleep(min(2 ** attempt, 30))
                continue
            reason = f"HTTP {status}" if isinstance(status, int) else type(exc).__name__
            hint = " Check API credentials, quota and model access." if status in {401, 402, 403, 429} else ""
            # SDK exception bodies may include requests; do not echo them or the key.
            raise RuntimeError(
                f"Mistral request failed for row key {row_key} after {attempt + 1} attempt(s) ({reason}).{hint}"
            ) from None
        choices = getattr(response, "choices", None)
        message = getattr(choices[0], "message", None) if choices else None
        return parse_response(getattr(message, "content", None), sources.names)


async def run_async(batch: MetadataBatch, sources: SourceVocabulary,
                    config: MistralConfig = MistralConfig()) -> SourceResult:
    """Classify the batch using at most concurrency worker tasks; preserve row keys."""
    output = empty_source_table(sources, batch.metadata.index)
    output["source_method"] = "mistral"
    if not len(batch):
        return SourceResult(output)

    prompt = system_prompt(sources.labels, sources.names, structured=True)
    response_format = make_response_format(sources.names)
    rows = iter(batch.records.items())
    completed = 0

    # Mistral owns separate sync and async HTTP clients; close both even on error.
    with create_client(config) as client:
        async with client:
            async def worker():
                nonlocal completed
                for key, record in rows:
                    label, evidence = await classify_one(client, record, key, sources, config, prompt, response_format)
                    output.loc[key, ["best_hit", "source_evidence"]] = [label, evidence]
                    completed += 1
                    if completed % config.progress_every == 0 or completed == len(batch):
                        print(f"Mistral classified {completed}/{len(batch)}", flush=True)

            tasks = [asyncio.create_task(worker()) for _ in range(min(config.concurrency, len(batch)))]
            try:
                await asyncio.gather(*tasks)
            finally:
                # Stop remaining requests on failure/cancellation before closing HTTP clients.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    result = SourceResult(output)
    result.validate(batch, sources)
    return result


def run(batch: MetadataBatch, sources: SourceVocabulary,
        config: MistralConfig = MistralConfig()) -> SourceResult:
    """Synchronous CLI entry point. Inside an event loop, await run_async instead."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_async(batch, sources, config))
    raise RuntimeError("Use await modules.mistral.run_async(...) inside an active event loop.")
