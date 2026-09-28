"""Zero-shot NLI stage; only this module loads the NLI model."""
from dataclasses import dataclass

import pandas as pd

from .contracts import MetadataBatch, SourceResult, SourceVocabulary, canonical_source_names


@dataclass(frozen=True)
class NLIConfig:
    device: int = 0
    batch_size: int = 64
    min_score: float = 0.2

    def __post_init__(self):
        if self.device < -1 or self.batch_size < 1:
            raise ValueError("device must be >= -1 and batch_size must be positive")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")


def pipeline(*args, **kwargs):
    # Taxonomy-only runs and downloads do not need to load Transformers/PyTorch.
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(*args, **kwargs)


def parse_source_scores(
    scores: pd.DataFrame,
    id_col: str,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """Shorten score headers and select the highest-scoring source per row."""
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    names = canonical_source_names(scores.columns, id_col)

    parsed = scores.copy()
    parsed.columns = names
    # idxmax resolves ties in column order, which follows sources.tsv.
    if len(parsed):
        best_hit = parsed.idxmax(axis=1)
        parsed["best_hit"] = best_hit.where(parsed.max(axis=1) >= min_score, "unknown")
    else:
        parsed["best_hit"] = pd.Series(index=parsed.index, dtype=str)
    return parsed


def run(batch: MetadataBatch, sources: SourceVocabulary, config: NLIConfig = NLIConfig()) -> SourceResult:
    """Original zero-shot inference, including its dtype and threshold logic."""
    records = batch.records.tolist()
    source_labels = list(sources.labels)
    score_rows = []
    if records:
        classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
            device=config.device,
            dtype="float32" if config.device < 0 else "auto",
        )
        for i in range(0, len(records), config.batch_size):
            record_batch = records[i:i + config.batch_size]
            results = classifier(
                record_batch, candidate_labels=source_labels,
                hypothesis_template="The biological host or environmental source of this sample is {}.",
                multi_label=False, batch_size=config.batch_size,
            )
            if isinstance(results, dict):
                results = [results]
            if len(results) != len(record_batch):
                raise RuntimeError("NLI returned an unexpected number of responses.")
            for result in results:
                scores = dict.fromkeys(source_labels, 0.0)
                scores.update({label: float(score) for label, score in zip(result["labels"], result["scores"])})
                score_rows.append(scores)
            print(f"Batch {i // config.batch_size + 1}: source classification done", flush=True)

    output = parse_source_scores(pd.DataFrame(score_rows, columns=source_labels, dtype=float), sources.id_col, config.min_score)
    if len(output) != len(batch):
        raise RuntimeError("NLI returned an unexpected number of responses.")
    output.index = batch.metadata.index
    output["source_method"] = "nli"
    output["source_evidence"] = ""
    result = SourceResult(output[sources.columns])
    result.validate(batch, sources)
    return result
