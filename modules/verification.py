"""Binary NLI support for each final, non-unknown source assignment."""
from __future__ import annotations

import math

import pandas as pd

from . import nli
from .contracts import (
    MetadataBatch, SourceResult, SourceVocabulary, VerificationResult,
    VERIFICATION_COLUMN,
)


def run(batch: MetadataBatch, sources: SourceVocabulary, result: SourceResult,
        config: nli.NLIConfig = nli.NLIConfig(), *, disabled: bool = False,
        model: nli.NLIModel | None = None) -> VerificationResult:
    """Score only the assigned full label; unknown and disabled rows remain NA."""
    result.validate(batch, sources)
    if not result.table.index.equals(batch.metadata.index):
        raise ValueError("Verification requires one final source decision per input row")

    scores = pd.Series(float("nan"), index=batch.metadata.index,
                       name=VERIFICATION_COLUMN, dtype=float)
    if disabled or not len(batch):
        return VerificationResult(scores)

    classifier = None
    for name, full_label in zip(sources.names, sources.labels):
        if name == "unknown":
            continue
        keys = result.table.index[result.table.best_hit == name].tolist()
        for start in range(0, len(keys), config.batch_size):
            selected = keys[start:start + config.batch_size]
            if classifier is None:
                classifier = (model if model is not None else nli.NLIModel(config)).get()
            responses = classifier(
                batch.records.loc[selected].tolist(), candidate_labels=[full_label],
                hypothesis_template=nli.HYPOTHESIS_TEMPLATE,
                multi_label=True, batch_size=config.batch_size,
            )
            if isinstance(responses, dict):
                responses = [responses]
            if len(responses) != len(selected):
                raise RuntimeError("NLI verification returned an unexpected number of responses")
            for key, response in zip(selected, responses):
                if response.get("labels") != [full_label] or len(response.get("scores", [])) != 1:
                    raise RuntimeError("NLI verification returned an unexpected candidate")
                score = float(response["scores"][0])
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("NLI verification score must be between 0 and 1")
                scores.loc[key] = score

    verified = VerificationResult(scores)
    verified.validate(batch)
    return verified
