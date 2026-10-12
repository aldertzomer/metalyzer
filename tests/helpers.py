"""Adapters keeping the pre-refactor regression fixtures on the public contracts."""
import pandas as pd

import metalyzer
from modules.contracts import MetadataBatch, SourceVocabulary
from modules.llm import LLMConfig
from modules.nli import NLIConfig


def adapt_taxonomy_mock(taxonomy):
    """Adapt existing numeric-stage mocks to the expanded host-resolution contract."""
    from modules.deterministic_source import HostResolution
    def resolve(row, anchors):
        result = taxonomy.classify_host_taxid(row.get('host_tax_id'), anchors)
        return (HostResolution(result, 'host_tax_id', result.evidence)
                if result is not None else HostResolution())
    taxonomy.resolve_host_row.side_effect = resolve
    return taxonomy


def llm_config(args):
    return LLMConfig(model=getattr(args, "llm_model", LLMConfig.model),
                     revision=getattr(args, "llm_revision", None), device=args.device,
                     batch_size=getattr(args, "llm_batch_size", 1),
                     max_new_tokens=getattr(args, "llm_max_new_tokens", 16),
                     min_score=getattr(args, "llm_min_score", 0.75))


def classify(df, records, labels, args, taxonomy=None, anchors=None):
    batch = MetadataBatch(df, pd.Series(records, index=df.index, dtype=object))
    return metalyzer.classify_sources(
        batch, SourceVocabulary(tuple(labels), args.id_col, tuple(anchors or ())),
        method=getattr(args, "method", "nli"), taxonomy=taxonomy,
        nli_config=NLIConfig(args.device, args.batch_size, args.min_score),
        llm_config=llm_config(args),
    ).table
