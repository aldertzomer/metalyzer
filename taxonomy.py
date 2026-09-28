"""Compatibility imports; taxonomy implementation now lives in modules."""
from modules.deterministic_source import (
    ANCHORS, FILES, TAXDUMP_URL, NCBITaxonomy, TaxonomySourceResult,
    download_taxonomy, load_taxonomy,
)

__all__ = [
    "ANCHORS", "FILES", "TAXDUMP_URL", "NCBITaxonomy", "TaxonomySourceResult",
    "download_taxonomy", "load_taxonomy",
]
