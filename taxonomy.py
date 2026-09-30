"""Compatibility imports; taxonomy implementation now lives in modules."""
from modules.deterministic_source import (
    FILES, TAXDUMP_URL, NCBITaxonomy, TaxonomySourceResult,
    download_taxonomy, load_taxonomy,
)

__all__ = [
    "FILES", "TAXDUMP_URL", "NCBITaxonomy", "TaxonomySourceResult",
    "download_taxonomy", "load_taxonomy",
]
