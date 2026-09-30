"""Local NCBI taxonomy lookup using source-configured host anchors."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import urllib.request
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import MetadataBatch, SourceResult, SourceVocabulary

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
FILES = ("nodes.dmp", "names.dmp", "merged.dmp")


@dataclass(frozen=True)
class TaxonomySourceResult:
    source: str
    taxid: int
    scientific_name: str

    @property
    def evidence(self):
        return f"host_tax_id={self.taxid}; {self.scientific_name}"


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield [part.strip() for part in line.split("|")]


class NCBITaxonomy:
    def __init__(self, taxonomy_dir):
        directory = Path(taxonomy_dir)
        self.parent_by_taxid = {int(r[0]): int(r[1]) for r in _rows(directory / "nodes.dmp")}
        self.scientific_name_by_taxid = {}
        for r in _rows(directory / "names.dmp"):
            if r[3] == "scientific name":
                taxid, name = int(r[0]), r[1]
                self.scientific_name_by_taxid[taxid] = name
        self.merged_taxid_map = {int(r[0]): int(r[1]) for r in _rows(directory / "merged.dmp")}
        if self.parent_by_taxid.get(1) != 1 or not self.scientific_name_by_taxid:
            raise ValueError("Taxonomy dump is empty or lacks the NCBI root")

    def normalize_taxid(self, taxid):
        text = str(taxid).strip().strip('\"\'').strip()
        if not re.fullmatch(r"[0-9]+(?:\.0+)?", text):
            return None
        try:
            current = int(text.split(".")[0])
        except ValueError:
            return None
        seen = set()
        while current in self.merged_taxid_map:
            if current in seen:
                return None
            seen.add(current)
            current = self.merged_taxid_map[current]
        return current if current > 0 and current in self.parent_by_taxid else None

    def lineage(self, taxid):
        return self._lineage(self.normalize_taxid(taxid))

    @lru_cache(maxsize=100000)
    def _lineage(self, taxid):
        result, seen = [], set()
        while taxid is not None and taxid not in seen:
            seen.add(taxid)
            result.append(taxid)
            parent = self.parent_by_taxid.get(taxid)
            if taxid == 1 and parent == 1:
                return tuple(result)
            taxid = parent
        return ()  # Missing parent or cycle: never trust a partial lineage.

    def is_descendant_of(self, taxid, ancestor_taxid):
        return ancestor_taxid in self.lineage(taxid)

    def scientific_name(self, taxid):
        return self.scientific_name_by_taxid.get(self.normalize_taxid(taxid), "")

    def classify_host_taxid(self, taxid, anchor_sources):
        taxid = self.normalize_taxid(taxid)
        lineage = self.lineage(taxid)
        if not lineage or not self.scientific_name(taxid):
            return None
        # The first configured ancestor wins; an ambiguous first match stays unresolved.
        for ancestor in lineage:
            matches = anchor_sources.get(ancestor, ())
            if len(matches) == 1:
                return TaxonomySourceResult(next(iter(matches)), taxid, self.scientific_name(taxid))
            if matches:
                return None
        return None


def configured_anchors(sources: SourceVocabulary, taxonomy: NCBITaxonomy,
                       *, report: bool = False) -> dict[int, set[str]]:
    """Validate and normalize every configured anchor before classifying rows."""
    anchors: dict[int, set[str]] = {}
    invalid = []
    merged = []
    configured = 0
    valid = 0
    for source, taxids in zip(sources.names, sources.taxonomy_anchors):
        for taxid in taxids:
            configured += 1
            normalized = taxonomy.normalize_taxid(taxid)
            if normalized is None or not taxonomy.lineage(normalized) or not taxonomy.scientific_name(normalized):
                invalid.append(f"{taxid} ({source})")
                continue
            valid += 1
            if normalized != taxid:
                merged.append((taxid, normalized))
            anchors.setdefault(normalized, set()).add(source)
    conflicts = {taxid: names for taxid, names in anchors.items() if len(names) > 1}
    if report:
        from .runlog import emit
        emit("Taxonomy configuration:\n"
             f"  configured anchors: {configured}\n"
             f"  valid anchors:      {valid}\n"
             f"  merged IDs:         {len(merged)}\n"
             f"  invalid anchors:    {len(invalid)}\n"
             f"  conflicts:          {len(conflicts)}")
        for original, current in merged:
            emit(f"taxonomy anchor {original} -> {current} (merged)")
        for item in invalid:
            emit(f"invalid taxonomy anchor {item}")
        for taxid, names in conflicts.items():
            emit(f"conflicting taxonomy anchor {taxid}: {', '.join(sorted(names))}")
    if invalid or conflicts:
        details = []
        if invalid:
            details.append("unresolvable taxonomy anchors: " + ", ".join(invalid))
        if conflicts:
            details.append("conflicting taxonomy anchors: " + "; ".join(
                f"{taxid} ({', '.join(sorted(names))})" for taxid, names in conflicts.items()))
        raise ValueError("Invalid taxonomy_anchors configuration: " + "; ".join(details))
    return anchors


def load_taxonomy(directory, fallback_method="nli"):
    if directory is None:
        warnings.warn(f"No --taxonomy-dir supplied; using {fallback_method.upper()}-only source classification.")
        return None
    try:
        return NCBITaxonomy(directory)
    except (OSError, ValueError, IndexError) as exc:
        warnings.warn(f"Cannot load taxonomy from {directory}: {exc}; using {fallback_method.upper()}-only source classification.")
        return None


def download_taxonomy(directory):
    """Explicit download; copy only three regular members, never extract paths."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as staging:
        staging = Path(staging)
        archive = staging / "taxdump.tar.gz"
        with urllib.request.urlopen(TAXDUMP_URL, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with tarfile.open(archive, "r:gz") as handle:
            for name in FILES:
                member = handle.getmember(name)
                if not member.isfile():
                    raise ValueError(f"Not a regular taxonomy file: {name}")
                with handle.extractfile(member) as source, (staging / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
        # Validate before replacing any existing cache files.
        NCBITaxonomy(staging)
        for name in FILES:
            (staging / name).replace(destination / name)


def run(batch: MetadataBatch, sources: SourceVocabulary, taxonomy: NCBITaxonomy | None = None,
        *, anchor_sources: dict[int, set[str]] | None = None) -> SourceResult:
    """Return only resolved host rows; absent rows must be handled by a fallback."""
    from .contracts import SourceResult, empty_source_table

    resolved = {}
    if taxonomy is not None:
        anchors = anchor_sources if anchor_sources is not None else configured_anchors(sources, taxonomy)
        for key, row in batch.metadata.iterrows():
            result = taxonomy.classify_host_taxid(row.get("host_tax_id"), anchors)
            if result is not None and result.source in sources.names:
                resolved[key] = result
    table = empty_source_table(sources, resolved)
    for key, result in resolved.items():
        table.loc[key, ["best_hit", "source_method", "source_evidence"]] = [
            result.source, "host_tax_id", result.evidence,
        ]
    output = SourceResult(table)
    output.validate(batch, sources)
    return output
