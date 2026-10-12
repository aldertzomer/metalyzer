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
DETERMINISTIC_METHODS = ("host_tax_id", "host_name", "host_alias", "host_other_animal")
NUMERIC_HOST_FIELDS = ("host_tax_id", "host_taxid")
NAME_CLASSES = {"scientific name", "common name", "genbank common name", "synonym", "genbank synonym"}
MISSING_HOST_NAMES = {"", "na", "n/a", "nan", "<na>", "none", "null", "missing", "unknown",
                      "unknown host", "unspecified", "not provided", "not collected", "not applicable",
                      "not available", "not known", "not determined", "not recorded", "no host", "-"}


def normalized_host_name(value):
    """Exact name key: normalize case/whitespace, never infer or tokenize a taxon."""
    key = " ".join(str(value).split()).casefold() if value is not None else ""
    return None if key in MISSING_HOST_NAMES else key


def host_name_values(row):
    """Only explicit host fields, including genuine original scientific-name aliases."""
    values = []
    for field, value in row.items():
        name = " ".join(str(field).replace("_", " ").split()).casefold()
        # The container name contains an underscore; check it before normalization.
        if str(field).casefold().startswith("original_attributes."):
            suffix = " ".join(str(field).split(".", 1)[1].replace("_", " ").split()).casefold()
            accepted = suffix == "host scientific name"
        else:
            accepted = name in {"host", "host scientific name"}
        if accepted and normalized_host_name(value) is not None:
            values.append((str(field), str(value).strip()))
    # Canonical host first; original aliases never override it.
    return sorted(values, key=lambda item: (item[0] != "host", item[0] != "host_scientific_name"))


@dataclass(frozen=True)
class TaxonomyNameResult:
    taxid: int | None
    match_type: str  # scientific name, common name/synonym, ambiguous, unresolved, missing


@dataclass(frozen=True)
class HostResolution:
    result: TaxonomySourceResult | None = None
    method: str = ""
    evidence: str = ""
    numeric_taxids: tuple[int, ...] = ()
    name_taxids: tuple[int, ...] = ()
    scientific_name_resolved: bool = False
    ambiguous: bool = False
    conflict: str = ""
    numeric_text_conflict: bool = False
    match_type: str = ""


@dataclass(frozen=True)
class TaxonomySourceResult:
    source: str
    taxid: int | None
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
        self.names_path = (directory / "names.dmp").resolve()
        self._name_cache = {}
        self.parent_by_taxid = {}
        self.specific_host_taxids = set()
        for row in _rows(directory / "nodes.dmp"):
            taxid = int(row[0])
            self.parent_by_taxid[taxid] = int(row[1])
            if row[2] in {"species", "subspecies"}:
                self.specific_host_taxids.add(taxid)
        self.scientific_name_by_taxid = {}
        for r in _rows(self.names_path):
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

    def resolve_host_names(self, values):
        """Resolve a batch in one streaming pass, storing only requested exact keys.

        Scientific names take precedence over aliases. Ambiguity is determined
        across all normalized live taxids, even when they share a source anchor.
        Repeated batches with already-seen names do not reopen names.dmp.
        """
        wanted = {key for value in values if (key := normalized_host_name(value)) is not None}
        wanted.difference_update(self._name_cache)
        if not wanted:
            return
        scientific = {key: set() for key in wanted}
        aliases = {key: set() for key in wanted}
        for row in _rows(self.names_path):
            if row[3] not in NAME_CLASSES:
                continue
            key = normalized_host_name(row[1])
            if key not in wanted:
                continue
            taxid = self.normalize_taxid(row[0])
            if taxid is None or not self.lineage(taxid) or not self.scientific_name(taxid):
                continue
            (scientific if row[3] == "scientific name" else aliases)[key].add(taxid)
        for key in wanted:
            matches = scientific[key] or aliases[key]
            if len(matches) == 1:
                self._name_cache[key] = TaxonomyNameResult(
                    next(iter(matches)), "scientific name" if scientific[key] else "common name/synonym")
            else:
                self._name_cache[key] = TaxonomyNameResult(None, "ambiguous" if matches else "unresolved")

    def resolve_host_name(self, value):
        key = normalized_host_name(value)
        if key is None:
            return TaxonomyNameResult(None, "missing")
        self.resolve_host_names([value])
        return self._name_cache[key]

    def resolve_host_row(self, row, anchor_sources, *, host_alias_sources=None,
                         other_animal=False):
        """Combine explicit numeric and named host evidence without mutating a row."""
        numeric, named = [], []
        numeric_taxids, name_taxids = [], []
        ambiguous = scientific = False
        for field in NUMERIC_HOST_FIELDS:
            taxid = self.normalize_taxid(row.get(field))
            if taxid is not None and self.lineage(taxid) and self.scientific_name(taxid):
                numeric_taxids.append(taxid)
                result = self.classify_host_taxid(taxid, anchor_sources)
                if result is not None:
                    numeric.append((field, result))
        names = host_name_values(row)
        self.resolve_host_names(value for _, value in names)
        for field, value in names:
            match = self.resolve_host_name(value)
            ambiguous |= match.match_type == "ambiguous"
            if match.taxid is not None:
                scientific |= match.match_type == "scientific name"
                name_taxids.append(match.taxid)
                result = self.classify_host_taxid(match.taxid, anchor_sources)
                if result is not None:
                    named.append((field, value, result, match.match_type))
        facts = dict(numeric_taxids=tuple(dict.fromkeys(numeric_taxids)),
                     name_taxids=tuple(dict.fromkeys(name_taxids)),
                     scientific_name_resolved=scientific, ambiguous=ambiguous)
        classes = {result.source for _, result in numeric} | {result.source for _, _, result, _ in named}
        if len(classes) > 1:
            return HostResolution(conflict="different configured source classes: " + ", ".join(sorted(classes)),
                                  numeric_text_conflict=bool(numeric and named), **facts)
        # Conflicting host values also matter under a deliberately broad source
        # anchor. Ancestor/descendant evidence agrees; unrelated taxa do not.
        taxids = tuple(dict.fromkeys(numeric_taxids + name_taxids))
        incompatible = [(left, right) for i, left in enumerate(taxids) for right in taxids[i + 1:]
                        if not self.is_descendant_of(left, right) and not self.is_descendant_of(right, left)]
        if incompatible:
            numeric_text_conflict = any(left != right and not self.is_descendant_of(left, right)
                                        and not self.is_descendant_of(right, left)
                                        for left in numeric_taxids for right in name_taxids)
            return HostResolution(conflict="incompatible host taxa: " + ", ".join(f"{a}/{b}" for a, b in incompatible),
                                  numeric_text_conflict=numeric_text_conflict, **facts)
        aliases = [(field, value, host_alias_sources[key]) for field, value in names
                   if host_alias_sources and (key := normalized_host_name(value)) in host_alias_sources]
        alias_classes = {source for _, _, source in aliases}
        if len(alias_classes | classes) > 1:
            return HostResolution(conflict="configured host alias contradicts host source evidence", **facts)
        if aliases:
            alias_source = aliases[0][2]
            source_anchors = [anchor for anchor, sources in anchor_sources.items() if alias_source in sources]
            # An operational descriptor cannot override an explicit taxon outside
            # its configured lineage, including ancestors of domestic anchors.
            if any(not any(self.is_descendant_of(taxid, anchor) for anchor in source_anchors)
                   for taxid in taxids):
                return HostResolution(conflict="configured host alias contradicts resolved host taxon", **facts)
        if numeric:
            field, result = numeric[0]
            return HostResolution(result, "host_tax_id", f"{field}={result.taxid}; {result.scientific_name}", **facts)
        # An ambiguous host-name field cannot be bypassed by a second name alias.
        if named and not ambiguous:
            field, value, result, match_type = min(named, key=lambda item: item[3] != "scientific name")
            return HostResolution(result, "host_name",
                                  f"{field}={value}; taxid={result.taxid}; {result.scientific_name}",
                                  match_type=match_type, **facts)
        if aliases and not ambiguous:
            field, value, source = aliases[0]
            return HostResolution(TaxonomySourceResult(source, None, ""), "host_alias",
                                  f"{field}={value}; configured_host_alias={source}",
                                  match_type="configured host alias", **facts)
        if other_animal and name_taxids and not ambiguous:
            # Compatible fields may include an ancestor; choose the most specific
            # explicit named taxon. Never infer a name from numeric-only evidence.
            taxid = max(name_taxids, key=lambda value: len(self.lineage(value)))
            lineage = self.lineage(taxid)
            if (taxid in self.specific_host_taxids and 33208 in lineage and 8782 not in lineage
                    and not any(self.is_descendant_of(anchor, taxid) for anchor in anchor_sources)
                    and not any(anchor in lineage for anchor in anchor_sources)):
                field, value = next((field, value) for field, value in names
                                    if self.resolve_host_name(value).taxid == taxid)
                result = TaxonomySourceResult("other_animal", taxid, self.scientific_name(taxid))
                return HostResolution(result, "host_other_animal",
                                      f"{field}={value}; taxid={taxid}; {result.scientific_name}; specific_unlisted_nonbird_animal",
                                      match_type="specific unlisted animal", **facts)
        return HostResolution(**facts)

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
        *, anchor_sources: dict[int, set[str]] | None = None, diagnostics: list | None = None,
        use_host_aliases: bool = True, other_animal: bool = False) -> SourceResult:
    """Return only resolved host rows; absent rows must be handled by a fallback."""
    from .contracts import SourceResult, empty_source_table
    from . import runlog

    resolved = {}
    counts = dict(numeric=0, names=0, scientific=0, common=0, aliases=0, other_animal=0,
                  ambiguous_conflicting=0, unresolved=0)
    if taxonomy is not None:
        anchors = anchor_sources if anchor_sources is not None else configured_anchors(sources, taxonomy)
        aliases = {alias: source for source, values in zip(sources.names, sources.host_aliases)
                   for alias in values} if use_host_aliases else {}
        taxonomy.resolve_host_names(value for _, row in batch.metadata.iterrows()
                                    for _, value in host_name_values(row))
        conflict_examples = []
        for key, row in batch.metadata.iterrows():
            options = {}
            if aliases:
                options['host_alias_sources'] = aliases
            if other_animal and 'other_animal' in sources.names:
                options['other_animal'] = True
            resolution = taxonomy.resolve_host_row(row, anchors, **options)
            if diagnostics is not None:
                diagnostics.append(dict(row_key=key, resolution=resolution))
            if resolution.result is not None and resolution.result.source in sources.names:
                resolved[key] = resolution
                route = {'host_tax_id': 'numeric', 'host_name': 'names',
                         'host_alias': 'aliases', 'host_other_animal': 'other_animal'}[resolution.method]
                counts[route] += 1
                if route == 'names':
                    counts['scientific' if resolution.match_type == 'scientific name' else 'common'] += 1
            else:
                counts["unresolved"] += 1
                if resolution.conflict or resolution.ambiguous:
                    counts["ambiguous_conflicting"] += 1
                if resolution.conflict and len(conflict_examples) < 10:
                    conflict_examples.append(f"  row {key}: {resolution.conflict}")
        for message in conflict_examples:
            runlog.emit("Host taxonomy conflict" + message)
    else:
        counts["unresolved"] = len(batch)
    runlog.taxonomy_resolution(counts)
    table = empty_source_table(sources, resolved)
    for key, resolution in resolved.items():
        table.loc[key, ["best_hit", "source_method", "source_evidence"]] = [
            resolution.result.source, resolution.method, resolution.evidence,
        ]
    output = SourceResult(table)
    output.validate(batch, sources)
    return output
