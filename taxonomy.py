"""Local NCBI taxonomy lookup and conservative host-source classification."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import urllib.request
import warnings

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
FILES = ("nodes.dmp", "names.dmp", "merged.dmp")
ANCHORS = {
    "chicken": "Gallus gallus", "turkey": "Meleagris gallopavo",
    "cattle": "Bos taurus", "sheep": "Ovis aries", "goat": "Capra hircus",
    "human": "Homo sapiens", "dog": "Canis lupus familiaris",
    "cat": "Felis catus", "pig": "Sus scrofa domesticus",
}


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
        self.taxid_by_scientific_name = {}
        for r in _rows(directory / "names.dmp"):
            if r[3] == "scientific name":
                taxid, name = int(r[0]), r[1]
                self.scientific_name_by_taxid[taxid] = name
                # Ambiguous scientific names must never choose an arbitrary anchor.
                if name in self.taxid_by_scientific_name:
                    self.taxid_by_scientific_name[name] = None
                else:
                    self.taxid_by_scientific_name[name] = taxid
        self.merged_taxid_map = {int(r[0]): int(r[1]) for r in _rows(directory / "merged.dmp")}
        if self.parent_by_taxid.get(1) != 1 or not self.scientific_name_by_taxid:
            raise ValueError("Taxonomy dump is empty or lacks the NCBI root")
        def anchor(name):
            taxid = self.taxid_by_scientific_name.get(name)
            return taxid if self.lineage(taxid) else None

        self.anchors = {source: anchor(name) for source, name in ANCHORS.items()}
        self.animal = anchor("Metazoa") or anchor("Animalia")
        self.bird = anchor("Aves")
        self.boar = anchor("Sus scrofa")
        self.generic_safe = all(self.anchors.values()) and bool(self.animal and self.bird and self.boar)
        if not self.generic_safe:
            warnings.warn("Some taxonomy anchors are missing/ambiguous; generic other_animal assignments disabled.")

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

    def classify_host_taxid(self, taxid):
        taxid = self.normalize_taxid(taxid)
        lineage = self.lineage(taxid)
        if not lineage or not self.scientific_name(taxid):
            return None
        # Lineage runs from most specific to least specific.
        for ancestor in lineage:
            matches = [source for source, anchor in self.anchors.items() if anchor == ancestor]
            if len(matches) == 1:
                return TaxonomySourceResult(matches[0], taxid, self.scientific_name(taxid))
            if matches:
                return None
        if not self.generic_safe or self.animal not in lineage or self.bird in lineage:
            return None
        # Sus scrofa and its non-domestic descendants cannot reliably distinguish
        # wild boar from domestic pigs. Broad ancestors also remain ambiguous.
        if self.boar in lineage or any(taxid in self.lineage(anchor) for anchor in self.anchors.values()):
            return None
        if taxid in self.lineage(self.bird):
            return None
        return TaxonomySourceResult("other_animal", taxid, self.scientific_name(taxid))


def load_taxonomy(directory):
    if directory is None:
        warnings.warn("No --taxonomy-dir supplied; using NLI-only source classification.")
        return None
    try:
        return NCBITaxonomy(directory)
    except (OSError, ValueError, IndexError) as exc:
        warnings.warn(f"Cannot load taxonomy from {directory}: {exc}; using NLI-only source classification.")
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
