"""Variant C v1: exact administrative names and conservative original aliases.

This module selects text fields only. It never edits metadata or resolves taxa.
Unknown fields are retained. No matching against arbitrary name substrings.
"""
from html import unescape
import re

PROFILE = 'variant_c_v1'


def normalize_field(name):
    name = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', str(name))
    return ' '.join(re.sub(r'[_\-\s]+', ' ', name).casefold().split())


def original_field(name):
    prefix, dot, suffix = str(name).partition('.')
    return suffix if dot and normalize_field(prefix) == 'original attributes' else None


EXCLUDED = {normalize_field(name) for name in '''
accession
biosample_id
gap_accession
gap_sample_id
gap_subject_id
submitted sample id
submitted subject id
Submitter Id
submitter handle
External Id
subject_id
host_subject_id
sample_id
source_material_id
biospecimen repository sample id
submission identifier
SRA accession
ena_run_accession
bioproject_accession
gisaid_accession
sample accession
run accession
experiment accession
experiment alias
INSDC center name
INSDC center alias
INSDC status
ENA-CHECKLIST
ENA checklist
checklist_accession
broker name
owner
owner_abbreviation
collected_by
sequenced_by
submission_tool
submission_tool_version
access
status
status_date
biospecimen repository
submission_date
publication_date
last_update
last_updated
last_updated_by
INSDC first public
INSDC last update
ENA first public
ENA last update
receipt date
last update date
release date
ngdc_release_date
plate_id
well_id
library_id
lane
tube_id
barcode_id
barcode_sequence
BarcodeSequence
LinkerPrimerSequence
filename
filename2
file_location
sequencing run
sequencing batch
library name
instrument_model
library_layout
library_strategy
library_selection
assembly software
assembly_method
assembly quality
binning software
binning parameters
completeness software
completeness score
contamination score
number of contigs
contig N50
genome_length
MAG coverage software
MAG coverage depth
GUNC version
GUNC contamination score
GUNC clade separation score
GUNC reference representation score
taxonomic classification software
sequencing method
molecular data type
read base count
total number of reads
number of mapped reads
extraction_method
nucleic acid extraction
pcr primers
sequencing primer provider
adapters
library preparation protocol
sample storage temperature
samp_store_temp
samp_store_dur
samp_store_loc
model
package
package_display_name
package.display_name
attribute_package
env_package
environmental package
host-associated environmental package
human associated environmental package
soil environmental package
water environmental package
'''.strip().splitlines()}


def is_noisy_field(name):
    name = original_field(name) or str(name)
    key = normalize_field(name)
    return (key in EXCLUDED or bool(re.match(r'^(?:id|link)\.', key))
            or bool(re.fullmatch(r'diagnostic pcr protocol(?: [0-9]+)?', key)))


# Genuine NCBI harmonization aliases; not a whitelist of accepted metadata.
CONCEPTS = {
    'host scientific name': 'host',
    'sample description': 'description',
    'geographic location (country and/or sea)': 'geo loc name',
    'geographic location': 'geo loc name',
}


def concept(name):
    key = normalize_field(name)
    return CONCEPTS.get(key, key)


def equivalent_value(value):
    # Compare complete, untruncated values. Repeated map values have set semantics.
    text = ' '.join(unescape(str(value)).split()).casefold()
    return frozenset(part.strip() for part in text.split(' | '))


def select_fields(items):
    """Keep input order; omit only defined noise or equivalent original fields.

    Callers have already removed legacy-excluded fields and missing-like values.
    Conflicting canonical fields prevent suppressing an original alias.
    """
    canonical = {}
    for name, value in items:
        if original_field(name) is None and not is_noisy_field(name):
            canonical.setdefault(concept(name), set()).add(equivalent_value(value))
    kept, omitted = [], []
    for name, value in items:
        original = original_field(name)
        if is_noisy_field(name):
            omitted.append(('administrative_technical', str(name)))
        elif original is not None and canonical.get(concept(original)) == {equivalent_value(value)}:
            omitted.append(('redundant_original', str(name)))
        else:
            kept.append((name, value))
    return kept, omitted
