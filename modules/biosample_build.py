#!/usr/bin/env python3
"""Standalone streaming builder, adapted from the server's BioSample converter.

Keeps its Arrow schema, canonical maps, 10,000-row batches and zstd encoding.
Run with ``python -m modules.biosample_build``; classification never imports it.
"""

from collections import Counter
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq


BATCH_SIZE = 10_000


def clean(text):
    """Clean text for later TSV export."""
    if text is None:
        return None

    text = " ".join(text.split())
    return text if text else None


def add_value(d, key, value):
    """
    Add a value to a dictionary.

    If the same key occurs several times in one BioSample,
    concatenate unique values with ' | '.
    """
    value = clean(value)

    if not key or value is None:
        return

    if key not in d:
        d[key] = value
    elif value not in d[key].split(" | "):
        d[key] += " | " + value


def flatten_xml_attributes(element, prefix, output):
    """
    Preserve XML attributes from fixed elements.

    Example:
        <Package display_name="MIGS...">MIGS.ba.6.0</Package>

    becomes roughly:
        package = MIGS.ba.6.0
        package.display_name = MIGS...
    """
    if element is None:
        return

    for key, value in element.attrib.items():
        add_value(output, f"{prefix}.{key}", value)


def parse_biosample(bs):
    # ------------------------------------------------------------
    # Common / structural BioSample fields
    # ------------------------------------------------------------

    fixed = {}

    # BioSample-level attributes
    fixed["accession"] = bs.attrib.get("accession")
    fixed["biosample_id"] = bs.attrib.get("id")
    fixed["access"] = bs.attrib.get("access")
    fixed["submission_date"] = bs.attrib.get("submission_date")
    fixed["publication_date"] = bs.attrib.get("publication_date")
    fixed["last_update"] = bs.attrib.get("last_update")

    # Description
    title = bs.find("./Description/Title")
    fixed["title"] = clean(title.text if title is not None else None)

    organism = bs.find("./Description/Organism")
    if organism is not None:
        fixed["organism"] = organism.attrib.get("taxonomy_name")
        fixed["tax_id"] = organism.attrib.get("taxonomy_id")
    else:
        fixed["organism"] = None
        fixed["tax_id"] = None

    # Description/comment
    paragraphs = []

    for p in bs.findall("./Description/Comment/Paragraph"):
        value = clean("".join(p.itertext()))
        if value:
            paragraphs.append(value)

    fixed["description"] = " | ".join(paragraphs) if paragraphs else None

    # Owner
    owner = bs.find("./Owner/Name")
    fixed["owner"] = clean(owner.text if owner is not None else None)

    if owner is not None:
        fixed["owner_abbreviation"] = owner.attrib.get("abbreviation")
    else:
        fixed["owner_abbreviation"] = None

    # Models
    models = []
    for model in bs.findall("./Models/Model"):
        value = clean(model.text)
        if value:
            models.append(value)

    fixed["model"] = " | ".join(models) if models else None

    # Package
    package = bs.find("./Package")
    fixed["package"] = clean(package.text if package is not None else None)

    if package is not None:
        fixed["package_display_name"] = package.attrib.get("display_name")
    else:
        fixed["package_display_name"] = None

    # Status
    status = bs.find("./Status")

    if status is not None:
        fixed["status"] = status.attrib.get("status")
        fixed["status_date"] = status.attrib.get("when")
    else:
        fixed["status"] = None
        fixed["status_date"] = None

    # ------------------------------------------------------------
    # IDs
    #
    # Store variable IDs in sparse map:
    #   id.SRA
    #   id.BioSample
    #   id.WUGSC
    # etc.
    # ------------------------------------------------------------

    identifiers = {}

    for x in bs.findall("./Ids/Id"):
        db = x.attrib.get("db", "unknown")
        label = x.attrib.get("db_label")

        key = f"id.{db}"
        if label:
            key += f".{label}"

        add_value(identifiers, key, "".join(x.itertext()))

    # ------------------------------------------------------------
    # BioSample attributes
    #
    # Prefer harmonized_name when NCBI provides it.
    #
    # We ALSO keep original-name aliases separately so no
    # information about the submitted naming is lost.
    # ------------------------------------------------------------

    attributes = {}
    original_attributes = {}

    for attr in bs.findall("./Attributes/Attribute"):

        original_name = attr.attrib.get("attribute_name")
        harmonized_name = attr.attrib.get("harmonized_name")

        value = clean("".join(attr.itertext()))

        # canonical name
        name = harmonized_name or original_name

        if name:
            add_value(attributes, name, value)

        # preserve submitter's original name where different
        if original_name:
            add_value(original_attributes, original_name, value)

    # ------------------------------------------------------------
    # Links
    # ------------------------------------------------------------

    links = {}

    for link in bs.findall("./Links/Link"):
        link_type = link.attrib.get("type", "unknown")
        target = link.attrib.get("target")
        label = link.attrib.get("label")

        parts = [link_type]

        if target:
            parts.append(target)

        if label:
            parts.append(label)

        key = "link." + ".".join(parts)

        add_value(links, key, "".join(link.itertext()))

    # Preserve unexpected XML fields in the existing sparse attributes map.
    known_paths = {'Description', 'Description/Title', 'Description/Organism',
                   'Description/Comment', 'Description/Comment/Paragraph',
                   'Owner', 'Owner/Name', 'Models', 'Models/Model', 'Package',
                   'Status', 'Ids', 'Ids/Id', 'Attributes', 'Attributes/Attribute',
                   'Links', 'Links/Link'}
    known_attributes = {
        'Description/Organism': {'taxonomy_name', 'taxonomy_id'},
        'Owner/Name': {'abbreviation'}, 'Package': {'display_name'},
        'Status': {'status', 'when'}, 'Ids/Id': {'db', 'db_label'},
        'Attributes/Attribute': {'attribute_name', 'harmonized_name'},
        'Links/Link': {'type', 'target', 'label'},
    }
    for key, value in bs.attrib.items():
        if key not in {'accession', 'id', 'access', 'submission_date', 'publication_date', 'last_update'}:
            add_value(attributes, 'biosample.' + key, value)

    def preserve(node, path):
        prefix = path.replace('/', '.').lower()
        for key, value in node.attrib.items():
            if key not in known_attributes.get(path, set()):
                add_value(attributes, prefix + '.' + key, value)
        if path not in known_paths:
            add_value(attributes, prefix, node.text)
        for child in node:
            preserve(child, path + '/' + child.tag)
            add_value(attributes, prefix + '.tail', child.tail)

    for child in bs:
        preserve(child, child.tag)

    return {
        **fixed,
        "attributes": attributes,
        "original_attributes": original_attributes,
        "identifiers": identifiers,
        "links": links,
    }


# -----------------------------------------------------------------
# Arrow schema
# -----------------------------------------------------------------

map_type = pa.map_(pa.string(), pa.string())

schema = pa.schema([
    ("accession", pa.string()),
    ("biosample_id", pa.string()),
    ("access", pa.string()),
    ("submission_date", pa.string()),
    ("publication_date", pa.string()),
    ("last_update", pa.string()),

    ("title", pa.string()),
    ("organism", pa.string()),
    ("tax_id", pa.string()),
    ("description", pa.string()),

    ("owner", pa.string()),
    ("owner_abbreviation", pa.string()),

    ("model", pa.string()),
    ("package", pa.string()),
    ("package_display_name", pa.string()),

    ("status", pa.string()),
    ("status_date", pa.string()),

    # sparse metadata
    ("attributes", map_type),
    ("original_attributes", map_type),
    ("identifiers", map_type),
    ("links", map_type),
])


def dict_to_map(d):
    if not d:
        return []
    return list(d.items())


def rows_to_table(rows):

    converted = []

    for row in rows:
        r = dict(row)

        for field in (
            "attributes",
            "original_attributes",
            "identifiers",
            "links",
        ):
            r[field] = dict_to_map(r[field])

        converted.append(r)

    return pa.Table.from_pylist(converted, schema=schema)


# Standalone setup utility: never imported by classification.
import argparse
import gzip
import logging
from pathlib import Path
import subprocess
import tempfile
import time
import os

from .biosample import DEFAULT_STORE, canonical_from_parquet, ensure_index, load_biosamples

DEFAULT_URL = 'https://ftp.ncbi.nlm.nih.gov/biosample/biosample_set.xml.gz'
LOGGER = logging.getLogger('metalyzer.biosample_build')


def download_archive(destination, url=DEFAULT_URL, *, force=False, skip=False):
    destination = Path(destination)
    started = time.monotonic()
    LOGGER.info('Source URL: %s; destination: %s', url, destination)
    if skip:
        if not destination.is_file():
            raise FileNotFoundError(destination)
        action = 'reused'
    elif destination.exists() and not force:
        action = 'reused'
    else:
        partial = destination.with_name(destination.name + '.part')
        if force:
            partial.unlink(missing_ok=True)
        action = 'resumed' if partial.exists() else 'downloaded'
        subprocess.run(['curl', '-L', '--fail', '--retry', '5', '-C', '-', url,
                        '-o', str(partial)], check=True)
        partial.replace(destination)
    LOGGER.info('XML archive %s; compressed size: %s bytes; download duration: %.1f seconds',
                action, destination.stat().st_size, time.monotonic() - started)
    return destination


def iter_biosamples(archive):
    with gzip.open(archive, 'rb') as handle:
        stack = []
        for event, elem in ET.iterparse(handle, events=('start', 'end')):
            if event == 'start':
                stack.append(elem)
                continue
            if elem.tag.rsplit('}', 1)[-1] == 'BioSample':
                # The NCBI archive uses unqualified tags; normalize namespaces too.
                for child in elem.iter():
                    child.tag = child.tag.rsplit('}', 1)[-1]
                row = parse_biosample(elem)
                if len(stack) > 1:
                    stack[-2].remove(elem)
                elem.clear()
                yield row
            elif not any(parent.tag.rsplit('}', 1)[-1] == 'BioSample' for parent in stack[:-1]):
                if len(stack) > 1:
                    stack[-2].remove(elem)
                elem.clear()
            stack.pop()


def validate_database(path):
    samples = []
    with pq.ParquetFile(path) as parquet:
        if not parquet.metadata.num_rows or 'accession' not in parquet.schema_arrow.names:
            raise ValueError('BioSample database is empty or lacks accession')
        for group, last in ((0, False), (parquet.num_row_groups - 1, True)):
            chosen = None
            for batch in parquet.iter_batches(batch_size=1024, row_groups=[group]):
                chosen = batch.slice(batch.num_rows - 1 if last else 0, 1).to_pylist()[0]
                if not last:
                    break
            record = canonical_from_parquet(chosen)
            samples.append(record['accession'])
    return samples


def convert_archive(archive, output, *, batch_size=BATCH_SIZE, force=False):
    output = Path(output)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if output.exists() and not force:
        validate_database(output)
        ensure_index(output, report=LOGGER.info)
        LOGGER.info('Reused Parquet database: %s', output)
        return output
    started = time.monotonic()
    fd, name = tempfile.mkstemp(prefix=output.name + '.', suffix='.tmp', dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    attributes, originals = Counter(), Counter()
    count = 0
    temporary_index = None
    try:
        with pq.ParquetWriter(temporary, schema, compression='zstd', use_dictionary=True) as writer:
            rows = []
            for row in iter_biosamples(archive):
                rows.append(row)
                attributes.update(row['attributes'].keys())
                originals.update(row['original_attributes'].keys())
                count += 1
                if len(rows) >= batch_size:
                    writer.write_table(rows_to_table(rows), row_group_size=batch_size)
                    rows.clear()
                    if count % max(batch_size, 100000) == 0:
                        elapsed = time.monotonic() - started
                        LOGGER.info('Parsed/written BioSamples: %s; output size: %s bytes; elapsed: %.1fs; records/second: %.1f',
                                    f'{count:,}', temporary.stat().st_size, elapsed, count / elapsed)
            if rows:
                writer.write_table(rows_to_table(rows), row_group_size=batch_size)
        samples = validate_database(temporary)
        temporary_index = ensure_index(temporary, report=LOGGER.info)
        frame = load_biosamples(samples, temporary)
        if frame.accession.tolist() != samples:
            raise ValueError('BioSample accession round-trip failed')
        # Prepare the index before replacing an existing valid database.
        from .biosample import index_path
        import json
        import sqlite3
        stat = temporary.stat()
        signature = json.dumps([(str(output.resolve()), stat.st_size, stat.st_mtime_ns)])
        with sqlite3.connect(temporary_index) as db:
            db.execute('UPDATE accessions SET parquet_file=?', (str(output.resolve()),))
            db.execute('UPDATE metadata SET signature=?', (signature,))
        # Publish only after complete parsing, Parquet and index validation.
        temporary.replace(output)
        temporary_index.replace(index_path(output))
        for filename, counts in (('attribute_catalogue.tsv', attributes),
                                 ('original_attribute_catalogue.tsv', originals)):
            with (output.parent / filename).open('w', encoding='utf-8') as handle:
                handle.write('attribute\tn_samples\n')
                for key, number in counts.most_common():
                    handle.write(f'{key}\t{number}\n')
        elapsed = time.monotonic() - started
        LOGGER.info('BioSample database build complete\nXML archive: %s\nParquet database: %s\nBioSample records: %s\nParquet size: %.3f GB\nElapsed time: %s',
                    archive, output, f'{count:,}', output.stat().st_size / 1e9,
                    time.strftime('%H:%M:%S', time.gmtime(elapsed)))
    finally:
        temporary.unlink(missing_ok=True)
        if temporary_index:
            temporary_index.unlink(missing_ok=True)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Download and stream the complete NCBI BioSample database')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_STORE)
    parser.add_argument('--url', default=DEFAULT_URL)
    parser.add_argument('--force-download', action='store_true')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--force-convert', action='store_true')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.force_download and args.skip_download:
        parser.error('--force-download and --skip-download are mutually exclusive')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.StreamHandler(), logging.FileHandler(args.output_dir / 'biosample_build.log')])
    archive = download_archive(args.output_dir / 'biosample_set.xml.gz', args.url,
                               force=args.force_download, skip=args.skip_download)
    convert_archive(archive, args.output_dir / 'biosample.parquet',
                    batch_size=args.batch_size, force=args.force_convert)


if __name__ == '__main__':
    main()
