"""Local, bounded-memory BioSample lookup and lossless metadata flattening."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_STORE = Path('/mnt/sdd1/data/biosample')
PRIORITY = ('accession', 'host', 'host_scientific_name', 'host_tax_id',
            'isolation_source', 'sample_type', 'specimen', 'tissue', 'body_site',
            'sample_name', 'title', 'description', 'environmental_medium',
            'geo_loc_name', 'country', 'collection_date')


def nonempty_text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (date, datetime)):
        value = value.isoformat()
    text = str(value).strip()
    return text or None


def add_value(metadata, key, value):
    text = nonempty_text(value)
    if not key or text is None:
        return
    if key not in metadata:
        metadata[key] = text
    elif text not in metadata[key].split(' | '):
        metadata[key] += ' | ' + text


def flatten_value(metadata, key, value):
    if isinstance(value, Mapping):
        for subkey, subvalue in value.items():
            flatten_value(metadata, f'{key}.{subkey}' if key else str(subkey), subvalue)
    elif isinstance(value, (list, tuple)):
        if value and all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], str) for v in value):
            for subkey, subvalue in value:
                flatten_value(metadata, f'{key}.{subkey}' if key else subkey, subvalue)
        else:
            for item in value:
                flatten_value(metadata, key, item)
    else:
        add_value(metadata, key, value)


def canonical_from_parquet(row):
    """Adapt GenEpiO's canonical maps, protecting the structural accession."""
    metadata = {}
    for key, value in row.items():
        extracted = {}
        flatten_value(extracted, '' if key in {'attributes', 'identifiers', 'links'} else key, value)
        for name, text in extracted.items():
            if name == 'accession' and key != 'accession':
                name = f'{key}.accession'
            add_value(metadata, name, text)
    accession = nonempty_text(row.get('accession'))
    if accession is None:
        raise ValueError('BioSample row lacks structural accession')
    metadata['accession'] = accession
    return metadata


def store_files(path):
    path = Path(path).resolve()
    files = sorted(path.rglob('*.parquet')) if path.is_dir() else [path]
    if not files:
        raise ValueError(f'No BioSample Parquet files in {path}')
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    return files


def index_path(path):
    path = Path(path).resolve()
    if path.is_dir():
        files = store_files(path)
        if len(files) == 1:
            return files[0].with_suffix('.accessions.sqlite')
    return path / 'biosample.accessions.sqlite' if path.is_dir() else path.with_suffix('.accessions.sqlite')


def fingerprint(files):
    return json.dumps([(str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files])


def ensure_index(path, *, force=False, report=print):
    """Stream accession-only batches once; atomically publish a complete index."""
    files = store_files(path)
    target = index_path(path)
    signature = fingerprint(files)
    if target.exists() and not force:
        try:
            with sqlite3.connect(target.as_uri() + '?mode=ro', uri=True) as db:
                if db.execute('SELECT signature FROM metadata').fetchone() == (signature,):
                    report(f'BioSample accession index: {target}; index built/reused: reused')
                    return target
        except sqlite3.DatabaseError:
            pass
    report(f'Building BioSample accession index: {target}')
    fd, name = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    count = 0
    started = time.monotonic()
    try:
        with sqlite3.connect(temporary) as db:
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('PRAGMA synchronous=OFF')
            db.execute('CREATE TABLE metadata(signature TEXT NOT NULL)')
            db.execute('CREATE TABLE accessions(accession TEXT PRIMARY KEY, parquet_file TEXT NOT NULL, row_group INTEGER NOT NULL, row_index INTEGER NOT NULL) WITHOUT ROWID')
            for file in files:
                with pq.ParquetFile(file) as parquet:
                    if 'accession' not in parquet.schema_arrow.names:
                        raise ValueError(f'Missing structural accession column: {file}')
                    for group in range(parquet.num_row_groups):
                        offset = 0
                        for batch in parquet.iter_batches(batch_size=8192, columns=['accession'], row_groups=[group]):
                            values = batch.column(0).to_pylist()
                            db.executemany('INSERT INTO accessions VALUES (?,?,?,?)',
                                           ((a, str(file), group, offset + i) for i, a in enumerate(values) if a))
                            offset += len(values)
                            count += len(values)
                        if group % 100 == 0:
                            db.commit()
                            report(f'Indexed BioSamples: {count:,}; elapsed: {time.monotonic()-started:.1f}s')
            db.execute('INSERT INTO metadata VALUES (?)', (signature,))
            db.commit()
        if fingerprint(files) != signature:
            raise ValueError('BioSample store changed while indexing; retry with a stable store')
        temporary.replace(target)
    except sqlite3.IntegrityError as exc:
        raise ValueError('BioSample store contains duplicate structural accessions; accession lookup requires unique database records') from exc
    finally:
        temporary.unlink(missing_ok=True)
    report(f'BioSample accession index: {target}; index built/reused: built')
    return target


def load_biosamples(accessions: list[str], biosample_path: Path = DEFAULT_STORE) -> pd.DataFrame:
    from .runlog import emit
    started = time.monotonic()
    requested = [str(a).strip() for a in accessions]
    if not requested or any(not a for a in requested):
        raise ValueError('Provide nonblank BioSample accessions')
    emit('Input mode: BioSample')
    emit(f'BioSample store: {Path(biosample_path).resolve()}')
    emit(f'Requested BioSamples: {len(requested)}')
    target = ensure_index(biosample_path, report=emit)
    locations = {}
    missing = []
    with sqlite3.connect(target.as_uri() + '?mode=ro', uri=True) as db:
        for accession in dict.fromkeys(requested):
            row = db.execute('SELECT parquet_file,row_group,row_index FROM accessions WHERE accession=?', (accession,)).fetchone()
            if row is None:
                missing.append(accession)
            else:
                locations.setdefault((row[0], row[1]), {})[row[2]] = accession
    if missing:
        raise ValueError(f'Missing BioSample accessions ({len(missing)}): ' + ', '.join(missing[:20]) + (' ...' if len(missing) > 20 else ''))
    records = {}
    for (file, group), wanted in locations.items():
        with pq.ParquetFile(file) as parquet:
            offset = 0
            pending = dict(wanted)
            for batch in parquet.iter_batches(batch_size=1024, row_groups=[group]):
                selected = [i-offset for i in pending if offset <= i < offset + batch.num_rows]
                if selected:
                    for local, row in zip(selected, batch.take(selected).to_pylist()):
                        accession = pending.pop(offset + local)
                        record = canonical_from_parquet(row)
                        if record['accession'] != accession:
                            raise ValueError('BioSample index does not match database; rebuild the index')
                        records[accession] = record
                offset += batch.num_rows
                if not pending:
                    break
            if pending:
                raise ValueError('BioSample index row positions exceed database')
    frame = pd.DataFrame([records[a] for a in requested]).fillna('')
    columns = [c for c in PRIORITY if c in frame]
    frame = frame[columns + [c for c in frame if c not in columns]]
    emit(f'Resolved BioSamples: {len(frame)}')
    emit(f'BioSample lookup time: {time.monotonic()-started:.3f} seconds')
    emit(f'BioSample metadata columns: {len(frame.columns)}')
    return frame
