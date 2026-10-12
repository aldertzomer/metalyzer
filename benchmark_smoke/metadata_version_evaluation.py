"""Validate the documented metadata-version exclusions; never alter classification."""


def normalized(value):
    return ' '.join(str(value).split()).casefold()


def validated_conflict_runs(truth_rows, metadata_rows, mapping_rows, conflicts,
                            *, biosample_rows=None, expected_count=44):
    """Whitelist exact run/accession pairs only after checking old/new source evidence."""
    try:
        from .benchmark_evaluation import unique_by_id
    except ImportError:
        from benchmark_evaluation import unique_by_id
    truth = unique_by_id(truth_rows, 'truth')
    metadata = unique_by_id(metadata_rows, 'benchmark metadata')
    mapping = unique_by_id(mapping_rows, 'accession mapping')
    documented = unique_by_id(conflicts, 'documented conflicts')
    if len(documented) != expected_count:
        raise ValueError(f'Expected exactly {expected_count} documented conflict pairs, got {len(documented)}')
    current = {}
    if biosample_rows is not None:
        for row in biosample_rows:
            key = row['accession']
            if key in current and (current[key].get('host'), current[key].get('ifsac_category')) != (row.get('host'), row.get('ifsac_category')):
                raise ValueError(f'Inconsistent current metadata for {key}')
            current[key] = row
    for run, record in documented.items():
        accession = record['accession']
        if (run not in truth or run not in metadata or run not in mapping
                or metadata[run]['sample_accession'] != accession
                or mapping[run]['accession'] != accession):
            raise ValueError(f'Documented conflict pair does not match inputs: {run}/{accession}')
        if (normalized(truth[run]['final_species']) != 'goat'
                or normalized(record['final_species']) != 'goat'
                or normalized(metadata[run].get('host_scientific_name', '')) != 'capra hircus'
                or normalized(record.get('tsv_metadata_host_scientific_name', '')) != 'capra hircus'
                or normalized(record.get('bio_metadata_host', '')) != 'gallus gallus'
                or normalized(record.get('bio_metadata_ifsac_category', '')) != 'chicken'):
            raise ValueError(f'Documented pair is not an old-goat/current-chicken conflict: {run}/{accession}')
        if biosample_rows is not None:
            live = current.get(accession)
            if (live is None or normalized(live.get('host', '')) != 'gallus gallus'
                    or normalized(live.get('ifsac_category', '')) != 'chicken'
                    or not str(live.get('last_update', '')).startswith('2025-06-18')):
                raise ValueError(f'Current BioSample no longer matches documented 18 June 2025 conflict: {run}/{accession}')
    return frozenset(documented)
