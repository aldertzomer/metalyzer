"""Strict accession alignment shared by source evaluation and SVG rendering."""
import csv
from pathlib import Path


def read_tsv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def unique_by_id(rows, name):
    keyed = {}
    for row in rows:
        key = row['run_accession']
        if not key or key in keyed:
            raise ValueError(f'Blank or duplicate run accession in {name}: {key!r}')
        keyed[key] = row
    return keyed


def align_predictions(truth_rows, prediction_rows, mapping_rows=None, *, allow_subset=False):
    """Return predictions keyed by sequencing run, never by truth-file position.

    BioSample IDs can repeat across runs. Require the expected request sequence,
    then associate each occurrence with its corresponding mapped sequencing run.
    """
    truth = unique_by_id(truth_rows, 'truth')
    if prediction_rows and 'run_accession' not in prediction_rows[0]:
        if mapping_rows is None:
            raise ValueError('BioSample predictions require an accession mapping')
        unique_by_id(mapping_rows, 'accession mapping')
        if [r['accession'] for r in prediction_rows] != [r['accession'] for r in mapping_rows]:
            raise ValueError('BioSample predictions differ from mapped accession order or count')
        prediction_rows = [dict(prediction, run_accession=mapping['run_accession'])
                           for prediction, mapping in zip(prediction_rows, mapping_rows)]
    predicted = unique_by_id(prediction_rows, 'predictions')
    if not set(predicted) <= set(truth) or (not allow_subset and set(predicted) != set(truth)):
        raise ValueError('Truth and predictions have different run accession sets')
    return truth, predicted
