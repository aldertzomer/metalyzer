"""Value-free record filtering telemetry, separate from classification outputs."""
from collections import Counter
import json
from pathlib import Path
from statistics import fmean, median

import pandas as pd

from .noisy_fields import PROFILE


def summarize(batch):
    audit = batch.record_audit
    result = dict(records=len(batch))
    if audit is None:
        return result
    for name in ['fields_considered', 'nonempty_eligible_fields', 'administrative_technical',
                 'redundant_original', 'before_truncated', 'after_truncated']:
        result[name] = int(audit[name].sum()) if len(audit) else 0
    result['records_with_omissions'] = int((audit.administrative_technical + audit.redundant_original).gt(0).sum()) if len(audit) else 0
    for name in ['before_length', 'after_length']:
        result[name+'_median'] = float(audit[name].median()) if len(audit) else 0.0
        result[name+'_mean'] = float(audit[name].mean()) if len(audit) else 0.0
    return result


def report(batch, scope):
    from .runlog import emit
    if batch.record_audit is not None:
        emit(f'Record rendering statistics ({scope}): ' + json.dumps(summarize(batch), sort_keys=True))


def nli_exposure(batch, sources, model):
    """Measure any-label pair exposure without loading models or changing tokens."""
    from .nli import HYPOTHESIS_TEMPLATE
    classifier = getattr(model, '_classifier', None)
    tokenizer = getattr(classifier, 'tokenizer', None)
    limit = getattr(tokenizer, 'model_max_length', None)
    if not isinstance(limit, int) or not 1 <= limit <= 100000:
        return dict(available=False)
    # These are lengths before the pipeline's pair truncation. With longest
    # hypothesis this identifies records exposed for at least one candidate.
    hypotheses = [HYPOTHESIS_TEMPLATE.format(label) for label in sources.labels]
    max_hypothesis = max(len(tokenizer(text, add_special_tokens=False)['input_ids']) for text in hypotheses)
    extra = tokenizer.num_special_tokens_to_add(pair=True)
    lengths = {text: len(tokenizer(text, add_special_tokens=False)['input_ids']) for text in set(batch.records)}
    exposed = sum(lengths[text] + max_hypothesis + extra > limit for text in batch.records)
    token_lengths = [lengths[text] for text in batch.records]
    return dict(available=True, model_token_limit=limit, records_any_candidate_exceeds_limit=int(exposed),
                fraction=exposed / len(batch) if len(batch) else 0.0,
                premise_token_median=float(median(token_lengths)) if token_lengths else 0.0,
                premise_token_mean=float(fmean(token_lengths)) if token_lengths else 0.0,
                premise_token_max=max(token_lengths, default=0),
                measurement='pre-truncation premise+hypothesis+pair-special-tokens; any candidate')


def write(batch, pending, sources, model, output, enabled):
    from .runlog import emit
    statistics = dict(enabled=enabled, profile=PROFILE if enabled else 'legacy',
                      all_records=summarize(batch), source_model_records=summarize(pending),
                      nli_token_exposure_all=nli_exposure(batch, sources, model),
                      nli_token_exposure_source_model=nli_exposure(pending, sources, model))
    Path(str(output)+'.filtering.json').write_text(json.dumps(statistics, indent=2)+'\n')
    rows = []
    for scope, selected in [('all_records', batch), ('source_model_records', pending)]:
        counter = Counter(item for omissions in selected.record_audit.omissions for item in omissions) if len(selected) else Counter()
        rows.extend(dict(scope=scope, reason=reason, field=field, count=count)
                    for (reason, field), count in sorted(counter.items()))
    pd.DataFrame(rows, columns=['scope', 'reason', 'field', 'count']).to_csv(str(output)+'.excluded_fields.tsv', sep='\t', index=False)
    emit('NLI token-limit exposure: ' + json.dumps({key: value for key, value in statistics.items() if key.startswith('nli_token')}))
    emit(f'Filtering audit: {output}.filtering.json; excluded fields: {output}.excluded_fields.tsv')
