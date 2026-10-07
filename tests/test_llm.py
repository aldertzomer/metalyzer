import contextlib
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import MagicMock, patch

import pandas as pd

import metalyzer
from modules import country, records as record_module, deterministic_source, llm as llm_module, nli as nli_module, verification
from helpers import classify, llm_config
from modules.contracts import MetadataBatch, SourceResult, SourceVocabulary
from modules.deterministic_source import TaxonomySourceResult


CLI = ["--metadata", "input.tsv", "--sources", "sources.tsv", "--out", "out.tsv", "--id-col", "id"]
LABELS = ["turkey (poultry host including turkey meat)", "sheep (Ovis aries)", "other_animal (other hosts)"]
NAMES = ["turkey", "sheep", "other_animal"]


class LocalLLMTests(unittest.TestCase):
    def setUp(self):
        self.args = metalyzer.parse_args(CLI + ["--method", "llm", "--device", "-1"])
        self.console = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.console))

    def mocked_hf(self, responses, *, no_pad=False):
        # Tiny in-memory tensors exercise prompt slicing without model packages.
        import numpy as np

        class Tensor:
            def __init__(self, values):
                self.values = np.asarray(values)

            @property
            def shape(self):
                return self.values.shape

            def to(self, device):
                return self

            def __getitem__(self, key):
                return Tensor(self.values[key])

            def tolist(self):
                return self.values.tolist()

        inference = [False]
        class InferenceMode:
            def __enter__(self):
                inference[0] = True

            def __exit__(self, *args):
                inference[0] = False

        torch = SimpleNamespace(
            tensor=Tensor, full=lambda shape, value: Tensor(np.full(shape, value)),
            cat=lambda tensors, dim: Tensor(np.concatenate([x.values for x in tensors], axis=dim)),
            is_tensor=lambda value: isinstance(value, Tensor),
            inference_mode=InferenceMode, is_inference_mode_enabled=lambda: inference[0],
            device=lambda value: value,
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        )
        self.enterContext(patch.dict('sys.modules', {'torch': torch}))

        class BatchEncoding(dict):
            def to(self, device):
                return BatchEncoding({key: value.to(device) for key, value in self.items()})

        tokenizer = MagicMock()
        tokenizer.pad_token_id = None if no_pad else 0
        tokenizer.eos_token_id = 9
        tokenizer.all_special_ids = [0, 9]
        tokenizer.eos_token = "<eos>"
        tokenizer.apply_chat_template.side_effect = lambda conversations, **kw: BatchEncoding({
            "input_ids": torch.tensor([[0, 0, 11, 12], [21, 22, 23, 24]][:len(conversations)]),
            "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]][:len(conversations)]),
        })
        model = MagicMock()
        model.generation_config.max_length = 262144
        remaining = iter(responses)
        response_by_token = {}
        def generate(**kw):
            self.assertTrue(torch.is_inference_mode_enabled())
            count = kw["input_ids"].shape[0]
            tokens = []
            for _ in range(count):
                token = 77 + len(response_by_token)
                response_by_token[token] = next(remaining)
                tokens.append([token, token])
            return SimpleNamespace(
                sequences=torch.cat([kw["input_ids"], torch.tensor(tokens)], dim=1),
                scores=(object(), object()),
            )
        model.generate.side_effect = generate
        model.compute_transition_scores.side_effect = lambda sequences, scores, normalize_logits: torch.full(
            (sequences.shape[0], 2), math.log(0.8),
        )
        def decode(tokens, **kw):
            self.assertEqual(kw, {"skip_special_tokens": True})
            self.assertEqual(tokens, [tokens[0]] * len(tokens))
            return response_by_token[tokens[0]] if len(tokens) == 2 else ""
        tokenizer.decode.side_effect = decode
        tok_factory = MagicMock(return_value=tokenizer)
        model_factory = MagicMock(return_value=model)
        # Mock the module boundary too: offline tests must not initialize any
        # Transformers lazy imports or optional vision/audio dependencies.
        self.enterContext(patch.dict('sys.modules', {'transformers': SimpleNamespace(
            MistralCommonBackend=SimpleNamespace(from_pretrained=tok_factory),
            Mistral3ForConditionalGeneration=SimpleNamespace(from_pretrained=model_factory),
        )}))
        nli = self.enterContext(patch.object(nli_module, "pipeline", side_effect=AssertionError("NLI loaded in LLM mode")))
        return SimpleNamespace(tokenizer=tokenizer, model=model, tok_factory=tok_factory,
                               model_factory=model_factory, nli=nli)

    def classify(self, records, taxonomy=None, df=None):
        if df is None:
            df = pd.DataFrame({"id": range(len(records))})
        return classify(df, records, LABELS, self.args, taxonomy)

    def test_cli_defaults_and_validation(self):
        args = metalyzer.parse_args(CLI)
        self.assertEqual(args.method, "nli")
        self.assertEqual(args.llm_model, "mistralai/Ministral-3-8B-Instruct-2512")
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.llm_batch_size, 1)
        self.assertEqual(args.llm_max_new_tokens, 16)
        self.assertEqual(args.llm_min_score, 0.75)
        self.assertIsNone(args.llm_revision)
        self.assertIsNone(args.limit)
        for option, value in [("--device", "-2"), ("--batch-size", "0"),
                              ("--llm-batch-size", "0"), ("--llm-max-new-tokens", "0"),
                              ("--limit", "0"), ("--limit", "-1"), ("--min-score", "1.1"),
                              ("--llm-min-score", "-0.01"), ("--llm-min-score", "1.01"),
                              ("--method", "api")]:
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    metalyzer.parse_args(CLI + [option, value])
                self.assertEqual(error.exception.code, 2)

    def test_response_parser(self):
        valid = {'turkey': 'turkey', '"turkey"': 'turkey', "'turkey'": 'turkey',
                 '`turkey`': 'turkey', ' TURKEY ': 'turkey', '{"source":"turkey"}': 'turkey',
                 'other animal': 'other_animal', 'other-animal': 'other_animal',
                 'other_animal': 'other_animal', 'unknown': 'unknown'}
        for response, expected in valid.items():
            with self.subTest(response=response):
                self.assertEqual(llm_module.parse_llm_response(response, NAMES), (expected, ""))
        for response in ['I think turkey is the likely source.', 'turkey or sheep', '', 'turkey.',
                         '{"source":12}', '{"source":"turkey","why":"food"}', '{broken',
                         '["turkey"]', '```turkey```', 'turkey\nexplanation', 'not_in_sources']:
            with self.subTest(response=response):
                result, evidence = llm_module.parse_llm_response(response, NAMES)
                self.assertEqual(result, 'unknown')
                self.assertTrue(evidence.startswith('invalid_llm_output='))

    def test_ambiguous_normalization_and_exact_labels(self):
        labels = ['other-animal', 'other_animal', 'Turkey', 'turkey']
        for response in ['other animal', 'TURKEY']:
            self.assertEqual(llm_module.parse_llm_response(response, labels)[0], 'unknown')
        for response in labels:
            self.assertEqual(llm_module.parse_llm_response(response, labels), (response, ''))

    def test_invalid_evidence_sanitized_and_truncated(self):
        label, evidence = llm_module.parse_llm_response('why\t\n\r\x00\u202e' + 'x' * 300, NAMES)
        self.assertEqual(label, 'unknown')
        self.assertLessEqual(len(evidence), len('invalid_llm_output=') + 200)
        self.assertFalse(any(ch in evidence for ch in '\t\n\r\x00\u202e'))

    def test_generation_batches_and_model_loading(self):
        hf = self.mocked_hf(['turkey', 'other animal', 'unknown'])
        self.args.llm_batch_size = 2
        self.args.llm_revision = 'pinned-revision'
        self.args.llm_model = 'another/compatible-instruct'
        result = self.classify(['short record', 'longer sample metadata record', 'third'])
        self.assertEqual(result.best_hit.tolist(), ['turkey', 'other_animal', 'unknown'])
        self.assertTrue(result[NAMES].isna().all().all())
        self.assertNotIn('unknown', result.columns)
        self.assertEqual(result.source_method.tolist(), ['llm'] * 3)
        self.assertEqual(result.source_evidence.tolist(), [''] * 3)
        self.assertEqual(result.columns.tolist(), [*NAMES, 'best_hit', 'source_method',
                                                  'source_evidence', 'source_llm_score'])
        self.assertEqual(result.source_llm_score.tolist(), [0.8] * 3)
        self.assertTrue(result.source_llm_score.between(0, 1).all())
        self.assertEqual(hf.model.compute_transition_scores.call_count, 2)
        for call in hf.model.compute_transition_scores.call_args_list:
            self.assertTrue(call.kwargs['normalize_logits'])
        hf.nli.assert_not_called()
        hf.tok_factory.assert_called_once_with(self.args.llm_model, revision='pinned-revision')
        hf.model_factory.assert_called_once_with(self.args.llm_model, revision='pinned-revision',
                                                 dtype='auto', low_cpu_mem_usage=True)
        self.assertEqual(str(hf.model.to.call_args.args[0]), 'cpu')
        hf.model.eval.assert_called_once()
        self.assertIsNone(hf.model.generation_config.max_length)
        self.assertEqual(hf.tokenizer.padding_side, 'left')
        self.assertEqual(hf.model.generate.call_count, 2)
        for call in hf.model.generate.call_args_list:
            self.assertFalse(call.kwargs['do_sample'])
            self.assertTrue(call.kwargs['return_dict_in_generate'])
            self.assertTrue(call.kwargs['output_scores'])
            self.assertEqual(call.kwargs['max_new_tokens'], 16)
            self.assertEqual(call.kwargs['pad_token_id'], 0)
        for call in hf.tokenizer.apply_chat_template.call_args_list:
            self.assertEqual(call.kwargs, dict(tokenize=True, padding=True, return_tensors='pt', return_dict=True))
        self.assertEqual(self.console.getvalue().count('LLM model:'), 1)

    def test_mixed_taxonomy_invalid_output_and_order(self):
        hf = self.mocked_hf(['"turkey"', 'I think turkey is the likely source.'])
        self.args.llm_batch_size = 2
        taxonomy = MagicMock()
        tax_result = TaxonomySourceResult('sheep', 9940, 'Ovis aries')
        taxonomy.classify_host_taxid.side_effect = [None, tax_result, None]
        df = pd.DataFrame({'id': ['a', 'b', 'c'], 'host_tax_id': ['', '9940', '']}, index=[7, 8, 9])
        result = self.classify(['first', 'taxonomy-only record', 'third'], taxonomy, df)
        self.assertEqual(taxonomy.classify_host_taxid.call_count, 3)
        self.assertEqual(result.index.tolist(), [7, 8, 9])
        self.assertEqual(result.best_hit.tolist(), ['turkey', 'sheep', 'unknown'])
        self.assertEqual(result.source_method.tolist(), ['llm', 'host_tax_id', 'llm'])
        self.assertEqual(result.source_evidence.iloc[1], tax_result.evidence)
        self.assertTrue(result.source_evidence.iloc[2].startswith('invalid_llm_output='))
        self.assertTrue(pd.isna(result.source_llm_score.iloc[0]))
        self.assertTrue(pd.isna(result.source_llm_score.iloc[1]))
        self.assertTrue(pd.isna(result.source_llm_score.iloc[2]))
        self.assertTrue(result[NAMES].isna().all().all())
        self.assertEqual(hf.tokenizer.apply_chat_template.call_count, 1)
        self.assertNotIn('taxonomy-only record', str(hf.tokenizer.apply_chat_template.call_args_list))
        self.assertIn('invalid LLM outputs:  1', self.console.getvalue())

    def test_no_model_for_empty_or_all_taxonomy_both_methods(self):
        self.args.llm_min_score = 1.0
        for method in ('nli', 'llm'):
            for count in (0, 2):
                with self.subTest(method=method, count=count):
                    self.args.method = method
                    taxonomy = MagicMock()
                    taxonomy.classify_host_taxid.return_value = TaxonomySourceResult('sheep', 9940, 'Ovis aries')
                    with patch.object(nli_module, 'pipeline') as nli, patch.object(llm_module, 'load_local_llm') as llm:
                        result = self.classify(['sheep'] * count, taxonomy)
                    nli.assert_not_called()
                    llm.assert_not_called()
                    self.assertEqual(len(result), count)
                    self.assertTrue(result[NAMES].isna().all().all())

    def test_nli_regression_no_causal_model(self):
        self.args.method = 'nli'
        self.args.llm_min_score = 1.0
        self.args.batch_size = 2
        with patch.object(nli_module, 'pipeline') as nli, patch.object(llm_module, 'load_local_llm') as llm:
            nli.return_value.return_value = [
                {'labels': LABELS[::-1], 'scores': [0.1, 0.2, 0.7]},
                {'labels': LABELS, 'scores': [0.1, 0.1, 0.1]},
            ]
            result = self.classify(['a', 'b'])
        llm.assert_not_called()
        nli.assert_called_once_with('zero-shot-classification',
            model='MoritzLaurer/deberta-v3-large-zeroshot-v2.0', device=-1, dtype='float32')
        nli.return_value.assert_called_once_with(['a', 'b'], candidate_labels=LABELS,
            hypothesis_template='The biological host or environmental source of this sample is {}.',
            multi_label=False, batch_size=2)
        expected = nli_module.parse_source_scores(pd.DataFrame([[0.7, 0.2, 0.1], [0.1, 0.1, 0.1]], columns=LABELS), 'id', 0.2)
        pd.testing.assert_frame_equal(result[expected.columns], expected)
        self.assertEqual(result.source_evidence.tolist(), ['', ''])
        self.assertEqual(result.source_method.tolist(), ['nli', 'nli'])
        self.assertTrue(result.source_llm_score.isna().all())

    def test_generated_label_score_uses_only_label_tokens(self):
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=9, all_special_ids=[0, 9])
        fragments = {1: '{"source":"', 2: 'other', 3: '_', 4: 'animal', 5: '"}', 9: ''}
        tokenizer.decode = lambda ids, **kw: ''.join(fragments[part] for part in ids)
        ids = [1, 2, 3, 4, 5, 9]
        response = tokenizer.decode(ids, skip_special_tokens=True)
        score = llm_module.generated_label_score(tokenizer, ids,
                    [math.log(p) for p in (0.01, 0.91, 0.87, 0.95, 0.02, 0.001)],
                    response, 'other_animal')
        self.assertAlmostEqual(score, (0.91 * 0.87 * 0.95) ** (1 / 3))
        self.assertTrue(math.isnan(llm_module.generated_label_score(
            tokenizer, [1, 9], [math.log(0.8), math.log(0.8)],
            '{"source":"other_animal"}', 'other_animal')))

    def test_quoted_label_score_excludes_quotes_and_eos(self):
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=9, all_special_ids=[0, 9])
        fragments = {1: '"', 2: 'tur', 3: 'key', 4: '"', 9: ''}
        tokenizer.decode = lambda ids, **kw: ''.join(fragments[part] for part in ids)
        ids = [1, 2, 3, 4, 9]
        score = llm_module.generated_label_score(tokenizer, ids,
                    [math.log(p) for p in (0.01, 0.91, 0.87, 0.02, 0.001)],
                    tokenizer.decode(ids, skip_special_tokens=True), 'turkey')
        self.assertAlmostEqual(score, (0.91 * 0.87) ** 0.5)

    def test_min_score_does_not_change_llm_calls(self):
        self.mocked_hf(['turkey', 'turkey'])
        for threshold in (0.0, 1.0):
            self.args.min_score = threshold
            self.assertEqual(self.classify(['record']).best_hit.tolist(), ['turkey'])
        self.assertIn('--min-score is not applicable', self.console.getvalue())

    def test_llm_confidence_cutoff_preserves_score_and_rejected_label(self):
        hf = self.mocked_hf(['turkey'] * 5 + ['unknown', 'bad response'])
        with patch.object(llm_module, 'generated_label_score',
                          side_effect=[0.75, 0.93, 0.43, 0.43, 0.0, 0.2]) as score:
            first = self.classify(['record'] * 4)
            self.args.llm_min_score = 0
            disabled = self.classify(['record'])
            last = self.classify(['record'] * 2)
        self.assertEqual(first.best_hit.tolist(), ['turkey', 'turkey', 'unknown', 'unknown'])
        self.assertEqual(first.source_llm_score.tolist(), [0.75, 0.93, 0.43, 0.43])
        self.assertEqual(first.source_evidence.tolist(), ['', '', 'llm_low_score=turkey', 'llm_low_score=turkey'])
        self.assertEqual(disabled.best_hit.tolist(), ['turkey'])
        self.assertEqual(disabled.source_llm_score.tolist(), [0.0])
        self.assertEqual(last.best_hit.tolist(), ['unknown', 'unknown'])
        self.assertEqual(last.source_evidence.iloc[0], '')
        self.assertEqual(last.source_llm_score.iloc[0], 0.2)
        self.assertTrue(last.source_evidence.iloc[1].startswith('invalid_llm_output='))
        self.assertTrue(pd.isna(last.source_llm_score.iloc[1]))
        self.assertEqual(score.call_count, 6)
        self.assertEqual(hf.model.generate.call_count, 7)

    def test_cli_logs_llm_threshold_and_filter_totals(self):
        self.mocked_hf(['turkey', 'sheep', 'unknown'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'id': ['a', 'b', 'c'], 'host': ['first', 'second', 'third']}).to_csv(
                root / 'input.tsv', sep='\t', index=False)
            pd.DataFrame({'source': LABELS}).to_csv(root / 'sources.tsv', sep='\t', index=False)
            with patch.object(deterministic_source, 'load_taxonomy', return_value=None), \
                 patch.object(llm_module, 'generated_label_score', side_effect=[0.43, 0.93, 0.2]):
                metalyzer.main(['--metadata', str(root / 'input.tsv'), '--sources', str(root / 'sources.tsv'),
                                '--out', str(root / 'out.tsv'), '--id-col', 'id', '--method', 'llm',
                                '--device', '-1', '--disable-verify-source'])
            out = pd.read_csv(root / 'out.tsv', sep='\t', keep_default_na=False)
            log = (root / 'out.tsv.log').read_text(encoding='utf-8')
        self.assertEqual(out.best_hit.tolist(), ['unknown', 'sheep', 'unknown'])
        self.assertEqual(out.source_evidence.tolist(), ['llm_low_score=turkey', '', ''])
        self.assertEqual(out.source_llm_score.tolist(), [0.43, 0.93, 0.2])
        self.assertEqual(out.nli_verification_score.tolist(), ['NA'] * 3)
        self.assertIn('LLM minimum generation score: 0.75', log)
        self.assertIn('llm_min_score: 0.75', log)
        self.assertIn('LLM assignments before confidence filtering: 2', log)
        self.assertIn('LLM low-score calls converted to unknown: 1', log)
        self.assertIn('LLM assignments retained after confidence filtering: 1', log)

    def test_low_score_unknown_skips_nli_verification(self):
        self.mocked_hf(['turkey'])
        with patch.object(llm_module, 'generated_label_score', return_value=0.43):
            table = self.classify(['record'])
        batch = MetadataBatch(pd.DataFrame({'id': [0]}), pd.Series(['record']))
        sources = SourceVocabulary(tuple(LABELS), 'id')
        with patch.object(nli_module, 'pipeline') as nli:
            scores = verification.run(batch, sources, SourceResult(table)).values
        nli.assert_not_called()
        self.assertTrue(scores.isna().all())

    def test_eos_padding_and_revision_omitted(self):
        hf = self.mocked_hf([], no_pad=True)
        llm_module.load_local_llm(llm_config(self.args))
        self.assertEqual(hf.tokenizer.pad_token, hf.tokenizer.eos_token)
        hf.tok_factory.assert_called_once_with(llm_module.DEFAULT_LLM_MODEL)
        hf.model_factory.assert_called_once_with(llm_module.DEFAULT_LLM_MODEL, dtype='auto', low_cpu_mem_usage=True)

    def test_missing_padding_and_eos_rejected(self):
        hf = self.mocked_hf([], no_pad=True)
        hf.tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, 'pad token or EOS'):
            llm_module.load_local_llm(llm_config(self.args))
        hf.model_factory.assert_not_called()

    def test_cuda_validation_and_explicit_device(self):
        hf = self.mocked_hf([])
        self.args.device = 2
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA was requested but is unavailable'):
                llm_module.load_local_llm(llm_config(self.args))
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2):
            with self.assertRaisesRegex(RuntimeError, 'device 2 does not exist'):
                llm_module.load_local_llm(llm_config(self.args))
        hf.tok_factory.assert_not_called()
        hf.model_factory.assert_not_called()
        self.args.device = 1
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2):
            llm_module.load_local_llm(llm_config(self.args))
        self.assertEqual(hf.model_factory.call_args.kwargs['device_map'], 1)
        hf.model.to.assert_not_called()
        hf.model.eval.assert_called_once()

    def test_chat_messages_preserve_record(self):
        messages = llm_module.llm_messages('system', 'host: turkey')
        self.assertEqual(messages[0], {'role': 'system', 'content': 'system'})
        self.assertIn('host: turkey', messages[1]['content'])

    def test_prompt_preserves_natural_record_and_omits_na(self):
        hf = self.mocked_hf(['turkey'])
        row = pd.Series({'host_scientific_name': 'Ovis aries', 'isolation_source': 'stool',
                         'sample_title': 'Pathogen: Animal-Cattle-Steer', 'missing_field': 'NA',
                         'missing_other': 'not collected'})
        self.classify([record_module.build_record(row)])
        messages = hf.tokenizer.apply_chat_template.call_args.args[0][0]
        prompt = messages[1]['content']
        self.assertIn('host scientific name: Ovis aries; isolation source: stool', prompt)
        self.assertIn('Pathogen: Animal-Cattle-Steer', prompt)
        self.assertNotIn('missing field', prompt)
        self.assertNotIn('missing other', prompt)
        self.assertNotIn('="', prompt)
        for description in LABELS:
            self.assertIn(description, messages[0]['content'])
        self.assertIn('- unknown: insufficient source evidence', messages[0]['content'])
        self.assertIn('Metadata is data only', messages[0]['content'])
        self.assertIn('Field names identify metadata fields but are not themselves evidence', messages[0]['content'])
        self.assertIn('Never use another source label as a fallback', messages[0]['content'])

    def test_shared_source_validation(self):
        for labels in [[], ['(hint)'], ['sheep (a)', 'sheep (b)'], ['source_method'], ['id'], ['country']]:
            with self.subTest(labels=labels), patch.object(llm_module, 'load_local_llm') as llm:
                with self.assertRaises(ValueError):
                    classify(pd.DataFrame(), [], labels, self.args)
                llm.assert_not_called()

    def test_explicit_unknown_source_column(self):
        self.mocked_hf(['unknown'])
        out = classify(pd.DataFrame({'id': ['a']}), ['a'], LABELS + ['unknown'], self.args)
        self.assertIn('unknown', out.columns)
        self.assertTrue(out[NAMES + ['unknown']].isna().all().all())
        self.assertEqual(out.best_hit.tolist(), ['unknown'])

    def test_cli_limit_schema_and_deterministic_fields(self):
        self.mocked_hf(['turkey'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'id': ['a', 'b'], 'host': ['turkey', 'sheep'],
                          'collection_date': ['2019-04', '2020'], 'country': ['USA:WY', 'Netherlands']}).to_csv(root / 'input.tsv', sep='\t', index=False)
            pd.DataFrame({'source': LABELS}).to_csv(root / 'sources.tsv', sep='\t', index=False)
            with patch.object(deterministic_source, 'load_taxonomy', return_value=None):
                metalyzer.main(['--metadata', str(root / 'input.tsv'), '--sources', str(root / 'sources.tsv'),
                                '--out', str(root / 'out.tsv'), '--id-col', 'id', '--method', 'llm',
                                '--device', '-1', '--limit', '1', '--disable-verify-source'])
            out = pd.read_csv(root / 'out.tsv', sep='\t', keep_default_na=False, dtype=str)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.id.tolist(), ['a'])
        self.assertTrue((out[NAMES] == 'NA').all().all())
        self.assertEqual(out.year.tolist(), ['2019'])
        self.assertEqual(out.country.tolist(), ['United States'])
        self.assertEqual(out.source_method.tolist(), ['llm'])
        self.assertEqual(out.source_llm_score.tolist(), ['0.8'])
        self.assertEqual(out.nli_verification_score.tolist(), ['NA'])
        self.assertNotIn('source_verification_score', out.columns)

    def test_taxonomy_unavailable_label_falls_back_to_llm(self):
        self.mocked_hf(['unknown'])
        taxonomy = MagicMock()
        taxonomy.classify_host_taxid.return_value = TaxonomySourceResult('cat', 9685, 'Felis catus')
        out = self.classify(['unlisted host'], taxonomy)
        self.assertEqual(out.best_hit.tolist(), ['unknown'])
        self.assertEqual(out.source_method.tolist(), ['llm'])

    def test_empty_cli_preserves_schema_and_method_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(columns=['id', 'host']).to_csv(root / 'input.tsv', sep='\t', index=False)
            pd.DataFrame({'source': LABELS}).to_csv(root / 'sources.tsv', sep='\t', index=False)
            for method in ('nli', 'llm'):
                with patch.object(nli_module, 'pipeline') as nli, patch.object(llm_module, 'load_local_llm') as llm:
                    with warnings.catch_warnings(record=True) as seen:
                        warnings.simplefilter('always')
                        metalyzer.main(['--metadata', str(root / 'input.tsv'), '--sources', str(root / 'sources.tsv'),
                                        '--out', str(root / 'out.tsv'), '--id-col', 'id', '--method', method])
                    self.assertIn(f'{method.upper()}-only source classification', str(seen[0].message))
                    nli.assert_not_called()
                    llm.assert_not_called()
                out = pd.read_csv(root / 'out.tsv', sep='\t')
                self.assertEqual(len(out), 0)
                self.assertEqual(out.columns.tolist(), ['id', *NAMES, 'best_hit', 'source_method',
                                            'source_evidence', 'source_llm_score',
                                            'nli_verification_score', 'year', 'country'])


if __name__ == '__main__':
    unittest.main()
