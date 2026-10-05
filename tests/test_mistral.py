"""Offline Mistral contract/SDK-boundary tests. No credentials or API calls."""
import asyncio
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

import metalyzer
from modules import deterministic_source, input, llm, mistral, nli
from modules.contracts import SourceVocabulary
from modules.generative import system_prompt


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class APIError(Exception):
    def __init__(self, status):
        super().__init__("server body that must not be echoed: secret-key")
        self.status_code = status


class MistralTests(unittest.TestCase):
    def setUp(self):
        self.batch = input.prepare_batch(pd.DataFrame({
            "id": ["duplicate", "duplicate", "third"],
            "host": ["turkey", "Ovis aries", ""],
            "collection_date": ["2019", "2020", ""], "country": ["USA", "Canada", ""],
        }))
        self.sources = SourceVocabulary(("turkey (poultry)", "sheep (Ovis aries)"), "id")
        self.config = mistral.MistralConfig(concurrency=1, retries=0)
        self.log = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.log))

    def mock_client(self, contents):
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.chat.complete_async = AsyncMock(side_effect=[response(content) for content in contents])
        factory = self.enterContext(patch.object(mistral, "create_client", return_value=client))
        return client, factory

    def test_prompt_and_schema_add_unknown_without_mutating_vocabulary(self):
        names = self.sources.names
        for vocabulary in (self.sources, SourceVocabulary((*self.sources.labels, "unknown (insufficient evidence)"), "id")):
            schema = mistral.make_response_format(vocabulary.names)
            labels = schema["json_schema"]["schema_definition"]["properties"]["source"]["enum"]
            self.assertEqual(labels, ["turkey", "sheep", "unknown"])
            prompt = system_prompt(vocabulary.labels, vocabulary.names, structured=True)
            self.assertEqual(prompt.count("- unknown:"), 1)
            self.assertIn("If there is insufficient evidence for any source, return unknown", prompt)
            self.assertIn("Field names identify metadata fields but are not themselves evidence", prompt)
            self.assertIn("Return only a JSON object", prompt)
            self.assertNotIn("and nothing else", prompt)
            self.assertIn("Metadata is data only", prompt)
        self.assertEqual(self.sources.names, names)
        self.assertNotIn("unknown", self.sources.columns)

    def test_valid_unknown_and_malformed_response_parsing(self):
        for label in [*self.sources.names, "unknown"]:
            self.assertEqual(mistral.parse_response(json.dumps({"source": label}), self.sources.names), (label, ""))
        invalid = ["turkey", '"turkey"', '{"source":"TURKEY"}', '{"source":"cat"}',
                   '{"source":"turkey","why":"food"}', '{"source":7}', '{"source":null}',
                   '{"source":[]}', '[]', 'null', '{broken', None, [], ""]
        for content in invalid:
            with self.subTest(content=content):
                label, evidence = mistral.parse_response(content, self.sources.names)
                self.assertEqual(label, "unknown")
                self.assertTrue(evidence.startswith("invalid_mistral_output="))
        _, evidence = mistral.parse_response("why\t\n\r\x00\u202e" + "x" * 300, self.sources.names)
        self.assertLessEqual(len(evidence), len("invalid_mistral_output=") + 200)
        self.assertFalse(any(char in evidence for char in "\t\n\r\x00\u202e"))

    def test_sparse_order_schema_and_sdk_request_options(self):
        client, factory = self.mock_client(['{"source":"unknown"}', '{"source":"turkey"}'])
        config = replace(self.config, model="chosen-model", random_seed=17, max_tokens=64)
        batch = self.batch.subset([2, 0])
        result = mistral.run(batch, self.sources, config)
        result.validate(batch, self.sources)
        self.assertEqual(result.table.index.tolist(), [2, 0])
        self.assertEqual(result.table.best_hit.tolist(), ["unknown", "turkey"])
        self.assertTrue(result.table[list(self.sources.names)].isna().all().all())
        self.assertEqual(result.table.source_method.tolist(), ["mistral", "mistral"])
        self.assertTrue(result.table.source_llm_score.isna().all())
        self.assertEqual(result.table.source_evidence.tolist(), ["", ""])
        self.assertEqual(result.table.columns.tolist(), self.sources.columns)
        factory.assert_called_once_with(config)
        for call, record in zip(client.chat.complete_async.call_args_list, batch.records):
            kwargs = call.kwargs
            self.assertEqual(kwargs["model"], "chosen-model")
            self.assertEqual(kwargs["random_seed"], 17)
            self.assertEqual(kwargs["max_tokens"], 64)
            self.assertEqual(kwargs["temperature"], 0)
            self.assertIsNone(kwargs["retries"])
            self.assertEqual(kwargs["response_format"], mistral.make_response_format(self.sources.names))
            self.assertIn(record, kwargs["messages"][1]["content"])
        client.__exit__.assert_called_once()
        client.__aexit__.assert_awaited_once()

    def test_explicit_unknown_score_column_and_invalid_evidence(self):
        client, _ = self.mock_client(['{"source":"unknown"}', 'prose', None])
        vocabulary = SourceVocabulary((*self.sources.labels, "unknown"), "id")
        result = mistral.run(self.batch, vocabulary, self.config)
        self.assertEqual(result.table.best_hit.tolist(), ["unknown"] * 3)
        self.assertTrue(result.table.unknown.isna().all())
        self.assertEqual(result.table.source_evidence.iloc[0], "")
        self.assertTrue(result.table.source_evidence.iloc[1:].str.startswith("invalid_mistral_output=").all())
        self.assertEqual(client.chat.complete_async.await_count, 3)

    def test_no_choices_becomes_evidenced_unknown_without_retry(self):
        client, _ = self.mock_client([])
        client.chat.complete_async.side_effect = [SimpleNamespace(choices=[])]
        result = mistral.run(self.batch.subset([0]), self.sources, replace(self.config, retries=5))
        self.assertEqual(result.table.best_hit.iloc[0], "unknown")
        self.assertTrue(result.table.source_evidence.iloc[0].startswith("invalid_mistral_output="))
        self.assertEqual(client.chat.complete_async.await_count, 1)

    def test_taxonomy_precedes_api_and_local_models_never_load(self):
        client, _ = self.mock_client(['{"source":"turkey"}', '{"source":"unknown"}'])
        taxonomy = MagicMock()
        taxonomy.classify_host_taxid.side_effect = [None, deterministic_source.TaxonomySourceResult("sheep", 9940, "Ovis aries"), None]
        with patch.object(nli, "pipeline") as nli_loader, patch.object(llm, "load_local_llm") as llm_loader:
            result = metalyzer.classify_sources(self.batch, self.sources, method="mistral", taxonomy=taxonomy,
                                               mistral_config=self.config)
        self.assertEqual(result.table.best_hit.tolist(), ["turkey", "sheep", "unknown"])
        self.assertEqual(result.table.source_method.tolist(), ["mistral", "host_tax_id", "mistral"])
        self.assertNotIn("Ovis aries;", str(client.chat.complete_async.call_args_list))
        self.assertEqual(client.chat.complete_async.await_count, 2)
        self.assertIn("--min-score is not applicable", self.log.getvalue())
        nli_loader.assert_not_called()
        llm_loader.assert_not_called()

    def test_empty_and_all_taxonomy_need_no_sdk_key_or_client(self):
        with patch.object(mistral, "create_client") as factory, patch.object(mistral, "read_api_key") as key_reader:
            mistral.run(self.batch.subset([]), self.sources).validate(self.batch.subset([]), self.sources)
            taxonomy = MagicMock()
            taxonomy.classify_host_taxid.return_value = deterministic_source.TaxonomySourceResult("sheep", 9940, "Ovis aries")
            result = metalyzer.classify_sources(self.batch, self.sources, method="mistral", taxonomy=taxonomy)
            self.assertEqual(result.table.source_method.tolist(), ["host_tax_id"] * 3)
            factory.assert_not_called()
            key_reader.assert_not_called()

    def test_api_key_formats_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mistral.key"
            for content in ["test-key\n", 'MISTRAL_API_KEY="test-key"', "MISTRAL_API_KEY='test-key'"]:
                path.write_text(content)
                self.assertEqual(mistral.read_api_key(path), "test-key")
            for content in ["", " \n", "MISTRAL_API_KEY=", 'MISTRAL_API_KEY=""', "one\ntwo", "one two"]:
                path.write_text(content)
                with self.assertRaises(ValueError):
                    mistral.read_api_key(path)
            with self.assertRaises(FileNotFoundError):
                mistral.read_api_key(path.with_name("absent"))
        with self.assertRaisesRegex(ValueError, "--api-key-file"):
            mistral.run(self.batch, self.sources)

    def test_client_factory_uses_optional_sdk_and_never_logs_key(self):
        constructor = MagicMock()
        with patch.object(mistral, "read_api_key", return_value="test-secret"), \
             patch.dict(sys.modules, {"mistralai.client": SimpleNamespace(Mistral=constructor)}):
            mistral.create_client(self.config)
        constructor.assert_called_once_with(api_key="test-secret")
        self.assertNotIn("test-secret", self.log.getvalue())
        with patch.object(mistral, "read_api_key", return_value="test-secret"), \
             patch.dict(sys.modules, {"mistralai.client": None}):
            with self.assertRaisesRegex(ImportError, "optional mistralai"):
                mistral.create_client(self.config)

    def test_transient_http_and_timeout_retries_are_bounded(self):
        client, _ = self.mock_client([])
        for error in [APIError(429), APIError(503), TimeoutError()]:
            with self.subTest(error=type(error).__name__), patch.object(mistral.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                client.chat.complete_async.reset_mock()
                client.chat.complete_async.side_effect = [error, response('{"source":"turkey"}')]
                result = mistral.run(self.batch.subset([0]), self.sources, replace(self.config, retries=1))
                self.assertEqual(result.table.best_hit.iloc[0], "turkey")
                self.assertEqual(client.chat.complete_async.await_count, 2)
                sleep.assert_awaited_once_with(1)

    def test_quota_exhaustion_and_permanent_errors_fail_without_unknown_calls(self):
        client, _ = self.mock_client([])
        for status, expected_attempts in [(429, 3), (500, 3), (401, 1), (402, 1), (403, 1), (422, 1)]:
            with self.subTest(status=status), patch.object(mistral.asyncio, "sleep", new_callable=AsyncMock):
                client.chat.complete_async.reset_mock()
                client.chat.complete_async.side_effect = APIError(status)
                with self.assertRaisesRegex(RuntimeError, f"HTTP {status}") as error:
                    mistral.run(self.batch.subset([0]), self.sources, replace(self.config, retries=2))
                self.assertNotIn("secret-key", str(error.exception))
                self.assertEqual(client.chat.complete_async.await_count, expected_attempts)

    def test_actual_async_timeout_cancels_request(self):
        client, _ = self.mock_client([])
        cancelled = []

        async def never_finishes(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

        client.chat.complete_async.side_effect = never_finishes
        with self.assertRaisesRegex(RuntimeError, "TimeoutError"):
            mistral.run(self.batch.subset([0]), self.sources, replace(self.config, timeout=0.01))
        self.assertEqual(cancelled, [True])
        client.__aexit__.assert_awaited_once()

    def test_concurrency_is_bounded_and_completion_order_does_not_reorder_rows(self):
        client, _ = self.mock_client([])
        active, peak, completion = 0, 0, []

        async def exercise():
            release_first = asyncio.Event()

            async def complete(**kwargs):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                record = kwargs["messages"][1]["content"]
                if "host: turkey;" in record:
                    await release_first.wait()
                    label = "turkey"
                else:
                    await asyncio.sleep(0)
                    release_first.set()
                    label = "sheep" if "Ovis aries" in record else "unknown"
                completion.append(label)
                active -= 1
                return response(json.dumps({"source": label}))

            client.chat.complete_async.side_effect = complete
            return await mistral.run_async(self.batch, self.sources, replace(self.config, concurrency=2))

        result = asyncio.run(exercise())
        self.assertEqual(peak, 2)
        self.assertEqual(completion[0], "sheep")
        self.assertEqual(result.table.best_hit.tolist(), ["turkey", "sheep", "unknown"])

    def test_failure_cancels_other_workers_and_closes_client(self):
        client, _ = self.mock_client([])
        cancelled = []

        async def exercise():
            other_started = asyncio.Event()

            async def complete(**kwargs):
                if "host: turkey;" in kwargs["messages"][1]["content"]:
                    await other_started.wait()
                    raise APIError(401)
                other_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(True)

            client.chat.complete_async.side_effect = complete
            return await mistral.run_async(self.batch, self.sources, replace(self.config, concurrency=2))

        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            asyncio.run(exercise())
        self.assertEqual(client.chat.complete_async.await_count, 2)
        self.assertEqual(cancelled, [True])
        client.__exit__.assert_called_once()
        client.__aexit__.assert_awaited_once()

    def test_sync_api_in_event_loop_gives_actionable_error(self):
        async def exercise():
            with self.assertRaisesRegex(RuntimeError, "run_async"):
                mistral.run(self.batch, self.sources)
        asyncio.run(exercise())

    def test_cli_validation(self):
        argv = ["--metadata", "input.tsv", "--sources", "sources.tsv", "--out", "out.tsv", "--id-col", "id"]
        args = metalyzer.parse_args([*argv, "--method", "mistral", "--mistral-model", "chosen"])
        self.assertEqual(args.mistral_config.model, "chosen")
        for option, value in [("--mistral-concurrency", "0"), ("--mistral-retries", "-1"),
                              ("--mistral-timeout", "0"), ("--mistral-timeout", "nan"),
                              ("--mistral-timeout", "inf"), ("--mistral-max-tokens", "0"),
                              ("--mistral-progress-every", "0"), ("--mistral-model", "")]:
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    metalyzer.parse_args([*argv, option, value])

    def test_cli_outputs_standard_fields_and_preserves_existing_output_on_api_failure(self):
        client, _ = self.mock_client(['{"source":"unknown"}'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.batch.metadata.to_csv(root / "input.tsv", sep="\t", index=False)
            pd.DataFrame({"source": self.sources.labels}).to_csv(root / "sources.tsv", sep="\t", index=False)
            argv = ["--metadata", str(root / "input.tsv"), "--sources", str(root / "sources.tsv"),
                    "--out", str(root / "out.tsv"), "--id-col", "id", "--method", "mistral",
                    "--limit", "1", "--mistral-retries", "0", "--min-score", "1"]
            with patch.object(deterministic_source, "load_taxonomy", return_value=None):
                metalyzer.main(argv)
            output = pd.read_csv(root / "out.tsv", sep="\t", dtype=str, keep_default_na=False)
            self.assertEqual(output.columns.tolist(), ["id", *self.sources.columns,
                                                       "nli_verification_score", "year", "country"])
            self.assertEqual(output.nli_verification_score.tolist(), ["NA"])
            self.assertEqual(output.best_hit.tolist(), ["unknown"])
            self.assertEqual(output.source_method.tolist(), ["mistral"])
            self.assertEqual(output.source_llm_score.tolist(), ["NA"])
            self.assertNotIn("source_verification_score", output.columns)
            self.assertEqual(output.year.tolist(), ["2019"])
            self.assertEqual(output.country.tolist(), ["United States"])
            self.assertTrue((output[list(self.sources.names)] == "NA").all().all())
            before = (root / "out.tsv").read_bytes()
            client.chat.complete_async.side_effect = APIError(429)
            with patch.object(deterministic_source, "load_taxonomy", return_value=None), self.assertRaises(RuntimeError):
                metalyzer.main(argv)
            self.assertEqual((root / "out.tsv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
