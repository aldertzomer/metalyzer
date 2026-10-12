"""Validate TSV benchmark commands using a mocked GPU and subprocess, offline."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark_smoke import run_host_terminology_benchmarks as runner


class TSVBenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.audit = self.root / 'audit'
        self.audit.mkdir()
        self.out = self.root / 'results'
        self.taxonomy = self.root / 'taxonomy'
        self.taxonomy.mkdir()
        for name in ['nodes.dmp', 'names.dmp', 'merged.dmp']:
            (self.taxonomy / name).write_text('synthetic taxonomy\n')
        inputs = {}
        for name in ['sources.tsv', 'benchmark.tsv', 'benchmark_accessions.tsv', 'benchmark_true_labels.tsv']:
            content = 'synthetic input\n'
            (self.root / name).write_text(content)
            inputs[name] = hashlib.sha256(content.encode()).hexdigest()
        (self.root / 'metalyzer.py').write_text('# Never executed in this test\n')
        (self.root / 'modules').mkdir()
        (self.audit / 'summary.json').write_text(json.dumps(dict(input_mode='metadata', input_sha256=inputs)))
        self.argv = ['benchmark', '--input-mode', 'metadata', '--audit-dir', str(self.audit),
                     '--out-dir', str(self.out), '--taxonomy-dir', str(self.taxonomy), '--other-animal-taxonomy']
        self.gpu = SimpleNamespace(__version__='mock', version=SimpleNamespace(cuda='mock'),
                                   cuda=SimpleNamespace(is_available=lambda: True,
                                                        get_device_name=lambda _: 'CPU test mock GPU',
                                                        get_device_properties=lambda _: SimpleNamespace(total_memory=16 * 1024**3)))

    def invoke(self):
        with patch.object(runner, 'ROOT', self.root), patch.object(sys, 'argv', self.argv), patch.dict(sys.modules, {'torch': self.gpu}):
            runner.main()

    def test_tsv_commands_settings_no_paid_api_and_safe_resume(self):
        commands = []
        def fake_run(command, **kwargs):
            commands.append(command)
            self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
            self.assertEqual(kwargs['env']['TRANSFORMERS_OFFLINE'], '1')
            output = Path(command[command.index('--out') + 1])
            output.write_text('mock result\n')
            return SimpleNamespace(returncode=0)
        with patch.object(runner.subprocess, 'run', side_effect=fake_run) as launched:
            self.invoke()
            self.assertEqual(launched.call_count, 2)
            self.invoke()
            self.assertEqual(launched.call_count, 2)  # completed files reused
        self.assertEqual([cmd[cmd.index('--method')+1] for cmd in commands], ['nli', 'llm'])
        for cmd in commands:
            self.assertIn('--metadata', cmd)
            self.assertEqual(cmd[cmd.index('--id-col')+1], 'run_accession')
            self.assertNotIn('--biosample-file', cmd)
            self.assertNotIn('--api-key-file', cmd)
            self.assertEqual(cmd[cmd.index('--llm-min-score')+1], '0.75')
            self.assertEqual(cmd[cmd.index('--batch-size')+1], '64')
            self.assertIn('--other-animal-taxonomy', cmd)
        manifest = json.loads((self.out / 'manifest.json').read_text())
        self.assertEqual(manifest['inputs']['input_mode'], 'metadata')

    def test_preexisting_output_is_preserved(self):
        self.out.mkdir()
        existing = self.out / 'benchmark_tsv_nli_host_terminology.tsv'
        existing.write_text('previous output\n')
        with patch.object(runner.subprocess, 'run') as launched, self.assertRaises(FileExistsError):
            self.invoke()
        launched.assert_not_called()
        self.assertEqual(existing.read_text(), 'previous output\n')

    def test_no_cpu_inference_when_cuda_unavailable(self):
        self.gpu.cuda.is_available = lambda: False
        with patch.object(runner.subprocess, 'run') as launched, self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
            self.invoke()
        launched.assert_not_called()
