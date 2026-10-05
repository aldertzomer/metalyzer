"""CLI version, run provenance, and failure logging without model downloads."""
import contextlib
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import metalyzer
from modules import nli
from modules.version import __version__


class RunLoggingTests(unittest.TestCase):
    def test_version_exits_without_analysis_arguments(self):
        screen = io.StringIO()
        with contextlib.redirect_stdout(screen), self.assertRaises(SystemExit) as result:
            metalyzer.parse_args(["--version"])
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(screen.getvalue().strip(), f"metalyzer {__version__}")
        command = subprocess.run([sys.executable, "metalyzer.py", "--version"],
                                 cwd=Path(__file__).resolve().parents[1],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(command.stdout.strip(), f"metalyzer {__version__}")

    def test_analysis_creates_log_with_provenance_warning_and_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(columns=["id", "host", "collection_date"]).to_csv(
                root / "input.tsv", sep="\t", index=False)
            pd.DataFrame({"source": ["sheep (Ovis aries)"]}).to_csv(
                root / "sources.tsv", sep="\t", index=False)
            argv = ["--metadata", str(root / "input.tsv"), "--sources", str(root / "sources.tsv"),
                    "--out", str(root / "out.tsv"), "--id-col", "id"]
            with patch.object(nli, "pipeline") as model, contextlib.redirect_stdout(io.StringIO()) as screen:
                metalyzer.main(argv)
            model.assert_not_called()
            log = (root / "out.tsv.log").read_text(encoding="utf-8")
            self.assertIn("Metalyzer version: " + __version__, log)
            self.assertIn("Command line: " + sys.executable, log)
            self.assertIn("metalyzer.py --metadata", log)
            self.assertIn("Python version:", log)
            self.assertIn("Platform:", log)
            self.assertIn("NLI model: " + nli.DEFAULT_NLI_MODEL, log)
            self.assertIn("Sources SHA256: " + hashlib.sha256((root / "sources.tsv").read_bytes()).hexdigest(), log)
            self.assertIn("Metadata rows: 0", log)
            self.assertIn("Metadata columns: 3", log)
            self.assertIn("NLI-only source classification", log)
            self.assertIn("NLI verification scores: 0", log)
            self.assertIn("log path: " + str(root / "out.tsv.log"), log)
            self.assertEqual(screen.getvalue().count("Wrote " + str(root / "out.tsv")), 1)
            self.assertEqual((root / "out.tsv").read_text(encoding="utf-8").count("Run provenance"), 0)

    def test_selected_model_provenance_and_progress_reach_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": ["a"], "host": ["sheep"]}).to_csv(
                root / "input.tsv", sep="\t", index=False)
            pd.DataFrame({"source": ["sheep (Ovis aries)"]}).to_csv(
                root / "sources.tsv", sep="\t", index=False)
            argv = ["--metadata", str(root / "input.tsv"), "--sources", str(root / "sources.tsv"),
                    "--out", str(root / "out.tsv"), "--id-col", "id", "--device", "-1",
                    "--disable-verify-source"]
            with patch.object(nli, "pipeline") as factory, contextlib.redirect_stdout(io.StringIO()) as screen:
                factory.return_value.return_value = [{"labels": ["sheep (Ovis aries)"], "scores": [0.9]}]
                metalyzer.main(argv)
            log = (root / "out.tsv.log").read_text(encoding="utf-8")
            self.assertIn("Batch 1: source classification done", log)
            self.assertIn("Source classification:", log)
            self.assertIn("NLI assignments: 1", log)
            self.assertIn("NLI verification scores: 0", log)
            self.assertEqual(screen.getvalue().count("Batch 1: source classification done"), 1)

            pd.DataFrame(columns=["id", "host"]).to_csv(root / "input.tsv", sep="\t", index=False)
            for method, option, name in (("llm", "--llm-model", "local-model"),
                                         ("mistral", "--mistral-model", "api-model")):
                with self.subTest(method=method), contextlib.redirect_stdout(io.StringIO()):
                    metalyzer.main([*argv, "--method", method, option, name])
                log = (root / "out.tsv.log").read_text(encoding="utf-8")
                self.assertIn(("Local LLM model" if method == "llm" else "Mistral API model")
                              + ": " + name, log)

    def test_input_failure_keeps_nonzero_exit_and_full_traceback_in_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": ["a"], "host": ["sheep"]}).to_csv(
                root / "input.tsv", sep="\t", index=False)
            (root / "sources.tsv").write_text("source\ttaxonomy_anchors\nsheep\tbroken\n", encoding="utf-8")
            argv = [sys.executable, "metalyzer.py", "--metadata", str(root / "input.tsv"),
                    "--sources", str(root / "sources.tsv"), "--out", str(root / "out.tsv"),
                    "--id-col", "id"]
            process = subprocess.run(argv, cwd=Path(__file__).resolve().parents[1],
                                     capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("Invalid taxonomy_anchors value", process.stdout)
            log = (root / "out.tsv.log").read_text(encoding="utf-8")
            self.assertIn("Traceback (most recent call last)", log)
            self.assertIn("Invalid taxonomy_anchors value", log)
            self.assertFalse((root / "out.tsv").exists())


if __name__ == "__main__":
    unittest.main()
