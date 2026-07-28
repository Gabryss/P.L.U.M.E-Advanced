"""Tests for completed-run reproducibility metadata."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config
from run_manifest import write_run_manifest


class RunManifestTests(unittest.TestCase):
    def test_manifest_records_source_dependencies_and_output_hashes(self) -> None:
        config = load_project_config(ROOT / "config" / "project.toml")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = directory / "artifact.bin"
            artifact.write_bytes(b"plume")
            manifest = write_run_manifest(
                config,
                directory / "run_manifest.json",
                outputs=(artifact,),
                elapsed_seconds=1.25,
                source_root=ROOT,
            )

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["elapsed_seconds"], 1.25)
            self.assertTrue(payload["source"]["sha256"])
            self.assertIn("numpy", payload["dependencies"])
            self.assertEqual(payload["outputs"][0]["bytes"], 5)
            self.assertEqual(len(payload["outputs"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
