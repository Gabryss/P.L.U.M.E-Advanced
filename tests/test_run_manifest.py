"""Tests for completed-run reproducibility metadata."""

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.config import load_project_config
from plume_advanced.run_manifest import write_run_manifest


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
                inputs=(ROOT / "uv.lock",),
            )

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["elapsed_seconds"], 1.25)
            self.assertTrue(payload["source"]["sha256"])
            self.assertIn("numpy", payload["dependencies"])
            self.assertTrue(payload["inputs"][0]["path"].endswith("uv.lock"))
            self.assertEqual(len(payload["inputs"][0]["sha256"]), 64)
            self.assertEqual(payload["outputs"][0]["bytes"], 5)
            self.assertEqual(len(payload["outputs"][0]["sha256"]), 64)

    def test_manifest_can_record_a_failed_stage(self) -> None:
        config = load_project_config(ROOT / "config" / "project.toml")
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_run_manifest(
                config,
                Path(temporary) / "run_manifest.json",
                outputs=(),
                elapsed_seconds=0.5,
                source_root=ROOT,
                status="failed",
                failed_stage="geometry",
                error="RuntimeError: synthetic failure",
            )

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["failure"]["stage"], "geometry")
            self.assertIn("synthetic failure", payload["failure"]["error"])


if __name__ == "__main__":
    unittest.main()
