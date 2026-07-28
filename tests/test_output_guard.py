"""Regression tests for safe generation-output handling."""

import tempfile
import unittest
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.output_guard import (
    OutputOverwriteRefused,
    populated_output_directories,
    require_output_overwrite_confirmation,
)


class OutputGuardTests(unittest.TestCase):
    def test_empty_or_missing_directories_do_not_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            empty = Path(temp_dir) / "empty"
            empty.mkdir()
            missing = Path(temp_dir) / "missing"

            require_output_overwrite_confirmation(
                (empty, missing),
                interactive=False,
            )

            self.assertEqual(populated_output_directories((empty, missing)), ())

    def test_noninteractive_generation_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "outputs"
            output.mkdir()
            existing = output / "existing.txt"
            existing.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(
                OutputOverwriteRefused,
                "--force-overwrite",
            ):
                require_output_overwrite_confirmation(
                    (output,),
                    interactive=False,
                )

            self.assertEqual(existing.read_text(encoding="utf-8"), "keep")

    def test_interactive_confirmation_accepts_yes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            prompts: list[str] = []

            require_output_overwrite_confirmation(
                (output,),
                interactive=True,
                input_func=lambda prompt: prompts.append(prompt) or "yes",
            )

            self.assertEqual(len(prompts), 1)
            self.assertIn(str(output), prompts[0])

    def test_interactive_default_refuses_and_bypass_skips_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            error_stream = StringIO()

            with self.assertRaises(OutputOverwriteRefused):
                require_output_overwrite_confirmation(
                    (output,),
                    interactive=True,
                    input_func=lambda _prompt: "",
                    error_stream=error_stream,
                )
            self.assertIn("left unchanged", error_stream.getvalue())

            require_output_overwrite_confirmation(
                (output,),
                allow_overwrite=True,
                interactive=False,
                input_func=lambda _prompt: self.fail("bypass must not prompt"),
            )


if __name__ == "__main__":
    unittest.main()
