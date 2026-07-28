#!/usr/bin/env python3
"""Repository wrapper for the installed PLUME command-line entry point."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plume_advanced.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
