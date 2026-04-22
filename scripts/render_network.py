#!/usr/bin/env python3
"""Compatibility wrapper for the single cave-network entrypoint."""

from __future__ import annotations

from generate_cave import main


if __name__ == "__main__":
    raise SystemExit(main())
