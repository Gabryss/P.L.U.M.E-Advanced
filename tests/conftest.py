"""Retain structured rejection evidence when a CI regression fails."""

import hashlib
import json
import os
from pathlib import Path

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    yield
    destination = os.environ.get("PLUME_TEST_EVIDENCE")
    if not destination or call.excinfo is None:
        return
    inspection = getattr(call.excinfo.value, "report", None)
    if inspection is None:
        return
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(item.nodeid.encode()).hexdigest()[:16]
    (root / f"{name}.json").write_text(json.dumps(dict(
        test=item.nodeid, phase=call.when, error=str(call.excinfo.value), inspection=inspection),
        indent=2, default=str) + "\n")
