from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plume_advanced.evaluation import (  # noqa: E402
    diagnostics_to_json,
    longitudinal_continuity,
    network_diagnostics,
    section_diagnostics,
)


class _Point:
    def __init__(self, x, y, z, s):
        self.x, self.y, self.elevation, self.arc_length = x, y, z, s


class _Path:
    def __init__(self, kind, points, branch_id=None, target=None):
        self.branch_kind, self.points = kind, tuple(points)
        self.branch_id, self.target_trunk_index = branch_id, target


class EvaluationDiagnosticsTests(unittest.TestCase):
    def test_network_sinuosity_and_uphill_runs(self):
        straight = _Path("trunk", [_Point(0, 0, 10, 0), _Point(1, 0, 9, 1), _Point(2, 0, 8, 2)])
        bent = _Path("spur", [_Point(0, 0, 5, 0), _Point(1, 1, 6, 1), _Point(2, 0, 5, 2)], branch_id=2)
        report = network_diagnostics([bent, straight])
        self.assertEqual(report["sinuosity"]["by_kind"]["trunk"]["median"], 1.0)
        self.assertGreater(report["sinuosity"]["by_kind"]["spur"]["median"], 1.0)
        uphill = next(path["uphill"] for path in report["paths"] if path["kind"] == "spur")
        self.assertEqual(uphill["sustained_run_count"], 1)
        self.assertAlmostEqual(uphill["uphill_rise"], 1.0)

    def test_section_quantiles_are_order_invariant_and_derived_features_work(self):
        records = [
            {"arc_length": 2, "width": 4, "height": 2, "floor_z": 1, "expected_floor_z": 0, "roof_left": 3, "roof_right": 2},
            {"arc_length": 0, "width": 2, "height": 2, "floor_residual": -1, "roof_asymmetry": 0.0},
            {"arc_length": 1, "width": 3, "height": 3, "area": 7, "compactness": 0.8, "floor_residual": 0.5, "roof_asymmetry": 0.2},
        ]
        first = section_diagnostics(records)
        second = section_diagnostics(list(reversed(records)))
        self.assertEqual(first["features"], second["features"])
        self.assertAlmostEqual(first["features"]["width"]["median"], 3.0)
        self.assertEqual(first["missing_count"]["floor_residual"], 0)

    def test_continuity_distinguishes_smooth_and_repetitive_signals(self):
        smooth = [{"arc_length": i, "width": i / 10.0} for i in range(32)]
        repetitive = [{"arc_length": i, "width": 1.0 + (1.0 if i % 2 else -1.0)} for i in range(32)]
        smooth_report = longitudinal_continuity(smooth, fields=("width",))
        repetitive_report = longitudinal_continuity(repetitive, fields=("width",))
        smooth_metric = smooth_report["metrics"]["width"]
        repetitive_metric = repetitive_report["metrics"]["width"]
        self.assertGreater(smooth_metric["low_frequency_evolution_score"], repetitive_metric["low_frequency_evolution_score"])
        self.assertGreater(repetitive_metric["short_period_energy_ratio"], smooth_metric["short_period_energy_ratio"])

    def test_json_is_canonical_and_strict(self):
        report = {"b": 2, "a": 1.0, "nan": float("nan")}
        encoded = diagnostics_to_json(report)
        self.assertEqual(encoded, '{"a":1.0,"b":2,"nan":null}')
        self.assertEqual(json.loads(encoded)["nan"], None)


if __name__ == "__main__":
    unittest.main()
