from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plume_advanced.evaluation.metrics.continuity import longitudinal_continuity  # noqa: E402
from plume_advanced.evaluation.serialization import diagnostics_to_json  # noqa: E402

try:
    from plume_advanced.evaluation.metrics.network import network_metrics  # noqa: E402
    from plume_advanced.evaluation.metrics.sections import (
        pdc_comparable_section_summary,  # noqa: E402
    )
    from plume_advanced.stages.network import (
        CaveNetwork,
        CaveNetworkConfig,
        CaveNode,
        CavePoint,
        CaveSegment,
    )
    from plume_advanced.stages.section_field import (
        SectionField,
        SectionFieldConfig,
        SectionSample,
        SegmentSectionField,
    )
except ModuleNotFoundError:  # Optional geometry dependencies may be absent in minimal CI.
    network_metrics = None
    pdc_comparable_section_summary = None
    CaveNetwork = CaveNetworkConfig = CaveNode = CavePoint = CaveSegment = None
    SectionField = SectionFieldConfig = SegmentSectionField = SectionSample = None


class _Point:
    def __init__(self, x, y, z, s):
        self.x, self.y, self.elevation, self.arc_length = x, y, z, s


class EvaluationDiagnosticsTests(unittest.TestCase):
    def test_network_sinuosity_and_uphill_runs(self):
        if CaveNetwork is None or network_metrics is None:
            self.skipTest("stage network optional dependencies unavailable")

        def segment(segment_id, kind, points):
            return CaveSegment(segment_id, 0, 1, kind, 0, tuple(points), {})

        points = tuple(
            CavePoint(i, float(i), float(i % 2), float(i), 0, 1, 1, 0, float(i), 2.0, 1, 1400, i)
            for i in range(3)
        )
        network = CaveNetwork(
            CaveNetworkConfig(),
            (CaveNode(0, 0, 0, 0, 0, "entry"), CaveNode(1, 2, 0, 2, 0, "exit")),
            (segment(0, "backbone", points),),
            (),
            np.zeros((2, 2), dtype=bool),
            np.zeros((2, 2)),
            (0, 1),
            (),
            (),
            (),
        )
        report = network_metrics(network)
        self.assertIn("backbone", report["sinuosity_by_kind"])
        self.assertEqual(
            report["sustained_uphill_by_kind"]["backbone"]["uphill_length"]["count"], 1
        )

    def test_section_quantiles_are_order_invariant_and_derived_features_work(self):
        if pdc_comparable_section_summary is None:
            self.skipTest("stage section optional dependencies unavailable")
        records = [
            {
                "arc_length": 2,
                "width": 4,
                "height": 2,
                "floor_z": 1,
                "expected_floor_z": 0,
                "roof_left": 3,
                "roof_right": 2,
            },
            {"arc_length": 0, "width": 2, "height": 2, "floor_residual": -1, "roof_asymmetry": 0.0},
            {
                "arc_length": 1,
                "width": 3,
                "height": 3,
                "area": 7,
                "compactness": 0.8,
                "floor_residual": 0.5,
                "roof_asymmetry": 0.2,
            },
        ]
        first = pdc_comparable_section_summary(records)
        second = pdc_comparable_section_summary(list(reversed(records)))
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["width"]["median"], 3.0)

    def test_continuity_distinguishes_smooth_and_repetitive_signals(self):
        smooth = [{"arc_length": i, "width": i / 10.0} for i in range(32)]
        repetitive = [{"arc_length": i, "width": 1.0 + (1.0 if i % 2 else -1.0)} for i in range(32)]
        smooth_report = longitudinal_continuity(smooth, fields=("width",))
        repetitive_report = longitudinal_continuity(repetitive, fields=("width",))
        smooth_metric = smooth_report["metrics"]["width"]
        repetitive_metric = repetitive_report["metrics"]["width"]
        self.assertGreater(
            smooth_metric["low_frequency_evolution_score"],
            repetitive_metric["low_frequency_evolution_score"],
        )
        self.assertGreater(
            repetitive_metric["short_period_energy_ratio"],
            smooth_metric["short_period_energy_ratio"],
        )

    def test_section_field_continuity_uses_real_stage_c_object(self):
        if SectionField is None:
            self.skipTest("stage section optional dependencies unavailable")
        profile = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0))
        samples = []
        for index, arc in enumerate((0.0, 10.0, 20.0)):
            samples.append(
                SectionSample(
                    index,
                    4,
                    arc,
                    arc,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    (1, 0, 0),
                    (0, 1, 0),
                    (0, 0, 1),
                    2.0 + index * 0.2,
                    1.0,
                    0.2,
                    1.0,
                    0.0,
                    0.0,
                    (),
                    profile,
                )
            )
        field = SectionField(
            SectionFieldConfig(), (SegmentSectionField(4, (), tuple(samples)),), (4,)
        )
        from plume_advanced.evaluation.metrics.sections import section_longitudinal_continuity

        report = section_longitudinal_continuity(field)
        self.assertEqual(report["segments"]["4"]["metrics"]["width"]["count"], 3)

    def test_json_is_canonical_and_strict(self):
        report = {"b": 2, "a": 1.0, "nan": float("nan")}
        encoded = diagnostics_to_json(report)
        self.assertEqual(encoded, '{"a":1.0,"b":2,"nan":null}')
        self.assertEqual(json.loads(encoded)["nan"], None)


if __name__ == "__main__":
    unittest.main()
