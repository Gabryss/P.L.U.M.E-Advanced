"""Smoke tests for the stage-B trunk graph."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config
from stages import HostFieldGenerator, TrunkGraphGenerator


class TrunkGraphTests(unittest.TestCase):
    def test_default_config_loads_and_generates_trunk_graph(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        trunk_graph = TrunkGraphGenerator(project_config.graph).generate(host_field)

        self.assertGreater(len(trunk_graph.points), 10)
        self.assertEqual(len(trunk_graph.edges), len(trunk_graph.points) - 1)
        self.assertGreater(trunk_graph.total_length, 0.0)

        start_point = trunk_graph.points[0]
        end_point = trunk_graph.points[-1]
        self.assertGreater(end_point.x, start_point.x)
        self.assertLess(end_point.elevation, start_point.elevation)

        arc_lengths = [point.arc_length for point in trunk_graph.points]
        self.assertEqual(arc_lengths, sorted(arc_lengths))


if __name__ == "__main__":
    unittest.main()
