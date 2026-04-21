"""Smoke tests for the branch / merge sub-stage."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config
from stages import (
    DOWNSTREAM_RECONNECT_LOOP,
    LOCAL_BYPASS_LOOP,
    SPUR,
    BranchMergeGenerator,
    HostFieldGenerator,
    TrunkGraphGenerator,
)


class BranchMergeTests(unittest.TestCase):
    def test_branching_substage_runs_without_mutating_trunk_stage(self) -> None:
        project_config = load_project_config(ROOT / "config" / "project.toml")
        host_field = HostFieldGenerator(project_config.host_field).generate()
        trunk_graph = TrunkGraphGenerator(project_config.graph).generate(host_field)
        branch_network = BranchMergeGenerator(project_config.branching).generate(
            host_field,
            trunk_graph,
        )

        summary = branch_network.summary()
        self.assertLessEqual(
            len(branch_network.branches),
            project_config.branching.max_branch_count,
        )
        self.assertGreaterEqual(len(branch_network.candidates), len(branch_network.branches))

        for branch in branch_network.branches:
            self.assertGreater(len(branch.points), 1)
            self.assertGreater(branch.total_length, 0.0)
            if branch.branch_kind in {LOCAL_BYPASS_LOOP, DOWNSTREAM_RECONNECT_LOOP}:
                self.assertIsNotNone(branch.merge_event)
                self.assertIsNotNone(branch.target_trunk_index)
                self.assertGreater(branch.target_trunk_index, branch.source_trunk_index)
            else:
                self.assertEqual(branch.branch_kind, SPUR)
                self.assertIsNone(branch.merge_event)
                self.assertIsNone(branch.target_trunk_index)

        self.assertGreater(len(trunk_graph.points), 10)
        self.assertEqual(len(trunk_graph.edges), len(trunk_graph.points) - 1)


if __name__ == "__main__":
    unittest.main()
