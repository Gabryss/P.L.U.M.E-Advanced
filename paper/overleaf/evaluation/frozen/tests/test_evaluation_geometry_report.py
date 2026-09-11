from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plume_advanced.evaluation.artifacts import export_geometry_report
from plume_advanced.stages.geometry_types import CaveGeometry, GeometryConfig, VoxelGrid


def test_geometry_report_serializes_per_junction_records_and_preserves_none(tmp_path: Path) -> None:
    grid = VoxelGrid((0.0, 0.0, 0.0), 1.0, np.zeros((3, 3, 3), dtype=float), 0.0)
    geometry = CaveGeometry(
        config=GeometryConfig(),
        voxel_grid=grid,
        chunk_meshes=(),
        assembled_vertices=(),
        assembled_faces=(),
        component_count=0,
        stamped_sample_count=0,
        stamped_segment_ids=(),
        junction_records=(
            (("junction_id", 7), ("kind", "braid"), ("daughter_parent_area_ratio", None)),
        ),
    )
    report_path = export_geometry_report(geometry, tmp_path / "geometry.json")
    payload = json.loads(report_path.read_text())
    assert payload["junction_records"] == [
        {"junction_id": 7, "kind": "braid", "daughter_parent_area_ratio": None}
    ]
