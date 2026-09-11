"""Coarse performance tripwires for deterministic hot paths.

The budgets intentionally allow substantial CI variance. They catch accidental
order-of-magnitude regressions; they are not microbenchmarks.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.exporters import export_target_asset
from plume_advanced.stages.geometry_types import CaveGeometry, GeometryConfig, VoxelGrid
from plume_advanced.stages.host_field import GridConfig, HostFieldConfig, HostFieldGenerator
from plume_advanced.world import ExportConfig


def _tiny_geometry() -> CaveGeometry:
    return CaveGeometry(
        config=GeometryConfig(
            cave_diffuse_texture="",
            cave_normal_texture="",
            cave_roughness_texture="",
            cave_displacement_texture="",
        ),
        voxel_grid=VoxelGrid(
            origin=(0.0, 0.0, 0.0),
            voxel_size=1.0,
            density=np.zeros((2, 2, 2), dtype=np.float32),
            iso_level=0.0,
        ),
        chunk_meshes=(),
        assembled_vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        assembled_faces=((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
        component_count=1,
        stamped_sample_count=0,
        stamped_segment_ids=(),
    )


@pytest.mark.performance
def test_vectorized_host_field_generation_stays_within_regression_budget() -> None:
    config = HostFieldConfig(grid=GridConfig(width=500.0, height=400.0, nx=128, ny=96))

    started = time.perf_counter()
    host = HostFieldGenerator(config).generate()
    elapsed = time.perf_counter() - started

    assert host.elevation.shape == (96, 128)
    assert elapsed < 3.0, f"host field generation took {elapsed:.2f}s (budget: 3.0s)"


@pytest.mark.performance
def test_all_target_tiny_export_stays_within_time_and_size_budget(
    tmp_path: Path,
) -> None:
    started = time.perf_counter()
    result = export_target_asset(
        _tiny_geometry(),
        ExportConfig(target="all", file_format="auto"),
        tmp_path / "all-targets",
        asset_name="benchmark_tube",
    )
    elapsed = time.perf_counter() - started
    package_bytes = sum(path.stat().st_size for path in result.files if path.is_file())

    assert elapsed < 15.0, f"all-target export took {elapsed:.2f}s (budget: 15.0s)"
    assert package_bytes < 5_000_000
