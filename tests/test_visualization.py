"""Artifact-level tests for the user-facing diagnostic renderers."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from plume_advanced.stages.events import GeologicalEventConfig, GeologicalEventField
from plume_advanced.stages.floor_map import FloorAtlas, FloorCell, FloorMapConfig
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator
from plume_advanced.visualization.events import (
    GeologicalEventPlotConfig,
    GeologicalEventPlotter,
)
from plume_advanced.visualization.floor_map import FloorMapPlotter
from plume_advanced.visualization.host_field import HostFieldPlotConfig, HostFieldPlotter
from plume_advanced.visualization.network import CaveNetworkPlotConfig, CaveNetworkPlotter
from plume_advanced.visualization.section_field import (
    SectionFieldPlotConfig,
    SectionFieldPlotter,
)


@pytest.fixture(scope="module")
def generated_stages():
    host = HostFieldGenerator().generate()
    network = CaveNetworkGenerator().generate(host)
    sections = SectionFieldGenerator().generate(network)
    return host, network, sections


def _assert_valid_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 1_000
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width > 100
        assert image.height > 100
        image.verify()


def test_stage_diagnostic_renderers_write_valid_png_artifacts(
    tmp_path: Path,
    generated_stages,
) -> None:
    host, network, sections = generated_stages
    outputs = (
        HostFieldPlotter(HostFieldPlotConfig((6.0, 4.0), 55)).render(
            host,
            tmp_path / "host.png",
        ),
        CaveNetworkPlotter(CaveNetworkPlotConfig((6.0, 4.0), 55)).render(
            host,
            network,
            tmp_path / "network.png",
        ),
        SectionFieldPlotter(SectionFieldPlotConfig((6.0, 4.0), 55)).render(
            network,
            sections,
            tmp_path / "sections.png",
        ),
        GeologicalEventPlotter(GeologicalEventPlotConfig((6.0, 4.0), 55)).render(
            network,
            sections,
            GeologicalEventField(
                GeologicalEventConfig(enabled=False, include_rock_props=False),
                events=(),
            ),
            tmp_path / "events-empty.png",
        ),
    )

    for output in outputs:
        _assert_valid_png(output)


def test_floor_map_renderer_handles_geological_cells_and_empty_atlas(
    tmp_path: Path,
) -> None:
    base = FloorCell(
        cell_id=0,
        segment_id=3,
        sample_index=1,
        z_level=0,
        distance_along_m=2.0,
        lateral_offset_m=0.0,
        atlas_x_m=2.0,
        atlas_y_m=0.0,
        x=1.0,
        y=2.0,
        z=-3.0,
        normal_x=0.0,
        normal_y=0.0,
        normal_z=1.0,
        clearance_m=4.0,
        tube_width_m=6.0,
        grounded=True,
        geology_class="sediment",
        event_influence=0.25,
        sediment_thickness_m=0.1,
    )
    populated = FloorAtlas(
        FloorMapConfig(),
        cells=(base,),
        segment_band_offsets_m=((3, 0.0),),
    )
    empty = FloorAtlas(FloorMapConfig(), cells=(), segment_band_offsets_m=())

    plotter = FloorMapPlotter()
    _assert_valid_png(plotter.render(populated, tmp_path / "floor.png"))
    _assert_valid_png(plotter.render(empty, tmp_path / "floor-empty.png"))
