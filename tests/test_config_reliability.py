"""Reject unusable inputs before allocation or generation starts."""

from pathlib import Path

import pytest

from plume_advanced.config import load_project_config
from plume_advanced.world import build_export_config


@pytest.mark.parametrize(
    "key",
    [
        "host_field.base_cover_thickness",
        "host_field.grid.width",
        "network.target_route_length_m",
        "section_field.sample_spacing",
        "geometry.voxel_size",
        "events.rock_density_per_m2",
    ],
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_input_names_offending_field(tmp_path, key, value):
    section, name = key.rsplit(".", 1)
    path = tmp_path / "invalid.toml"
    path.write_text(f"schema_version = 4\n[{section}]\n{name} = {value}\n")
    with pytest.raises(ValueError, match=key):
        load_project_config(path)


@pytest.mark.parametrize(
    "section,name,value",
    [
        ("geometry", "use_section_profiles", "true"),
        ("geometry", "junction_irregularity_amplitude", "0.3"),
        ("geometry", "junction_irregularity_frequency", "0.1"),
        ("section_field", "level_transition_fraction", "0.18"),
    ],
)
def test_retired_settings_are_rejected_explicitly(tmp_path, section, name, value):
    path = tmp_path / "retired.toml"
    path.write_text(f"schema_version = 4\n[{section}]\n{name} = {value}\n")
    with pytest.raises(ValueError, match=f"{section}.{name}"):
        load_project_config(path)


@pytest.mark.parametrize("path", sorted((Path(__file__).parents[1] / "config").glob("*.toml")))
def test_maintained_presets_have_no_retired_fields(path):
    assert load_project_config(path).schema_version == 4


@pytest.mark.parametrize("field", ["max_visual_triangles", "max_asset_bytes"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "10"])
def test_export_budgets_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        build_export_config({field: value})
