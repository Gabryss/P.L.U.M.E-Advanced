"""Fast contracts for world physics, deterministic seeds, and export support."""

from dataclasses import fields

import pytest

from plume_advanced.world import (
    APPLICATION_EXPORT_FORMATS,
    EXPORT_FORMATS_BY_TARGET,
    SUPPORTED_EXPORT_TARGETS,
    ExportConfig,
    build_export_config,
    derive_stage_seeds,
    resolve_world_config,
)


def test_application_export_formats_are_complete_and_accepted() -> None:
    application_targets = SUPPORTED_EXPORT_TARGETS - {"all", "neutral"}
    assert set(APPLICATION_EXPORT_FORMATS) == application_targets

    for target, file_format in APPLICATION_EXPORT_FORMATS.items():
        config = build_export_config({"target": target, "format": file_format})
        assert config == ExportConfig(target=target, file_format=file_format)
        assert file_format in EXPORT_FORMATS_BY_TARGET[target]


def test_every_incompatible_target_format_pair_is_rejected() -> None:
    concrete_formats = {"glb", "obj", "usd"}
    for target, supported_formats in EXPORT_FORMATS_BY_TARGET.items():
        for file_format in concrete_formats - set(supported_formats):
            with pytest.raises(ValueError, match="supports format"):
                build_export_config({"target": target, "format": file_format})


def test_stage_seeds_are_named_deterministic_and_independent() -> None:
    first = derive_stage_seeds(1234)
    repeated = derive_stage_seeds(1234)
    changed = derive_stage_seeds(1235)
    names = tuple(field.name for field in fields(first))

    assert names == ("host", "network", "sections", "events", "geometry")
    assert first == repeated
    assert len({getattr(first, name) for name in names}) == len(names)
    assert all(
        getattr(first, name) != getattr(changed, name)
        for name in names
    )
    assert all(
        getattr(derive_stage_seeds(None), name) is None
        for name in names
    )


def test_roof_demand_responds_monotonically_to_physical_inputs() -> None:
    earth = resolve_world_config(
        {"body": "earth", "material": "terrestrial_basalt"}
    )
    moon = resolve_world_config(
        {"body": "moon", "material": "terrestrial_basalt"}
    )

    baseline = earth.roof_demand_ratio(span_m=10.0, roof_thickness_m=5.0)
    assert earth.roof_demand_ratio(20.0, 5.0) == pytest.approx(4.0 * baseline)
    assert earth.roof_demand_ratio(10.0, 10.0) == pytest.approx(0.5 * baseline)
    assert moon.roof_demand_ratio(10.0, 5.0) < baseline
