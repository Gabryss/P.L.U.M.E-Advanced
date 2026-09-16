"""Inspection tools must find current exports and never choose a stale copy silently."""

import pytest

from plume_advanced.asset_paths import find_export_asset


@pytest.mark.parametrize("layout", ["export_blender", "export_all/blender"])
@pytest.mark.parametrize("filename", ["custom_cave.glb", "plume_continuous_inspection.blend"])
def test_resolves_assets_in_either_export_layout(tmp_path, layout, filename):
    asset = tmp_path / layout / filename
    asset.parent.mkdir(parents=True)
    asset.touch()
    kwargs = {"filename": filename} if asset.suffix == ".blend" else {}
    assert find_export_asset(tmp_path, **kwargs) == asset


def test_empty_single_target_directory_does_not_hide_all_target_asset(tmp_path):
    (tmp_path / "export_blender").mkdir()
    asset = tmp_path / "export_all/blender/cave.glb"
    asset.parent.mkdir(parents=True)
    asset.touch()
    assert find_export_asset(tmp_path) == asset


def test_missing_export_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_all/blender"):
        find_export_asset(tmp_path)


@pytest.mark.parametrize("second", ["export_blender/stale.glb", "export_all/blender/cave.glb"])
def test_ambiguous_exports_are_rejected(tmp_path, second):
    for filename in ("export_blender/cave.glb", second):
        asset = tmp_path / filename
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.touch()
    with pytest.raises(ValueError, match="Ambiguous blender exports"):
        find_export_asset(tmp_path)


def test_neutral_export_is_selected_only_when_requested(tmp_path):
    asset = tmp_path / "export_neutral/cave.glb"
    asset.parent.mkdir()
    asset.touch()
    with pytest.raises(FileNotFoundError):
        find_export_asset(tmp_path)
    assert find_export_asset(tmp_path, target="neutral") == asset
