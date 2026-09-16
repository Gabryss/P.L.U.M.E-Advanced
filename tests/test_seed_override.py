"""A campaign root seed must resolve the same host as an edited production TOML."""

from pathlib import Path

import pytest

from plume_advanced.config import load_project_config, project_config_manifest


@pytest.mark.parametrize("seed", [0, 17, 20260912, 4294967295])
def test_override_matches_config_edit_including_host_ranges(tmp_path, seed):
    source = Path("config/short-single.toml")
    original = source.read_text()
    # Resolve asset paths before moving the temporary TOML to another directory.
    original = original.replace("../texture/", str(Path("texture").resolve()) + "/")
    original = original.replace("../outputs/", str(Path("outputs").resolve()) + "/")
    first = tmp_path / "original.toml"
    edited = tmp_path / "edited.toml"
    first.write_text(original)
    edited.write_text(original.replace("procedural_seed = 20260911", f"procedural_seed = {seed}"))
    actual = load_project_config(first, seed_override=seed)
    expected = load_project_config(edited)
    assert project_config_manifest(actual) == project_config_manifest(expected)
    assert actual.host_field != load_project_config(first).host_field


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_override_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="nonnegative integer"):
        load_project_config("config/short-single.toml", seed_override=seed)
