"""Recipe edits must preserve resolved physics, paths and seeded sampling."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from plume_advanced import cli, identity
from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.recipes import available_presets, expand_recipe

ROOT = Path(__file__).resolve().parents[1]
BASELINES = json.loads((ROOT / "tests/fixtures/configuration/presets.json").read_text())


@pytest.mark.parametrize("case", BASELINES["cases"], ids=lambda case: case["id"])
def test_presets_preserve_pre_cleanup_resolved_configuration(case, tmp_path):
    recipe = tmp_path / "case.toml"
    recipe.write_text(f'recipe_version = 1\npreset = "{case["preset"]}"\n')
    options = dict(seed_override=case["seed"], world_body=case["body"])
    manifest = project_config_manifest(load_project_config(recipe, **options))
    # Relative paths are intentionally owned by the user's file, not the package.
    for table, keys in (
        (
            "geometry",
            (
                "cave_diffuse_texture",
                "cave_normal_texture",
                "cave_roughness_texture",
                "cave_displacement_texture",
            ),
        ),
        ("events", ("rocky_source_path", "rocky_texture_dir", "rocky_output_dir")),
    ):
        for key in keys:
            if manifest[table][key]:
                manifest[table][key] = os.path.relpath(manifest[table][key], recipe.parent)
    text = json.dumps(manifest, sort_keys=True)
    assert hashlib.sha256(text.encode()).hexdigest() == case["sha256"]


def write_recipe(tmp_path, text):
    path = tmp_path / "recipe.toml"
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    "text,match",
    [
        ('recipe_version = true\npreset = "preview"', "recipe_version"),
        ('recipe_version = 2\npreset = "preview"', "recipe_version"),
        ('recipe_version = 1\nschema_version = 4\npreset = "preview"', "not both"),
        ('recipe_version = 1\npreset = "not-a-preset"', "Unknown recipe preset"),
        ('recipe_version = 1\npreset = "../preview"', "Unknown recipe preset"),
        ("recipe_version = 1\npreset = 2", "Unknown recipe preset"),
        ('recipe_version = 1\npreset = "preview"\nprocedural_seed = -1', "nonnegative integer"),
        ('recipe_version = 1\npreset = "preview"\nprocedural_seed = true', "nonnegative integer"),
        ('recipe_version = 1\npreset = "preview"\nprocedural_seed = 1.1', "nonnegative integer"),
        ('recipe_version = 1\npreset = "preview"\nnetwork = 5', "network must be a TOML table"),
        (
            'recipe_version = 1\npreset = "preview"\n[geometry]\nvoxel_szie = 0.2',
            "geometry.voxel_szie",
        ),
        ('recipe_version = 1\npreset = "preview"\n[geometry]\nvoxel_size = nan', "finite"),
        (
            'recipe_version = 1\npreset = "preview"\n[network]\nsystems = false',
            "network.systems must be a TOML table",
        ),
        (
            'recipe_version = 1\npreset = "preview"\n[export]\ntarget = []',
            "export.target must be a string",
        ),
        (
            'recipe_version = 1\npreset = "preview"\n[export]\ntarget = false',
            "export.target must be a string",
        ),
    ],
)
def test_recipe_errors_are_explicit(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_project_config(write_recipe(tmp_path, text))


def test_precedence_arrays_and_no_shared_mutation(tmp_path):
    raw = dict(
        recipe_version=1,
        preset="short-single",
        procedural_seed=5,
        network=dict(topology=dict(island_count=[0, 0])),
        geometry=dict(cave_diffuse_texture="maps/rock.png"),
    )
    original = copy.deepcopy(raw)
    resolved = expand_recipe(raw)
    assert resolved["network"]["topology"]["island_count"] == [0, 0]
    resolved["network"]["topology"]["island_count"][0] = 999
    assert expand_recipe(raw)["network"]["topology"]["island_count"] == [0, 0]
    assert raw == original
    path = write_recipe(
        tmp_path,
        """recipe_version = 1
preset = "short-single"
procedural_seed = 5
[geometry]
cave_diffuse_texture = "maps/rock.png"
""",
    )
    cfg = load_project_config(path, seed_override=42)
    assert cfg.procedural_seed == 42
    assert cfg.geometry.cave_diffuse_texture == str(tmp_path / "maps/rock.png")
    path.write_text(path.read_text().replace("procedural_seed = 5", "procedural_seed = 42"))
    assert cfg == load_project_config(path)


def test_body_edit_matches_cli_override(tmp_path):
    path = write_recipe(tmp_path, 'recipe_version = 1\npreset = "research"\n')
    overridden = load_project_config(path, world_body="moon")
    path.write_text(path.read_text() + '\n[world]\nbody = "moon"\n')
    assert load_project_config(path) == overridden


def test_inspection_commands_do_not_generate_or_create_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_run_pipeline", lambda *a: pytest.fail("must not generate"))
    output = tmp_path / "out" / "network.png"
    assert cli.main(["--show-config", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == 4
    assert cli.main(["--list-presets"]) == 0
    assert set(capsys.readouterr().out.splitlines()) == set(available_presets())
    assert not output.parent.exists()


def test_invalid_inspection_command_does_not_write_failure_artifacts(tmp_path):
    path = write_recipe(tmp_path, 'recipe_version = 1\npreset = "missing"\n')
    with pytest.raises(ValueError):
        cli.main(
            ["--config", str(path), "--show-config", "--output", str(tmp_path / "out/network.png")]
        )
    assert not (tmp_path / "out").exists()


def test_catalog_changes_invalidate_generation_identity(tmp_path):
    (tmp_path / "module.py").write_text("x = 1")
    catalog = tmp_path / "presets.json"
    catalog.write_text('{"a": 1}')
    old = identity.package_source_hash(tmp_path)
    catalog.write_text('{"a": 2}')
    assert old != identity.package_source_hash(tmp_path)


def test_checkout_and_installed_defaults_agree():
    assert (ROOT / "config/project.toml").read_bytes() == cli.PACKAGED_CONFIG.read_bytes()


@pytest.mark.parametrize(
    "target,expected",
    [
        ("neutral", "glb"),
        ("unity", "glb"),
        ("blender", "glb"),
        ("ue5", "glb"),
        ("gazebo", "obj"),
        ("omniverse", "usd"),
        ("all", "auto"),
        (" GAZEBO ", "obj"),
        ("ALL", "auto"),
    ],
)
def test_recipe_target_selects_compatible_format(tmp_path, target, expected):
    path = write_recipe(
        tmp_path, f'recipe_version = 1\npreset = "preview"\n[export]\ntarget = "{target}"\n'
    )
    assert load_project_config(path).export.file_format == expected


def test_explicit_export_format_replaces_inherited_alias(tmp_path):
    path = write_recipe(
        tmp_path, 'recipe_version = 1\npreset = "preview"\n[export]\nfile_format = "obj"\n'
    )
    assert load_project_config(path).export.file_format == "obj"
    path.write_text(path.read_text() + 'format = "glb"\n')
    with pytest.raises(ValueError, match="only one"):
        load_project_config(path)
