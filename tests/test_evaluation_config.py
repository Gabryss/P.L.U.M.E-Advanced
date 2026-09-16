import json
from dataclasses import asdict
from pathlib import Path

import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.cli import main
from plume_advanced.evaluation.config import DEFAULT_CONFIG, load_evaluation_config

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_pdc_partitions_are_disjoint_and_complete() -> None:
    config = load_evaluation_config()
    calibration = set(config.pdc_cave_partition("calibration"))
    evaluation = set(config.pdc_cave_partition("evaluation"))

    assert len(calibration) == 76
    assert len(evaluation) == 19
    assert calibration.isdisjoint(evaluation)
    assert len(calibration | evaluation) == 95


@pytest.mark.parametrize("section", [
    "morphometry", "controllability", "host_ablation", "sampling_ablation",
    "scalability", "export_consistency", "determinism",
])
def test_bundled_experiments_have_reproducible_seeds(section):
    seeds = load_evaluation_config().seeds(section)
    assert seeds and len(seeds) == len(set(seeds))
    assert all(0 <= seed < 2**32 for seed in seeds)


def test_bundled_research_resolves_exactly_like_public_recipe(monkeypatch):
    monkeypatch.chdir(ROOT)
    bundled = load_evaluation_config().load_project()
    public = load_project_config(ROOT / "config/research.toml")
    assert asdict(bundled) == asdict(public)
    assert Path(bundled.geometry.cave_diffuse_texture).is_file()
    assert Path(bundled.events.rocky_output_dir).is_relative_to(ROOT / "outputs")


def test_default_audit_writes_to_working_directory(tmp_path, monkeypatch, capsys):
    (tmp_path / "texture").symlink_to(ROOT / "texture", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    assert main(["audit"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ready"]
    assert report["pdc_partition"] == {
        "calibration_caves": 76, "evaluation_caves": 19, "overlap_caves": 0,
    }
    destination = tmp_path / "outputs/evaluation"
    assert report["output_root"] == str(destination)
    assert (destination / "evaluation_audit.json").is_file()
    assert load_evaluation_config(DEFAULT_CONFIG).output_root == destination


def test_custom_evaluation_paths_resolve_beside_config(tmp_path, monkeypatch):
    directory = tmp_path / "custom"
    directory.mkdir()
    path = directory / "experiment.toml"
    path.write_text('schema_version = 1\n[general]\noutput_root = "results"\n'
                    'project_config = "research.toml"\n')
    monkeypatch.chdir(tmp_path)
    config = load_evaluation_config(path)
    assert config.output_root == directory / "results"
    assert config.project_config == directory / "research.toml"


def test_audit_missing_materials_is_not_ready(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["audit"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is False and len(report["missing_inputs"]) >= 4
    assert all(Path(p).is_relative_to(tmp_path) for p in report["missing_inputs"])
    assert load_evaluation_config().load_project().events.rocky_output_dir == str(
        tmp_path / "outputs/rocky_stage_e")


def test_explicit_asset_directory_is_independent_of_recipe_location(tmp_path):
    path = tmp_path / "experiments.toml"
    path.write_text(f'schema_version = 1\n[general]\nproject_config = "{DEFAULT_CONFIG.parent / "research.toml"}"\n'
                    'asset_directory = "assets/config"\n')
    config = load_evaluation_config(path)
    project = config.load_project()
    assert config.asset_directory == tmp_path / "assets/config"
    assert Path(project.geometry.cave_diffuse_texture).is_relative_to(tmp_path / "assets/texture")
    assert Path(project.events.rocky_output_dir).is_relative_to(tmp_path / "assets/outputs")


@pytest.mark.parametrize("settings, key", [
    ('[scalability]\ntimeot_s = 10', 'scalability.timeot_s'),
    ('[datasets.pdc]\npath_en = "PDC"', 'datasets.pdc.path_en'),
    ('[general]\nconfidence_level = true', 'confidence_level'),
    ('[general]\nbootstrap_iterations = 1.5', 'bootstrap_iterations'),
    ('[general]\nbootstrap_seed = -1', 'bootstrap_seed'),
    ('[scalability]\ntimeout_s = "nan"', 'timeout_s'),
    ('[scalability]\ntimeout_s = nan', 'timeout_s'),
    ('[scalability]\nmemory_limit_gib = inf', 'memory_limit_gib'),
    ('[scalability]\ntimeout_s = 0', 'timeout_s'),
    ('[scalability]\nroute_lengths_m = []', 'route_lengths_m'),
    ('[scalability]\nroute_lengths_m = [100, -2]', 'route_lengths_m'),
    ('[scalability]\nroute_lengths_m = [100, 100]', 'route_lengths_m'),
    ('[scalability]\nstorage_modes = ["unknown"]', 'storage_modes'),
    ('[scalability]\nquality = "prodution"', 'quality'),
    ('[export_consistency]\ntargets = ["unknown"]', 'targets'),
    ('[export_consistency]\ntargets = "unity"', 'targets'),
    ('[morphometry]\nexclude_self_intersections = 1', 'exclude_self_intersections'),
    ('[morphometry]\nmetrics = ["heigth_m"]', 'metrics'),
    ('[morphometry]\nmetrics = ["width_m"]\naggregate_metrics = ["height_m"]', 'aggregate_metrics'),
    ('[controllability]\ndistributary_values = [1.1]', 'distributary_values'),
    ('[controllability]\nsupply_values = [false]', 'supply_values'),
    ('[sampling_ablation]\nreference_spacing_m = -1', 'reference_spacing_m'),
    ('[host_ablation]\nconditions = ["no_slop"]', 'conditions'),
    ('[determinism]\nbody = "mooon"', 'body'),
    ('scalability = 3', 'scalability'),
])
def test_scientific_config_rejects_invalid_nested_settings(tmp_path, settings, key):
    path = tmp_path / "experiments.toml"
    path.write_text('schema_version = 1\n' + settings + '\n')
    with pytest.raises(ValueError, match=key):
        load_evaluation_config(path)


@pytest.mark.parametrize("version", ['true', '1.0', '"1"', '2'])
def test_schema_version_requires_exact_integer(tmp_path, version):
    path = tmp_path / "experiments.toml"
    path.write_text(f'schema_version = {version}\n')
    with pytest.raises(ValueError, match='schema_version'):
        load_evaluation_config(path)


@pytest.mark.parametrize("seeds", ['-1\n', '4294967296\n', '1\n1\n', '# empty\n'])
def test_invalid_seed_file_is_rejected(tmp_path, seeds):
    (tmp_path / "seeds.txt").write_text(seeds)
    path = tmp_path / "experiments.toml"
    path.write_text('schema_version = 1\n[determinism]\nseed_file = "seeds.txt"\n')
    with pytest.raises(ValueError):
        load_evaluation_config(path).seeds('determinism')
