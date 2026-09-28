"""Export retry reuses authenticated geometry and rechecks the package."""

import json
from pathlib import Path

import pytest
import trimesh
from test_embedded_inspection import geometry
from test_network_quality import network_fixture

from plume_advanced.cli import SOURCE_ROOT, _run_inputs
from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.identity import sha256_file
from plume_advanced.pipeline import StageCheckpointStore, pipeline_fingerprint
from plume_advanced.pipeline.recovery import AcceptedBase
from plume_advanced.reexport import main
from plume_advanced.stages.section_field import SectionFieldGenerator


@pytest.fixture
def saved_cave(tmp_path):
    recipe = tmp_path/'source.toml'
    recipe.write_text('schema_version = 4\nprocedural_seed = 9\n[export]\ntarget = "neutral"\n'
                      'max_visual_triangles = 4\nvisual_max_error_m = 0.000001\n'
                      '[run]\nmax_attempt_seconds = 1\n')
    project = load_project_config(recipe)
    source = tmp_path/'source'
    source.mkdir()
    (source/'run_manifest.json').write_text(json.dumps(dict(resolved_config=project_config_manifest(project))))
    fingerprint = pipeline_fingerprint(project, inputs=_run_inputs(recipe, project), source_root=SOURCE_ROOT)
    store = StageCheckpointStore(source/'.plume-checkpoints', fingerprint)
    network = network_fixture()
    sections = SectionFieldGenerator().generate(network)
    surface = trimesh.creation.box(extents=(4, 4, 4))
    for _ in range(3):
        surface = surface.subdivide()
    mesh = geometry(surface, smoothing=0)
    accepted = AcceptedBase(network, sections, mesh, dict(accepted_identity=dict(sections='fixture')))
    store.save('accepted_base', accepted)
    store.save('final_geometry_'+accepted.context_sha256[:16], mesh)
    return recipe, source, store


def arguments(saved, output):
    recipe, source, _ = saved
    return ['--config', str(recipe), '--source', str(source), '--output', str(output)]


def test_failed_export_can_be_retried_without_regenerating_cave(saved_cave, tmp_path, monkeypatch):
    from plume_advanced.stages.geometry import GeometryGenerator
    def forbidden(*args, **kwargs):
        raise AssertionError('Re-export must not generate geometry')
    monkeypatch.setattr(GeometryGenerator, 'build_base_volume', forbidden)
    recipe, source, _ = saved_cave
    hashes = {p: sha256_file(p) for p in source.rglob('*') if p.is_file()}
    output = tmp_path/'export'
    assert main(arguments(saved_cave, output)) == 2
    assert not output.exists()
    assert main(arguments(saved_cave, output)+['--max-visual-triangles', '1000']) == 0
    trace = [json.loads(line) for line in output.with_name('export.progress.jsonl').read_text().splitlines()]
    assert any(row.get('step') == 'Load final checkpoint' for row in trace)
    assert trace[-1]['event'] == 'stage_finish'
    receipt = json.loads((output/'reexport.json').read_text())
    assert receipt['root_seed'] == 9
    assert receipt['original_export']['max_visual_triangles'] == 4
    assert receipt['effective_export']['max_visual_triangles'] == 1000
    assert receipt['overrides'] == dict(max_visual_triangles=1000)
    assert receipt['max_seconds'] == 0
    assert json.loads((output/'pipeline_inspection.json').read_text())['serialized']['passed']
    sizes = json.loads((output/'export_size_report.json').read_text())
    assert sizes['files']['reexport.json'] == (output/'reexport.json').stat().st_size
    assert json.loads((output/'robot_qualification.json').read_text())['qualified'] is False
    assert all(sha256_file(p) == digest for p, digest in hashes.items())


@pytest.mark.parametrize('damage', ['recipe', 'payload', 'missing_final'])
def test_changed_inputs_or_damaged_checkpoints_cannot_be_reexported(saved_cave, tmp_path, damage):
    recipe, source, _ = saved_cave
    if damage == 'recipe':
        recipe.write_text(recipe.read_text()+'\n[geometry]\nvoxel_size = 0.3\n')
    elif damage == 'payload':
        (source/'.plume-checkpoints/accepted_base.pickle').write_bytes(b'invalid')
    else:
        next((source/'.plume-checkpoints').glob('final_geometry_*.pickle')).unlink()
    output = tmp_path/'export'
    with pytest.raises(ValueError, match='matching'):
        main(arguments(saved_cave, output))
    assert not output.exists()


def test_reexport_cannot_replace_source_or_existing_directory(saved_cave, tmp_path):
    for output in (saved_cave[1], tmp_path, Path(saved_cave[1]/'.plume-checkpoints')):
        with pytest.raises(ValueError, match='new output'):
            main(arguments(saved_cave, output))


@pytest.mark.parametrize('limit', [None, '0', '1'])
def test_reexport_deadline_is_explicit_not_inherited(saved_cave, tmp_path, monkeypatch, capsys, limit):
    import plume_advanced.progress as progress
    import plume_advanced.reexport as module

    now = [0.]
    monkeypatch.setattr(progress.time, 'perf_counter', lambda: now[0])
    calls = []

    def export(*args, **kwargs):
        now[0] = 100000.
        progress.report_progress('Collision surface comparison', 1, 1)
        calls.append(kwargs['reexport_provenance']['max_seconds'])

    monkeypatch.setattr(module, 'export_target_asset', export)
    argv = arguments(saved_cave, tmp_path/'export')
    if limit is not None:
        argv += ['--max-seconds', limit]
    assert main(argv) == (2 if limit == '1' else 0)
    assert calls == ([] if limit == '1' else [0])
    assert progress.TerminalProgress._current is None
    progress.check_work_budget()  # no leaked deadline after success or timeout
    if limit == '1':
        assert 'Omit --max-seconds' in capsys.readouterr().err


@pytest.mark.parametrize('value', ['-1', 'nan', 'inf', '-inf'])
def test_invalid_export_time_limit_is_rejected(saved_cave, tmp_path, value):
    with pytest.raises(SystemExit) as error:
        main(arguments(saved_cave, tmp_path/'export') + ['--max-seconds='+value])
    assert error.value.code == 2


def test_base_mesh_is_released_before_loading_final_mesh(saved_cave, tmp_path, monkeypatch):
    import weakref

    load = StageCheckpointStore.load
    references = []

    def checked_load(store, stage):
        if stage.startswith('final_geometry_'):
            assert references and references[0]() is None
        value = load(store, stage)
        if stage == 'accepted_base':
            references.append(weakref.ref(value))
        return value

    monkeypatch.setattr(StageCheckpointStore, 'load', checked_load)
    assert main(arguments(saved_cave, tmp_path/'export') + ['--max-visual-triangles', '1000']) == 0


def test_cli_retains_oversized_export_and_recovers_without_checkpoint_loading(saved_cave, tmp_path, monkeypatch, capsys):
    output = tmp_path / 'too_large'
    assert main(arguments(saved_cave, output) + ['--max-visual-triangles', '1000', '--max-asset-bytes', '16']) == 2
    assert 'preserved' in capsys.readouterr().err
    retained = next(tmp_path.glob('too_large.size-rejected-*'))

    def forbidden(*args, **kwargs):
        raise AssertionError('Size recovery must not load geometry checkpoints')

    monkeypatch.setattr(StageCheckpointStore, 'load', forbidden)
    recovered = tmp_path / 'recovered'
    assert main(['--recover-package', str(retained), '--output', str(recovered),
                 '--max-asset-bytes', '10000000']) == 0
    report = json.loads((recovered / 'reexport.json').read_text())
    assert report['original_export']['max_visual_triangles'] == 4
    assert report['effective_export']['max_asset_bytes'] == 10000000
    assert report['overrides']['max_asset_bytes'] == 10000000


@pytest.mark.parametrize('damage', [None, 'new_config', 'payload', 'metadata', 'missing_final', 'receipt'])
def test_reviewed_compatibility_is_bound_to_current_request_and_exact_payloads(saved_cave, tmp_path, damage):
    recipe, source, store = saved_cave
    checkpoints = {p.stem: json.loads(p.read_text())['payload_sha256'] for p in store.root.glob('*.json')}
    # Simulate an explicitly reviewed operational-only change, not blanket
    # permission to load arbitrary checkpoints with mismatching fingerprints.
    recipe.write_text(recipe.read_text().replace('max_attempt_seconds = 1', 'max_attempt_seconds = 0'))
    project = load_project_config(recipe)
    current = pipeline_fingerprint(project, inputs=_run_inputs(recipe, project), source_root=SOURCE_ROOT)
    receipt = dict(schema='plume.reexport-compatibility.v1', source_fingerprint=store.fingerprint,
                   reexport_fingerprint=current, checkpoints=checkpoints)
    if damage == 'new_config':
        recipe.write_text(recipe.read_text()+'\n[geometry]\nvoxel_size = 0.3\n')
    elif damage == 'payload':
        (store.root/'accepted_base.pickle').write_bytes(b'damaged')
    elif damage == 'metadata':
        metadata_path = store.root/'accepted_base.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['payload_sha256'] = 'different'
        metadata_path.write_text(json.dumps(metadata))
    elif damage == 'missing_final':
        next(store.root.glob('final_geometry_*.pickle')).unlink()
    elif damage == 'receipt':
        receipt['reexport_fingerprint'] = 'different_source'
    (store.root/'reexport_compatibility.json').write_text(json.dumps(receipt))
    output = tmp_path/'export'
    argv = arguments(saved_cave, output) + ['--max-visual-triangles', '1000']
    if damage:
        with pytest.raises(ValueError, match='matching'):
            main(argv)
        assert not output.exists()
    else:
        assert main(argv) == 0
        provenance = json.loads((output/'reexport.json').read_text())
        assert provenance['source_fingerprint'] == store.fingerprint
        assert provenance['reexport_fingerprint'] == current != store.fingerprint
