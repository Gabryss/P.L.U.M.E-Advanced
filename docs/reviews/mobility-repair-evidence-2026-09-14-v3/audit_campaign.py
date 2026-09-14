"""Independently verify frozen recipes, every asset receipt and native coverage."""
import importlib.util
import json
from pathlib import Path

from plume_advanced.identity import package_source_hash, sha256_file

EVIDENCE = Path(__file__).resolve().parent
REPO = EVIDENCE.parents[2]
ROOT = REPO/'outputs/mobility_repair_campaign_20260914_v3'
BASELINE = REPO/'outputs/textured_campaign_20260913'


def read(path):
    return json.loads(path.read_text())


def audit():
    spec = importlib.util.spec_from_file_location('native_campaign_audit', REPO/'scripts/check_native_engines.py')
    native_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native_runner)
    protocol = read(EVIDENCE/'protocol.json')
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
    check('executing source unchanged', package_source_hash() == protocol['source_sha256'])
    for recipe in protocol['recipes']:
        check('recipe '+recipe['config'], sha256_file(recipe['config']) == recipe['sha256'])
    for name, digest in protocol['files'].items():
        check('source/dependency '+name, sha256_file(REPO/name) == digest)
    tests = read(EVIDENCE/'test_results.json')
    check('test source matches campaign', tests['source_sha256'] == protocol['source_sha256'])
    for name, digest in tests['log_sha256'].items():
        check('test log '+name, sha256_file(EVIDENCE/name) == digest)
    check('complete regression suite', '899 passed, 20 skipped, 59 subtests passed' in
          (EVIDENCE/'full_suite.txt').read_text())
    check('additional native material tests', '25 passed, 2 skipped' in
          (EVIDENCE/'native_material.txt').read_text())
    review = read(EVIDENCE/'visual_review.json')
    check('all six contact/full-size reviews completed', {r['case'] for r in review['cases']} ==
          {f'{group}_seed{seed}' for group in ('single','multi') for seed in protocol['seeds']})
    for row in review['cases']:
        for name, digest in row['files'].items():
            check('reviewed image '+name, sha256_file(REPO/name) == digest)
    for seed in protocol['seeds']:
        group_root = ROOT/f'seed_{seed}'
        summary = read(group_root/'summary.json')
        plan = read(group_root/'plan.json')
        check(f'{seed} campaign complete and passed', summary['complete'] and summary['passed'])
        check(f'{seed} both cases completed', len(summary['cases']) == 2)
        check(f'{seed} inputs/source unchanged', summary.get('inputs_unchanged') and summary.get('source_unchanged'))
        check(f'{seed} cold replay required', plan['replay'] and summary['replay'])
        check(f'{seed} expected config order', [Path(c['config']).name for c in plan['cases']] ==
              ['multi_250m_4k.toml', 'single_250m_4k.toml'])
        for name, digest in plan['inputs'].items():
            check(f'{seed} unchanged input {name}', sha256_file(name) == digest)
    comparisons = []
    receipt_count = 0
    for group in ('single','multi'):
        for i, seed in enumerate(protocol['seeds']):
            tag = f'{group}/{seed}'
            run = ROOT/f'seed_{seed}'/f"case_{0 if group=='multi' else 1:04d}"/'attempt_0000'
            replay = ROOT/f'seed_{seed}'/f"replay_{0 if group=='multi' else 1:04d}"/'attempt_0000'
            records = []
            for label, path in [('original',run),('replay',replay)]:
                check(tag+' '+label+' result exists', (path/'result.json').is_file())
                if not (path/'result.json').is_file():
                    records.append(dict(status='missing'))
                    continue
                result = read(path/'result.json')
                records.append(result)
                check(tag+' '+label+' passed', result['status']=='passed')
                if result['status'] != 'passed':
                    continue
                check(tag+' '+label+' source', result['source_sha256']==protocol['source_sha256'])
                check(tag+' '+label+' seed', result['case']['seed']==seed)
                for relative, digest in result['artifacts'].items():
                    asset = path/relative
                    check(tag+' '+label+' '+relative, asset.resolve().is_relative_to(path.resolve())
                          and asset.is_file() and sha256_file(asset)==digest)
                    receipt_count += 1
            original, repeated = records
            if any(r['status'] != 'passed' for r in records):
                comparisons.append(dict(group=group, seed=seed,
                    status='generation_or_replay_failed',
                    original_status=original['status'], replay_status=repeated['status'],
                    original_failure=original.get('failure', original.get('error')),
                    replay_failure=repeated.get('failure', repeated.get('error'))))
                continue
            check(tag+' exact cold identities', original['identity']==repeated['identity'])
            check(tag+' exact cold collider bytes',
                  sha256_file(run/'export/plume_cave_collision.obj') ==
                  sha256_file(replay/'export/plume_cave_collision.obj'))
            check(tag+' different Python hash seed', original['python_hash_seed']!=repeated['python_hash_seed'])
            check(tag+' replay runtime', original['runtime']==repeated['runtime'])
            old_group = 'single_250' if group=='single' else 'multi'
            baseline = read(BASELINE/old_group/f'case_{i:04d}'/'attempt_0000'/'result.json')
            baseline_quality = read(BASELINE/old_group/f'case_{i:04d}'/'attempt_0000'/'pipeline_quality_report.json')
            check(tag+' preserved original host', original['identity']['host']==baseline['identity']['host'])
            quality = read(run/'pipeline_quality_report.json')
            check(tag+' pipeline accepted', quality['passed'])
            check(tag+' resolution warning count retained', quality['resolution']['under_resolved_count'] ==
                  len(quality['resolution']['under_resolved_sections']))
            surface = next(a for a in quality['surface_repairs'] if a['accepted'])
            old_surface = next(a for a in baseline_quality['surface_repairs'] if a['accepted'])
            capsule = quality['required_route_clearance']
            check(tag+' half-metre requirement', capsule['enabled'] and capsule['passed']
                  and capsule['height_m']==.5 and capsule['width_m']==.5 and capsule['margin_m']==.02)
            check(tag+' every required path swept', bool(capsule['paths']) and
                  all(p['passed'] and not p['blocked_edges'] and not p['invalid_samples'] for p in capsule['paths']))
            check(tag+' placement search stayed bounded',
                  capsule.get('placement_queries',0) <= protocol['route_placement_max_sweeps'])
            check(tag+' every changed path independently verified', all(
                  not p.get('placement_repair',{}).get('passed') or p['placement_repair'].get('verified')
                  for p in capsule['paths']))
            collision = quality['export_inspection']['collision']
            check(tag+' collider passed', collision['inspection']['passed'])
            check(tag+' exact cold collider inspection', collision ==
                  read(replay/'pipeline_quality_report.json')['export_inspection']['collision'])
            check(tag+' collision capsule', collision['inspection']['traversal']['passed'])
            check(tag+' bounded collider attempts', len(collision['attempts']) <= 4)
            check(tag+' collider reduction reported', collision['method']=='quadric_edge_collapse'
                  and collision['output_triangles'] <= collision['source_triangles'])
            if not collision['used_raw_fallback']:
                selected = next(row for row in collision['attempts'] if row['accepted'])
                check(tag+' collision surface tolerance', selected['surface_deviation']['passed']
                      and selected['surface_deviation']['max_sampled_error_m'] <= .03)
                precision = selected['precision']
            else:
                check(tag+' full collider surface tolerance', collision['surface_deviation']['passed']
                      and collision['surface_deviation']['max_sampled_error_m'] <= .03)
                precision = collision['precision']
            check(tag+' collider engine precision', precision['passed'] and
                  all(count==0 for count in precision['attempts'][-1]['invalid_faces'].values()))
            check(tag+' texture acceptance', quality['texture_recovery']['passed'])
            native_root = ROOT/'native'/f'{group}_seed{seed}'
            check(tag+' native summary exists', (native_root/'native_summary.json').is_file())
            if not (native_root/'native_summary.json').is_file():
                comparisons.append(dict(group=group, seed=seed, status='native_evidence_missing'))
                continue
            native = read(native_root/'native_summary.json')
            receipt = read(native_root/'native_input_receipt.json')
            plan = read(native_root/'view_plan.json')
            check(tag+' native passed', native['passed'])
            if not native['passed']:
                comparisons.append(dict(group=group, seed=seed, status='native_failed',
                                        failures=native.get('failures')))
                continue
            check(tag+' native source GLB', receipt['glb_sha256']==original['identity']['glb'])
            check(tag+' view receipt', receipt['view_plan_sha256']==sha256_file(native_root/'view_plan.json'))
            check(tag+' native collider conversion', receipt['collision_obj_sha256']==sha256_file(run/'export/plume_cave_collision.obj')
                  and receipt['collision_glb_sha256']==sha256_file(native_root/'plume_collision.glb'))
            for name, digest in receipt['fixture_sha256'].items():
                check(tag+' native fixture '+name, sha256_file(REPO/'tests/fixtures'/name)==digest)
            for engine, folder in [('unity','unity_project'),('unreal','unreal_run_01')]:
                result = native['checks'][engine]
                check(tag+' '+engine+' all views', len(result['images'])==plan['view_count'])
                for name in result['images']:
                    check(tag+' '+engine+' capture '+name, (native_root/folder/name).is_file())
                try:
                    captures = native_runner.validate_captures(native_root/folder)
                    check(tag+' '+engine+' capture measurements unchanged', captures==result['images'])
                except (OSError, ValueError) as error:
                    check(tag+' '+engine+' capture validation: '+str(error), False)
                details = result['native']
                check(tag+' '+engine+' native visual triangle count',
                      details['triangles'] == original['metrics']['triangles'])
                check(tag+' '+engine+' native texture dimensions/conventions',
                      len(details['textures']) == 3 and
                      all(t['width'] == protocol['texture_resolution'] and
                          t['height'] == protocol['texture_resolution'] for t in details['textures']) and
                      [t['srgb'] for t in details['textures']] == [True, False, False])
                if engine == 'unity':
                    check(tag+' Unity collider triangle count',
                          details['collisionTriangles'] == collision['output_triangles'])
                    check(tag+' Unity visual queries', details['passageSamples'] > 0 and
                          details['passagePassed'] == details['passageSamples'])
                    check(tag+' Unity collision queries', details['collisionSamples'] > 0 and
                          details['collisionPassed']==details['collisionSamples'] and
                          details['collisionSamples'] == original['metrics']['inspected_centers'])
                else:
                    check(tag+' Unreal collider triangle count',
                          details['collision_triangles'] == collision['output_triangles'])
                    check(tag+' Unreal visual queries', details['passage_samples'] > 0 and
                          details['passage_passed'] == details['passage_samples'])
                    check(tag+' Unreal collision queries', details['collision_samples'] > 0 and
                          details['collision_passed']==details['collision_samples'] and
                          details['collision_samples'] == original['metrics']['inspected_centers'])
            comparisons.append(dict(group=group, seed=seed, baseline=baseline['metrics'],
                current=original['metrics'], original_seconds=original['elapsed_s'],
                replay_seconds=repeated['elapsed_s'], peak_rss_mib=max(original['peak_rss_mib'],repeated['peak_rss_mib']),
                required_path_count=len(capsule['paths']),
                route_placement_queries=capsule.get('placement_queries',0),
                repaired_path_count=sum(p.get('placement_repair',{}).get('passed',False) for p in capsule['paths']),
                required_swept_edges=sum(p['samples']-1 for p in capsule['paths']),
                collider_source_triangles=collision['source_triangles'],
                collider_output_triangles=collision['output_triangles'],
                collider_reduction=collision['achieved_reduction'],
                collider_raw_fallback=collision['used_raw_fallback'],
                collider_max_sampled_error_m=(collision if collision['used_raw_fallback'] else selected)['surface_deviation']['max_sampled_error_m'],
                precision_relaxed_vertices=precision['relaxed_vertices'],
                precision_maximum_vertex_change_m=precision['maximum_vertex_change_m'],
                original_network_matches_baseline=original['identity']['network']==baseline['identity']['network'],
                global_relief_scale=surface['relief_scale'],
                baseline_global_relief_scale=old_surface['relief_scale'],
                resolution_section_count=quality['resolution']['section_count'],
                under_resolved_count=quality['resolution']['under_resolved_count'],
                warnings=quality['warnings'],
                local_relief_regions=len(surface.get('local_relief_regions',[])),
                native_views_per_engine=plan['view_count'],
                native_results={engine:value['native'] for engine,value in native['checks'].items()}))
    convergence = read(EVIDENCE/'resolution_original.json')
    check('real refinement source', convergence['source_sha256']==protocol['source_sha256'])
    check('real refinement exact cold replay', convergence==read(EVIDENCE/'resolution_replay.json'))
    check('real refinement accepted', convergence['positive']['passed'] and convergence['clearance']['passed'])
    check('real insufficient convergence rejected', not convergence['unconverged_rejected']['passed'])
    check('real allocation budget rejected', not convergence['allocation_rejected']['passed'])
    placement = read(EVIDENCE/'production_path_repair.json')
    check('previously rejected exact mesh route repaired', placement['passed'] and
          placement['placement_queries'] == 129 and all(p['passed'] for p in placement['paths']))
    return dict(passed=all(c['passed'] for c in checks), checks=checks,
                verified_artifact_receipts=receipt_count, comparisons=comparisons)


if __name__ == '__main__':
    result = audit()
    (EVIDENCE/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"{sum(c['passed'] for c in result['checks'])}/{len(result['checks'])} checks passed")
    raise SystemExit(int(not result['passed']))
