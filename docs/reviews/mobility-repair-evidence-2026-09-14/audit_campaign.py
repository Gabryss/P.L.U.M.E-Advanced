"""Independently verify frozen recipes, every asset receipt and native coverage."""
import json
from pathlib import Path

from plume_advanced.identity import package_source_hash, sha256_file

EVIDENCE = Path(__file__).resolve().parent
REPO = EVIDENCE.parents[2]
ROOT = REPO/'outputs/mobility_repair_campaign_20260914'
BASELINE = REPO/'outputs/textured_campaign_20260913'


def read(path):
    return json.loads(path.read_text())


def audit():
    protocol = read(EVIDENCE/'protocol.json')
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
    check('executing source unchanged', package_source_hash() == protocol['source_sha256'])
    for recipe in protocol['recipes']:
        check('recipe '+recipe['config'], sha256_file(recipe['config']) == recipe['sha256'])
    for name, digest in protocol['files'].items():
        check('source/dependency '+name, sha256_file(REPO/name) == digest)
    comparisons = []
    receipt_count = 0
    for group in ('single','multi'):
        for i, seed in enumerate(protocol['seeds']):
            tag = f'{group}/{seed}'
            run = ROOT/group/f'case_{i:04d}'/'attempt_0000'
            replay = ROOT/group/f'replay_{i:04d}'/'attempt_0000'
            records = []
            for label, path in [('original',run),('replay',replay)]:
                result = read(path/'result.json')
                records.append(result)
                check(tag+' '+label+' passed', result['status']=='passed')
                check(tag+' '+label+' source', result['source_sha256']==protocol['source_sha256'])
                check(tag+' '+label+' seed', result['case']['seed']==seed)
                for relative, digest in result['artifacts'].items():
                    asset = path/relative
                    check(tag+' '+label+' '+relative, asset.resolve().is_relative_to(path.resolve())
                          and asset.is_file() and sha256_file(asset)==digest)
                    receipt_count += 1
            original, repeated = records
            check(tag+' exact cold identities', original['identity']==repeated['identity'])
            check(tag+' different Python hash seed', original['python_hash_seed']!=repeated['python_hash_seed'])
            check(tag+' replay runtime', original['runtime']==repeated['runtime'])
            old_group = 'single_250' if group=='single' else 'multi'
            baseline = read(BASELINE/old_group/f'case_{i:04d}'/'attempt_0000'/'result.json')
            check(tag+' preserved original host', original['identity']['host']==baseline['identity']['host'])
            quality = read(run/'pipeline_quality_report.json')
            check(tag+' pipeline accepted', quality['passed'])
            capsule = quality['required_route_clearance']
            check(tag+' half-metre requirement', capsule['enabled'] and capsule['passed']
                  and capsule['height_m']==.5 and capsule['width_m']==.5 and capsule['margin_m']==.02)
            check(tag+' every required path swept', bool(capsule['paths']) and
                  all(p['passed'] and not p['blocked_edges'] and not p['invalid_samples'] for p in capsule['paths']))
            collision = quality['export_inspection']['collision']
            check(tag+' collider passed', collision['inspection']['passed'])
            check(tag+' collision capsule', collision['inspection']['traversal']['passed'])
            check(tag+' bounded collider attempts', len(collision['attempts']) <= 4)
            check(tag+' collider reduction reported', collision['method']=='quadric_edge_collapse'
                  and collision['output_triangles'] <= collision['source_triangles'])
            if not collision['used_raw_fallback']:
                selected = next(row for row in collision['attempts'] if row['accepted'])
                check(tag+' collision surface tolerance', selected['surface_deviation']['passed']
                      and selected['surface_deviation']['max_sampled_error_m'] <= .03)
            check(tag+' texture acceptance', quality['texture_recovery']['passed'])
            native_root = ROOT/'native'/f'{group}_seed{seed}'
            native = read(native_root/'native_summary.json')
            receipt = read(native_root/'native_input_receipt.json')
            plan = read(native_root/'view_plan.json')
            check(tag+' native passed', native['passed'])
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
                details = result['native']
                if engine == 'unity':
                    check(tag+' Unity collision queries', details['collisionSamples'] > 0 and
                          details['collisionPassed']==details['collisionSamples'])
                else:
                    check(tag+' Unreal collision queries', details['collision_samples'] > 0 and
                          details['collision_passed']==details['collision_samples'])
            comparisons.append(dict(group=group, seed=seed, baseline=baseline['metrics'],
                current=original['metrics'], original_seconds=original['elapsed_s'],
                replay_seconds=repeated['elapsed_s'], peak_rss_mib=max(original['peak_rss_mib'],repeated['peak_rss_mib']),
                required_path_count=len(capsule['paths']),
                required_swept_edges=sum(p['samples']-1 for p in capsule['paths']),
                collider_source_triangles=collision['source_triangles'],
                collider_output_triangles=collision['output_triangles'],
                collider_reduction=collision['achieved_reduction'],
                collider_raw_fallback=collision['used_raw_fallback'],
                native_views_per_engine=plan['view_count'],
                native_results={engine:value['native'] for engine,value in native['checks'].items()}))
    convergence = read(EVIDENCE/'resolution_original.json')
    check('real refinement source', convergence['source_sha256']==protocol['source_sha256'])
    check('real refinement exact cold replay', convergence==read(EVIDENCE/'resolution_replay.json'))
    check('real refinement accepted', convergence['positive']['passed'] and convergence['clearance']['passed'])
    check('real insufficient convergence rejected', not convergence['unconverged_rejected']['passed'])
    check('real allocation budget rejected', not convergence['allocation_rejected']['passed'])
    return dict(passed=all(c['passed'] for c in checks), checks=checks,
                verified_artifact_receipts=receipt_count, comparisons=comparisons)


if __name__ == '__main__':
    result = audit()
    (EVIDENCE/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"{sum(c['passed'] for c in result['checks'])}/{len(result['checks'])} checks passed")
    raise SystemExit(int(not result['passed']))
