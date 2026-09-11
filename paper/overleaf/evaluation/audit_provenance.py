"""Reconstruct provenance/configuration identities without generating geometry.

Run with the matching frozen PYTHONPATH and thread environment. Original case
records are read only; reconstructed identities are verified before saving.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.evaluation.config import load_evaluation_config
from plume_advanced.evaluation.experiments.common import config_hash, for_seed, with_routing_condition
from plume_advanced.evaluation.experiments.scalability_worker import _benchmark_project
from plume_advanced.evaluation.provenance import capture_provenance, directory_identity

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[2]
CONTROL_KEYS = {'distributary':'distributary_tendency', 'duration':'duration_scale',
                'inflation':'inflation', 'supply':'supply_rate_scale'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    args = parser.parse_args()
    name = args.experiment
    config = load_evaluation_config(CAMPAIGN/'frozen/paper/experiments.toml')
    section = config.section(name)
    resolved = {'experiment': section}
    inputs = [config.project_config, config.seed_file(name)]
    if name == 'morphometry':
        dataset = directory_identity(ROOT/'data/reference/pdc_v2/extracted/Pyroduct Digital Catalog in .txt')
        resolved.update(dataset_sha256=dataset['sha256'], reference_partition='evaluation', max_seeds=None)
        inputs.append(config.pdc_partition_path('evaluation'))
    provenance = capture_provenance(config.project_config.parent.parent,
                                    resolved_config=resolved, inputs=inputs)
    source = config.output_root/name
    records = [json.loads(p.read_text()) for p in sorted((source/'cases').glob('*.json'))]
    expected = json.loads((CAMPAIGN/'freeze_manifest.json').read_text())['planned_cases'][name]
    if len(records) != expected:
        raise ValueError(f'{name}: {len(records)}/{expected} records; wait for completion')
    bases = {}
    if name == 'controllability':
        for row in records:
            control, value = row['condition_id'].split('=')
            bases.setdefault(row['condition_id'], None)
            if bases[row['condition_id']] is None:
                bases[row['condition_id']] = load_project_config(config.project_config,
                    world_body='earth', dev_mode=False,
                    flow_regime_overrides={CONTROL_KEYS[control]:float(value)})
    else:
        bases['default'] = load_project_config(config.project_config, world_body='earth',
                                               dev_mode=name in {'export_consistency','determinism'})
    verified = []
    for row in records:
        if row['provenance_sha256'] != provenance['identity_sha256']:
            raise ValueError(f'{name}/{row["run_id"]}: provenance mismatch')
        if name == 'controllability':
            project = for_seed(bases[row['condition_id']], row['seed'])
        elif name == 'scalability':
            length, mode = row['condition_id'].split('m-')
            project = _benchmark_project(bases['default'], row['seed'], float(length), mode)
        else:
            project = for_seed(bases['default'], row['seed'])
            if name == 'host_ablation':
                project = with_routing_condition(project, row['condition_id'])
        digest = config_hash(project)
        if name == 'scalability':
            if row['resolved_config_sha256'] != provenance['resolved_config_sha256']:
                raise ValueError('Benchmark declaration hash mismatch')
            if row['status'] == 'complete' and row['project_config_sha256'] != digest:
                raise ValueError('Benchmark worker project hash mismatch')
        elif row['resolved_config_sha256'] != digest:
            raise ValueError(f'{name}/{row["run_id"]}: project configuration mismatch')
        verified.append({'run_id':row['run_id'], 'project_config_sha256':digest,
                         'status':row['status']})
    target = CAMPAIGN/'verified_provenance'
    target.mkdir(exist_ok=True)
    result = {'experiment':name,'verified_records':len(records),
        'statuses':dict(Counter(r['status'] for r in records)),
        'provenance':provenance, 'verified_configurations':verified,
        'resolved_base_configurations':{key:project_config_manifest(value) for key,value in bases.items()},
        'scope':'Post-execution reconstruction, checked against every saved case identity. No generation or mutation of case records. Benchmark failed-case project hashes are reconstructed from frozen inputs; complete-case worker hashes also match.'}
    path = target/f'{name}.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'experiment':name,'verified':len(records),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))

if __name__ == '__main__':
    main()
