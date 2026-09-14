"""Run the frozen campaign's accepted originals through native editors as they finish."""
import json
import subprocess
import sys
import time
from pathlib import Path

from plume_advanced.evaluation.reliability_reports import verify_result
from plume_advanced.identity import package_source_hash, sha256_file

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
protocol = json.loads((ROOT / 'protocol_v2.json').read_text())
remaining = [(group, i, case) for group, cases in protocol['groups'].items() for i, case in enumerate(cases)]
report = {'complete': False, 'passed': False, 'cases': [], 'scope': 'All six accepted originals; no native replay claim'}
started = time.monotonic()

def save():
    (ROOT / 'native_campaign.json').write_text(json.dumps(report, indent=2)+'\n')

save()
while remaining:
    found = False
    for group, index, case in list(remaining):
        run = ROOT / group / f'case_{index:04}' / 'attempt_0000'
        result_path = run / 'result.json'
        if not (run / 'result.sha256').exists():
            if time.monotonic() - started > 25000:
                raise TimeoutError('Generation inputs did not finish within the campaign limit')
            continue
        found = True
        remaining.remove((group, index, case))
        row = {'group': group, 'index': index, 'seed': case['seed'], 'run': str(run), 'passed': False}
        result, error = verify_result(run)
        if result is None or error or result['status'] != 'passed':
            row.update(status='input_rejected', failure=error or result.get('reason',result['status']))
            report['cases'].append(row)
            save()
            continue
        if package_source_hash() != protocol['source_sha256']:
            raise RuntimeError('Production source changed during native evaluation')
        output = ROOT / 'native_v2' / f'{group}_seed{case["seed"]}'
        log = ROOT / f'native_v2_{group}_seed{case["seed"]}.log'
        report['active'] = {'case': row, 'log': str(log)}
        save()
        command = [sys.executable, str(REPO / 'scripts/check_native_engines.py'), str(run),
            '--output', str(output),
            '--unity', '/home/gabriel/Unity/Hub/Editor/6000.6.0f1/Editor/Unity',
            '--unreal', '/home/gabriel/Downloads/Linux_Unreal_Engine_5.8.2/Engine/Binaries/Linux/UnrealEditor',
            '--timeout', '1800']
        print(f'Native inspection: {group} seed {case["seed"]}', flush=True)
        with log.open('w') as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=5500)
        summary = output / 'native_summary.json'
        row.update(exit_code=process.returncode, output=str(output), log=str(log),
                   input_glb_sha256=sha256_file(run / 'export/plume_cave.glb'))
        if summary.exists():
            native = json.loads(summary.read_text())
            row.update(passed=process.returncode==0 and native['passed'], summary=native)
        else:
            row.update(status='setup_failed', failure=log.read_text()[-3000:])
        report['cases'].append(row)
        report.pop('active',None)
        save()
    if not found:
        time.sleep(10)
report.update(complete=True, passed=all(row['passed'] for row in report['cases']))
save()
print(f'Native campaign complete: {sum(r["passed"] for r in report["cases"])} / {len(report["cases"])} passed',flush=True)
raise SystemExit(0 if report['passed'] else 1)
