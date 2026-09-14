"""Run the six independent native inspections as their original assets finish."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO/'outputs/mobility_repair_campaign_20260914'
JOURNAL = ROOT/'native_jobs.json'
UNITY = '/home/gabriel/Unity/Hub/Editor/6000.6.0f1/Editor/Unity'
UNREAL = '/home/gabriel/Downloads/Linux_Unreal_Engine_5.8.2/Engine/Binaries/Linux/UnrealEditor'


def main():
    pending = [(group, i, seed) for i, seed in enumerate([1,42,4294967295])
               for group in ['single','multi']]
    jobs = []
    deadline = time.monotonic()+14400
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(f'Generation/native campaign deadline; pending {pending}')
        for item in list(pending):
            group, i, seed = item
            run = ROOT/group/f'case_{i:04d}'/'attempt_0000'
            result_path = run/'result.json'
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text())
            if result.get('status') not in ('passed','failed','timeout'):
                continue
            job = dict(group=group, seed=seed, run=str(run), status='running')
            jobs.append(job)
            pending.remove(item)
            JOURNAL.write_text(json.dumps(jobs,indent=2)+'\n')
            if result['status'] != 'passed':
                job.update(status='skipped_failed_generation', reason=result.get('failure','Generation did not pass'))
            else:
                output = ROOT/'native'/f'{group}_seed{seed}'
                command = [sys.executable, str(REPO/'scripts/check_native_engines.py'), str(run),
                           '--output', str(output), '--unity', UNITY, '--unreal', UNREAL,
                           '--timeout','1800']
                log = ROOT/f'native_{group}_seed{seed}.log'
                started = time.monotonic()
                print(f'Inspecting {group} seed {seed}',flush=True)
                with log.open('w') as stream:
                    outcome = subprocess.run(command,cwd=REPO,env=os.environ.copy(),stdout=stream,
                                             stderr=subprocess.STDOUT,timeout=7200)
                job.update(status='passed' if outcome.returncode==0 else 'failed',
                           returncode=outcome.returncode, elapsed_s=time.monotonic()-started,
                           output=str(output),log=str(log))
            JOURNAL.write_text(json.dumps(jobs,indent=2)+'\n')
            print(f'{group} seed {seed}: {job["status"]}',flush=True)
        if pending:
            time.sleep(5)
    return int(any(j['status']!='passed' for j in jobs))


if __name__ == '__main__':
    raise SystemExit(main())
