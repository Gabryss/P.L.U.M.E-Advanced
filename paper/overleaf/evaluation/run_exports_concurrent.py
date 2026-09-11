"""Overlap non-benchmark package work with the remaining network-only sweep.

The main driver will later reuse matching completed cases and refresh the export
summary. Its command entry is not edited by this helper.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

campaign=Path(__file__).resolve().parent
root=campaign.parents[2]
state=json.loads((campaign/'execution.json').read_text())
assert state['commands']['scalability'].get('finished_at_utc')
assert 'export-consistency' not in state['commands']
env=os.environ.copy()
env.update(PYTHONPATH=str(campaign/'frozen/src'),PYTHONDONTWRITEBYTECODE='1',
    PYTHONUNBUFFERED='1',PLUME_EVALUATION_WORKERS='4',OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
    MPLCONFIGDIR='/tmp/plume-evaluation-mpl')
command=[sys.executable,'-m','plume_advanced.evaluation.cli','--config',
         str(campaign/'frozen/paper/experiments.toml'),'export-consistency']
record={'started_at_utc':datetime.datetime.now(datetime.UTC).isoformat(),
        'argv':command,'package_source':env['PYTHONPATH'],
        'thread_environment':{k:env[k] for k in ('PLUME_EVALUATION_WORKERS','OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        'reason':'Overlap sequential non-benchmark exports with controllability after all isolated resource measurements have finished. Matching completed records will be reused by the main driver; no case is forced or replaced.',
        'log':str(campaign/'logs/exports-concurrent.log')}
path=campaign/'concurrent_exports.json'
path.write_text(json.dumps(record,indent=2)+'\n')
started=time.monotonic()
with (campaign/'logs/exports-concurrent.log').open('w') as log:
    result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
record.update(returncode=result.returncode,elapsed_s=time.monotonic()-started,
              finished_at_utc=datetime.datetime.now(datetime.UTC).isoformat())
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
