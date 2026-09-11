"""Execute the remaining non-benchmark diagnostics alongside controllability.

The main driver later reuses matching case records. All isolated benchmarks
have already finished. No command entry or original case is overwritten here.
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
assert state['active_command']=='controllability'
assert json.loads((campaign/'concurrent_exports.json').read_text()).get('returncode')==0
env=os.environ.copy()
env.update(PYTHONPATH=str(campaign/'frozen/src'),PYTHONDONTWRITEBYTECODE='1',
    PYTHONUNBUFFERED='1',PLUME_EVALUATION_WORKERS='4',OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
    MPLCONFIGDIR='/tmp/plume-evaluation-mpl')
record={'reason':'Independent sampling and A-C repeatability cases overlap the remaining control sweep after all isolated resource benchmarks and export generation finish. The main driver later reuses matching completed cases.',
        'package_source':env['PYTHONPATH'], 'commands':{}}
path=campaign/'concurrent_auxiliary.json'
for name in ('sampling-ablation','determinism'):
    assert name not in json.loads((campaign/'execution.json').read_text())['commands']
    command=[sys.executable,'-m','plume_advanced.evaluation.cli','--config',
             str(campaign/'frozen/paper/experiments.toml'),name]
    item={'started_at_utc':datetime.datetime.now(datetime.UTC).isoformat(),
          'argv':command,'configured_workers':4,
          'log':str(campaign/f'logs/{name}-concurrent.log')}
    record['commands'][name]=item
    path.write_text(json.dumps(record,indent=2)+'\n')
    started=time.monotonic()
    with Path(item['log']).open('w') as log:
        result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    item.update(returncode=result.returncode,elapsed_s=time.monotonic()-started,
                finished_at_utc=datetime.datetime.now(datetime.UTC).isoformat())
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(name,result.returncode,flush=True)
record['finished_at_utc']=datetime.datetime.now(datetime.UTC).isoformat()
path.write_text(json.dumps(record,indent=2)+'\n')
