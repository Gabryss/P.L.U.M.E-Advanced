"""Replot all resource outcomes using the preserved reporting implementation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[2]
SOURCE = ROOT/'paper/outputs/scalability/raw_results.csv'
PLOTTER = CAMPAIGN/'reporting_source/plotting.py'


def main():
    spec = importlib.util.spec_from_file_location('campaign_plotting',PLOTTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = ROOT/'paper/outputs/figures'
    output.mkdir(exist_ok=True)
    module._scalability_figure(SOURCE,output)
    target = ROOT/'paper/overleaf/figures/f10_scalability.png'
    shutil.copyfile(output/'figure_scalability.png',target)
    manifest = {'figure':str(target.relative_to(ROOT)),
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'source':str(SOURCE.relative_to(ROOT)),
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'reporting_implementation_sha256':hashlib.sha256(PLOTTER.read_bytes()).hexdigest(),
        'scope':'All 40 outcomes, including five dense 5 km memory-limit failures. Failure times are stopped costs, not completed reconstruction times.'}
    (CAMPAIGN/'scalability_figure.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__ == '__main__':
    main()
