"""Update delivery metadata after the campaign and PDF checks are complete."""
from pathlib import Path
import hashlib
import json
import shutil

CAMPAIGN=Path(__file__).resolve().parent
ROOT=CAMPAIGN.parents[2]
PAPER=ROOT/'paper/overleaf'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    report=read(CAMPAIGN/'report.json')
    execution=read(CAMPAIGN/'execution.json')
    assert execution['status']=='finished'
    qa=read(CAMPAIGN/'release_qa.json')
    assert qa['visual_review_completed']
    assert qa['pdf_sha256']==digest(ROOT/'tmp/pdfs/build/main.pdf')
    assert qa['main_tex_sha256']==digest(PAPER/'main.tex')
    path=PAPER/'provenance.json'
    legacy=CAMPAIGN/'paper_provenance_before_campaign.json'
    if not legacy.exists():
        shutil.copyfile(path,legacy)
    previous=read(legacy)
    payload=dict(previous)
    payload['review_date']='2026-09-07'
    payload['generator_modified_by_manuscript_review']=True
    payload['quantitative_campaign_executed_by_manuscript_review']=True
    payload['campaign_changes_scope']='Evaluation instrumentation, failure accounting, statistics, and full control-input resolution. No generator parameter fitting to evaluation data. Default resolved configuration unchanged by control amendment.'
    payload['reviewed_source_sha256']={name:digest(ROOT/name)
        for name in previous['reviewed_source_sha256'] if (ROOT/name).is_file()}
    payload['previous_event_illustration_validation']=previous['validation']
    payload['validation']=qa
    payload['notes']=[
        'All 2,076 declared cases were attempted; completion and failed checks are reported separately.',
        'The PDC evaluation partition was not historically unseen: earlier whole-catalog exploration is disclosed.',
        'Both executing source snapshots include preexisting uncommitted development changes.',
        'No model parameters were fitted after examining campaign outcomes.',
        'Frozen-source hashes, reconstructed per-case configurations and input identities are bundled under evaluation/.',
        'Application packages were checked automatically; no application imports or simulator contact tests were completed.',
        'Earlier figures retain their declared illustrative/development scope; Figures 9 and 10 use campaign measurements.',
        'Recorded absolute paths refer to the original workstation; see evaluation/PACKAGE_SCOPE.md for reproduction scope.'
    ]
    payload['figure_revision']='Measured morphology and resource figures added; Tables IV–VI populated from audited campaign results. Existing rock and atlas illustrations preserved.'
    payload['inserted_figures']=previous['inserted_figures']+[{
        'file':'figures/f09_morphometry.png',
        'source':'evaluation/measurements/morphometry/raw_*_sections.csv',
        'status':'Measured Stage-C morphology: 100 Earth worlds, 19 PDC evaluation caves, 200 reference contours.'}, {
        'file':'figures/f10_scalability.png',
        'source':'evaluation/measurements/scalability/raw_results.csv',
        'status':'All 40 benchmark attempts at 0.6 m; 35 complete and five dense 5 km memory-limit failures.'}]
    for item in payload['inserted_figures']:
        item['sha256']=digest(PAPER/item['file'])
    payload['campaign']={
        'date':'2026-09-07', 'inventory':report['inventory'],
        'original_source_freeze':read(CAMPAIGN/'freeze_manifest.json')['file_manifest_sha256'],
        'control_source_freeze':read(CAMPAIGN/'controls_amendment.json')['file_manifest_sha256'],
        'report_sha256':digest(CAMPAIGN/'report.json'),
        'execution_sha256':digest(CAMPAIGN/'execution.json'),
        'source_integrity':read(CAMPAIGN/'source_integrity.json'),
        'reporting_source_sha256':{str(p.relative_to(CAMPAIGN)):digest(p) for p in sorted(CAMPAIGN.rglob('*.py'))
            if 'frozen' not in p.parts and 'frozen_controls_v2' not in p.parts},
        'verified_provenance_sha256':{p.name:digest(p) for p in sorted((CAMPAIGN/'verified_provenance').glob('*.json'))},
        'figure_inputs':{'morphometry':read(CAMPAIGN/'morphometry_figure.json'),
                        'scalability':read(CAMPAIGN/'scalability_figure.json')},
    }
    pdf=ROOT/'tmp/pdfs/build/main.pdf'
    for target in (PAPER/'PLUME_Advanced_With_Figures.pdf',ROOT/'output/pdf/PLUME_Advanced_With_Figures.pdf'):
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(pdf,target)
    files=['main.tex','references.tex','PLUME_Advanced_With_Figures.pdf','README.md','FIGURE_GUIDE.md','REVIEW_NOTES.md']
    files.extend(str(p.relative_to(PAPER)) for p in sorted((PAPER/'results').glob('*.tex')))
    payload['deliverable_sha256']={name:digest(PAPER/name) for name in files}
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print('Paper provenance updated from verified campaign and release QA.')

if __name__=='__main__':
    main()
