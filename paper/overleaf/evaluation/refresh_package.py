"""Bundle the audited campaign evidence with the finished Overleaf manuscript.

Run after final PDF rendering and visual review. Does not start any experiment.
"""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

CAMPAIGN=Path(__file__).resolve().parent
ROOT=CAMPAIGN.parents[2]
PAPER=ROOT/'paper/overleaf'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    execution=json.loads((CAMPAIGN/'execution.json').read_text())
    report=json.loads((CAMPAIGN/'report.json').read_text())
    if execution['status']!='finished':
        raise ValueError('The campaign driver has not finished')
    if sum(x['planned'] for x in report['inventory'].values())!=2076:
        raise ValueError('Unexpected campaign denominator')
    evidence=PAPER/'evaluation'
    evidence.mkdir(exist_ok=True)
    names=['README.md','REPORT.md','report.json','case_inventory.csv','execution.json',
           'freeze_manifest.json','controls_amendment.json','hardware.json',
           'requirements_frozen.txt','experiments_original.toml','morphometry_figure.json',
           'scalability_figure.json','scalability_diagnostics.json','source_integrity.json',
           'concurrent_exports.json','concurrent_auxiliary.json','release_qa.json',
           'sampling_diagnostics.json','export_asset_manifest.json']
    for name in names:
        shutil.copyfile(CAMPAIGN/name,evidence/name)
    for directory in ('verified_provenance','reporting_source'):
        shutil.copytree(CAMPAIGN/directory,evidence/directory,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('build_report.py','plot_morphometry.py','plot_scalability.py','audit_provenance.py',
                 'run_campaign.py','run_campaign_v1.py','refresh_package.py','verify_source_integrity.py',
                 'run_exports_concurrent.py','run_auxiliary_concurrent.py','update_paper_provenance.py',
                 'hash_export_assets.py'):
        shutil.copyfile(CAMPAIGN/name,evidence/name)
    for snapshot in ('frozen','frozen_controls_v2'):
        manifest=json.loads((CAMPAIGN/('freeze_manifest.json' if snapshot=='frozen' else 'controls_amendment.json')).read_text())
        for relative in manifest['files']:
            source=CAMPAIGN/snapshot/relative
            target=evidence/snapshot/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    for name in report['inventory']:
        destination=evidence/'measurements'/name
        destination.mkdir(parents=True,exist_ok=True)
        origin=ROOT/'paper/outputs'/name
        for filename in ('summary.json','summary.csv','raw_results.csv'):
            if (origin/filename).exists():
                shutil.copyfile(origin/filename,destination/filename)
        if name=='morphometry':
            for source in origin.glob('raw_*_sections.csv'):
                shutil.copyfile(source,destination/source.name)
        if name in ('scalability','sampling_ablation','determinism','export_consistency'):
            shutil.copytree(origin/'cases',destination/'cases',dirs_exist_ok=True)
    (evidence/'PACKAGE_SCOPE.md').write_text('''# Evidence package scope

This directory preserves sources, manifests, resolved configurations, summary
statistics, per-case CSV measurements and morphology descriptor tables from the
7 September 2026 campaign. The full local campaign retains all original case
JSON files, logs and emitted application packages under `paper/outputs/`.
This compact Overleaf bundle omits the large morphology contour arrays and
application meshes; their measurements and configuration identities are retained.

The two frozen source trees document exactly what executed. Texture assets and
the original PDC source dataset are external, content-addressed inputs and are
not bundled. Recorded absolute paths refer to the original workstation. The
scripts preserve their original repository-relative layout; they are not
advertised as runnable from this relocated Overleaf evidence directory. To rerun
experiments, use the repository layout and the recorded inputs/dependencies.
PDF compilation requires none of these runtime dependencies or external inputs.

Figures 9 and 10 use the saved descriptor/resource measurements. Earlier figures
remain explicitly labelled method illustrations and development scenarios.
''')
    pdf=ROOT/'tmp/pdfs/build/main.pdf'
    for target in (PAPER/'PLUME_Advanced_With_Figures.pdf',ROOT/'output/pdf/PLUME_Advanced_With_Figures.pdf'):
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(pdf,target)
    manifest={str(p.relative_to(evidence)):digest(p) for p in sorted(evidence.rglob('*'))
              if p.is_file() and p.name!='BUNDLE_MANIFEST.json' and '__pycache__' not in p.parts}
    (evidence/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    zip_path=ROOT/'paper/PLUME_Advanced_Overleaf_With_Figures.zip'
    excluded={'PLUME_Advanced_Expanded.pdf'}
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(PAPER.rglob('*')):
            if not path.is_file() or path.name in excluded or '__pycache__' in path.parts:
                continue
            if path.suffix in {'.aux','.log','.out','.fls','.fdb_latexmk','.synctex.gz'}:
                continue
            archive.write(path,path.relative_to(PAPER))
    print(json.dumps({'pdf_sha256':digest(pdf),'zip_sha256':digest(zip_path),
                      'zip_size_mib':zip_path.stat().st_size/1024**2,'evidence_files':len(manifest)},indent=2))

if __name__=='__main__':
    main()
