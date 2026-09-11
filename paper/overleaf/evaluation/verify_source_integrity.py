"""Verify both preserved package snapshots against their original manifests."""
from pathlib import Path
import hashlib
import json

CAMPAIGN=Path(__file__).resolve().parent


def main():
    report={}
    for folder, filename in [('frozen','freeze_manifest.json'),('frozen_controls_v2','controls_amendment.json')]:
        manifest=json.loads((CAMPAIGN/filename).read_text())
        mismatches=[]
        for relative, expected in manifest['files'].items():
            source=CAMPAIGN/folder/relative
            actual=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
            if actual!=expected:
                mismatches.append(relative)
        if mismatches:
            raise ValueError(f'{folder}: source integrity failed for {mismatches}')
        report[folder]={'verified_files':len(manifest['files']),
                         'file_manifest_sha256':manifest['file_manifest_sha256'],
                         'all_content_hashes_match':True}
    original=json.loads((CAMPAIGN/'freeze_manifest.json').read_text())
    root=CAMPAIGN.parents[2]
    for relative, expected in original['external_asset_sha256'].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'External asset changed: {relative}')
    report['external_assets']={'verified_files':len(original['external_asset_sha256']),
                               'all_content_hashes_match':True}
    (CAMPAIGN/'source_integrity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
