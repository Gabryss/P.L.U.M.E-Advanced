"""Record the exact bytes of the three completed exported scene packages."""
from pathlib import Path
import hashlib
import json

campaign=Path(__file__).resolve().parent
root=campaign.parents[2]
source=root/'paper/outputs/export_consistency'
summary=json.loads((source/'summary.json').read_text())
assert summary['complete_n']==summary['planned_n']==3
records=[]
for path in sorted((source/'packages').rglob('*')):
    if not path.is_file():
        continue
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(block)
    records.append({'path':str(path.relative_to(root)), 'bytes':path.stat().st_size,
                    'sha256':digest.hexdigest()})
manifest={'scope':'File identity of the completed emitted packages; not an application-import, rendering or simulator-contact test.',
          'file_count':len(records),'bytes':sum(r['bytes'] for r in records),'files':records}
(campaign/'export_asset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({key:value for key,value in manifest.items() if key!='files'},indent=2))
