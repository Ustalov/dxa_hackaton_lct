import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'runtime-data'
manifest=json.loads((root/'manifest.json').read_text())
for row in manifest:
    path=root/row['path']
    if not path.resolve().is_relative_to(root.resolve()):raise SystemExit('Unsafe manifest path')
    assert path.stat().st_size==row['bytes'],path
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],path
print('Verified runtime files:',len(manifest))
