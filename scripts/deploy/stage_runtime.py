"""Collect the selected local weights; never collects medical images or passwords."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2]
local=['artifacts/model.joblib','artifacts/png_full_v8/hip/best.pt','artifacts/png_full_v8/spine/best.pt']
external=['dxa-ct-synth/model/region_cls/region_cls.pt','dxa-ct-synth/model/region_cls/info.json',
'dxa-ct-synth/model/v4/best.pt','dxa-ct-synth/model/spine_v1/best.pt','dxa-ct-synth/model/spine_v1/split.json',
'dxa-three-stage/stage3/best.pt','dxa-foreign-model-v3/best.pt',
'dxa-foreign-threshold-audit-v1/policy_validation_f1.json','dxa-explicit-rules-v2/config.json']
manifest=[]
for source_root,prefix,names in [(root,'app',local),(Path('/data'),'data',external)]:
    for name in names:
        source=source_root/name;dest=root/'runtime-data'/prefix/name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        manifest.append(dict(path=str(dest.relative_to(root/'runtime-data')),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
(root/'runtime-data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Runtime files:',len(manifest),'bytes:',sum(r['bytes'] for r in manifest))
