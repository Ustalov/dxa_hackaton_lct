"""Native-image entry point for explicit-v2, without loading classification weights."""
import json
from pathlib import Path
import torch
from .unified import UnifiedAnalyzer
from .explicit_rules import apply_iliac_rules, evaluate_explicit
from .imaging import read_dicom

DEFAULT_CONFIG='/data/dxa-explicit-rules-v2/config.json'

class ExplicitAnalyzer(UnifiedAnalyzer):
    def __init__(self,config=DEFAULT_CONFIG,device=None):
        self.config=json.loads(Path(config).read_text())
        torch.set_num_threads(4)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.nets={};self.load_anatomy()

    def load_anatomy(self):
        from .anatomy_network import AnatomyNet
        from .hip_v6 import HipNet
        from .isbi2020_adapter import make_model
        root=Path(__file__).resolve().parents[1]
        for name,net in [('spine',AnatomyNet('spine')),('hip',HipNet())]:
            cp=torch.load(root/'artifacts/png_full_v8'/name/'best.pt',map_location='cpu',weights_only=False)
            net.load_state_dict(cp['state_dict']);self.nets[name]=net.to(self.device).eval()
        self.nets['corners']=make_model(Path('/data/dxa-three-stage/stage3/best.pt')).to(self.device).eval()

    def analyze(self,path,destination=None):
        from .explicit_render import save_explicit_outputs
        import numpy as np
        raw,meta=read_dicom(path);case,bone,labels=self.anatomy(raw,meta)
        iliac=np.zeros(raw.shape,'uint8')
        if case['region']=='spine':case,iliac=apply_iliac_rules(case,bone,self.config['iliac_area_threshold_mm2'])
        result,foreign=evaluate_explicit(case,raw)
        result.update(source_path=str(path),metadata=meta,measurements=case['measurements'])
        if destination:save_explicit_outputs(Path(destination),raw,case,bone,labels,iliac,foreign,result)
        return result

    def classify(self,*args,**kwargs):
        raise RuntimeError('Classification heads are disabled in explicit-v2; use analyze/evaluate_explicit')
