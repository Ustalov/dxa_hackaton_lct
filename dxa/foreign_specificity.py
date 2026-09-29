"""Foreign-object inference under a retrospective zero-FP calibration policy."""
import json,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from .foreign_objects import ForeignObjectNet,input_tensor

VARIANTS=['original','gamma_080','gamma_120','contrast_085','contrast_115','blur_060','unsharp_050','horizontal_flip']

def variants(image):
    a=image.astype('float32')/255
    def u(x):return np.uint8(np.rint(np.clip(x,0,1)*255))
    return [image,u(a**.8),u(a**1.2),u((a-.5)*.85+.5),u((a-.5)*1.15+.5),
            cv2.GaussianBlur(image,(0,0),.6),u(a+.5*(a-cv2.GaussianBlur(a,(0,0),.8))),np.fliplr(image).copy()]

class ForeignSpecificity:
    def __init__(self,model_dir='/data/dxa-foreign-model-v3',policy=None):
        torch.set_num_threads(4);self.models={};self.model_dir=Path(model_dir)
        self.policy=json.loads(Path(policy).read_text()) if policy else None
        for key,name in [('best','best.pt'),('last','last.pt')]:
            if self.policy and ((key=='last' and self.policy['best_weight']==1) or (key=='best' and self.policy['best_weight']==0)):continue
            if self.policy and hashlib.sha256((self.model_dir/name).read_bytes()).hexdigest()!=self.policy['weight_sha256'][name]:
                raise ValueError('Model weights no longer match the calibrated policy')
            cp=torch.load(self.model_dir/name,map_location='cpu',weights_only=True)
            net=ForeignObjectNet();net.load_state_dict(cp['state_dict']);self.models[key]=net.eval()

    @torch.no_grad()
    def features(self,image):
        x=torch.stack([input_tensor(im)[0] for im in variants(image)])
        return {key:net(x)['logit'].tolist() for key,net in self.models.items()}

    @staticmethod
    def score(features,config):
        w=config['best_weight']
        combined=np.array(features['best']) if w==1 else np.array(features['last']) if w==0 else w*np.array(features['best'])+(1-w)*np.array(features['last'])
        method=config['aggregation']
        return float(combined[0] if method=='original' else combined.min() if method=='minimum' else np.quantile(combined,.25) if method=='q25' else combined.mean())

    @staticmethod
    def decide(features,density_present,policy):
        score=ForeignSpecificity.score(features,policy)
        key=str(int(density_present)) if policy['density_stratified'] else 'all'
        threshold=policy['thresholds'][key]
        return dict(present=bool(score>=threshold),score=score,threshold=threshold,score_units='combined logit',
                    density_present=bool(density_present),calibration_scope=policy.get('calibration_scope','all 99 historical labeled spine DICOM; not independent test'),
                    zero_fp_guaranteed_on_new_data=False)
