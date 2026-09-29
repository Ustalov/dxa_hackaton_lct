from pathlib import Path
import json,re
import numpy as np
import joblib,torch
from .imaging import handcrafted,canonical,letterbox
from .features import FeatureExtractor,context_features

def available_models(artifacts):
    root=Path(artifacts);registry=root/'model_registry.json'
    if registry.exists():return json.loads(registry.read_text())
    return {p.stem:{'file':str(p.relative_to(root)),'label':p.stem} for p in (root/'models').iterdir() if p.suffix in ('.joblib','.pt') and re.fullmatch(r'[A-Za-z0-9_.-]+',p.stem)}

def model_feature_keys(bundle):
    if 'feature_keys' in bundle:return set(bundle['feature_keys'])
    if 'components' in bundle:return set().union(*(model_feature_keys(c) for c in bundle['components']))
    if 'model' not in bundle:return {'geo'}
    model=bundle['model']
    if hasattr(model,'configs'):return {k for c in model.configs for k in c['features']}
    if hasattr(model,'components'):return set().union(*(model_feature_keys({'model':m}) for m in model.components))
    from .modeling import VARIANTS
    key=VARIANTS[model.variant][0]
    return {'deep','geo','context'} if key=='context' else {'deep','geo'} if key=='hybrid' else {key}

class ModelRunner:
    def __init__(self,artifacts,variant=None,bundle=None):
        self.artifacts=Path(artifacts);torch.set_num_threads(4)
        self.foundations={};self.legacy=None;self.net=None;self.neural_info=None;self.neural_component=None
        if bundle is None:
            if variant:
                registry=available_models(self.artifacts)
                if variant not in registry:raise ValueError('Unknown model variant')
                path=(self.artifacts/registry[variant]['file']).resolve()
                if not path.is_relative_to(self.artifacts.resolve()):raise ValueError('Invalid model path')
            else:path=self.artifacts/'model.joblib'
            if path.suffix=='.pt':
                checkpoint=torch.load(path,map_location='cpu',weights_only=False);info=checkpoint['info'];self.neural_info=info
                if info.get('kind')=='dino_finetune':
                    from .fine_dino import FineDINO
                    self.net=FineDINO()
                else:
                    from .neural import DXANet
                    self.net=DXANet(info['hybrid'])
                self.net.load_state_dict(checkpoint['state_dict']);self.net.eval()
                bundle={'name':info.get('name',variant),'thresholds':np.asarray(info['thresholds']),'policy':info.get('policy','independent'),'feature_keys':['geo']}
            else:bundle=joblib.load(path)
        self.bundle=bundle;self.keys=model_feature_keys(bundle)
        if bundle.get('specialist_count')==5:
            bundle['model'].artifacts=self.artifacts;bundle['model']._tasks=None
        if bundle.get('neural_component'):self.neural_component=ModelRunner(self.artifacts,bundle['neural_component'])

    def _foundation(self,name):
        if name not in self.foundations:
            from .advanced_features import FoundationExtractor
            self.foundations[name]=FoundationExtractor(self.artifacts,name,device='cpu',download=False)
        return self.foundations[name]

    def extract(self,images,metas):
        hog=[];geo=[];details=[]
        for a,m in zip(images,metas):
            hh,gg,detail=handcrafted(a,m);hog.append(hh);geo.append(gg);details.append(detail)
        features={'hog':np.asarray(hog),'geo':np.asarray(geo)}
        if 'deep' in self.keys:
            if self.legacy is None:self.legacy=FeatureExtractor(self.artifacts/'resnet18-f37072fd.pth')
            features.update(self.legacy.extract(images,metas)[0])
        if self.keys & {'shape','contour'}:
            from .advanced_features import shape_features,contour_features
            if 'shape' in self.keys:features['shape']=np.asarray([shape_features(a,m) for a,m in zip(images,metas)])
            if 'contour' in self.keys:features['contour']=np.asarray([contour_features(a,m) for a,m in zip(images,metas)])
        from .advanced_features import FOUNDATIONS,detail_view,contrast_view
        for name in FOUNDATIONS:
            if any(k.startswith(name+'_') for k in self.keys):
                extractor=self._foundation(name);features.update(extractor.extract(images,metas))
                if any(k.startswith(name+'_detail_') for k in self.keys):
                    cropped=[detail_view(a,m) for a,m in zip(images,metas)];a,m=zip(*cropped)
                    extra=extractor.extract(a,m);features.update({k.replace(name+'_',name+'_detail_'):v for k,v in extra.items()})
                if any(k.startswith(name+'_clahe_') for k in self.keys):
                    extra=extractor.extract([contrast_view(a) for a in images],metas)
                    features.update({k.replace(name+'_',name+'_clahe_'):v for k,v in extra.items()})
        if self.net is not None:features['_direct_prob']=self.neural_predict(images,metas,features['geo'])
        if self.neural_component:features['_neural_prob']=self.neural_component.neural_predict(images,metas,features['geo'])
        return features,details

    def contextualize(self,features,metas):
        groups=[m['study_uid'] for m in metas];regions=[m['region'] for m in metas]
        if 'context' in self.keys:features['context']=context_features(features,groups,regions)
        if 'dino_small_context' in self.keys:
            from .advanced_features import representation_context
            features['dino_small_context']=representation_context(features,groups,regions)

    def neural_predict(self,images,metas,geo):
        if self.neural_info.get('kind')=='dino_finetune':
            from .fine_dino import image_tensor,predict
            tensors=torch.stack([image_tensor(a,m) for a,m in zip(images,metas)])
            return predict(self.net,tensors,np.arange(len(images)),device='cpu')
        from .neural import infer
        tensors=[]
        for a,m in zip(images,metas):
            t=torch.from_numpy(letterbox(canonical(a,m),224,m['spacing']).copy()).float().div(255).repeat(3,1,1)
            tensors.append((t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None])
        normalized=torch.tensor(np.clip((geo-self.neural_info['mean'])/self.neural_info['scale'],-5,5),dtype=torch.float32)
        return infer(self.net,torch.stack(tensors),normalized,np.arange(len(images)))

    def predict(self,features):
        if self.net is not None:p=features['_direct_prob']
        elif self.neural_component:p=self.bundle['model'].predict(features,neural_prob=features['_neural_prob'])
        else:
            from .service import predict_bundle
            p=predict_bundle(self.bundle,features)
        p=np.asarray(p)
        if p.ndim!=2 or p.shape[1]!=7 or not np.isfinite(p).all():raise ValueError('Invalid model output')
        if (p<0).any() or (p>1).any():raise ValueError('Model score outside [0, 1]')
        from .advanced_model import consistent_predictions
        b=consistent_predictions(p,self.bundle['thresholds'],self.bundle.get('policy','independent'))
        return p,b
