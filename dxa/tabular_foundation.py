"""A local TabICL experiment; public pretrained weights, no remote prediction API."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'vendor').exists():sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLClassifier

_MODELS={}
class SharedTabICL(TabICLClassifier):
    def _load_model(self):
        key=(str(self.model_path),str(self.device))
        if key not in _MODELS:
            super()._load_model();_MODELS[key]=(self.model_,self.model_config_,self.model_path_)
        else:self.model_,self.model_config_,self.model_path_=_MODELS[key]

class TabularModel:
    def __init__(self,representation='dino',n_estimators=4):
        self.representation=representation;self.n_estimators=n_estimators;self.heads=[]
    def fit(self,features,labels,indices):
        self.heads=[]
        for j in range(7):
            keep=np.asarray(indices)[np.isfinite(labels[indices,j])];y=labels[keep,j]
            if len(np.unique(y))<2:self.heads.append(float(y.mean()) if len(y) else 0.);continue
            if self.representation=='geometry':x=features['geo'][keep];scaler=pca=None
            else:
                key='dino_small_cls' if self.representation=='dino' else 'contour'
                scaler=StandardScaler();z=scaler.fit_transform(features[key][keep]);pca=PCA(n_components=min(24,len(keep)-1,z.shape[1]),svd_solver='full');z=pca.fit_transform(z)
                x=np.c_[z,features['geo'][keep]]
            self.heads.append(dict(x=x.astype(np.float32),y=y.astype(int),scaler=scaler,pca=pca))
        return self
    def predict(self,features,indices=None,device='cpu'):
        n=len(features['geo']) if indices is None else len(indices);out=[]
        for head in self.heads:
            if isinstance(head,float):out.append(np.full(n,head));continue
            geo=features['geo'] if indices is None else features['geo'][indices]
            if self.representation=='geometry':x=geo
            else:
                key='dino_small_cls' if self.representation=='dino' else 'contour';z=features[key] if indices is None else features[key][indices]
                x=np.c_[head['pca'].transform(head['scaler'].transform(z)),geo]
            model=SharedTabICL(n_estimators=self.n_estimators,device=device,use_amp=False,use_fa3=False,n_jobs=4,
                              model_path=ROOT/'artifacts/foundation/tabicl-classifier-v2-20260212.ckpt',allow_auto_download=False,random_state=20260925)
            model.fit(head['x'],head['y']);out.append(model.predict_proba(x.astype(np.float32))[:,1])
        return np.stack(out,axis=1)
