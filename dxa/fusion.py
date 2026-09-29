import numpy as np
from scipy.special import logit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from .advanced_model import AdvancedModel

def thresholds_from_scores(y,p):
    grid=np.r_[.025,.05,.075,.10,.125,.15,.175,np.arange(.20,.81,.05)]
    out=[]
    for j in range(7):
        valid=np.isfinite(y[:,j]);yy=y[valid,j];pp=p[valid,j]
        if len(np.unique(yy))<2:out.append(.5);continue
        out.append(float(max(grid,key=lambda t:(f1_score(yy,pp>=t,zero_division=0),-abs(t-.5)))))
    return np.array(out)

class ProbabilityFusion:
    def __init__(self,mode='stack',strength=.3):self.mode=mode;self.strength=strength;self.heads=[]
    def fit(self,p,y):
        self.heads=[]
        if self.mode=='mean':return self
        for j in range(7):
            valid=np.isfinite(y[:,j]);yy=y[valid,j]
            if len(np.unique(yy))<2:self.heads.append(float(yy.mean()) if len(yy) else 0);continue
            x=logit(np.clip(p[valid,:,j],1e-5,1-1e-5))
            model=make_pipeline(StandardScaler(),LogisticRegression(C=self.strength,solver='liblinear',max_iter=1000,random_state=1))
            model.fit(x,yy);self.heads.append(model)
        return self
    def predict(self,p):
        if self.mode=='mean':return p.mean(1)
        return np.stack([np.full(len(p),h) if isinstance(h,float) else h.predict_proba(logit(np.clip(p[:,:,j],1e-5,1-1e-5)))[:,1] for j,h in enumerate(self.heads)],axis=1)

class FusionModel:
    def __init__(self,components,fusion,neural=False):self.components=components;self.fusion=fusion;self.neural=neural
    def predict(self,features,indices=None,neural_prob=None):
        values=[m.predict(features,indices) for m in self.components]
        if self.neural:
            if neural_prob is None:raise ValueError('Neural component probabilities are required')
            values.append(neural_prob)
        return self.fusion.predict(np.stack(values,axis=1))
