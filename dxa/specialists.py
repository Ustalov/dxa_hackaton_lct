"""Exactly five independently trained violation models; overall quality is derived."""
from pathlib import Path
import numpy as np
import joblib
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from .advanced_model import fit_head,predict_head

ROOT=Path(__file__).resolve().parents[1]
TASKS=['spine_position','spine_axis','spine_artifact','hip_position','hip_roi']

def overall_scores(types):
    return np.c_[1-np.prod(1-types[:,:3],axis=1),1-np.prod(1-types[:,3:],axis=1),types]

def decisions(types,thresholds):
    t=types>=np.asarray(thresholds);return np.c_[t[:,:3].any(1),t[:,3:].any(1),t]

def fit_calibrator(y,p):
    valid=np.isfinite(y);yy=y[valid];pp=p[valid]
    if len(np.unique(yy))<2:return None
    model=LogisticRegression(C=1.,solver='liblinear',max_iter=500,random_state=20260925)
    model.fit(logit(np.clip(pp,1e-5,1-1e-5)).reshape(-1,1),yy)
    return model

def calibrated(model,p):
    if model is None:return p
    return model.predict_proba(logit(np.clip(p,1e-5,1-1e-5)).reshape(-1,1))[:,1]

def objective(y,pred):
    # Equal weight per type within macro-F1; overall score counts each image once.
    def f1(yy,bb):
        valid=np.isfinite(yy);yy=yy[valid].astype(bool);bb=bb[valid]
        tp=(yy&bb).sum();fp=(~yy&bb).sum();fn=(yy&~bb).sum()
        return 2*tp/max(2*tp+fp+fn,1)
    macro=np.mean([f1(y[:,j+2],pred[:,j+2]) for j in range(5)])
    overall=f1(y[:,:2].flatten(),pred[:,:2].flatten())
    return .7*overall+.3*macro

def joint_thresholds(y,types):
    grid=np.array([.01,.025,.05,.075,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95])
    independent=[]
    for j in range(5):
        valid=np.isfinite(y[:,j+2]);yy=y[valid,j+2].astype(bool);pp=types[valid,j]
        def score(t):
            bb=pp>=t;tp=(yy&bb).sum();fp=(~yy&bb).sum();fn=(yy&~bb).sum();return 2*tp/max(2*tp+fp+fn,1)
        independent.append(max(grid,key=lambda t:(score(t),-abs(t-.5))))
    starts=[np.array(independent),np.full(5,.5),np.minimum(np.array(independent)+.15,.95)]
    best=None;best_score=-1
    for threshold in starts:
        for _ in range(3):
            previous=threshold.copy()
            for j in range(5):
                choices=[]
                for value in grid:
                    candidate=threshold.copy();candidate[j]=value
                    score=objective(y,decisions(types,candidate))-.0001*np.abs(candidate-.5).mean()
                    choices.append((score,-abs(value-.5),value))
                threshold[j]=max(choices)[2]
            if np.array_equal(previous,threshold):break
        score=objective(y,decisions(types,threshold))
        if score>best_score:best_score=score;best=threshold.copy()
    return best

class TaskModel:
    def __init__(self,target,configs,calibrator=None,threshold=.5):
        self.target=target;self.configs=configs;self.calibrator=calibrator;self.threshold=float(threshold);self.estimators=[]
    def fit(self,features,labels,indices):
        self.estimators=[fit_head(features,labels,indices,self.target+2,c) for c in self.configs];return self
    def predict(self,features,indices=None):
        values=[predict_head(m,features,c,indices) for c,m in zip(self.configs,self.estimators)]
        return calibrated(self.calibrator,np.mean(values,axis=0))

class SpecialistSet:
    def __init__(self,files):self.files=files;self._tasks=None;self.artifacts=None
    def tasks(self):
        if self._tasks is None:self._tasks=[joblib.load(Path(self.artifacts or ROOT/'artifacts')/f) for f in self.files]
        return self._tasks
    def predict(self,features,indices=None):
        return overall_scores(np.stack([m.predict(features,indices) for m in self.tasks()],axis=1))
