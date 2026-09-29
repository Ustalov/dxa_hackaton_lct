import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

VARIANTS = {
    'geometry_trees': ('geo', 'trees', True),
    'hog_linear': ('hog', 'linear', True),
    'resnet_linear': ('deep', 'linear', False),
    'resnet_balanced': ('deep', 'linear', True),
    'hybrid_linear': ('hybrid', 'linear', True),
    'hybrid_trees': ('hybrid', 'trees', True),
    'study_context': ('context', 'linear', True),
}

def matrix(features, variant):
    feature=VARIANTS[variant][0]
    keys={'hybrid':['deep','geo'], 'context':['deep','geo','context']}.get(feature,[feature])
    return np.concatenate([features[k] for k in keys],axis=1)

class MultiHead:
    def __init__(self, variant): self.variant=variant; self.heads=[]
    def fit(self, features, labels, indices):
        x=matrix(features,self.variant)
        _,kind,balanced=VARIANTS[self.variant]
        for j in range(labels.shape[1]):
            keep=indices[np.isfinite(labels[indices,j])]
            y=labels[keep,j].astype(int)
            if len(np.unique(y))<2:
                self.heads.append(float(y.mean()) if len(y) else 0.)
                continue
            if kind=='trees':
                model=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=2,max_features=.6,
                                           class_weight='balanced' if balanced else None,random_state=240924,n_jobs=4)
            else:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.03,solver='liblinear',max_iter=1000,
                                     class_weight='balanced' if balanced else None,random_state=240924))
            model.fit(x[keep],y);self.heads.append(model)
        return self
    def predict(self, features, indices=None):
        x=matrix(features,self.variant)
        if indices is not None:x=x[indices]
        out=[]
        for head in self.heads:
            out.append(np.full(len(x),head) if isinstance(head,float) else head.predict_proba(x)[:,1])
        return np.array(out).T

def choose_thresholds(labels, probabilities):
    from sklearn.metrics import f1_score
    thresholds=[]
    # Coarse grid prevents thresholds chasing individual scores on tiny rare classes.
    for j in range(labels.shape[1]):
        valid=np.isfinite(labels[:,j]) & np.isfinite(probabilities[:,j])
        y,p=labels[valid,j],probabilities[valid,j]
        if len(np.unique(y))<2: thresholds.append(.5);continue
        candidates=np.arange(.10,.81,.05)
        best=max(candidates,key=lambda t:(f1_score(y,p>=t,zero_division=0),-abs(t-.5)))
        thresholds.append(float(best))
    return np.array(thresholds)

def balanced_group_folds(labels, groups, n_folds, seed):
    """Split assignment uses labels only for group-level stratification, never predictions."""
    unique=np.unique(groups)
    vectors=np.array([np.nansum(labels[groups==g],axis=0) for g in unique])
    sizes=np.array([(groups==g).sum() for g in unique])
    rng=np.random.default_rng(seed);best=None;best_cost=np.inf
    for _ in range(1600):
        order=rng.permutation(len(unique));assignment=np.empty(len(unique),int)
        assignment[order]=np.arange(len(unique))%n_folds
        sums=np.array([vectors[assignment==k].sum(0) for k in range(n_folds)])
        target=vectors.sum(0)/n_folds
        cost=np.mean(((sums-target)/(target+1))**2)
        cost+=.15*np.std([sizes[assignment==k].sum() for k in range(n_folds)])/max(1,sizes.mean())
        # Encourage rare positives in every fold whenever possible.
        feasible=(vectors>0).sum(0)>=n_folds
        cost+=10*np.sum((sums[:,feasible]==0))
        if cost<best_cost:best_cost=cost;best=assignment.copy()
    lookup=dict(zip(unique,best))
    return np.array([lookup[g] for g in groups])
