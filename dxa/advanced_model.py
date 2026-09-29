import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

def candidate_configs(keys):
    configs=[]
    def add(name,features,kind='linear',strength=.03):
        if all(k in keys for k in features):configs.append(dict(name=name,features=features,kind=kind,strength=strength))
    for prefix in ['dino_small','dino_base','convnext']:
        for c in [.01,.1]:add(f'{prefix}_linear_{c}',[prefix+'_cls'],'linear',c)
        for c in [1.,10.]:add(f'{prefix}_rbf_{c}',[prefix+'_cls'],'rbf',c)
        if prefix!='convnext':
            add(prefix+'_spatial_linear',[prefix+'_cls',prefix+'_spatial'],'linear',.01)
            add(prefix+'_spatial_rbf',[prefix+'_cls',prefix+'_spatial'],'rbf',10)
    add('dino_base_mean_linear',['dino_base_mean'],'linear',.03)
    add('dino_base_mean_rbf',['dino_base_mean'],'rbf',10)
    add('foundation_fusion',['dino_base_cls','dino_small_cls','convnext_cls'],'linear',.01)
    add('dino_geometry',['dino_base_cls','geo'],'linear',.01)
    add('dino_shape',['dino_base_cls','shape'],'rbf',10)
    add('shape_trees',['shape','geo'],'trees',300)
    add('shape_linear',['shape','geo'],'linear',.01)
    add('shape_rbf',['shape','geo'],'rbf',1)
    add('hog_shape_rbf',['hog','shape','geo'],'rbf',1)
    add('dino_detail_linear',['dino_small_cls','dino_small_detail_cls'],'linear',.03)
    add('dino_detail_rbf',['dino_small_cls','dino_small_detail_cls'],'rbf',10)
    add('dino_context_linear',['dino_small_cls','dino_small_context'],'linear',.03)
    add('dino_context_rbf',['dino_small_cls','dino_small_context'],'rbf',10)
    add('contour_trees',['contour','geo'],'trees',300)
    add('contour_linear',['contour','geo'],'linear',.01)
    add('dino_contour',['dino_small_cls','contour'],'linear',.01)
    add('dino_small_clahe_linear',['dino_small_clahe_cls'],'linear',.1)
    add('dino_base_clahe_linear',['dino_base_clahe_cls'],'linear',.1)
    add('dino_clahe_dual',['dino_small_cls','dino_small_clahe_cls'],'linear',.03)
    add('dino_clahe_rbf',['dino_base_clahe_cls'],'rbf',10)
    add('dino_lda',['dino_small_cls'],'lda',1)
    add('dino_mean_lda',['dino_base_mean'],'lda',1)
    return configs

def matrix(features,config,indices=None):
    x=np.concatenate([features[k] for k in config['features']],axis=1)
    return x if indices is None else x[indices]

def fit_head(features,labels,indices,target,config,sample_weights=None):
    indices=np.asarray(indices);keep=indices[np.isfinite(labels[indices,target])]
    y=labels[keep,target].astype(int)
    if len(np.unique(y))<2:return float(y.mean()) if len(y) else 0.
    if config['kind']=='linear':
        model=make_pipeline(StandardScaler(),LogisticRegression(C=config['strength'],class_weight='balanced',solver='liblinear',max_iter=1500,random_state=20260925))
    elif config['kind']=='rbf':
        model=make_pipeline(StandardScaler(),SVC(C=config['strength'],kernel='rbf',gamma='scale',class_weight='balanced',cache_size=256))
    elif config['kind']=='lda':
        model=make_pipeline(StandardScaler(),PCA(n_components=min(24,len(y)-2),svd_solver='full'),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5]))
    else:
        model=ExtraTreesClassifier(n_estimators=int(config['strength']),class_weight='balanced',max_features=.5,min_samples_leaf=2,n_jobs=4,random_state=20260925)
    kwargs={}
    if sample_weights is not None:
        param='logisticregression__sample_weight' if config['kind']=='linear' else 'svc__sample_weight' if config['kind']=='rbf' else 'sample_weight'
        kwargs[param]=np.asarray(sample_weights)[keep]
    model.fit(matrix(features,config,keep),y,**kwargs);return model

def predict_head(model,features,config,indices=None):
    n=len(next(iter(features.values()))) if indices is None else len(indices)
    if isinstance(model,float):return np.full(n,model)
    x=matrix(features,config,indices)
    if config['kind']=='rbf':return expit(model.decision_function(x))
    return model.predict_proba(x)[:,1]

class AdvancedModel:
    def __init__(self,configs):self.configs=configs;self.heads=[]
    def fit(self,features,labels,indices,sample_weights=None):
        self.heads=[fit_head(features,labels,indices,j,c,sample_weights) for j,c in enumerate(self.configs)];return self
    def predict(self,features,indices=None):
        return np.stack([predict_head(h,features,c,indices) for h,c in zip(self.heads,self.configs)],axis=1)

def consistent_predictions(prob,thresholds,policy='quality_gate'):
    """Declared decision rule, scored as part of the pipeline. Not expert relabeling."""
    pred=prob>=np.asarray(thresholds)
    for q,ids in [(0,[2,3,4]),(1,[5,6])]:
        if policy=='types_or':pred[:,q]=pred[:,ids].any(1)
        elif policy=='quality_gate':
            pred[np.ix_(~pred[:,q],ids)]=False
            missing=pred[:,q]&~pred[:,ids].any(1)
            rows=np.flatnonzero(missing)
            # Greatest normalized distance to the learned threshold; deterministic.
            margin=prob[:,ids]-np.asarray(thresholds)[ids]
            best=np.asarray(ids)[margin.argmax(1)]
            pred[rows,best[rows]]=True
        elif policy!='independent':raise ValueError(policy)
    return pred
