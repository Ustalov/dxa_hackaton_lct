import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score
from .imaging import TARGETS

def binary(y,p,pred):
    valid=np.isfinite(y)&np.isfinite(p); y=np.asarray(y)[valid].astype(int);p=np.asarray(p)[valid];pred=np.asarray(pred)[valid]
    if not len(y): return {'n':0}
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sensitivity=tp/(tp+fn) if tp+fn else None
    specificity=tn/(tn+fp) if tn+fp else None
    return {'n':len(y),'positives':int(y.sum()),'f1':float(f1_score(y,pred,zero_division=0)),
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'pr_auc_ap':float(average_precision_score(y,p)) if y.sum() else None,
            'precision':float(precision_score(y,pred,zero_division=0)), 'sensitivity':sensitivity,'specificity':specificity,
            'balanced_accuracy':(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn)}

def evaluate(labels,prob,pred,groups):
    result={t:binary(labels[:,j],prob[:,j],pred[:,j]) for j,t in enumerate(TARGETS)}
    y=labels[:,:2].flatten();p=prob[:,:2].flatten();b=pred[:,:2].flatten()
    result['overall']=binary(y,p,b)
    result['macro_f1_types']=float(np.mean([result[t]['f1'] for t in TARGETS[2:]]))
    study_y=[];study_p=[];study_b=[]
    for group in np.unique(groups):
        ids=np.where(groups==group)[0]
        ys=labels[ids,:2]; ps=prob[ids,:2];bs=pred[ids,:2]
        valid=np.isfinite(ys)
        # Unlabeled images prevent a reliable negative study-level target.
        if valid.any() and (np.nanmax(ys)>0 or valid.sum()==len(ids)):
            study_y.append(np.nanmax(ys));study_p.append(ps[valid].max());study_b.append(bs[valid].max())
    result['study_any']=binary(np.array(study_y),np.array(study_p),np.array(study_b))
    return result

def bootstrap(labels,prob,pred,groups,repeats=500):
    rng=np.random.default_rng(917);unique=np.unique(groups);values=[]
    for _ in range(repeats):
        draw=rng.choice(unique,len(unique),replace=True)
        ids=np.concatenate([np.where(groups==g)[0] for g in draw])
        score=binary(labels[ids,:2].flatten(),prob[ids,:2].flatten(),pred[ids,:2].flatten())
        if score.get('roc_auc') is not None: values.append([score['f1'],score['roc_auc']])
    arr=np.array(values)
    return {'unit':'study','repeats':repeats,'valid_auc_repeats':len(values),
            'f1_95ci':np.quantile(arr[:,0],[.025,.975]).tolist(),
            'roc_auc_95ci':np.quantile(arr[:,1],[.025,.975]).tolist()}
