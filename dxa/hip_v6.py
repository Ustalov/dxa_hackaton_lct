"""PNG neck-ROI supervised femur/pelvis partition and anatomy-aware geometry."""
import numpy as np
import torch
from torch import nn
from scipy import ndimage as ndi
from .anatomy_network import AnatomyNet
from .geometry_targets import feature, lesser_candidate
from .anatomy_measurements import measure

class HipNet(AnatomyNet):
    def __init__(self):
        super().__init__('hip',geometry_channels=6,geometry_context=True)
        # AnatomyNet already applies this optional decoder to the same full-resolution features.
        self.vertebrae=nn.Conv2d(24,3,1)

def partition_from_roi(bone,roi,side):
    y,x=np.indices(bone.shape);a,b=np.asarray(roi['cut']);d=b-a
    signed=(x-a[0])*d[1]-(y-a[1])*d[0]
    femur=signed>0 if side=='right' else signed<0
    return np.where(bone,np.where(femur,1,2),0).astype('uint8')

def geometry_from_parts(parts,spacing=(1.05,.6),neck=None):
    h,w=parts.shape;femur=parts==1;pelvis=parts==2
    fy,fx=np.where(femur);py,px=np.where(pelvis)
    features={k:feature(label,n) for k,label,n in [('greater_trochanter','Большой вертел',1),('ischium','Седалищная кость',1),('lateral_femur','Латеральный край',1),('lesser_trochanter','Малый вертел',3)]}
    side=None;diag={'neck_roi':neck,'side_source':'pelvis relative to distal femoral shaft'}
    if len(fx)>30 and len(px)>30:
        distal=fx[fy>=np.quantile(fy,.7)];center=float(np.median(distal))
        pelvis_center=float(np.median(px));side='right' if pelvis_center<center else 'left'
        lateral=fx>=center if side=='right' else fx<=center
        # Lateral upper component excludes medial femoral head across the neck ROI.
        if lateral.any():
            top=fy[lateral].min();xx=fx[lateral & (fy==top)]
            features['greater_trochanter']=feature('Большой вертел',1,[[float(np.median(xx)),float(top)]])
        bottom=int(py.max());features['ischium']=feature('Седалищная кость',1,[[float(np.median(px[py==bottom])),bottom]])
        edge=int(fx.max() if side=='right' else fx.min())
        features['lateral_femur']=feature('Латеральный край',1,[[edge,float(np.median(fy[fx==edge]))]])
        if neck and features['greater_trochanter']['points']:
            mask=parts>0;roi={**neck};flip=side=='left'
            if flip:
                mask=np.fliplr(mask);center=w-1-center
                cut=np.asarray(neck['cut']).copy();cut[:,0]=w-1-cut[:,0];roi['cut']=cut.tolist()
            pts,info=lesser_candidate(mask,center,roi,features['greater_trochanter']['points'][0][1],spacing)
            if pts is not None:
                if flip:pts[:,0]=w-1-pts[:,0]
                features['lesser_trochanter']=feature('Малый вертел',3,pts)
            diag['lesser_detection']=info
    case=dict(region='hip',shape=[h,w],spacing_yx_mm=list(spacing),features=features,lateral_image_side=side,diagnostics=diag,review_status='automatic')
    case['measurements']=measure(case)
    return case

def neck_from_parts(parts):
    """Approximate ROI cut from the learned interface, not a new visible rectangle."""
    border=(parts==1)&ndi.binary_dilation(parts==2,iterations=2)
    y,x=np.where(border)
    if len(x)<12:return None
    coords=np.column_stack([x,y]).astype(float);center=coords.mean(0)
    _,_,vectors=np.linalg.svd(coords-center,full_matrices=False);direction=vectors[0]
    if direction[1]<0:direction=-direction
    t=(coords-center)@direction;low,high=np.quantile(t,[.05,.95]);cut=np.array([center+low*direction,center+high*direction])
    if np.linalg.norm(cut[1]-cut[0])<10:return None
    return dict(cut=cut.tolist(),polygon=None,method='interface of PNG ROI-supervised femur/pelvis model; approximate',confidence='unverified')

def clean_parts(parts):
    """Remove isolated class islands while preserving every foreground bone pixel."""
    out=parts.copy();seeds=[]
    for k in (1,2):
        lab,n=ndi.label(parts==k)
        if not n:return out
        counts=np.bincount(lab.ravel());counts[0]=0;seeds.append(lab==counts.argmax())
    # Stray pelvic predictions on the distal shaft must not become the ischium.
    stray=(parts>0)&~seeds[0]&~seeds[1]
    distance=np.stack([ndi.distance_transform_edt(~s) for s in seeds]);nearest=distance.argmin(0)+1
    out[stray]=nearest[stray]
    return out
