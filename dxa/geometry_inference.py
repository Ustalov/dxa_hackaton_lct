"""Inference with fixed confidence gates selected before native DICOM evaluation."""
import cv2,numpy as np,torch
from torch.nn import functional as F
from .anatomy_measurements import measure
from .geometry_targets import feature

@torch.no_grad()
def predict_geometry(net,image,checkpoint,spacing=(1.05,.6)):
    h,w=image.shape;size=checkpoint['input_size'];scale=size/max(h,w);nh,nw=round(h*scale),round(w*scale);top=(size-nh)//2;left=(size-nw)//2
    canvas=np.zeros((size,size),'float32');canvas[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh))/255
    x=torch.from_numpy(canvas).repeat(3,1,1);x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    out=net(x[None].to(next(net.parameters()).device));bone=out['bone'][0,0].sigmoid().cpu().numpy()[top:top+nh,left:left+nw]
    bone=cv2.resize(bone,(w,h))>.5
    labels=np.zeros((h,w),'uint8')
    if out['vertebrae'] is not None:
        probs=out['vertebrae'][0].softmax(0).cpu().numpy()[:,top:top+nh,left:left+nw]
        labels=np.stack([cv2.resize(p,(w,h)) for p in probs]).argmax(0).astype('uint8');labels[~bone]=0
    heat=F.interpolate(out['landmarks'].float(),(80,80),mode='bilinear',align_corners=False)[0].flatten(1).softmax(1).reshape(-1,80,80).cpu().numpy()
    spec=checkpoint['audit']['channels'];counts=checkpoint['audit']['channel_training_counts'];features={};confidence={}
    for (key,idx) in spec:
        arity=3 if key=='lesser_trochanter' else 2 if key.startswith('L') else 1
        if key not in features:features[key]=feature(key,arity);features[key]['points']=[None]*arity
    for j,((key,idx),n) in enumerate(zip(spec,counts)):
        y,x=np.unravel_index(heat[j].argmax(),(80,80));y0,y1=max(0,y-3),min(80,y+4);x0,x1=max(0,x-3),min(80,x+4)
        patch=heat[j,y0:y1,x0:x1];mass=float(patch.sum())
        # Local expectation avoids rounding millimetre measurements to 4 input pixels.
        py_grid,px_grid=np.mgrid[y0:y1,x0:x1];hx=float((patch*px_grid).sum()/max(mass,1e-12));hy=float((patch*py_grid).sum()/max(mass,1e-12))
        px=((hx+.5)*size/80-left)*w/nw-.5;py=((hy+.5)*size/80-top)*h/nh-.5
        valid=n>=20 and mass>=.15 and 0<=px<w and 0<=py<h
        confidence[f'{key}:{idx}']=dict(local_probability_mass=mass,training_references=n,accepted=bool(valid))
        if valid:features[key]['points'][idx]=[float(px),float(py)]
    for key,f in features.items():
        if any(p is None for p in f['points']):f['points']=[]
        else:
            f['visibility']='proposed'
            if key.startswith('L'):f['points'].sort(key=lambda p:p[0])
        f['provenance']='PNG-trained neural landmark heatmap; unvalidated on target domain'
    # Lateral image side uses the distal femoral shaft; no patient-side question.
    profile=bone[int(.78*h):int(.94*h)].sum(0).astype(float)
    center=int(np.argmax(cv2.GaussianBlur(profile[None],(21,1),0)[0])) if profile.sum() else w/2
    side='left' if center<w/2 else 'right'
    case=dict(region=checkpoint['region'],shape=[h,w],spacing_yx_mm=list(spacing),features=features,
              lateral_image_side=side if checkpoint['region']=='hip' else None,review_status='neural_prediction',
              curve_endpoints_confirmed=False,cobb_top='L1',cobb_bottom='L4',confidence=confidence)
    # Missing/implausible predictions are unknown, not fabricated negative labels.
    if case['region']=='hip' and features['lesser_trochanter']['points']:
        m=measure(case)['measurements']['lesser_trochanter']
        if not m.get('passes_visibility_shape_rule'):
            features['lesser_trochanter']['visibility']='uncertain'
            case['lesser_rejection']='Predicted triplet fails the agreed geometric shape rule'
    case['measurements']=measure(case)
    return case,bone,labels
