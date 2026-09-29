"""Restricted foreign-object synthesis, including genuine field-edge truncation."""
import cv2
import numpy as np


def bra_clasp_template(rng):
    """Procedural hook-and-eye closure, not an expert annotated radiograph."""
    a=np.zeros((180,120),'float32')
    n=int(rng.integers(1,4));pitch=int(rng.integers(38,50));thickness=int(rng.integers(5,9))
    y0=(180-n*pitch)//2
    for j in range(n):
        y=y0+j*pitch+pitch//2
        # Eye plus opposing J hook; no solid fabric rectangle / strap slider.
        cv2.ellipse(a,(38,y),(int(rng.integers(11,16)),12),0,0,360,1.,thickness)
        cv2.line(a,(77,y-15),(77,y+10),.95,thickness)
        cv2.ellipse(a,(67,y+10),(10,8),0,0,190,.95,thickness)
    yy,xx=np.where(a>0)
    return cv2.GaussianBlur(a[max(0,yy.min()-3):yy.max()+4,max(0,xx.min()-3):xx.max()+4],(0,0),.6)


def transformed(donor,rng,pair=False,poor=False):
    a=donor['signal'].copy();kind=donor['family']
    h,w=a.shape;sy,sx=1.05,.6
    if kind=='underwire':
        if donor.get('native_fragment'):
            scale=float(rng.uniform(.8,1.25));height=max(2,round(h*scale));width=max(2,round(w*scale))
        else:
            width_mm=float(rng.uniform(55,82) if pair else rng.uniform(65,115))
            width=max(5,round(width_mm/sx));height=max(3,round(width_mm*h/w/sy*rng.uniform(.8,1.15)))
    elif kind=='bra_clasp':
        length=float(rng.uniform(10,27));scale=length/max(h,w)
        height=max(3,round(h*scale/sy));width=max(3,round(w*scale/sx))
    else:
        length=float(rng.uniform(7,24));scale=length/max(h,w)
        height=max(3,round(h*scale/sy));width=max(3,round(w*scale/sx))
    a=cv2.resize(a,(width,height),interpolation=cv2.INTER_AREA)
    if rng.random()<.5:a=np.fliplr(a).copy()
    angle=float(rng.uniform(-18,18) if kind=='underwire' else rng.uniform(-65,65))
    pad=max(a.shape);canvas=np.zeros((height+2*pad,width+2*pad),'float32');canvas[pad:pad+height,pad:pad+width]=a
    matrix=cv2.getRotationMatrix2D((canvas.shape[1]/2,canvas.shape[0]/2),angle,1)
    a=cv2.warpAffine(canvas,matrix,(canvas.shape[1],canvas.shape[0]))
    morphology=str(rng.choice(['none','widen','narrow'],p=[.6,.25,.15]))
    if morphology!='none':
        changed=cv2.dilate(a,np.ones((3,3),np.uint8)) if morphology=='widen' else cv2.erode(a,np.ones((3,3),np.uint8))
        a=.8*a+.2*changed
    sigma=float(rng.uniform(.5,.9) if poor else rng.uniform(.2,.5))
    a=cv2.GaussianBlur(a,(0,0),sigma)
    yy,xx=np.where(a>.03)
    a=a[max(0,yy.min()-1):yy.max()+2,max(0,xx.min()-1):xx.max()+2]
    a=np.clip(a/max(float(a.max()),1e-6),0,1);a[a<.035]=0
    return a,dict(angle=angle,morphology=morphology,sigma_native_px=sigma,native_fragment=donor.get('native_fragment',False))


def synthesize(image,donors,rng,kind=None,force_pair=None,force_edge=None):
    if kind is None:kind=str(rng.choice(['underwire','bra_clasp','dense_object'],p=[.6,.25,.15]))
    pool=[r for r in donors if r['family']==kind]
    pair=kind=='underwire' and bool(rng.random()<.5 if force_pair is None else force_pair)
    if pair:pool=[r for r in pool if not r.get('native_fragment')]
    donor=pool[int(rng.integers(len(pool)))]
    poor=kind!='dense_object' and rng.random()<.3
    edge=kind=='underwire' and bool(rng.random()<.5 if force_edge is None else force_edge)
    a,params=transformed(donor,rng,pair,poor)
    h,w=image.shape;ah,aw=a.shape;center=w/2
    # Both paired wires share the field boundary. Clipping is applied to the signal,
    # not to the original anatomy or to a donor's surrounding CXR background.
    y= -int(rng.uniform(.25,.80)*ah) if edge else int(rng.uniform(.01,.30)*max(1,h-ah)) if kind=='underwire' else int(rng.uniform(.04,.7)*max(1,h-ah))
    if donor.get('native_fragment'):y=int(rng.integers(0,max(2,int(h*.1))))
    out=image.astype('float32').copy();union=np.zeros_like(out);objects=[]
    strength=float(rng.uniform(225,255) if kind=='dense_object' else rng.uniform(100,190) if kind=='bra_clasp' else rng.uniform(35,75) if poor else rng.uniform(75,145))
    for side in range(2 if pair else 1):
        signal=np.fliplr(a).copy() if side else a
        x=int(center-rng.uniform(1,6)-aw) if pair and side==0 else int(center+rng.uniform(1,6)) if pair else int(rng.uniform(-.1*w,max(1,w-aw+.1*w)))
        if kind=='bra_clasp':x=int(np.clip(w/2-aw/2+rng.normal(0,w*.15),0,max(0,w-aw)))
        x0=max(0,x);x1=min(w,x+aw);y0=max(0,y);y1=min(h,y+ah)
        if x1<=x0 or y1<=y0:continue
        alpha=signal[y0-y:y1-y,x0-x:x1-x]
        before=out[y0:y1,x0:x1]
        # Dense group explicitly reaches near-white intensity; wires remain realistic.
        if kind=='dense_object':changed=before+(255-before)*np.clip(alpha*1.35,0,1)
        else:changed=before+strength*alpha*(1-before/300.)
        out[y0:y1,x0:x1]=np.clip(changed,0,255)
        union[y0:y1,x0:x1]=np.maximum(union[y0:y1,x0:x1],alpha)
        objects.append(dict(donor_id=donor['id'],source_group=donor['source_group'],family=kind,x=x,y=y,
                            width=aw,height=ah,clipped=bool(x<0 or y<0 or x+aw>w or y+ah>h),**params))
    out=np.uint8(np.rint(out));difference=out.astype(int)-image.astype(int)
    mask=difference>=3
    peak=int(difference.max());visible=int(mask.sum())
    dense_valid=True
    if kind=='dense_object':
        # 235/255 is a normalized image-brightness criterion, NOT HU or material density.
        dense_valid=int(((out>=235)&mask).sum())>=8 and peak>=100
    return out,mask.astype('uint8'),dict(family=kind,pair=pair,poor=poor,edge_clipped=any(o['clipped'] for o in objects),
              visible_pixels=visible,max_added_signal=peak,dense_valid=dense_valid,objects=objects)
