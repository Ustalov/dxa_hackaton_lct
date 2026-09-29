"""Object-only DXA augmentation, including bilateral underwires and poor visibility."""
import json
from pathlib import Path
import cv2,numpy as np
from scipy import ndimage as ndi

APPROVED_WIRES={'train_05544_0','train_07088_1','train_07088_2','train_01649_0','train_01649_1',
                'dev_08690_0','dev_08638_0'}

def clean_signal(path):
    with np.load(path) as z:signal=z['signal'].astype('float32')
    mask=signal>.18;ids,n=ndi.label(ndi.binary_closing(mask,structure=np.ones((3,3))))
    if n:
        sizes=ndi.sum(mask,ids,np.arange(1,n+1));core=ndi.binary_dilation(ids==(int(np.argmax(sizes))+1),iterations=2);signal*=core
    yy,xx=np.where(signal>.08)
    if len(xx)<8:return None
    return signal[max(0,yy.min()-2):yy.max()+3,max(0,xx.min()-2):xx.max()+3]

def transform(signal,kind,rng,poor=False,pair=False,spacing=(1.05,.6)):
    sy,sx=spacing;s=signal.copy()
    if not pair and rng.random()<.5:s=np.fliplr(s).copy()
    h,w=s.shape;side=max(h,w)*2;canvas=np.zeros((side,side),'float32');y,x=(side-h)//2,(side-w)//2;canvas[y:y+h,x:x+w]=s
    mat=cv2.getRotationMatrix2D((side/2,side/2),float(rng.uniform(-18,18)),1)
    s=cv2.warpAffine(canvas,mat,(side,side));yy,xx=np.where(s>.05)
    s=s[yy.min():yy.max()+1,xx.min():xx.max()+1];h,w=s.shape
    length=float(rng.uniform(35,65) if pair else rng.uniform(55,115) if kind=='wire' else rng.uniform(7,28))
    scale=length/max(h,w);nh=max(3,round(h*scale/sy));nw=max(3,round(w*scale/sx))
    s=cv2.resize(s,(nw,nh),interpolation=cv2.INTER_AREA)
    # Low-resolution branch smooths/samples the object, not just a crisp resized outline.
    if poor:
        factor=float(rng.uniform(.5,.85));small=cv2.resize(s,(max(2,round(nw*factor)),max(2,round(nh*factor))),interpolation=cv2.INTER_AREA)
        s=cv2.resize(small,(nw,nh),interpolation=cv2.INTER_LINEAR)
    sigma=float(rng.uniform(.8,1.35) if poor else rng.uniform(.35,.7));s=cv2.GaussianBlur(s,(0,0),sigma)
    pos=s[s>.005]
    if len(pos):s=np.clip(s/max(float(np.percentile(pos,95)),.01),0,1)
    s[s<.025]=0
    if rng.random()<.2 and min(s.shape)>5:
        # Partial field visibility, kept as a positive if signal still survives.
        if rng.random()<.5:s=s[:max(3,round(len(s)*rng.uniform(.65,.95)))]
        else:s=s[:,:max(3,round(s.shape[1]*rng.uniform(.65,.95)))]
    return s,dict(length_mm=length,sigma_native_px=sigma,poor=poor)

def synthesize(image,donors,rng,force_pair=None,force_poor=None):
    """Return known insertion mask and metadata. No native test-image access."""
    wires=[d for d in donors if d['pair_eligible']]
    pair=bool(wires and (rng.random()<.30 if force_pair is None else force_pair))
    poor=bool(rng.random()<.45 if force_poor is None else force_poor)
    donor=wires[int(rng.integers(len(wires)))] if pair else donors[int(rng.integers(len(donors)))]
    h,w=image.shape;out=image.astype('float32').copy();objects=[]
    amplitude=float(rng.uniform(28,85) if poor else rng.uniform(90,200))
    center=w/2;pair_y=int(rng.uniform(.02,.20)*h)
    for k in range(2 if pair else 1):
        signal=donor['signal']
        if pair and k==1:signal=np.fliplr(signal)
        alpha,params=transform(signal,donor['family'],rng,poor,pair)
        ah,aw=alpha.shape
        limit_w=int(w*.43) if pair else w-4
        if ah>h-4 or aw>limit_w:
            scale=min((h-4)/ah,limit_w/aw);alpha=cv2.resize(alpha,(max(2,round(aw*scale)),max(2,round(ah*scale))),interpolation=cv2.INTER_AREA);ah,aw=alpha.shape
        if pair:
            x=int(max(0,center-rng.uniform(5,18)-aw)) if k==0 else int(min(w-aw,center+rng.uniform(5,18)))
            y=int(np.clip(pair_y+rng.integers(-6,7),0,h-ah))
        else:
            x=int(rng.integers(0,max(1,w-aw+1)))
            upper=donor['family']=='wire' and rng.random()<.8
            y=int(rng.integers(0,max(1,min(h-ah+1,round(h*.35)) if upper else h-ah+1)))
        before=out[y:y+ah,x:x+aw];strength=amplitude*float(rng.uniform(.85,1.15))
        out[y:y+ah,x:x+aw]=before+strength*alpha*(1-before/255.)
        objects.append(dict(donor_id=donor['id'],source_group=donor['source_group'],x=x,y=y,width=aw,height=ah,amplitude=strength,**params))
    out=np.uint8(np.rint(np.clip(out,0,255)));diff=out.astype(int)-image.astype(int);mask=(diff>=3).astype('float32')
    return out,mask,dict(pair=pair,poor=poor,objects=objects,visible_pixels=int(mask.sum()),max_added_signal=int(diff.max()))

def load_donors(manifest,partition):
    rows=json.loads(Path(manifest).read_text());out=[]
    for r in rows:
        if r['partition']!=partition:continue
        with np.load(r['clean_path']) as z:signal=z['signal'].astype('float32')
        out.append(dict(**r,signal=signal))
    return out
