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
    length=float(rng.uniform(35,65) if pair else rng.uniform(35,65) if kind=='belt' else rng.uniform(65,135) if kind=='zipper' else rng.uniform(55,115) if kind=='wire' else rng.uniform(10,28))
    scale=length/max(h,w);scale_y=float(rng.uniform(.85,1.20));scale_x=float(rng.uniform(.85,1.20))
    nh=max(3,round(h*scale/sy*scale_y));nw=max(3,round(w*scale/sx*scale_x))
    s=cv2.resize(s,(nw,nh),interpolation=cv2.INTER_AREA)
    morphology=str(rng.choice(['none','widen','narrow'],p=[.35,.40,.25]));blend=0.
    if morphology!='none':
        kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3));blend=float(rng.uniform(.15,.45))
        changed=cv2.dilate(s,kernel) if morphology=='widen' else cv2.erode(s,kernel)
        s=(1-blend)*s+blend*changed
    bend=float(rng.uniform(-1.0,1.0)) if rng.random()<.35 else 0.
    if bend:
        yy,xx=np.indices(s.shape,dtype='float32');mapx=xx+bend*np.sin(np.pi*yy/max(1,nh-1))
        s=cv2.remap(s,mapx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    # Low-resolution branch smooths/samples the object, not just a crisp resized outline.
    if poor:
        factor=float(rng.uniform(.5,.85));small=cv2.resize(s,(max(2,round(nw*factor)),max(2,round(nh*factor))),interpolation=cv2.INTER_AREA)
        s=cv2.resize(small,(nw,nh),interpolation=cv2.INTER_LINEAR)
    sigma=float(rng.uniform(.8,1.35) if poor else rng.uniform(.35,.7));s=cv2.GaussianBlur(s,(0,0),sigma)
    pos=s[s>.005]
    if len(pos):s=np.clip(s/max(float(np.percentile(pos,95)),.01),0,1)
    s[s<.025]=0
    cropped=False;crop_fraction=1.
    if rng.random()<.4 and min(s.shape)>5:
        cropped=True;crop_fraction=float(rng.uniform(.72,.97));axis=int(rng.integers(2));side=int(rng.integers(2));size=max(3,round(s.shape[axis]*crop_fraction))
        if axis==0:s=s[:size] if side==0 else s[-size:]
        else:s=s[:,:size] if side==0 else s[:,-size:]
    return s,dict(length_mm=length,sigma_native_px=sigma,poor=poor,scale_y=scale_y,scale_x=scale_x,
        morphology=morphology,morphology_blend=blend,bend_native_px=bend,cropped=cropped,crop_fraction=crop_fraction)


def synthesize(image,donors,rng,force_pair=None,force_poor=None):
    """Return known insertion mask and metadata. No native test-image access."""
    wires=[d for d in donors if d['pair_eligible']]
    pair=bool(wires and (rng.random()<.30 if force_pair is None else force_pair))
    poor=bool(rng.random()<.45 if force_poor is None else force_poor)
    if pair:pool=wires
    else:
        # Prevent numerous procedural templates from dominating real donors.
        groups={key:[d for d in donors if (d['family']==key if key in ['belt','zipper'] else d['family'] not in ['belt','zipper'])] for key in ['belt','zipper','observed']}
        keys=[k for k,v in groups.items() if v];weights=np.array([.15 if k in ['belt','zipper'] else .70 for k in keys]);weights/=weights.sum()
        pool=groups[str(rng.choice(keys,p=weights))]
    donor=pool[int(rng.integers(len(pool)))]
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
            if donor['family'] in ['belt','zipper']:
                # Waist / fly: lower field, occasionally partly below the scan.
                y=int(rng.integers(max(0,min(h-ah,round(h*.55))),max(1,h-ah+1)))
                x=int(np.clip(w/2-aw/2+rng.normal(0,w*.08),0,w-aw))
            else:y=int(rng.integers(0,max(1,min(h-ah+1,round(h*.35)) if upper else h-ah+1)))
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
