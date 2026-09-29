"""Frozen PNG-only models and explicit contrast / intensity postprocessing trials."""
import cv2,numpy as np,torch
from scipy import ndimage as ndi
from .png_spine_v6 import resize_aligned,tensor_channels,bbox,component
from .geometry_targets import feature,plate
from .anatomy_measurements import measure

@torch.no_grad()
def global_prob(net,image):
    zeros=np.zeros_like(image,'uint8');(im,_,_),box=resize_aligned(image,zeros,zeros,320)
    x=torch.from_numpy(tensor_channels(im))[None].to(next(net.parameters()).device)
    out=net(x);y,x0,nh,nw=box;h,w=image.shape
    p=out['vertebrae'][0].softmax(0).cpu().numpy()[:,y:y+nh,x0:x0+nw]
    bone=out['bone'][0,0].sigmoid().cpu().numpy()[y:y+nh,x0:x0+nw]
    return np.stack([cv2.resize(a,(w,h)) for a in p]),cv2.resize(bone,(w,h))

@torch.no_grad()
def local_prob(net,image,seed_labels):
    h,w=image.shape;maps=np.zeros((4,h,w),'float32');inputs=[];boxes=[]
    for k in range(1,5):
        seed=component(seed_labels==k)
        box=bbox(seed)
        if box is None:continue
        y0,y1,x0,x1=box;crop=image[y0:y1,x0:x1];zero=np.zeros_like(crop,'uint8')
        (im,_,_),letter=resize_aligned(crop,zero,zero,192)
        inputs.append(tensor_channels(im,True));boxes.append((k,box,letter,seed))
    if not inputs:return maps
    x=torch.from_numpy(np.stack(inputs)).to(next(net.parameters()).device);pred=net(x)['bone'][:,0].sigmoid().cpu().numpy()
    for probability,(k,(y0,y1,x0,x1),(y,x,nh,nw),seed) in zip(pred,boxes):
        p=cv2.resize(probability[y:y+nh,x:x+nw],(x1-x0,y1-y0))
        maps[k-1,y0:y1,x0:x1]=p
    return maps

def labels_from_prob(maps,seeds):
    scores=maps.copy()
    for k in range(1,5):
        keep=component(maps[k-1]>.5,seeds==k);scores[k-1]*=keep
    labels=(scores.argmax(0)+1).astype('uint8');labels[scores.max(0)<=.5]=0
    return labels

def edge_path(gradient,initial,xs,height,upper):
    """Smooth path through signed vertical contrast near a proposed endplate."""
    h,w=gradient.shape;radius=max(3,int(.23*height));center=float(np.median(initial))
    lo=max(1,int(center-radius));hi=min(h-2,int(center+radius)+1)
    if hi-lo<3:return initial
    ys=np.arange(lo,hi);g=gradient[lo:hi,xs]
    if not upper:g=-g
    scale=max(float(np.percentile(abs(g),90)),.015);reward=g/scale
    reward-=.65*((ys[:,None]-initial[None,:])/max(radius,1))**2
    transitions=.18*(ys[:,None]-ys[None,:])**2
    score=reward[:,0];pointers=[]
    for col in range(1,len(xs)):
        candidates=score[:,None]-transitions;ptr=candidates.argmax(0);score=candidates.max(0)+reward[:,col];pointers.append(ptr)
    path=[int(score.argmax())]
    for ptr in reversed(pointers):path.append(int(ptr[path[-1]]))
    return ys[np.array(path[::-1])].astype(float)

def refine_boundaries(raw,labels):
    p=raw.astype('float32');lo,hi=np.percentile(p,[1,99]);p=np.clip((p-lo)/max(hi-lo,1),0,1)
    gradient=cv2.Sobel(cv2.GaussianBlur(p,(0,0),1),cv2.CV_32F,0,1,ksize=3)/4
    result=labels.copy();plates={};details={}
    for k in range(1,5):
        mask=component(labels==k);yy,xx=np.where(mask)
        if len(xx)<40:continue
        x0,x1=np.quantile(xx,[.15,.85]);xs=np.arange(int(x0),int(x1)+1)
        if len(xs)<8:continue
        upper=[];lower=[]
        for x in xs:
            ys=np.flatnonzero(mask[:,x]);upper.append(float(ys.min()) if len(ys) else float(yy.min()));lower.append(float(ys.max()) if len(ys) else float(yy.max()))
        bh=max(5,float(np.median(np.array(lower)-upper)))
        up=edge_path(gradient,np.array(upper),xs,bh,True);down=edge_path(gradient,np.array(lower),xs,bh,False)
        if np.any(down-up<.35*bh):continue
        slopes=[]
        for side,path in [('upper',up),('lower',down)]:
            slope,intercept=np.polyfit(xs,path,1);slopes.append((slope,intercept))
            plates[f'L{k}_{side}']=[[float(xs[0]),float(slope*xs[0]+intercept)],[float(xs[-1]),float(slope*xs[-1]+intercept)]]
        refined=mask.copy()
        for x in range(int(xx.min()),int(xx.max())+1):
            ys=np.flatnonzero(mask[:,x])
            if not len(ys):continue
            top=int(np.clip(np.interp(x,xs,up),0,raw.shape[0]-1));bottom=int(np.clip(np.interp(x,xs,down),0,raw.shape[0]-1))
            # Respect tapering on lateral processes; central columns carry endplate refinement.
            if x<xs[0] or x>xs[-1]:continue
            refined[:,x]=False;refined[top:bottom+1,x]=True
        refined&=(labels==0)|(labels==k)
        result[result==k]=0;result[refined]=k
        details[f'L{k}']=dict(mean_upper_shift_px=float(np.mean(up-upper)),mean_lower_shift_px=float(np.mean(down-lower)),method='signed-gradient smooth path within model boundary neighborhood')
    return result,plates,details

def geometry_from_masks(labels,bone,spacing=(1.05,.6),plates=None):
    h,w=labels.shape;f={};contours={};plates=plates or {}
    for k in range(1,6):
        for side in ['upper','lower']:
            key=f'L{k}_{side}';pts=plates.get(key,plate(labels==k,side=='upper') if k<5 else None)
            f[key]=feature(key,2,pts,source='PNG-trained segmentation boundary'+('; DICOM gradient refinement' if key in plates else ''))
    p3=f['L3_lower']['points'];p4=f['L4_upper']['points'];ygrid,xgrid=np.mgrid[:h,:w]
    for side in ['left','right']:
        point=None;key='iliac_'+side
        if p3 and p4:
            level=(np.mean(np.array(p3)[:,1])+np.mean(np.array(p4)[:,1]))/2;ys,xs=np.where(labels==4)
            if len(xs):
                edge=xs.min() if side=='left' else xs.max()
                area=bone&(ygrid>level)&((xgrid<edge-20/spacing[1]) if side=='left' else (xgrid>edge+20/spacing[1]))
                candidate=component(ndi.binary_opening(area,structure=np.ones((3,3))))
                cy,cx=np.where(candidate)
                if len(cx)>40:
                    line=[]
                    for x in range(int(cx.min()),int(cx.max())+1):
                        ys=np.flatnonzero(candidate[:,x])
                        if len(ys):line.append([float(x),float(ys.min())])
                    if len(line)>3:
                        curve=np.array(line);curve[:,1]=ndi.median_filter(curve[:,1],size=3);contours[key]=curve.tolist();point=[curve[curve[:,1].argmin()].tolist()]
        f[key]=feature('Верхний контур подвздошной кости '+side,1,point,source='upper envelope of lateral bone below L3-L4; preliminary')
    case=dict(region='spine',shape=[h,w],spacing_yx_mm=list(spacing),features=f,contours=contours,review_status='model_prediction',curve_endpoints_confirmed=False,cobb_top='L1',cobb_bottom='L4')
    case['measurements']=measure(case)
    case['unavailable_tasks']=['L5 and complete L1-L5 axis','validated Cobb curve selection','metal/rib/Th12 segmentation']
    return case
