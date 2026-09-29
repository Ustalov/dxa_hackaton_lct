"""Conservative PNG geometry pseudo-labels. These are not expert ground truth."""
import itertools
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from .anatomy_measurements import lesser_trochanter


def feature(label, arity, pts=None, source='automatic geometry pseudo-label'):
    return dict(label=label, arity=arity,
                points=[] if pts is None else np.asarray(pts).round(3).tolist(),
                visibility='uncertain' if pts is None else 'proposed', provenance=source)


def neck_rectangle(gray):
    """Pair oblique parallel burned-in strokes; canonical femur is on right."""
    h, w = gray.shape
    residual = abs(gray.astype(float) - cv2.medianBlur(gray, 7))
    strokes = ((residual > 12) & ((gray < 115) | (gray > 220))).astype('uint8') * 255
    batches=[]
    for threshold, residual_limit in [(22,12),(25,15),(18,20)]:
        s=((residual>residual_limit)&((gray<115)|(gray>220))).astype('uint8')*255
        detected=cv2.HoughLinesP(s,1,np.pi/360,threshold,minLineLength=int(.18*h),maxLineGap=7)
        if detected is not None:batches.append(detected.reshape(-1,4))
    lines=np.concatenate(batches) if batches else None
    candidates = []
    if lines is None:
        return None
    for x, y, u, v in lines.reshape(-1, 4):
        if v < y:
            x, y, u, v = u, v, x, y
        dx, dy = float(u-x), float(v-y)
        if dx >= -8 or not .65 < dy/-dx < 2.7:
            continue
        if not .14*h < y < .52*h or not .38*h < v < .76*h:
            continue
        if not .12*w < u < .70*w or not .35*w < x < .85*w:
            continue
        a = np.array([x, y], float); b = np.array([u, v], float)
        d = (b-a)/np.linalg.norm(b-a)
        candidates.append((a, b, d))
    best = None
    for (a,b,d), (c,e,f) in itertools.combinations(candidates, 2):
        if np.dot(d, f) < .992:
            continue
        n = np.array([d[1], -d[0]])
        separation = abs(np.dot(c-a, n))
        overlap = min(np.dot(b-a,d), np.dot(e-a,d))-max(0,np.dot(c-a,d))
        if not 7 < separation < .15*w or overlap < .20*h:
            continue
        score = overlap - 2*abs(np.dot(a-c,d)) - abs(separation-.07*w)
        if best is None or score > best[0]:
            best = score, np.array([a,b,e,c]), (a+c)/2, (b+e)/2
    if best is None:
        return None
    _, polygon, top, bottom = best
    return dict(polygon=polygon.tolist(), cut=[top.tolist(), bottom.tolist()],
                method='paired oblique PNG ROI strokes', confidence='weak')


def shaft_contour(mask, start, center):
    """Follow the femoral run from bottom upward, avoiding isolated ROI strokes."""
    h,w=mask.shape;rows=[];medial=[];last=center
    for y in range(h-3, start-1, -1):
        lab,n=ndi.label(mask[y]);runs=[np.flatnonzero(lab==k) for k in range(1,n+1)]
        runs=[r for r in runs if len(r)>=max(8,.05*w)]
        if not runs:continue
        run=min(runs,key=lambda r:abs(float(r.mean())-last))
        if abs(float(run.mean())-last)>.18*w:continue
        last=float(run.mean());rows.append(y);medial.append(float(run.min()))
    return np.array(rows[::-1]), np.array(medial[::-1])


def lesser_candidate(mask, center, neck, gt_y, spacing=(1.05,.6)):
    """Multi-scale chord prominence below the neck; reject short tall artifacts."""
    if neck is None:
        return None, {'reason':'Neck boundary unavailable; neck bulge cannot be excluded'}
    h,w=mask.shape;sy,sx=spacing
    bottom=np.asarray(neck['cut'][1]);start=max(int(gt_y+.12*h),int(bottom[1]-.10*h))
    rows,x=shaft_contour(mask,start,center)
    if len(rows)<18:return None, {'reason':'Medial shaft contour incomplete'}
    x=ndi.gaussian_filter1d(x,.9)
    candidates=[]
    for length_mm in [8,12,18,24,32,40,48]:
        radius=max(3,int(length_mm/(2*sy)))
        if 2*radius+1>=len(rows):continue
        for i in range(radius,len(rows)-radius):
            a,b=i-radius,i+radius
            if rows[b]-rows[a]!=b-a:continue
            # Search first proximal bump reached from the shaft, not distal noise.
            if not gt_y+.18*h<rows[i]<min(gt_y+.48*h,.80*h):continue
            if rows[i]>bottom[1]+.12*h:continue
            # A rectangle can extend below the actual prominence. Use its
            # distal half-plane, never its lowest corner as the anatomy level.
            neck_top,neck_bottom=np.asarray(neck['cut']);direction=neck_bottom-neck_top
            distance=((x[i]-neck_top[0])*direction[1]-(rows[i]-neck_top[1])*direction[0])/np.linalg.norm(direction)
            if distance<2:continue
            pts=np.array([[x[a],rows[a]],[x[i],rows[i]],[x[b],rows[b]]])
            dims=lesser_trochanter(*pts,spacing_yx=spacing,medial_direction=-1)
            if not dims['passes_visibility_shape_rule']:continue
            axy=pts[0]*[sx,sy];bxy=pts[2]*[sx,sy];v=bxy-axy
            n=np.array([-v[1],v[0]])/np.linalg.norm(v)
            curve=np.column_stack([x[a:b+1]*sx,rows[a:b+1]*sy])
            dev=(curve-axy)@n
            peak=int(np.argmax(dev))
            if abs(peak-radius)>2 or peak<3 or peak>len(dev)-4:continue
            if (dev<-.6).mean()>.15:continue
            # Both ends must return toward a shaft baseline, not stay on a neck ramp.
            if dev[peak]-dev[2]<.5 or dev[peak]-dev[-3]<.5:continue
            score=dims['medial_protrusion_mm']-.015*dims['length_mm']
            candidates.append((score,pts,dims))
    if not candidates:return None, {'reason':'No below-neck bump satisfies length/protrusion criteria'}
    _,pts,dims=max(candidates,key=lambda item:item[0])
    return pts,dict(dimensions=dims,reason=None,confidence='weak; needs visual assessment')


def hip_geometry(rgb, bone):
    h,w=bone.shape
    clean=ndi.binary_opening(bone.astype(bool),structure=np.ones((5,5)))
    # Restore broad bone boundaries without re-growing long disconnected ROI lines.
    clean=ndi.binary_dilation(clean,iterations=1)&bone
    profile=clean[int(.78*h):int(.94*h)].sum(0).astype(float)
    center=int(np.argmax(ndi.gaussian_filter1d(profile,6)))
    flip=center<w/2
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    if flip:
        clean=np.fliplr(clean).copy();gray=np.fliplr(gray).copy();center=w-1-center
    neck=neck_rectangle(gray)
    yy,xx=np.mgrid[:h,:w]
    if neck is not None:
        a,b=np.array(neck['cut']);d=b-a
        femoral_side=(xx-a[0])*d[1]-(yy-a[1])*d[0]>0
    else:
        femoral_side=xx>center-.16*w
    femur=clean&femoral_side;pelvis=clean&~femoral_side
    # Keep components attached to distal shaft, so stray annotations cannot win extrema.
    lab,n=ndi.label(femur);seed=lab[int(.80*h):int(.94*h),max(0,center-10):min(w,center+11)]
    counts=np.bincount(seed.ravel(),minlength=n+1);counts[0]=0
    if counts.max()>0:femur=lab==counts.argmax()
    ys,xs=np.where(femur&(xx>center)&(yy>.08*h)&(yy<.60*h))
    gt=None if not len(xs) else [float(np.median(xs[ys==ys.min()])),float(ys.min())]
    ys,xs=np.where(pelvis&(xx<center-.22*w))
    isch=None if not len(xs) else [float(np.median(xs[ys==ys.max()])),float(ys.max())]
    ys,xs=np.where(femur&(yy>=(gt[1] if gt else .25*h)))
    lat=None if not len(xs) else [float(xs.max()),float(np.median(ys[xs==xs.max()]))]
    # Use the real bone contour: the artificial neck cut must never create a bump.
    lesser,diagnostic=lesser_candidate(clean,center,neck,gt[1]) if gt else (None,{'reason':'Greater trochanter unavailable'})
    def restore(p):
        if p is None:return None
        p=np.array(p,float)
        if flip:p[...,0]=w-1-p[...,0]
        return p
    f={'greater_trochanter':feature('Большой вертел',1,None if gt is None else restore([gt])),
       'ischium':feature('Седалищная кость',1,None if isch is None else restore([isch])),
       'lateral_femur':feature('Латеральный край бедра',1,None if lat is None else restore([lat])),
       'lesser_trochanter':feature('Малый вертел: начало, вершина, конец',3,restore(lesser))}
    if isch and isch[1]>=h-3:f['ischium']['visibility']='uncertain'
    if neck:
        neck['polygon']=restore(neck['polygon']).tolist();neck['cut']=restore(neck['cut']).tolist()
    partition=np.where(femur,1,np.where(pelvis,2,0)).astype('uint8')
    if flip:partition=np.fliplr(partition).copy()
    return f, 'left' if flip else 'right', partition, dict(neck_roi=neck,lesser_detection=diagnostic)


def plate(mask,upper):
    ys,xs=np.where(mask)
    if len(xs)<20:return None
    x0,x1=np.quantile(xs,[.15,.85]);edges=[]
    for x in range(int(x0),int(x1)+1):
        yy=np.flatnonzero(mask[:,x])
        if len(yy):edges.append((x,yy.min() if upper else yy.max()))
    a=np.asarray(edges,float)
    if len(a)<5:return None
    slope,intercept=np.polyfit(a[:,0],a[:,1],1)
    return [[float(x0),float(np.clip(slope*x0+intercept,0,mask.shape[0]-1))],
            [float(x1),float(np.clip(slope*x1+intercept,0,mask.shape[0]-1))]]


def spine_geometry(labels,bone):
    """No equal-band L5 synthesis. A separated full bone component is required."""
    f={};h,w=bone.shape;yy,xx=np.mgrid[:h,:w]
    for k in range(1,6):
        for side in ['upper','lower']:
            f[f'L{k}_{side}']=feature(f'L{k} '+side,2,plate(labels==k,side=='upper') if k<5 else None,
                                    source='estimated from bone contour; not verified anatomical endplate')
    l4=labels==4;ys,xs=np.where(l4);l5=np.zeros_like(bone,bool)
    if len(xs)>30:
        left,right=np.quantile(xs,[.05,.95]);bottom=ys.max();height=ys.max()-ys.min()+1
        # Actual connected components separated from L4, without an artificial upper cut.
        candidate=ndi.binary_opening(bone&(labels==0),structure=np.ones((3,3)))
        lab,n=ndi.label(candidate)
        for k in range(1,n+1):
            cy,cx=np.where(lab==k)
            if len(cx)<100:continue
            bw=cx.max()-cx.min()+1;bh=cy.max()-cy.min()+1
            if (bottom<cy.min()<bottom+.35*height and cy.max()<h-3
                and .6*height<bh<1.5*height and .6*(right-left)<bw<1.6*(right-left)
                and abs(cx.mean()-xs.mean())<.25*(right-left)):
                l5=lab==k;break
        if l5.any():
            for side in ['upper','lower']:
                f['L5_'+side]=feature('L5 '+side,2,plate(l5,side=='upper'),source='separate bone component below L4; weak L5 identity')
    for side in ['left','right']:
        pts=None
        if len(xs)>30 and f['L3_lower']['points']:
            level=(np.mean(np.array(f['L3_lower']['points'])[:,1])+np.mean(np.array(f['L4_upper']['points'])[:,1]))/2
            edge=float(xs.min() if side=='left' else xs.max())
            area=bone&(yy>level)&((xx<edge-20/.6) if side=='left' else (xx>edge+20/.6))
            lab,n=ndi.label(area);sizes=np.bincount(lab.ravel());sizes[0]=0
            if n and sizes.max()>=30:
                cy,cx=np.where(lab==sizes.argmax());pts=[[float(np.median(cx[cy==cy.min()])),float(cy.min())]]
        f['iliac_'+side]=feature('Подвздошная кость '+side,1,pts)
    return f,l5
