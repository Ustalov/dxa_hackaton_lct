"""Recover burned-in lumbar outlines; retain partial contours with explicit weak regions."""
import itertools
import cv2,numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from .png_segmentation import recover_bands

def recover_roi_lines(rgb):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=g.shape
    residual=cv2.medianBlur(g,7).astype(float)-g
    color=(rgb.max(2).astype(float)-rgb.min(2)>8)&(rgb[:,:,0].astype(float)>rgb[:,:,2]+5.)&(rgb[:,:,1].astype(float)>rgb[:,:,2]+3.)
    dark=(((residual>5)&(g<210))|((residual<-8)&(g>210))|color).astype('uint8')*255
    lines=cv2.HoughLinesP(dark,1,np.pi/720,threshold=15,minLineLength=int(.16*w),maxLineGap=18)
    candidates=[]
    if lines is not None:
        for x,y,u,v in lines.reshape(-1,4):
            if abs(u-x)<.16*w or abs(v-y)>.35*abs(u-x):continue
            if min(x,u)>.29*w and max(x,u)<.75*w:continue
            slope=(v-y)/(u-x);yc=y+(w/2-x)*slope
            if -.08*h<yc<1.03*h:candidates.append((float(yc),float(slope),float(abs(u-x))))
    candidates.sort();groups=[]
    for line in candidates:
        if groups and line[0]-np.mean([a[0] for a in groups[-1]])<5:groups[-1].append(line)
        else:groups.append([line])
    merged=[]
    for group in groups:
        best=max(group,key=lambda a:a[2]);merged.append(best)
    old=recover_bands(rgb)
    if old:
        recovered=[]
        for y in old:
            nearby=[a for a in merged if abs(a[0]-y)<7]
            a=min(nearby,key=lambda a:abs(a[0]-y)) if nearby else (float(y),0.,0.)
            recovered.append(dict(y=a[0],slope=a[1]))
        return recovered,'original ROI levels plus recovered line slopes'
    if len(merged)>16:merged=sorted(sorted(merged,key=lambda a:a[2],reverse=True)[:16])
    best=None
    for choices in itertools.combinations(merged,5):
        ys=np.array([a[0] for a in choices]);d=np.diff(ys)
        if min(d)<.045*h or max(d)>.36*h or ys[-1]-ys[0]<.30*h:continue
        score=sum(a[2]/w for a in choices)-3*np.std(d)/np.mean(d)+.3*(ys[-1]-ys[0])/h
        if best is None or score>best[0]:best=score,choices
    if best:return [dict(y=a[0],slope=a[1]) for a in best[1]],'peripheral dark ROI strokes'
    return None,'ROI boundaries not recovered'

def trace(score,prior,low,high):
    xs=np.arange(low,high);cost=-score[:,low:high]+.035*(xs[None]-prior[:,None])**2
    transition=.10*(xs[:,None]-xs[None,:])**2;value=cost[0];pointers=[]
    for row in cost[1:]:
        choices=value[:,None]+transition;ptr=choices.argmin(0);value=choices.min(0)+row;pointers.append(ptr)
    path=[int(value.argmin())]
    for ptr in reversed(pointers):path.append(int(ptr[path[-1]]))
    return xs[np.array(path[::-1])]

def recover_contours(rgb,image,bone,roi_hints=None):
    h,w=image.shape
    if roi_hints:
        lines=[dict(y=float(a[0]),slope=float(a[1])) for a in roi_hints];method='assistant read existing source ROI boundaries; not expert anatomy labels'
    else:lines,method=recover_roi_lines(rgb)
    if lines is None:return None,dict(method=method,available=False)
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
    # Whiteness alone is insufficient on a white background: require local stroke contrast.
    horizontal=cv2.medianBlur(g.astype('uint8'),5).astype(float)
    stroke=np.clip((g-horizontal-5)/25,0,3)*(g>205)
    residual=(g-np.minimum(ndi.shift(g,(0,-2),order=0,mode='nearest'),ndi.shift(g,(0,2),order=0,mode='nearest')))
    stroke+=np.clip((residual-15)/45,0,1)*(g>235)
    color=(rgb.max(2).astype(float)-rgb.min(2)>8)&(rgb[:,:,0].astype(float)>rgb[:,:,2]+5.)&(rgb[:,:,1].astype(float)>rgb[:,:,2]+3.)
    stroke[color]=5
    score=ndi.maximum_filter(stroke,size=(1,3));labels=np.zeros((h,w),'uint8');details=[]
    xgrid=np.arange(w);ygrid=np.arange(h)[:,None]
    for k,(upper,lower) in enumerate(zip(lines[:-1],lines[1:]),1):
        top=upper['y']+(xgrid-w/2)*upper['slope'];bottom=lower['y']+(xgrid-w/2)*lower['slope']
        y0=max(0,int(np.min(top))+1);y1=min(h,int(np.max(bottom)))
        if y1-y0<8:return None,dict(available=False,method=method,reason='invalid recovered ROI')
        center=int(np.argmax(ndi.gaussian_filter1d(image[y0:y1,int(.2*w):int(.8*w)].mean(0).astype(float),8)))+int(.2*w)
        left=[];right=[]
        for y in range(y0,y1):
            row=ndi.binary_closing(bone[y].astype(bool),iterations=2);lab,n=ndi.label(row)
            runs=[np.flatnonzero(lab==j) for j in range(1,n+1)];runs=[r for r in runs if len(r)>=.10*w and abs(np.mean(r)-center)<.23*w]
            if runs:
                r=min(runs,key=lambda r:abs(np.mean(r)-center));left.append(float(r.min()));right.append(float(r.max()))
            else:left.append(np.nan);right.append(np.nan)
        valid=np.isfinite(left)
        if valid.sum()<.35*(y1-y0):
            ls=score[y0:y1,int(.15*w):int(.49*w)].sum(0);rs=score[y0:y1,int(.51*w):int(.86*w)].sum(0)
            if min(ls.max(),rs.max())<.15*(y1-y0):return None,dict(available=False,method=method,reason='insufficient bone or stroke support')
            left=np.full(y1-y0,int(.15*w)+ls.argmax(),float);right=np.full(y1-y0,int(.51*w)+rs.argmax(),float);center=int((left[0]+right[0])/2)
        else:
            ys=np.arange(y1-y0);left=np.interp(ys,ys[valid],np.array(left)[valid]);right=np.interp(ys,ys[valid],np.array(right)[valid])
        left=ndi.median_filter(left,7);right=ndi.median_filter(right,7)
        lo=max(2,int(min(np.quantile(left,.05)-12,center-.12*w)));hi=min(center-3,int(np.quantile(left,.95)+14))
        rl=max(center+3,int(np.quantile(right,.05)-14));rh=min(w-2,int(max(np.quantile(right,.95)+12,center+.12*w)))
        if hi-lo<4 or rh-rl<4:return None,dict(available=False,method=method,reason='side search degenerate')
        l=trace(score[y0:y1],left,lo,hi);r=trace(score[y0:y1],right,rl,rh)
        mask=np.zeros((h,w),bool)
        for y,a,b in zip(range(y0,y1),l,r):mask[y,a:b+1]=True
        mask&=(ygrid>top[None])&(ygrid<bottom[None]);labels[mask]=k
        support=float(np.mean(np.r_[score[np.arange(y0,y1),l]>0.3,score[np.arange(y0,y1),r]>0.3]))
        details.append(dict(level=f'L{k}',stroke_support=support,missing_strokes='bone-intensity-supported contour continuation',pixels=int(mask.sum())))
    if any((labels==k).sum()<50 for k in range(1,5)):return None,dict(available=False,reason='missing vertebra after contour reconstruction')
    return labels,dict(available=True,method=method,roi_lines=lines,bands=details,numbering='L1-L4 top to bottom; user confirmed dataset content',kind='recovered white strokes plus intensity-assisted continuation, weak reference')
