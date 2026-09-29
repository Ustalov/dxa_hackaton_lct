"""Inspectable intensity-based proposals. No learned segmentation weights or expert masks.

Burned-in contours are weak references. Inpainted pixels are always marked unknown
for future training. All numbered vertebrae and metal candidates require review.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu,threshold_multiotsu,threshold_sauvola
from skimage.morphology import remove_small_objects,remove_small_holes,skeletonize
from skimage.segmentation import watershed

LABELS={0:'background',1:'L1',2:'L2',3:'L3',4:'L4',5:'Th12_candidate',
        6:'ribs_or_other_bone',7:'bone_unassigned',8:'femur',9:'pelvis_candidate',10:'foreign_metal_candidate'}
COLORS={1:(245,103,95),2:(255,194,73),3:(89,211,118),4:(66,171,255),5:(199,125,250),
        6:(84,216,212),7:(184,191,201),8:(75,208,143),9:(85,155,249),10:(255,50,190)}

def groups(values,gap=3):
    out=[]
    for v in values:
        if out and v-out[-1][-1]<=gap:out[-1].append(int(v))
        else:out.append([int(v)])
    return out

def annotation_layers(rgb,region,has_annotations=True,reviewed_bounds=None):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=gray.shape
    if not has_annotations:return gray,np.zeros_like(gray),np.zeros_like(gray),[],{'method':'unannotated DICOM'}
    color=(rgb.max(2).astype(int)-rgb.min(2).astype(int)>20)&(rgb.max(2)>100)
    med=cv2.medianBlur(gray,5)
    white=(gray>=225)&(gray.astype(float)-med>15)
    # Thin white contour strokes remain distinguishable even on negative images.
    white|=(gray>=248)&(cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,np.ones((3,3),np.uint8))>18)
    white=remove_small_objects(white,min_size=5,connectivity=2)
    dark=(med.astype(float)-gray>20)&(gray<110)
    thin=(white|dark|color).astype('uint8')
    horizontal=cv2.morphologyEx(thin,cv2.MORPH_OPEN,np.ones((1,max(25,int(.35*w))),np.uint8))
    vertical=cv2.morphologyEx(thin,cv2.MORPH_OPEN,np.ones((max(25,int(.25*h)),1),np.uint8))
    annotation=(white|color|horizontal.astype(bool)|vertical.astype(bool)).astype('uint8')
    lines=cv2.HoughLinesP(thin*255,1,np.pi/180,threshold=35,minLineLength=int(.28*w),maxLineGap=5)
    if lines is not None and region=='hip':
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            angle=abs(np.degrees(np.arctan2(y2-y1,x2-x1)))%180
            if 15<angle<75 or 105<angle<165:
                line=np.zeros_like(gray);cv2.line(line,(x1,y1),(x2,y2),1,2)
                annotation|=(line.astype(bool)&ndi.binary_dilation(thin,iterations=1)).astype('uint8')
    ys=[]
    if region=='spine':
        scores=horizontal.sum(1)
        candidates=groups(np.flatnonzero(scores>max(35,.4*w)))
        ys=[int(max(g,key=lambda y:scores[y])) for g in candidates]
        if len(ys)>5:
            # Keep five long lines; no intensity-based inference of vertebral numbers.
            ys=sorted(sorted(ys,key=lambda y:scores[y],reverse=True)[:5])
        if reviewed_bounds is not None:
            ys=list(reviewed_bounds)
            # Pilot-only manual recovery of existing overlay bands, not new anatomy labels.
            for y in ys:
                annotation[max(0,y-1):min(h,y+2),int(.03*w):int(.96*w)]=1
        # L1-L4 label boxes occupy a small strip to the left of vertebral contours.
        xcounts=vertical.sum(0);xlines=np.flatnonzero(xcounts>.25*h)
        left=int(xlines.min()) if len(xlines) else int(.03*w)
        for y in ys[:-1]:annotation[max(0,y+1):min(h,y+22),max(0,left):min(w,left+24)]=1
    annotation=ndi.binary_dilation(annotation,iterations=1).astype('uint8')
    cleaned=cv2.inpaint(gray,annotation*255,3,cv2.INPAINT_TELEA)
    # Reference edge extraction uses the original exact strokes, never the inpaint.
    edges=(white|color).astype('uint8')
    for y in ys:edges[max(0,y-2):min(h,y+3)]=0
    return cleaned,annotation,edges,ys,{'method':'color + thin-stroke morphology + long-line geometry',
                                      'annotated_pixel_fraction':float(annotation.mean()),'horizontal_boundaries':ys,
                                      'band_source':'assistant read from original L1-L4 overlay' if reviewed_bounds else 'automatic line detection',
                                      'inpaint':'Telea radius 3; reconstructed pixels excluded from trusted supervision'}

def edge_path(edges,y0,y1,x0,x1):
    roi=edges[y0:y1,x0:x1];distance=ndi.distance_transform_edt(~roi.astype(bool))
    cost=np.minimum(distance,10).astype(float)
    n=roi.shape[1];xx=np.arange(n)
    transition=.06*(xx[:,None]-xx[None,:])**2
    scores=cost[0].copy();pointers=[]
    for row in cost[1:]:
        values=scores[:,None]+transition;pointers.append(values.argmin(0));scores=row+values.min(0)
    path=[int(scores.argmin())]
    for ptr in reversed(pointers):path.append(int(ptr[path[-1]]))
    path=np.asarray(path[::-1])+x0
    support=float(np.mean(distance[np.arange(len(path)),path-x0]<=1.5))
    return ndi.median_filter(path,size=3),support

def spine_reference(edges,bounds):
    h,w=edges.shape;reference=np.zeros((h,w),np.uint8);details=[]
    if len(bounds)!=5:return reference,{'available':False,'reason':'Could not recover exactly five ROI band boundaries'}
    for k,(y0,y1) in enumerate(zip(bounds[:-1],bounds[1:]),1):
        if y1-y0<8:return np.zeros_like(reference),{'available':False,'reason':'Implausible band spacing'}
        left,ls=edge_path(edges,y0+3,y1-2,int(.17*w),int(.49*w))
        right,rs=edge_path(edges,y0+3,y1-2,int(.51*w),int(.86*w))
        left=np.pad(left,(3,2),mode='edge');right=np.pad(right,(3,2),mode='edge')
        for yy,l,r in zip(range(y0,y1),left,right):reference[yy,l:r+1]=k
        details.append({'label':f'L{k}','left_edge_support':ls,'right_edge_support':rs})
    return reference,{'available':True,'kind':'weak contour reference; band order L1-L4 assumed from source labels; not expert ground truth','bands':details}

def normalize(gray,region):
    h,w=gray.shape
    # All pilot PNG polarities are verified against source contact sheets in the manifest.
    inverted=float(np.median(np.r_[gray[:,:max(1,w//12)].ravel(),gray[:,-max(1,w//12):].ravel()]))>150
    image=255-gray if inverted else gray.copy()
    low,high=np.percentile(image,[2,99]);scaled=np.uint8(np.clip((image.astype(float)-low)*255/max(high-low,1),0,255))
    return scaled,{'inverted':inverted,'low_percentile_2':float(low),'high_percentile_99':float(high),'units':'normalized display intensity, not physical bone density'}

def metal_candidates(image,region,enabled):
    if not enabled or region!='spine':return np.zeros_like(image,dtype=bool)
    h,w=image.shape;top=cv2.morphologyEx(image,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    possible=(top>25)&(image>100)
    possible[int(.42*h):]=False
    lab,n=ndi.label(possible,np.ones((3,3)))
    result=np.zeros_like(possible)
    for i in range(1,n+1):
        obj=lab==i;yy,xx=np.where(obj)
        if len(xx)<15:continue
        lateral=np.mean((xx<.37*w)|(xx>.67*w))
        length=int(skeletonize(obj).sum());width=float(ndi.distance_transform_edt(obj).max())
        if lateral>.6 and length>.1*w and width<=5:result|=obj
    return ndi.binary_dilation(result,iterations=1)

def threshold_variants(image,valid,metal):
    smooth=cv2.GaussianBlur(image,(3,3),.65);v=smooth[valid&~metal]
    global_t=float(threshold_otsu(v));multi=threshold_multiotsu(v,classes=3).astype(float)
    window=min(41,(min(image.shape)//2)*2-1);window=max(window,3)
    local=np.maximum(threshold_sauvola(smooth,window_size=window,k=.12),global_t*.85)
    thresholds={'otsu':global_t,'multiotsu':float(multi[1]),'local':None}
    masks={}
    for name,threshold in [('otsu',global_t),('multiotsu',multi[1]),('local',local)]:
        m=(smooth>threshold)&~metal
        m=remove_small_objects(m,min_size=25,connectivity=2)
        m=ndi.binary_closing(m,structure=np.ones((3,3)))
        m=remove_small_holes(m,area_threshold=max(30,int(image.size*.003)))
        masks[name]=m&~metal
    return masks,{'thresholds':thresholds,'multiotsu_both_thresholds':multi.tolist(),'local_window':window,'local_k':.12}

def spine_instances(bone,reference,bounds,native_guides=None):
    h,w=bone.shape;out=np.zeros_like(reference)
    if reference.any():
        # ROI bands assist instance identity; foreground remains threshold-derived.
        for k,(y0,y1) in enumerate(zip(bounds[:-1],bounds[1:]),1):
            envelope=ndi.binary_dilation(reference==k,iterations=max(3,int(.035*w)))
            envelope[:y0]=False;envelope[y1:]=False
            out[bone&envelope]=k
        outside=bone&(out==0);out[outside]=7
    elif native_guides:
        x0,x1=native_guides['spine_x'];bounds=native_guides['lumbar_y']
        for k,(y0,y1) in enumerate(zip(bounds[:-1],bounds[1:]),1):
            area=np.zeros_like(bone);area[y0:y1,x0:x1]=True;out[bone&area]=k
        top=np.zeros_like(bone);top[:bounds[0],x0:x1]=True;out[bone&top]=5
        out[bone&(out==0)]=6
    else:out[bone]=7
    return out

def hip_instances(bone,image):
    h,w=bone.shape;yy,xx=np.mgrid[:h,:w]
    # Distal shaft seed distinguishes the femur from adjacent pelvis.
    distal=bone[int(.76*h):int(.95*h)].sum(0)
    center=int(np.argmax(ndi.gaussian_filter1d(distal.astype(float),8)))
    markers=np.zeros_like(image,np.int32)
    shaft=bone&(yy>.80*h)&(abs(xx-center)<.12*w);markers[shaft]=1
    proximal=bone[int(.06*h):int(.35*h)].sum(0).astype(float)
    proximal_center=float(np.dot(proximal,np.arange(w))/max(proximal.sum(),1))
    pelvic_side=xx<.33*w if proximal_center<center else xx>.67*w
    pelvis=bone&pelvic_side&(yy<.60*h);markers[pelvis]=2
    if not shaft.any():return bone.astype('uint8')*7,{'distal_seed_available':False}
    # Label isolated bone components as well as connected structures.
    distance=ndi.distance_transform_edt(bone)
    labels=watershed(-distance,markers,mask=bone)
    out=np.zeros_like(image,np.uint8);out[labels==1]=8;out[labels==2]=9;out[bone&(labels==0)]=7
    return out,{'distal_seed_available':True,'shaft_center_x':center,'proximal_center_x':proximal_center,'method':'threshold foreground + distal-shaft/pelvic seeds + watershed; touching anatomy requires review'}

def overlay(rgb,labels,alpha=.48):
    out=rgb.astype(float).copy()
    for k,color in COLORS.items():
        mask=labels==k;out[mask]=out[mask]*(1-alpha)+np.array(color)*alpha
        edge=mask&~ndi.binary_erosion(mask);out[edge]=color
    return np.uint8(np.clip(out,0,255))
