"""PNG-only weak-reference preparation; no DICOM imports or trained QC weights.

Labels are proposals, not independent anatomical truth. Uncertain pixels have
zero supervision weight. Spine source ROI bands assist L1-L4 identity only.
"""
import itertools
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects,remove_small_holes
from .segmentation_pilot import annotation_layers,normalize,spine_reference,groups

def recover_bands(rgb):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=g.shape
    residual=np.abs(g.astype(float)-cv2.medianBlur(g,7).astype(float))
    color=(rgb.max(2).astype(int)-rgb.min(2).astype(int))>20
    stroke=((residual>12)&((g<115)|(g>225)))|color
    lines=cv2.HoughLinesP(stroke.astype('uint8')*255,1,np.pi/720,
                          threshold=max(30,int(.2*w)),minLineLength=int(.50*w),maxLineGap=12)
    candidates=[]
    if lines is not None:
        for x0,y0,x1,y1 in lines.reshape(-1,4):
            if abs(x1-x0)<.5*w or abs(y1-y0)>.10*abs(x1-x0):continue
            y=float(y0+(w/2-x0)*(y1-y0)/(x1-x0))
            if .035*h<y<.965*h:candidates.append(y)
    clusters=groups(sorted(int(round(y)) for y in candidates),gap=4)
    centers=[int(np.median(c)) for c in clusters]
    if len(centers)<5:return None
    # Restrict to a plausible sequence of four roughly similar-height ROI bands.
    best=None
    for b in itertools.combinations(centers,5):
        d=np.diff(b)
        if min(d)<.085*h or max(d)>.30*h or b[-1]-b[0]<.48*h:continue
        score=np.std(d)/np.mean(d)-.08*(b[-1]-b[0])/h
        if best is None or score<best[0]:best=(score,b)
    return list(best[1]) if best else None

def make_reference(rgb,region,reviewed_bounds=None,approved_labels=None,lower_ratio=.60):
    bounds=reviewed_bounds if reviewed_bounds is not None else recover_bands(rgb) if region=='spine' else None
    cleaned,annotation,edges,detected,info=annotation_layers(rgb,region,True,bounds)
    # Additional black ROI/neck strokes missed by the original thin-white detector.
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=gray.shape
    dark=((cv2.medianBlur(gray,7).astype(float)-gray)>12)&(gray<100)
    lines=cv2.HoughLinesP(dark.astype('uint8')*255,1,np.pi/360,threshold=30,minLineLength=int(.22*w),maxLineGap=6)
    extra=np.zeros_like(gray)
    if lines is not None:
        for x0,y0,x1,y1 in lines.reshape(-1,4):
            line=np.zeros_like(gray);cv2.line(line,(x0,y0),(x1,y1),1,2)
            extra|=(line.astype(bool)&ndi.binary_dilation(dark,iterations=1)).astype('uint8')
    annotation|=extra
    cleaned=cv2.inpaint(gray,annotation*255,3,cv2.INPAINT_TELEA)
    image,norm=normalize(cleaned,region)
    smooth=cv2.GaussianBlur(image,(3,3),.65)
    valid=~ndi.binary_dilation(annotation.astype(bool),iterations=1)
    if valid.sum()<.35*image.size:raise ValueError('Too many annotation/uncertain pixels')
    t=float(threshold_otsu(smooth[valid]));seeds=smooth>t*.95
    permissive=smooth>t*lower_ratio
    bone=ndi.binary_propagation(seeds,mask=permissive)
    bone=remove_small_objects(bone,min_size=70,connectivity=2)
    bone=ndi.binary_closing(bone,structure=np.ones((3,3)),border_value=0)
    bone=remove_small_holes(bone,area_threshold=max(50,int(image.size*.006)))
    if bone.mean()<.08 or bone.mean()>.75:raise ValueError('Implausible foreground fraction')
    # Interior foreground is learned weakly; intermediate boundary/soft-tissue
    # intensities remain ignored rather than being declared expert truth.
    confident_bg=(smooth<t*.40)&~ndi.binary_dilation(bone,iterations=2)
    confident_fg=ndi.binary_erosion(bone,iterations=1)
    bone_weight=(confident_bg|confident_fg)&valid
    labels=np.zeros_like(image,np.uint8);class_weight=np.zeros_like(image,np.uint8)
    reference=np.zeros_like(image,np.uint8);ref_meta={'available':False}
    if region=='spine':
        reference,ref_meta=spine_reference(edges,bounds or detected)
        if reference.any():
            supports=[min(v['left_edge_support'],v['right_edge_support']) for v in ref_meta['bands']]
            fraction=float((reference>0).mean())
            reliable=min(supports)>=.60 and .07<fraction<.50
            if reliable:
                labels=reference.copy()
                for k in range(1,5):class_weight[ndi.binary_erosion(reference==k,iterations=2)&valid]=1
                class_weight[confident_bg&valid]=1
            ref_meta.update({'passes_automatic_filter':bool(reliable),'minimum_edge_support':float(min(supports))})
        if approved_labels is not None:
            assert approved_labels.shape==labels.shape
            labels=np.where((approved_labels>=1)&(approved_labels<=4),approved_labels,0).astype('uint8')
            class_weight[:]=0
            for k in range(1,5):class_weight[ndi.binary_erosion(labels==k,iterations=1)&valid]=1
            class_weight[confident_bg&valid]=1
            # Approved L1-L4 foreground supplements the general weak bone mask.
            bone[labels>0]=True;bone_weight[(labels>0)&valid]=True
            ref_meta['expert_reviewed_pilot_labels']=True
    return {'image':image,'bone':bone.astype('uint8'),'bone_weight':bone_weight.astype('uint8'),
            'labels':labels,'class_weight':class_weight,'ignore':annotation.astype('uint8'),
            'reference':reference,'meta':{'threshold_otsu':t,'lower_threshold':t*lower_ratio,'normalization':norm,
            'line_extraction':info,'reference':ref_meta,'bone_fraction':float(bone.mean()),
            'bone_supervised_fraction':float(bone_weight.mean()),'class_supervised_pixels':int(class_weight.sum()),
            'label_kind':'expert-reviewed pilot + weak references' if approved_labels is not None else 'automatically generated weak reference',
            'native_spacing_yx_mm':[1.05,.6],'spacing_source':'user_assumed_dicom_equivalent'}}
