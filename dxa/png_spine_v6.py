"""Aligned PNG-only whole-spine / single-vertebra inputs and inference helpers."""
import random
import cv2
import numpy as np
from scipy import ndimage as ndi

def resize_aligned(image,mask,weight,size):
    if image.shape!=mask.shape or image.shape!=weight.shape:raise ValueError('Image and target must share native coordinates before resizing')
    h,w=image.shape;s=size/max(h,w);nh,nw=round(h*s),round(w*s);y=(size-nh)//2;x=(size-nw)//2
    outputs=[]
    for arr,method in [(image,cv2.INTER_LINEAR),(mask,cv2.INTER_NEAREST),(weight,cv2.INTER_NEAREST)]:
        out=np.zeros((size,size),arr.dtype);out[y:y+nh,x:x+nw]=cv2.resize(arr,(nw,nh),interpolation=method);outputs.append(out)
    return outputs,(y,x,nh,nw)

def bbox(mask,jitter=False):
    yy,xx=np.where(mask)
    if len(xx)<20:return None
    h,w=mask.shape;bh=yy.max()-yy.min()+1;bw=xx.max()-xx.min()+1
    mx=max(10,round(.3*bw));my=max(8,round(.32*bh))
    dx=round(random.uniform(-.10,.10)*bw) if jitter else 0;dy=round(random.uniform(-.1,.1)*bh) if jitter else 0
    return (max(0,int(yy.min())-my+dy),min(h,int(yy.max())+my+1+dy),max(0,int(xx.min())-mx+dx),min(w,int(xx.max())+mx+1+dx))

def intensity(a,augment=False):
    a=a.astype('float32')/255
    if augment:
        a=np.clip(a**random.uniform(.45,2.1)*random.uniform(.8,1.2)+random.uniform(-.06,.06),0,1)
        if random.random()<.5:a=1-a
        if random.random()<.25:a=cv2.GaussianBlur(a,(3,3),random.uniform(.3,1.0))
        a=np.clip(a+np.random.normal(0,random.uniform(0,.02),a.shape),0,1)
    return np.uint8(a*255)

def tensor_channels(image,local=False):
    a=image.astype('float32')/255
    if local:
        contrast=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(image)/255
        y,x=np.mgrid[:image.shape[0],:image.shape[1]]
        prior=np.exp(-(((x-(image.shape[1]-1)/2)/(.28*image.shape[1]))**2+((y-(image.shape[0]-1)/2)/(.28*image.shape[0]))**2)/2)
        out=np.stack([a,contrast,prior]).astype('float32')
    else:out=np.repeat(a[None],3,0)
    return (out-np.array([.485,.456,.406],dtype='float32')[:,None,None])/np.array([.229,.224,.225],dtype='float32')[:,None,None]

def contrast_views(raw):
    a=raw.astype('float32');lo,hi=np.percentile(a,[1,99]);p=np.uint8(np.clip((a-lo)/max(hi-lo,1)*255,0,255))
    return {'positive':p,'negative':255-p,'soft':np.uint8((p/255.)**.65*255),
            'hard':np.uint8((p/255.)**1.6*255),'clahe':cv2.createCLAHE(2,(8,8)).apply(p)}

def component(mask,seed=None):
    lab,n=ndi.label(mask)
    if not n:return np.zeros_like(mask,bool)
    count=np.bincount(lab[seed].ravel(),minlength=n+1) if seed is not None and seed.any() else np.bincount(lab.ravel())
    count[0]=0
    if count.max()==0:count=np.bincount(lab.ravel());count[0]=0
    return ndi.binary_fill_holes(lab==count.argmax())
