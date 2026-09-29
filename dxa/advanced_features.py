"""Frozen foundation representations and image-only shape descriptors.

All learned preprocessing belongs in the downstream training fold. This module
uses public pretrained weights, pixel intensities and acquisition geometry only.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
import timm
from scipy.ndimage import gaussian_filter1d
from .imaging import canonical, letterbox

FOUNDATIONS = {
    'dino_small': ('vit_small_patch14_dinov2.lvd142m', 378),
    'dino_base': ('vit_base_patch14_dinov2.lvd142m', 378),
    'convnext': ('convnext_tiny.fb_in22k_ft_in1k', 384),
}

def shape_features(image, meta):
    a = canonical(image, meta).astype(np.float32)
    a = cv2.resize(a, (128, 192), interpolation=cv2.INTER_AREA)/255
    h, w = a.shape; yy, xx = np.mgrid[:h, :w]; xx=xx/w; yy=yy/h
    vals=[]
    # Brightness robustness: both absolute and within-image quantile thresholds.
    for threshold in [.2, .35, .5, .65, .8, .93] + [float(np.quantile(a,q)) for q in [.6,.75,.9]]:
        mask=(a>threshold).astype(np.uint8)
        for low, high in [(0,1),(0,.35),(.2,.65),(.55,1)]:
            region=mask.copy(); region[:int(low*h)]=0; region[int(high*h):]=0
            count=region.sum(); cx=(region*xx).sum()/max(count,1); cy=(region*yy).sum()/max(count,1)
            vx=(region*(xx-cx)**2).sum()/max(count,1); vy=(region*(yy-cy)**2).sum()/max(count,1)
            cov=(region*(xx-cx)*(yy-cy)).sum()/max(count,1)
            vals.extend([count/(h*w),cx,cy,vx,vy,cov])
        components, _, stats, centroids=cv2.connectedComponentsWithStats(mask,8)
        areas=sorted(stats[1:,cv2.CC_STAT_AREA],reverse=True)
        vals.extend([min(components-1,50)/50]+[(areas[j] if j<len(areas) else 0)/(h*w) for j in range(4)])
        for band in np.array_split(mask,12,axis=0):
            weights=band.sum(0); nz=np.flatnonzero(weights)
            vals.extend([weights.sum()/band.size,float(nz[0]/w) if len(nz) else 0,
                         float(nz[-1]/w) if len(nz) else 0,float(np.dot(weights,np.arange(w))/(max(weights.sum(),1)*w))])
    # Row-profile spatial information; no cropping discards the acquisition edge.
    for band in np.array_split(a,12,axis=0):
        for patch in np.array_split(band,8,axis=1):
            vals.extend([patch.mean(),patch.std(),np.quantile(patch,.9)])
    gradx=cv2.Sobel(a,cv2.CV_32F,1,0);grady=cv2.Sobel(a,cv2.CV_32F,0,1)
    for arr in [a,abs(gradx),abs(grady),abs(cv2.Laplacian(a,cv2.CV_32F))]:
        for axis in [0,1]:
            profile=arr.mean(axis=axis)
            vals.extend(np.interp(np.linspace(0,1,32),np.linspace(0,1,len(profile)),profile))
    return np.asarray(vals,np.float32)

def detail_view(image,meta):
    h,w=image.shape
    # Fixed anatomy-specific view, complementary to (not replacing) the full field.
    if meta['region']=='hip':a=image[:max(32,int(.75*h))]
    else:a=image[int(.05*h):int(.90*h),int(.15*w):int(.85*w)]
    return a,{**meta,'shape':list(a.shape)}

def contrast_view(image):
    # Fixed local contrast transform; never estimates a parameter from other images.
    return cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(image)

def contour_features(image,meta):
    """Experimental image contours; no claim of verified anatomic landmarks."""
    a=canonical(image,meta).astype(np.float32);h,w=a.shape;sy,sx=meta['spacing'];vals=[]
    sm=cv2.GaussianBlur(a,(5,5),1.2)
    for threshold in [60,90,120,150,180]:
        ys=np.linspace(.94*h,.12*h,36).astype(int);centers=[];left=[];right=[];mass=[]
        distal=np.maximum(sm[int(.72*h):int(.93*h)]-threshold,0).sum(0)
        center=float(np.dot(distal,np.arange(w))/max(distal.sum(),1))
        if meta['region']=='spine':center=w/2
        for y in ys:
            radius=w*(.24 if meta['region']=='hip' else .2);low=max(0,int(center-radius));high=min(w,int(center+radius))
            strip=np.maximum(sm[max(0,y-2):min(h,y+3)].mean(0)-threshold,0);strip[:low]=0;strip[high:]=0
            total=strip.sum();cum=np.cumsum(strip)/max(total,1)
            if total>0:
                c=float(np.dot(strip,np.arange(w))/total);l=float(np.searchsorted(cum,.07));r=float(np.searchsorted(cum,.93));center=.75*c+.25*center
            else:c=center;l=r=center
            centers.append(c);left.append(l);right.append(r);mass.append(total/(255*w))
        left=np.asarray(left);right=np.asarray(right);centers=np.asarray(centers);ys=ys.astype(float)
        for values in [centers,left,right]:
            slope,intercept=np.polyfit(ys[:12]*sy,values[:12]*sx,1);residual=values*sx-(slope*ys*sy+intercept)
            vals.extend([np.degrees(np.arctan(slope))/30,np.std(residual)/30,np.max(residual)/30,np.min(residual)/30])
            vals.extend((values/w).tolist());vals.extend((residual[10:30]/30).tolist())
        vals.extend(((right-left)/w).tolist());vals.extend(mass)
    return np.nan_to_num(np.asarray(vals,np.float32))

def representation_context(features,groups,regions,key='dino_small_cls'):
    x=features[key];g=np.asarray(groups);regions=np.asarray(regions);out=[]
    for i in range(len(x)):
        blocks=[]
        for region in ['spine','hip']:
            ids=np.flatnonzero((g==g[i])&(regions==region)&(np.arange(len(x))!=i))
            mean=x[ids].mean(0) if len(ids) else np.zeros(x.shape[1],dtype=np.float32)
            blocks.extend([mean,abs(mean-x[i]) if len(ids) else np.zeros_like(mean),np.array([min(len(ids),2)],np.float32)])
        out.append(np.concatenate(blocks))
    return np.asarray(out,np.float32)

class FoundationExtractor:
    def __init__(self, artifacts, name, device='cpu', download=False):
        self.name=name;self.device=device
        model_name,self.size=FOUNDATIONS[name]
        path=Path(artifacts)/'foundation'/f'{name}.pth'
        official=Path(artifacts)/'foundation'/f'{name}_official.pth'
        kw={'num_classes':0}
        if name.startswith('dino'):kw['img_size']=self.size
        self.model=timm.create_model(model_name,pretrained=download and not path.exists() and not official.exists(),**kw)
        if path.exists():self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        elif official.exists():
            from timm.models.vision_transformer import checkpoint_filter_fn
            state=checkpoint_filter_fn(torch.load(official,map_location='cpu',weights_only=True),self.model)
            self.model.load_state_dict(state)
            if download:torch.save(self.model.state_dict(),path)
        elif download:
            path.parent.mkdir(parents=True,exist_ok=True);torch.save(self.model.state_dict(),path)
        else:raise FileNotFoundError(f'Foundation weights unavailable: {path}')
        self.model.to(device).eval()

    def extract(self, images, metas, batch_size=8):
        tensors=[]
        for image,meta in zip(images,metas):
            a=canonical(image,meta);box=letterbox(a,self.size,meta['spacing'])
            t=torch.from_numpy(box.copy()).float().div(255).repeat(3,1,1)
            tensors.append((t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None])
        cls_out=[];mean_out=[];spatial_out=[]
        with torch.inference_mode():
            for start in range(0,len(tensors),batch_size):
                x=torch.stack(tensors[start:start+batch_size]).to(self.device)
                z=self.model.forward_features(x)
                if z.ndim==3:
                    cls=z[:,0];tokens=z[:,self.model.num_prefix_tokens:];side=int(tokens.shape[1]**.5)
                    grid=tokens.transpose(1,2).reshape(len(z),-1,side,side);mean=tokens.mean(1)
                else:
                    grid=z;mean=z.mean((2,3));cls=mean
                spatial=torch.nn.functional.adaptive_avg_pool2d(grid,(2,2)).flatten(1)
                cls_out.append(cls.cpu().numpy());mean_out.append(mean.cpu().numpy());spatial_out.append(spatial.cpu().numpy())
        return {self.name+'_cls':np.concatenate(cls_out),self.name+'_mean':np.concatenate(mean_out),
                self.name+'_spatial':np.concatenate(spatial_out)}
