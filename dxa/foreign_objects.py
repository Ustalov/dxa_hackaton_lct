"""Binary foreign-object model with an auxiliary localization head."""
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

SIZE=320

def input_tensor(image):
    h,w=image.shape;s=SIZE/max(h,w);nh,nw=round(h*s),round(w*s);y,x=(SIZE-nh)//2,(SIZE-nw)//2
    im=cv2.resize(image,(nw,nh));canvas=np.zeros((SIZE,SIZE),'uint8');canvas[y:y+nh,x:x+nw]=im
    contrast=cv2.createCLAHE(2,(8,8)).apply(canvas)
    residual=cv2.absdiff(canvas,cv2.GaussianBlur(canvas,(0,0),4))
    channels=np.stack([canvas,contrast,np.minimum(residual.astype('float32')*3,255)]).astype('float32')/255
    return torch.from_numpy((channels-.5)/.25),(y,x,nh,nw)

class ForeignObjectNet(nn.Module):
    def __init__(self,initial=None):
        super().__init__();self.encoder=resnet18(weights=None)
        if initial:self.encoder.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True))
        self.encoder.fc=nn.Identity()
        self.classifier=nn.Sequential(nn.Linear(1024,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,1))
        self.low=nn.Conv2d(128,48,1);self.high=nn.Conv2d(512,48,1)
        self.mask=nn.Sequential(nn.Conv2d(48,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,1,1))
    def forward(self,x):
        b=self.encoder;x=b.maxpool(b.relu(b.bn1(b.conv1(x))));x=b.layer1(x);low=b.layer2(x);x=b.layer3(low);high=b.layer4(x)
        pooled=torch.cat([F.adaptive_avg_pool2d(high,1).flatten(1),F.adaptive_max_pool2d(high,1).flatten(1)],1)
        mask=self.mask(self.low(low)+F.interpolate(self.high(high),size=low.shape[-2:],mode='bilinear',align_corners=False))
        return dict(logit=self.classifier(pooled)[:,0],mask=mask)

@torch.no_grad()
def foreign_prediction(net,image):
    t,(y,x,nh,nw)=input_tensor(image);out=net(t[None].to(next(net.parameters()).device))
    score=float(out['logit'].sigmoid()[0]);heat=F.interpolate(out['mask'].sigmoid(),(SIZE,SIZE),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
    heat=cv2.resize(heat[y:y+nh,x:x+nw],(image.shape[1],image.shape[0]))
    return score,heat

def paste_object(image,signal,rng):
    """Add attenuating object residual only; never transfer its chest background."""
    h,w=image.shape;s=np.array(signal,copy=True)
    if rng.random()<.5:s=np.fliplr(s)
    # Mild anisotropy/cropping changes underwire length and width without relabelling anatomy.
    longest=float(rng.uniform(25,min(180,.8*min(h,w))))
    scale=longest/max(s.shape);nw=max(3,round(s.shape[1]*scale*rng.uniform(.8,1.15)));nh=max(3,round(s.shape[0]*scale*rng.uniform(.8,1.15)))
    s=cv2.resize(s,(nw,nh))
    if rng.random()<.25:s=s[:max(2,round(nh*rng.uniform(.7,1))),:max(2,round(nw*rng.uniform(.7,1)))]
    sh,sw=s.shape;side=max(sh,sw)*2;canvas=np.zeros((side,side),'float32');y0,x0=(side-sh)//2,(side-sw)//2;canvas[y0:y0+sh,x0:x0+sw]=s
    mat=cv2.getRotationMatrix2D((side/2,side/2),float(rng.uniform(-20,20)),1);s=cv2.warpAffine(canvas,mat,(side,side))
    yy,xx=np.where(s>.04)
    if not len(xx):return image.copy(),np.zeros(image.shape,'float32')
    s=s[yy.min():yy.max()+1,xx.min():xx.max()+1];sh,sw=s.shape
    if sh>=h or sw>=w:
        scale=min((h-2)/sh,(w-2)/sw);s=cv2.resize(s,(max(2,round(sw*scale)),max(2,round(sh*scale))));sh,sw=s.shape
    ylimit=max(1,h-sh);ylimit=min(ylimit,max(1,int(.5*h))) if rng.random()<.7 else ylimit
    y=int(rng.integers(0,ylimit));x=int(rng.integers(0,max(1,w-sw)))
    alpha=cv2.GaussianBlur(s,(0,0),float(rng.uniform(.2,.65)));out=image.astype('float32').copy();region=out[y:y+sh,x:x+sw]
    amplitude=float(rng.uniform(40,190));out[y:y+sh,x:x+sw]=region+amplitude*alpha*(1-region/300)
    mask=np.zeros((h,w),'float32');mask[y:y+sh,x:x+sw]=alpha>.10
    return np.uint8(np.clip(out,0,255)),mask
