"""Independent PNG-trained anatomy networks; ImageNet-only encoder initialization."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

class Block(nn.Sequential):
    def __init__(self,inc,out):
        super().__init__(nn.Conv2d(inc,out,3,padding=1,bias=False),nn.GroupNorm(8,out),nn.SiLU(),
                         nn.Conv2d(out,out,3,padding=1,bias=False),nn.GroupNorm(8,out),nn.SiLU())

class AnatomyNet(nn.Module):
    def __init__(self,region,imagenet_path=None,geometry_channels=0,geometry_context=False):
        super().__init__()
        if region not in ('hip','spine'):raise ValueError(region)
        self.region=region
        base=resnet18(weights=None)
        if imagenet_path:base.load_state_dict(torch.load(imagenet_path,map_location='cpu',weights_only=True))
        self.stem=nn.Sequential(base.conv1,base.bn1,base.relu)
        self.pool=base.maxpool;self.enc1=base.layer1;self.enc2=base.layer2;self.enc3=base.layer3;self.enc4=base.layer4
        self.dec3=Block(512+256,192);self.dec2=Block(192+128,128);self.dec1=Block(128+64,64);self.dec0=Block(64+64,32)
        self.full=Block(32,24);self.bone=nn.Conv2d(24,1,1)
        self.vertebrae=nn.Conv2d(24,5,1) if region=='spine' else None
        self.geometry_context=geometry_context
        self.landmarks=(nn.Sequential(Block(26,24),nn.Conv2d(24,geometry_channels,1)) if geometry_context else nn.Conv2d(24,geometry_channels,1)) if geometry_channels else None
    def forward(self,x):
        size=x.shape[-2:];s=self.stem(x);a=self.enc1(self.pool(s));b=self.enc2(a);c=self.enc3(b);d=self.enc4(c)
        def up(u,v,block):return block(torch.cat([F.interpolate(u,size=v.shape[-2:],mode='bilinear',align_corners=False),v],1))
        y=up(d,c,self.dec3);y=up(y,b,self.dec2);y=up(y,a,self.dec1);y=up(y,s,self.dec0)
        y=self.full(F.interpolate(y,size=size,mode='bilinear',align_corners=False))
        geo=y
        if self.landmarks is not None and self.geometry_context:
            gy,gx=torch.meshgrid(torch.linspace(-1,1,y.shape[2],device=y.device,dtype=y.dtype),torch.linspace(-1,1,y.shape[3],device=y.device,dtype=y.dtype),indexing='ij')
            geo=torch.cat([y,torch.stack([gx,gy])[None].expand(y.shape[0],-1,-1,-1)],1)
        return {'bone':self.bone(y),'vertebrae':self.vertebrae(y) if self.vertebrae is not None else None,
                'landmarks':self.landmarks(geo) if self.landmarks is not None else None}
