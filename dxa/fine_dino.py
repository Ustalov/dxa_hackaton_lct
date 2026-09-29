import numpy as np
import torch
from torch import nn
import timm
from .imaging import canonical,letterbox
from .advanced_model import AdvancedModel

class FineDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,img_size=378,num_classes=0)
        self.classifier=nn.Linear(384,7)
    def forward(self,x):return self.classifier(self.encoder(x))

def image_tensor(a,meta):
    box=letterbox(canonical(a,meta),378,meta['spacing'])
    return torch.from_numpy(box.copy()).float().div(255).repeat(3,1,1)

def normalize(x):
    return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]

def predict(net,images,indices,device='cuda',batch_size=8):
    net.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(indices),batch_size):
            x=images[indices[start:start+batch_size]].to(device)
            out.append(torch.sigmoid(net(normalize(x))).cpu().numpy())
    return np.concatenate(out)

def fit(images,features,labels,indices,weights_path,seed=20260925,epochs=8,device='cuda'):
    torch.manual_seed(seed);np.random.seed(seed)
    net=FineDINO();net.encoder.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True))
    config=dict(name='warm_start',features=['dino_small_cls'],kind='linear',strength=.1)
    probe=AdvancedModel([config]*7).fit(features,labels,indices)
    with torch.no_grad():
        for j,head in enumerate(probe.heads):
            if isinstance(head,float):
                net.classifier.weight[j].zero_();net.classifier.bias[j]=float(np.log((head+.001)/(1.001-head)));continue
            scaler,lr=head.steps[0][1],head.steps[1][1]
            coef=lr.coef_[0]/scaler.scale_;bias=lr.intercept_[0]-np.dot(coef,scaler.mean_)
            net.classifier.weight[j].copy_(torch.tensor(coef,dtype=torch.float32));net.classifier.bias[j]=float(bias)
    for parameter in net.encoder.parameters():parameter.requires_grad=False
    for block in net.encoder.blocks[-2:]:
        for parameter in block.parameters():parameter.requires_grad=True
    for parameter in net.encoder.norm.parameters():parameter.requires_grad=True
    net.to(device)
    optimizer=torch.optim.AdamW([{'params':[p for p in net.encoder.parameters() if p.requires_grad],'lr':1e-5},
                                {'params':net.classifier.parameters(),'lr':1e-4}],weight_decay=.02)
    observed=np.isfinite(labels[indices]);positive=np.nansum(labels[indices],0);negative=observed.sum(0)-positive
    pw=torch.tensor(np.clip(negative/np.maximum(positive,1),1,10),dtype=torch.float32,device=device)
    y=torch.tensor(np.nan_to_num(labels),dtype=torch.float32,device=device);mask=torch.tensor(np.isfinite(labels),dtype=torch.float32,device=device)
    history=[];rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        net.train();losses=[]
        for start in range(0,len(indices),8):
            if start==0:order=rng.permutation(indices)
            batch=order[start:start+8];x=images[batch].to(device)
            gamma=torch.empty((len(batch),1,1,1),device=device).uniform_(.85,1.15)
            gain=torch.empty((len(batch),1,1,1),device=device).uniform_(.93,1.07)
            x=(x.pow(gamma)*gain+torch.randn_like(x)*.004).clamp(0,1)
            logits=net(normalize(x));loss=nn.functional.binary_cross_entropy_with_logits(logits,y[batch],pos_weight=pw,reduction='none')
            loss=(loss*mask[batch]).sum()/mask[batch].sum()
            pp=torch.sigmoid(logits)
            consistency=0.
            for q,ids in [(0,[2,3,4]),(1,[5,6])]:
                consistency+=(((pp[:,q]-pp[:,ids].max(1).values)**2)*mask[batch,q]).sum()/mask[batch,q].sum().clamp_min(1)
            loss=loss+.05*consistency
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    net.eval();return net,history
