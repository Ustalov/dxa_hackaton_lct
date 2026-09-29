import copy
import numpy as np
import torch
from torchvision.models import resnet18
from .modeling import balanced_group_folds,choose_thresholds

class DXANet(torch.nn.Module):
    def __init__(self,hybrid=False):
        super().__init__();self.encoder=resnet18(weights=None);self.encoder.fc=torch.nn.Identity()
        self.hybrid=hybrid
        self.head=torch.nn.Linear(512+(77 if hybrid else 0),7)
    def forward(self,x,geo):
        z=self.encoder(x)
        return self.head(torch.cat([z,geo],1) if self.hybrid else z)

def infer(model,x,geo,indices):
    model.eval();results=[]
    with torch.inference_mode():
        for chunk in np.array_split(indices,max(1,int(np.ceil(len(indices)/16)))):
            if len(chunk):results.append(model(x[chunk],geo[chunk]).sigmoid().numpy())
    return np.concatenate(results)

def fit_neural(x,geo,labels,groups,indices,weights,hybrid=False,seed=44,epochs=14):
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4)
    folds=balanced_group_folds(labels[indices],groups[indices],4,seed)
    tr=indices[folds!=0];cal=indices[folds==0]
    # Calibration studies are never used in gradient updates.
    mean=geo[tr].mean(0);scale=geo[tr].std(0)+1e-4
    gg=torch.tensor(np.clip((geo-mean)/scale,-5,5),dtype=torch.float32)
    model=DXANet(hybrid)
    state=torch.load(weights,map_location='cpu',weights_only=True);state.pop('fc.weight');state.pop('fc.bias')
    model.encoder.load_state_dict(state,strict=False)
    for name,param in model.encoder.named_parameters():param.requires_grad=name.startswith('layer4')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.02)
    yy=torch.tensor(np.nan_to_num(labels),dtype=torch.float32);mask=torch.tensor(np.isfinite(labels),dtype=torch.float32)
    pos=np.nansum(labels[tr],0);neg=np.isfinite(labels[tr]).sum(0)-pos
    pw=torch.tensor(np.clip(neg/np.maximum(pos,1),1,6),dtype=torch.float32)
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=pw,reduction='none')
    best=None;bestloss=np.inf;bestepoch=0;history=[]
    for epoch in range(epochs):
        model.train()
        # Frozen batch statistics avoid destructive updates from tiny medical batches.
        for module in model.modules():
            if isinstance(module,torch.nn.BatchNorm2d):module.eval()
        losses=[]
        for chunk in np.array_split(np.random.permutation(tr),max(1,int(np.ceil(len(tr)/12)))):
            xx=x[chunk].clone()
            # Photometric only: geometric augmentation could change the quality label.
            contrast=torch.rand(len(chunk),1,1,1)*.12+.94
            xx=xx*contrast+torch.randn(len(chunk),1,1,1)*.025
            opt.zero_grad();logits=model(xx,gg[chunk]);loss=(lossfn(logits,yy[chunk])*mask[chunk]).sum()/mask[chunk].sum()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            cp=infer(model,x,gg,cal);ct=torch.tensor(np.clip(cp,1e-6,1-1e-6))
            closs=float((torch.nn.functional.binary_cross_entropy(ct,yy[cal],reduction='none')*mask[cal]).sum()/mask[cal].sum())
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'calibration_loss':closs})
        if closs<bestloss:bestloss=closs;best=copy.deepcopy(model.state_dict());bestepoch=epoch+1
        print('neural',hybrid,'seed',seed,'epoch',epoch+1,'cal_loss',round(closs,4),flush=True)
    model.load_state_dict(best);cp=infer(model,x,gg,cal);thresholds=choose_thresholds(labels[cal],cp)
    return model,gg,{'mean':mean,'scale':scale,'thresholds':thresholds,'hybrid':hybrid,'best_epoch':bestepoch,
                    'train_indices':tr.tolist(),'calibration_indices':cal.tolist(),'history':history}
