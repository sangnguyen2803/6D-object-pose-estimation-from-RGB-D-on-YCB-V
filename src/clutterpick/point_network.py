"""Small class-conditioned local point-attention foreground/background network."""
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from .common import resolve
from .detection import choose_device

def point_features(points,colors,membership):
    inside=np.asarray(membership)>0
    anchor=points[inside] if inside.any() else points
    center=np.median(anchor,axis=0)
    size=max(float(np.linalg.norm(np.quantile(anchor,.95,axis=0)-np.quantile(anchor,.05,axis=0))),.05)
    xyz=np.clip((points-center)/size,-5,5)
    distance,_=cKDTree(points).query(points,k=min(8,len(points)))
    density=distance.mean(1)/size if distance.ndim==2 else distance/size
    return np.column_stack([xyz,colors/255,np.asarray(membership),points[:,2]/3,density]).astype(np.float32)

class PointForegroundNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(9,64),nn.ReLU(),nn.Linear(64,64))
        self.q=nn.Linear(64,32); self.k=nn.Linear(64,32); self.v=nn.Linear(64,64)
        self.position=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,64))
        self.attention=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        self.class_embedding=nn.Embedding(22,16)
        self.classifier=nn.Sequential(nn.Linear(144,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,features,objects):
        xyz=features[:,:,:3].float()
        with torch.no_grad(): neighbors=torch.cdist(xyz,xyz).topk(min(16,xyz.shape[1]),largest=False).indices
        batch=torch.arange(features.shape[0],device=features.device)[:,None,None]
        h=self.input(features)
        neighbor=h[batch,neighbors]
        relative=xyz[:,:,None,:]-xyz[batch,neighbors]
        position=self.position(relative.to(h.dtype))
        differences=self.q(h)[:,:,None,:]-self.k(neighbor)
        weights=self.attention(torch.cat([differences,position[:,:,:,:32]],-1)).softmax(2)
        local=h+(weights*(self.v(neighbor)+position)).sum(2)
        global_features=local.max(1,keepdim=True).values.expand_as(local)
        embedded=self.class_embedding(objects)[:,None,:].expand(-1,h.shape[1],-1)
        return self.classifier(torch.cat([local,global_features,embedded],-1)).squeeze(-1)

class PointCleaner:
    def __init__(self,checkpoint,device='auto'):
        self.device=choose_device(device)
        payload=torch.load(resolve(checkpoint),map_location='cpu',weights_only=False)
        self.model=PointForegroundNetwork().to(self.device)
        self.model.load_state_dict(payload['model'],strict=True); self.model.eval()
        self.threshold=payload['threshold']; self.checkpoint=str(resolve(checkpoint))
        self.sample_points=payload.get('sample_points',512)
    def filter(self,points,colors,obj_id,membership=None,min_points=40):
        if len(points)<min_points:
            return np.ones(len(points),dtype=bool),{'used':False,'fallback':'insufficient_points'}
        features=point_features(points,colors,np.ones(len(points)) if membership is None else membership)
        rng=np.random.default_rng(42+obj_id)
        chosen=rng.choice(len(points),min(self.sample_points,len(points)),replace=False)
        with torch.inference_mode():
            tensor=torch.from_numpy(features[chosen])[None].to(self.device)
            probability=self.model(tensor,torch.tensor([obj_id],device=self.device)).sigmoid()[0].cpu().numpy()
        distances,ids=cKDTree(features[chosen,:3]).query(features[:,:3],k=min(3,len(chosen)))
        if ids.ndim==1: scores=probability[ids]
        else:
            weights=1/np.maximum(distances,1e-5); weights/=weights.sum(1,keepdims=True)
            scores=(probability[ids]*weights).sum(1)
        keep=scores>=self.threshold
        fallback=keep.sum()<min_points or keep.mean()<.25
        diagnostic={'used':not fallback,'checkpoint':self.checkpoint,'threshold':self.threshold,
                    'input_points':len(points),'predicted_foreground':int(keep.sum()),
                    'kept_fraction':float(keep.mean()),'fallback':'too_few_foreground' if fallback else None,
                    'uses_gt_labels':False}
        return np.ones(len(points),dtype=bool) if fallback else keep,diagnostic

