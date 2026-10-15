"""RGB/depth fusion Mask R-CNN and a lightweight spatial-query mask decoder."""
import copy
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from .detection import create_maskrcnn,choose_device
from .common import resolve
from .data import load_rgbd,BOPDetectionDataset


def depth_channels(depth,k):
    """Metric depth plus camera-facing normals; invalid pixels stay zero."""
    valid=np.isfinite(depth)&(depth>.1)&(depth<3)
    z=np.where(valid,depth,0)
    h,w=z.shape; v,u=np.mgrid[:h,:w]
    xyz=np.stack([(u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z],axis=-1)
    du=np.gradient(xyz,axis=1); dv=np.gradient(xyz,axis=0)
    normals=-np.cross(du,dv)
    length=np.linalg.norm(normals,axis=-1)
    reliable=valid&(length>1e-10)
    normals=np.divide(normals,np.maximum(length[...,None],1e-10))
    normals[~reliable]=0
    return (z/3).astype(np.float32),normals.astype(np.float32)

def make_input(rgb,depth,k,variant):
    image=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
    if variant.startswith('rgb'): return image
    scalar,normals=depth_channels(depth,k)
    # Keep identical depth information across early/late/mid-level ablations.
    extra=torch.from_numpy(scalar[None] if variant=='early' else np.repeat(scalar[None],3,axis=0))
    return torch.cat([image,extra])

class RGBDFPN(nn.Module):
    def __init__(self,original,mode):
        super().__init__()
        self.rgb=original; self.depth=copy.deepcopy(original); self.mode=mode
        self.out_channels=original.out_channels
        channels=[256]*5 if mode=='late' else [256,512,1024,2048]
        self.adapters=nn.ModuleList()
        for width in channels:
            if mode=='late':
                layer=nn.Conv2d(width*2,width,1)
                nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
                with torch.no_grad(): layer.weight[:,:width,0,0]=torch.eye(width)
            else:
                layer=nn.Sequential(nn.Conv2d(width*2,64,1),nn.SiLU(),nn.Conv2d(64,width,1))
                nn.init.zeros_(layer[-1].weight); nn.init.constant_(layer[-1].bias,-4.)
            self.adapters.append(layer)
    def forward(self,x):
        valid=(x[:,3:].abs().sum(1,keepdim=True)>0).float()
        if self.mode=='late':
            rgb=self.rgb(x[:,:3]); depth=self.depth(x[:,3:])
        else:
            rgb=self.rgb.body(x[:,:3]); depth=self.depth.body(x[:,3:])
        fused=OrderedDict()
        for index,(key,r) in enumerate(rgb.items()):
            d=depth[key]
            available=nn.functional.interpolate(valid,size=r.shape[-2:],mode='nearest')
            if self.mode=='late':
                candidate=self.adapters[index](torch.cat([r,d],1))
                fused[key]=r+available*(candidate-r)
            else:
                gate=torch.sigmoid(self.adapters[index](torch.cat([r,d],1)))
                fused[key]=r+available*gate*d
        return fused if self.mode=='late' else self.rgb.fpn(fused)

class QueryMaskPredictor(nn.Module):
    """Pretrained CNN logits plus residual 28x28 spatial-query decoder -> 56x56."""
    def __init__(self,original,classes):
        super().__init__()
        self.original=original
        self.project=nn.Conv2d(256,64,1)
        self.position=nn.Parameter(torch.zeros(1,784,64))
        layer=nn.TransformerDecoderLayer(64,4,128,dropout=0,batch_first=True)
        self.decoder=nn.TransformerDecoder(layer,1)
        self.output=nn.Conv2d(64,classes,1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    def forward(self,x):
        base=self.original(x)
        if x.shape[0]==0: return base
        features=self.project(x); tokens=features.flatten(2).transpose(1,2)
        decoded=self.decoder(tokens+self.position,tokens)
        decoded=decoded.transpose(1,2).reshape(x.shape[0],64,28,28)
        residual=nn.functional.interpolate(self.output(decoded),size=base.shape[-2:],mode='bilinear',align_corners=False)
        return base+residual

def create_fusion_model(config,variant,initialize=True):
    model=create_maskrcnn(config)
    if initialize:
        payload=torch.load(resolve(config['detector']['checkpoint']),map_location='cpu',weights_only=False)
        model.load_state_dict(payload['state_dict'],strict=True)
    # Equal-budget adaptation: freeze RGB trunk; adapt depth's final stage.
    for parameter in model.backbone.parameters(): parameter.requires_grad_(False)
    modality=variant.split('_')[0]
    if modality=='early':
        old=model.backbone.body.conv1
        conv=nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        with torch.no_grad(): conv.weight[:,:3]=old.weight; conv.weight[:,3:]=0
        model.backbone.body.conv1=conv
        model.transform.image_mean += [0.]
        model.transform.image_std += [1.]
    elif modality in ['late','gated']:
        model.backbone=RGBDFPN(model.backbone,modality)
        if config.get('research',{}).get('train_depth_layer4',True):
            for parameter in model.backbone.depth.body.layer4.parameters(): parameter.requires_grad_(True)
        model.transform.image_mean += [0.,0.,0.]
        model.transform.image_std += [1.,1.,1.]
    elif modality!='rgb': raise ValueError(variant)
    if variant.endswith('_query'):
        model.roi_heads.mask_roi_pool=MultiScaleRoIAlign(['0','1','2','3'],28,2)
        model.roi_heads.mask_predictor=QueryMaskPredictor(model.roi_heads.mask_predictor,22)
    if 'research' in config:
        model.roi_heads.fg_bg_sampler.batch_size_per_image=config['research'].get('roi_batch_size',256)
    return model

class RGBDDataset:
    def __init__(self,frames,variant,training=False):
        self.frames=frames; self.variant=variant; self.training=training
        self.base=BOPDetectionDataset(frames,training=False)
    def __len__(self): return len(self.frames)
    def __getitem__(self,index):
        rgb,depth,k=load_rgbd(self.frames[index])
        _,target=self.base[index]
        if self.training:
            # Identical random draws across variants; RGB-only ignores corrupted depth.
            missing=float(torch.rand(()))<.2
            drop=torch.rand(depth.shape).numpy()<.1
            noise=torch.randn(depth.shape).numpy()*.002
            valid=depth>0
            depth[valid]=np.maximum(depth[valid]+noise[valid],0)
            depth[drop]=0
            if missing: depth[:]=0
        image=make_input(rgb,depth,k,self.variant)
        if self.training and torch.rand(())<.5:
            image=image.flip(-1); target['masks']=target['masks'].flip(-1)
            target['boxes'][:,[0,2]]=image.shape[-1]-target['boxes'][:,[2,0]]
        return image,target

class FusionDetector:
    requires_depth=True
    def __init__(self,checkpoint,config,device='auto'):
        self.device=choose_device(device); self.checkpoint=str(resolve(checkpoint))
        payload=torch.load(resolve(checkpoint),map_location='cpu',weights_only=False)
        self.config=payload['config']; self.variant=payload['variant']
        if self.config['data']['object_ids']!=config['data']['object_ids']: raise ValueError('Class mapping mismatch')
        self.model=create_fusion_model(self.config,self.variant,initialize=False)
        self.model.load_state_dict(payload['model'],strict=True); self.model.to(self.device).eval()
    def predict(self,rgb,score_threshold=None,depth=None,k=None):
        if depth is None or k is None: raise ValueError('Fusion detector requires registered depth and K')
        image=make_input(rgb,depth,k,self.variant).to(self.device)
        with torch.inference_mode(): result=self.model([image])[0]
        threshold=self.config['detector']['score_threshold'] if score_threshold is None else score_threshold
        keep=result['scores']>=threshold
        output={key:value[keep].cpu().numpy() for key,value in result.items()}
        output['masks']=output['masks'][:,0]>=self.config['detector']['mask_threshold']
        return output

