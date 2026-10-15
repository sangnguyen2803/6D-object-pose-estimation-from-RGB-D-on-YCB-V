"""Train equal-budget RGB, early/late/gated RGB-D and query-head ablations."""
import argparse,copy,gc,time,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from clutterpick.common import load_config,resolve,write_json,set_seed
from clutterpick.data import load_index,select_frames,collate
from clutterpick.fusion import create_fusion_model,RGBDDataset,FusionDetector
from clutterpick.evaluation import evaluate_detector

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/research.yaml')
    parser.add_argument('--variants',nargs='+')
    parser.add_argument('--epochs',type=int)
    args=parser.parse_args()
    cfg=load_config(args.config); index,splits=load_index(cfg)
    research=cfg['research']
    train=select_frames(index,splits['train'],research['train_limit'],cfg['seed'])
    val=select_frames(index,splits['val'],research['val_limit'],cfg['seed'])
    assert train and val and not (set(splits['train'])&set(splits['val']))
    assert all(frame['group']=='pbr_ycbv' for frame in train+val)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(4)
    directory=resolve('artifacts/ycbv/research/fusion')
    completed={}
    for variant in args.variants or research['variants']:
        target=directory/variant; target.mkdir(parents=True,exist_ok=True)
        if (target/'history.json').exists():
            history=__import__('json').loads((target/'history.json').read_text())
            if len(history)>= (args.epochs or research['epochs']):
                completed[variant]=history; print('Already complete',variant,flush=True); continue
        set_seed(cfg['seed'])
        model=create_fusion_model(cfg,variant).to(device)
        optimizer=torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                     lr=research['learning_rate'],weight_decay=.0001)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        dataset=RGBDDataset(train,variant,training=True)
        set_seed(cfg['seed']) # Align minibatch order and flips independently of architecture initialization.
        loader=torch.utils.data.DataLoader(dataset,batch_size=research['batch_size'],shuffle=True,
                                          num_workers=0,collate_fn=collate)
        best=-1.; history=[]
        for epoch in range(args.epochs or research['epochs']):
            model.train(); total=0.; begun=time.perf_counter()
            for step,(images,targets) in enumerate(loader):
                images=[image.to(device) for image in images]
                targets=[{key:value.to(device) for key,value in item.items()} for item in targets]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type,enabled=device.type=='cuda'):
                    losses=model(images,targets); loss=sum(losses.values())
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss: {losses}')
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5)
                scaler.step(optimizer); scaler.update(); total+=float(loss.detach())
                if (step+1)%50==0: print(variant,'epoch',epoch+1,'step',step+1,'/',len(loader),'loss',round(total/(step+1),4),flush=True)
            payload={'model':model.cpu().state_dict(),'config':cfg,'variant':variant,'epoch':epoch+1,
                     'train_keys':[frame['key'] for frame in train],'val_keys':[frame['key'] for frame in val],
                     'trained_on_test':False,'pretraining_source':'CosyPose SYNT+REAL; published pretraining may include these PBR validation scenes'}
            torch.save(payload,target/'last.pt'); model.to(device)
            # Reuse trained model directly: avoid a second GPU copy during evaluation.
            detector=object.__new__(FusionDetector)
            detector.model=model.eval(); detector.device=device; detector.config=cfg
            detector.variant=variant; detector.checkpoint=str(target/'last.pt')
            evaluation=evaluate_detector(detector,val,target/f'val_epoch_{epoch+1}.json',boundary=True)
            score=evaluation['metrics']['segm']['AP_50_95']
            record={'epoch':epoch+1,'train_loss':total/len(loader),'seconds':time.perf_counter()-begun,
                    'val_mask_AP':score,'val_boundary_AP':evaluation['metrics']['boundary']['AP_50_95'],
                    'val_box_AP':evaluation['metrics']['bbox']['AP_50_95']}
            history.append(record)
            if score>best:
                best=score; torch.save(payload,target/'best.pt')
            write_json(target/'history.json',history)
            print(variant,record,flush=True)
        completed[variant]=history
        del detector,model,optimizer,scaler,payload
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
        write_json(resolve('reports/ycbv/research/fusion_training.json'),completed)
if __name__=='__main__': main()

