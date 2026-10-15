import csv
import random
import time
from pathlib import Path
import numpy as np
from .common import resolve,set_seed,write_json
from .data import BOPDetectionDataset,collate,load_index,select_frames
from .detection import choose_device,create_maskrcnn,MaskRCNNDetector
from .evaluation import evaluate_detector


def prefetched_batches(loader):
    """Overlap CPU decoding with GPU work without duplicating Windows processes."""
    import queue
    import threading
    pending = queue.Queue(maxsize=2)
    stopped = threading.Event()
    def send(item):
        while not stopped.is_set():
            try:
                pending.put(item,timeout=.1)
                return
            except queue.Full:
                pass
    def produce():
        try:
            for batch in loader:
                if stopped.is_set():
                    break
                send((True,batch))
            send((True,None))
        except BaseException as error:
            send((False,error))
    worker = threading.Thread(target=produce,daemon=True)
    worker.start()
    try:
        while True:
            ok,batch = pending.get()
            if not ok:
                raise batch
            if batch is None:
                break
            yield batch
    finally:
        stopped.set()
        worker.join(timeout=5)


def train_maskrcnn(config,epochs=None,limit=None,val_limit=100,resume=None,device="auto"):
    import torch
    from torch.utils.data import DataLoader
    set_seed(config["seed"])
    torch.set_num_threads(4)
    device = choose_device(device)
    index,splits = load_index(config)
    frames = select_frames(index,splits["train"],limit,config["seed"])
    if not frames or not splits['val']:
        raise ValueError('Training requires separate nonempty train and validation splits; test cannot be substituted')
    if config["data"].get("include_pbr_in_training",False):
        from .data import balanced_validation_frames
        val = balanced_validation_frames(index,splits,val_limit,config["seed"])
    else:
        val = select_frames(index,splits["val"],val_limit,config["seed"])
    cfg = config["training"]
    directory = resolve(cfg["output"])
    directory.mkdir(parents=True,exist_ok=True)
    initial_checkpoint = cfg.get("initial_checkpoint") if resume is None else None
    model = create_maskrcnn(config,pretrained=resume is None and not initial_checkpoint)
    if initial_checkpoint:
        model.load_state_dict(torch.load(resolve(initial_checkpoint),map_location="cpu",weights_only=False)["model"])
    if cfg["freeze_backbone"]:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=cfg["learning_rate"],
                                momentum=cfg["momentum"],weight_decay=cfg["weight_decay"])
    count_epochs = epochs or cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(1,count_epochs//2),gamma=0.3)
    amp = cfg["amp"] and device.type=="cuda"
    scaler = torch.amp.GradScaler("cuda",enabled=amp)
    start,best = 0,-1.
    history = []
    if resume:
        payload = torch.load(resume,map_location="cpu",weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        start,best = payload["epoch"]+1,payload.get("best_val_AP",-1.)
        history = payload.get("history",[])
        if "rng_state" in payload:
            random.setstate(payload["rng_state"]["python"])
            np.random.set_state(payload["rng_state"]["numpy"])
            torch.set_rng_state(payload["rng_state"]["torch"])
            if device.type=="cuda" and payload["rng_state"].get("cuda"):
                torch.cuda.set_rng_state_all(payload["rng_state"]["cuda"])
    dataset = BOPDetectionDataset(frames,True,config["data"]["min_visibility_train"],cfg)
    sampler = None
    if cfg.get("balanced_sampling",False):
        from collections import Counter
        counts = Counter(o["obj_id"] for frame in frames for o in frame["instances"])
        weights = [np.mean([1/counts[o["obj_id"]] for o in frame["instances"]]) for frame in frames]
        sampler = torch.utils.data.WeightedRandomSampler(weights,cfg.get("samples_per_epoch",len(frames)),replacement=True)
    loader = DataLoader(dataset,
                        batch_size=cfg["batch_size"],shuffle=sampler is None,sampler=sampler,num_workers=cfg["num_workers"],
                        collate_fn=collate,pin_memory=device.type=="cuda")
    # Preserve the configured LR schedule while honoring an explicit early stop.
    stop_epoch = min(count_epochs,cfg.get('stop_after_epoch',count_epochs))
    for epoch in range(start,stop_epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        epoch_started = time.perf_counter()
        batches = prefetched_batches(loader) if cfg.get('threaded_prefetch',False) else loader
        for step,(images,targets) in enumerate(batches):
            images = [image.to(device) for image in images]
            targets = [{k:v.to(device) for k,v in target.items()} for target in targets]
            accumulation = cfg["accumulation_steps"]
            block_start = (step//accumulation)*accumulation
            block_size = min(accumulation,len(loader)-block_start)
            with torch.autocast(device_type=device.type,enabled=amp):
                losses = model(images,targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")
            scaler.scale(loss/block_size).backward()
            if (step+1)%accumulation==0 or step+1==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total += loss.item()
            if step%25==0:
                print(f"epoch={epoch+1}/{count_epochs} step={step+1}/{len(loader)} loss={loss.item():.4f} elapsed={time.perf_counter()-epoch_started:.1f}s",flush=True)
                import json
                with (directory/"steps.jsonl").open("a",encoding="utf-8") as log:
                    log.write(json.dumps({"epoch":epoch+1,"step":step+1,"loss":loss.item(),
                              "components":{k:v.item() for k,v in losses.items()},"lr":optimizer.param_groups[0]["lr"]})+"\n")
        scheduler.step()
        record = {"epoch":epoch+1,"mean_training_loss":total/len(loader),"training_seconds":time.perf_counter()-epoch_started,
                  "batch_size":cfg["batch_size"],"effective_batch_size":cfg["batch_size"]*cfg["accumulation_steps"]}
        payload = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                   "scaler":scaler.state_dict(),"epoch":epoch,"config":config,"best_val_AP":best,"history":history,
                   "training_frame_keys":[f["key"] for f in frames],"validation_frame_keys":[f["key"] for f in val],
                   "dataset_revision":index["revision"],"device":str(device),"torch_version":torch.__version__,
                   "num_training_frames":len(frames),"samples_per_epoch":len(loader.sampler),
                   "initial_checkpoint":initial_checkpoint,"is_limited_training_run":bool(limit and limit<len(splits["train"]))}
        payload["rng_state"] = {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
                                "cuda":torch.cuda.get_rng_state_all() if device.type=="cuda" else None}
        torch.save(payload,directory/"last.pt")
        # Reuse the model during validation rather than allocating a second network.
        detector = object.__new__(MaskRCNNDetector)
        detector.model,detector.config,detector.device,detector.checkpoint = model,config,device,str(directory/"last.pt")
        model.eval()
        sources = {"real":[f for f in val if not f["group"].startswith("pbr")],"pbr":[f for f in val if f["group"].startswith("pbr")]}
        if config["data"].get("include_pbr_in_training",False) and all(sources.values()):
            scores = {name:evaluate_detector(detector,frames,directory/f"validation_{name}_epoch_{epoch+1:02d}.json") for name,frames in sources.items()}
            score = .7*scores["real"]["metrics"]["segm"]["AP_50_95"]+.3*scores["pbr"]["metrics"]["segm"]["AP_50_95"]
            box_score = .7*scores["real"]["metrics"]["bbox"]["AP_50_95"]+.3*scores["pbr"]["metrics"]["bbox"]["AP_50_95"]
            record["val_mask_AP_by_source"] = {name:e["metrics"]["segm"]["AP_50_95"] for name,e in scores.items()}
        else:
            evaluation = evaluate_detector(detector,val,directory/f"validation_epoch_{epoch+1:02d}.json")
            score = evaluation["metrics"]["segm"]["AP_50_95"]
            box_score = evaluation["metrics"]["bbox"]["AP_50_95"]
        record["val_mask_AP"] = score
        record["val_box_AP"] = box_score
        if device.type=="cuda":
            record["peak_allocated_vram_GB"] = torch.cuda.max_memory_allocated()/1e9
        history.append(record)
        payload["history"] = history
        if score>best:
            best = score
            payload["best_val_AP"] = best
            torch.save(payload,directory/"best.pt")
        payload["best_val_AP"] = best
        torch.save(payload,directory/"last.pt")
        write_json(directory/"history.json",history)
        with (directory/"history.csv").open("w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=list(history[-1]))
            writer.writeheader(); writer.writerows(history)
        print(f"epoch={epoch+1} val_mask_AP={score:.4f} val_box_AP={record['val_box_AP']:.4f}",flush=True)
    return directory/"best.pt"
