from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import yaml
from .common import resolve,object_ids
from .data import load_index,select_frames


def export_yolo(config,limit=None):
    import cv2
    from ultralytics.data.converter import merge_multi_segment
    index,splits = load_index(config)
    if not splits['train'] or not splits['val']:
        raise ValueError('YOLO export requires independent nonempty train/validation splits')
    root = resolve("data/ycbv/yolo" if not limit else f"data/ycbv/yolo_pilot_{limit}")
    for split in ["train","val"]:
        images,labels = root/"images"/split,root/"labels"/split
        images.mkdir(parents=True,exist_ok=True); labels.mkdir(parents=True,exist_ok=True)
        frames = select_frames(index,splits[split],limit,config["seed"])
        for frame in frames:
            name = f"{frame['group']}_{frame['scene_id']}_{frame['image_id']:06d}"
            destination = images/(name+Path(frame["rgb"]).suffix)
            if not destination.exists():
                shutil.copy2(frame["rgb"],destination)
            lines = []
            for obj in frame["instances"]:
                if split=="train" and obj["visib_fract"]<config["data"]["min_visibility_train"]:
                    continue
                mask = np.array(Image.open(obj["mask_visib"]),dtype=np.uint8)
                h,w = mask.shape
                contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                segments = [cv2.approxPolyDP(c,1.,True).reshape(-1,2) for c in contours if len(c)>=3]
                segments = [s for s in segments if len(s)>=3]
                if not segments:
                    continue
                polygon = np.concatenate(merge_multi_segment(segments),axis=0) if len(segments)>1 else segments[0]
                polygon = polygon/np.array([w,h])
                lines.append(str(obj["obj_id"]-1)+" "+" ".join(f"{x:.6f}" for x in polygon.ravel()))
            (labels/(name+".txt")).write_text("\n".join(lines),encoding="utf-8")
    definition = {"path":str(root),"train":"images/train","val":"images/val",
                  "names":{i-1:f"object_{i}" for i in object_ids(config)}}
    (root/"dataset.yaml").write_text(yaml.safe_dump(definition),encoding="utf-8")
    return root/"dataset.yaml"


def train_yolo(config,epochs=8,limit=None,device="auto",resume=None):
    from ultralytics import YOLO,settings
    settings.update({"sync":False,"clearml":False,"comet":False,"dvc":False,"mlflow":False,"raytune":False,"wandb":False})
    from .detection import choose_device
    data = export_yolo(config,limit)
    model = YOLO(str(resolve(resume)) if resume else "yolo11n-seg.pt")
    training_device = str(choose_device(device))
    results = model.train(data=str(data),epochs=epochs,resume=bool(resume),imgsz=config["detector"]["max_size"],batch=8,
                         device=training_device,workers=0,seed=config["seed"],deterministic=True,
                         project=str(resolve("artifacts/ycbv/training")),name="yolo11n_seg",exist_ok=True,
                         plots=True,cache=False,amp=config["training"]["amp"])
    return Path(results.save_dir)/"weights"/"best.pt"
