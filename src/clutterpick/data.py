"""Scene-wise BOP indexing and visible-mask instance supervision."""
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read_json, resolve, write_json, object_ids


def source_group(path, config=None):
    if config and config['data'].get('name') == 'ycbv':
        split = path.relative_to(resolve(config['data']['root'])).parts[0]
        mapping = {'test':'ycbv_test','test_bop19':'ycbv_test','train_real':'ycbv_train_real',
                   'train_pbr':'pbr_ycbv','train_synt':'ycbv_train_synt'}
        if split not in mapping:
            raise ValueError(f'Unsupported YCB-V split: {split}')
        return mapping[split]
    raise ValueError(f"Expected the YCB-V configuration: {path}")


def image_files(folder):
    return {str(int(p.stem)): p for p in folder.iterdir()
            if p.is_file() and p.stem.isdigit() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}}


def build_index(config):
    root = resolve(config["data"]["root"])
    frames, issues = [], []
    for gt_path in sorted(root.rglob("scene_gt.json")):
        scene_dir = gt_path.parent
        group = source_group(gt_path, config)
        scene = scene_dir.name
        gt = read_json(gt_path)
        cameras = read_json(scene_dir / "scene_camera.json")
        infos = read_json(scene_dir / "scene_gt_info.json")
        rgbs, depths = image_files(scene_dir/"rgb"), image_files(scene_dir/"depth")
        for frame_id, objects in sorted(gt.items(), key=lambda x: int(x[0])):
            if frame_id not in rgbs or frame_id not in depths or frame_id not in cameras or frame_id not in infos:
                issues.append({"group": group, "scene": scene, "frame": frame_id, "reason": "missing_frame_assets"})
                continue
            instances = []
            for j, obj in enumerate(objects):
                visible = scene_dir/"mask_visib"/f"{int(frame_id):06d}_{j:06d}.png"
                full = scene_dir/"mask"/f"{int(frame_id):06d}_{j:06d}.png"
                if not visible.exists() or not full.exists():
                    raise ValueError(f"Missing instance mask: {visible}")
                rotation = np.asarray(obj["cam_R_m2c"]).reshape(3,3)
                instances.append({"instance_id": f"{group}/{scene}/{frame_id}/{j}", "instance_index": j,
                                  "obj_id": int(obj["obj_id"]), "mask_visib": str(visible), "mask": str(full),
                                  "R_m2c": obj["cam_R_m2c"], "t_m2c_mm": obj["cam_t_m2c"],
                                  "rotation_orthogonality_error": float(np.linalg.norm(rotation.T@rotation-np.eye(3))),
                                  **infos[frame_id][j]})
            camera = cameras[frame_id]
            frames.append({"id": len(frames), "key": f"{group}/{scene}/{frame_id}", "group": group,
                           "scene_id": scene, "image_id": int(frame_id), "rgb": str(rgbs[frame_id]),
                           "depth": str(depths[frame_id]), "K": camera["cam_K"],
                           "depth_scale_mm": camera.get("depth_scale", 1), "instances": instances})
    if not frames:
        raise ValueError(f"No BOP scenes in {root}; run scripts/prepare_ycbv.py first")
    index = {"dataset":config['data']['name'],"object_ids":object_ids(config),
             "revision": config["data"]["revision"], "root": str(root), "frames": frames, "issues": issues,
             "counts": dict(Counter(f["group"] for f in frames))}
    write_json(resolve(config["data"]["index"]), index)
    validation = set(config["data"]["validation_scenes"])
    pbr_validation = set(config["data"].get("pbr_validation_scenes",[]))
    splits = {"train": [], "val": [], "test": []}
    holdout=set(config['data'].get('pbr_holdout_scenes',[]))
    if holdout:
        if holdout & pbr_validation: raise ValueError('PBR holdout and validation scenes overlap')
        splits['holdout']=[]
    for f in frames:
        group = f["group"]
        if group in {'ycbv_train_real','ycbv_train_synt'}:
            split = "val" if f["scene_id"] in validation else "train"
        elif group == 'ycbv_test':
            split = "test"
        else:
            split = 'holdout' if f['scene_id'] in holdout else ("val" if f["scene_id"] in pbr_validation else "train")
        splits[split].append(f["id"])
    for split in ["train", "val"]:
        classes = {o["obj_id"] for i in splits[split] for o in frames[i]["instances"]}
        if splits[split] and classes != set(object_ids(config)):
            raise ValueError(f"Split {split} lacks object classes: {classes}")
    write_json(resolve(config["data"]["splits"]), {"policy": "official test; scene-level holdout; test never reused for training/validation",
               "pbr_validation_scenes":sorted(pbr_validation),
               **splits})
    write_json(resolve(config["data"].get("index_report","data/index_report.json")), {"frame_counts": {k: len(v) for k,v in splits.items()},
               "multi_object_frames": {k:sum(len(frames[i]["instances"])>1 for i in ids) for k,ids in splits.items()},
               "source_groups": {k:dict(Counter(frames[i]["group"] for i in ids)) for k,ids in splits.items()},
               "class_instances": {k: dict(Counter(o["obj_id"] for i in ids for o in frames[i]["instances"]))
                                   for k,ids in splits.items()},
               "rotation_flags_train": sum(o["rotation_orthogonality_error"]>1e-3 for i in splits["train"] for o in frames[i]["instances"]),
               "issues": issues})
    return index, splits


def load_index(config):
    return read_json(resolve(config["data"]["index"])), read_json(resolve(config["data"]["splits"]))


def select_frames(index, ids, limit=None, seed=42):
    """Sample across scenes; None selects all frames and zero selects none."""
    if limit is not None and limit < 0:
        raise ValueError("Frame limit must be nonnegative or None")
    if limit is None or limit >= len(ids):
        return [index["frames"][i] for i in ids]
    rng = np.random.default_rng(seed)
    groups = {}
    for i in ids:
        f = index["frames"][i]
        groups.setdefault((f["group"], f["scene_id"]), []).append(i)
    queues = [list(rng.permutation(g)) for _,g in sorted(groups.items())]
    chosen = []
    while len(chosen) < limit and any(queues):
        for queue in queues:
            if queue and len(chosen) < limit:
                chosen.append(queue.pop())
    return [index["frames"][i] for i in chosen]


def load_rgbd(frame):
    rgb = np.array(Image.open(frame["rgb"]).convert("RGB"))
    raw = np.array(Image.open(frame["depth"]))
    if raw.ndim != 2 or raw.shape != rgb.shape[:2]:
        raise ValueError("Depth is not a registered single-channel image")
    if raw.dtype != np.uint16:
        raise ValueError(f"Expected uint16 metric depth, got {raw.dtype}")
    return rgb, raw.astype(np.float64)*frame["depth_scale_mm"]/1000, np.array(frame["K"]).reshape(3,3)


def balanced_validation_frames(index,splits,limit,seed):
    """Balance real/PBR frames, filling unused quotas from the other source."""
    if limit is None or limit >= len(splits['val']):
        return select_frames(index,splits['val'],limit,seed)
    if limit < 0:
        raise ValueError("Frame limit must be nonnegative or None")
    real = [i for i in splits['val'] if not index['frames'][i]['group'].startswith('pbr')]
    synthetic = [i for i in splits['val'] if index['frames'][i]['group'].startswith('pbr')]
    if real and synthetic:
        real_limit = min(len(real),limit//2)
        synthetic_limit = min(len(synthetic),limit-real_limit)
        real_limit = min(len(real),limit-synthetic_limit)
        return select_frames(index,real,real_limit,seed)+select_frames(index,synthetic,synthetic_limit,seed)
    return select_frames(index,splits['val'],limit,seed)


class BOPDetectionDataset:
    def __init__(self, frames, training=False, min_visibility=0, augmentation=None):
        self.frames, self.training, self.min_visibility = frames, training, min_visibility
        self.augmentation = augmentation or {}
        labels = sorted({obj['obj_id'] for frame in frames for obj in frame['instances']})
        self.class_frames = {label:[i for i,f in enumerate(frames) if any(o["obj_id"]==label for o in f["instances"])] for label in labels} if self.augmentation.get("copy_paste_probability",0) else {}

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        import torch
        frame = self.frames[item]
        rgb = np.array(Image.open(frame["rgb"]).convert("RGB"))
        h,w = rgb.shape[:2]
        masks, boxes, labels = [], [], []
        for obj in frame["instances"]:
            if self.training and obj["visib_fract"] < self.min_visibility:
                continue
            mask = np.array(Image.open(obj["mask_visib"]))>0
            y,x = np.nonzero(mask)
            if len(x)<2:
                continue
            masks.append(mask)
            boxes.append([x.min(),y.min(),x.max()+1,y.max()+1])
            labels.append(obj["obj_id"])
        if self.training and np.random.random()<self.augmentation.get("copy_paste_probability",0):
            import cv2
            for _ in range(np.random.randint(1,4)):
                label = int(np.random.choice([key for key,indices in self.class_frames.items() if indices]))
                donor = self.frames[int(np.random.choice(self.class_frames[label]))]
                objects = [o for o in donor["instances"] if o["obj_id"]==label and o["visib_fract"]>=.25]
                if not objects:
                    continue
                obj = objects[int(np.random.randint(len(objects)))]
                donor_mask = np.asarray(Image.open(obj["mask_visib"]))>0
                yy,xx = np.nonzero(donor_mask)
                if len(xx)<30:
                    continue
                patch = np.asarray(Image.open(donor["rgb"]).convert("RGB"))[yy.min():yy.max()+1,xx.min():xx.max()+1]
                patch_mask = donor_mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
                factor = np.random.uniform(.7,1.3)
                ph,pw = min(h-1,max(4,int(patch.shape[0]*factor))),min(w-1,max(4,int(patch.shape[1]*factor)))
                patch = cv2.resize(patch,(pw,ph),interpolation=cv2.INTER_LINEAR)
                patch_mask = cv2.resize(patch_mask.astype(np.uint8),(pw,ph),interpolation=cv2.INTER_NEAREST)>0
                if masks and np.random.random()<.75:
                    by,bx = np.nonzero(masks[int(np.random.randint(len(masks)))])
                    if not len(bx):
                        continue
                    left = int(np.clip(bx.mean()+np.random.uniform(-.5,.5)*pw-pw/2,0,w-pw))
                    top = int(np.clip(by.mean()+np.random.uniform(-.5,.5)*ph-ph/2,0,h-ph))
                else:
                    left,top = np.random.randint(w-pw),np.random.randint(h-ph)
                pasted = np.zeros((h,w),dtype=bool)
                pasted[top:top+ph,left:left+pw] = patch_mask
                region = rgb[top:top+ph,left:left+pw]
                region[patch_mask] = patch[patch_mask]
                masks = [mask & ~pasted for mask in masks]+[pasted]
                labels.append(label)
            retained = [(mask,label) for mask,label in zip(masks,labels) if mask.sum()>=10]
            masks,labels,boxes = [],[],[]
            for mask,label in retained:
                yy,xx = np.nonzero(mask)
                masks.append(mask); labels.append(label)
                boxes.append([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
        if self.training and self.augmentation.get("color_augmentation",False):
            from PIL import ImageEnhance
            enhanced = Image.fromarray(rgb)
            for operation in [ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color]:
                enhanced = operation(enhanced).enhance(float(np.random.uniform(.75,1.25)))
            rgb = np.asarray(enhanced)
        image = torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
        target = {"boxes": torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
                  "labels": torch.tensor(labels,dtype=torch.int64),
                  "masks": torch.from_numpy(np.stack(masks).astype(np.uint8)) if masks else torch.zeros((0,h,w),dtype=torch.uint8),
                  "image_id": torch.tensor(frame["id"],dtype=torch.int64),
                  "area": torch.tensor([m.sum() for m in masks],dtype=torch.float32),
                  "iscrowd": torch.zeros(len(labels),dtype=torch.int64)}
        if self.training and torch.rand(()) < 0.5:
            image = image.flip(-1)
            target["masks"] = target["masks"].flip(-1)
            target["boxes"][:, [0,2]] = w-target["boxes"][:, [2,0]]
        return image, target


def collate(batch):
    return tuple(zip(*batch))

