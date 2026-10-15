"""COCO detection/mask AP and independent CAD pose diagnostics."""
import contextlib
import io
import time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .common import write_json


def encode_mask(mask):
    from pycocotools import mask as mask_api
    encoded = mask_api.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def evaluate_detector(detector,frames,output=None,boundary=False):
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images,annotations,detections = [],[],[]
    elapsed = []
    ann_id = 1
    for frame in frames:
        rgb = np.array(Image.open(frame["rgb"]).convert("RGB"))
        h,w = rgb.shape[:2]
        images.append({"id":frame["id"],"width":w,"height":h})
        for obj in frame["instances"]:
            mask = np.array(Image.open(obj["mask_visib"]))>0
            y,x = np.nonzero(mask)
            if not len(x):
                continue
            annotations.append({"id":ann_id,"image_id":frame["id"],"category_id":obj["obj_id"],
                "segmentation":encode_mask(mask),"bbox":[int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1)],
                "area":int(mask.sum()),"iscrowd":0,"visibility":obj["visib_fract"]})
            ann_id += 1
        if torch.cuda.is_available() and str(detector.device).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        if getattr(detector,'requires_depth',False):
            from .data import load_rgbd
            _,depth,k=load_rgbd(frame)
            pred=detector.predict(rgb,score_threshold=.05,depth=depth,k=k)
        else:
            pred = detector.predict(rgb,score_threshold=0.05)
        if torch.cuda.is_available() and str(detector.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-started)
        for box,score,label,mask in zip(pred["boxes"],pred["scores"],pred["labels"],pred["masks"]):
            x1,y1,x2,y2 = box
            detections.append({"image_id":frame["id"],"category_id":int(label),"score":float(score),
                               "bbox":[float(x1),float(y1),float(x2-x1),float(y2-y1)],"segmentation":encode_mask(mask)})
    gt = COCO()
    from .common import object_ids
    categories = object_ids(detector.config) if hasattr(detector,'config') else sorted({a['category_id'] for a in annotations})
    gt.dataset = {"info":{},"images":images,"annotations":annotations,
                  "categories":[{"id":i,"name":f"object_{i}"} for i in categories]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    results = {}
    for kind in ["bbox","segm"]+(['boundary'] if boundary else []):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            if detections:
                inputs = detections if kind=="bbox" else [{k:v for k,v in d.items() if k!="bbox"} for d in detections]
                dt = gt.loadRes(inputs)
            else:
                dt = COCO()
                dt.dataset = {"info":{},"images":images,"annotations":[],"categories":gt.dataset["categories"]}
                dt.createIndex()
            if kind=='boundary':
                from .boundary import BoundaryCOCOeval
                ev=BoundaryCOCOeval(gt,dt,'segm')
            else: ev = COCOeval(gt,dt,kind)
            ev.params.imgIds = [f["id"] for f in frames]
            ev.evaluate(); ev.accumulate(); ev.summarize()
        precision = ev.eval["precision"]
        per_class = {}
        for j,cat in enumerate(ev.params.catIds):
            values = precision[:,:,j,0,-1]
            values = values[values>=0]
            per_class[str(cat)] = float(values.mean()) if len(values) else None
        results[kind] = {"AP_50_95":float(ev.stats[0]),"AP_50":float(ev.stats[1]),
                         "AP_75":float(ev.stats[2]),"AR_100":float(ev.stats[8]),"per_class_AP":per_class}
    # Recall at mask IoU .5 and confidence .5, grouped by GT visibility; misses counted.
    bins = {"0_25":{"gt":0,"matched":0},"25_50":{"gt":0,"matched":0},"50_75":{"gt":0,"matched":0},"75_100":{"gt":0,"matched":0}}
    from pycocotools import mask as mask_api
    matched_ious = []
    for image in images:
        used = set()
        image_gt = [a for a in annotations if a["image_id"]==image["id"]]
        image_dt = sorted([d for d in detections if d["image_id"]==image["id"] and d["score"]>=0.5],key=lambda d:d["score"],reverse=True)
        matched = set()
        for d in image_dt:
            candidates = [(j,a) for j,a in enumerate(image_gt) if j not in used and a["category_id"]==d["category_id"]]
            if not candidates:
                continue
            ious = mask_api.iou([d["segmentation"]],[a["segmentation"] for _,a in candidates],[0]*len(candidates))[0]
            k = int(np.argmax(ious))
            if ious[k]>=0.5:
                j,_ = candidates[k]
                used.add(j); matched.add(j)
                matched_ious.append(float(ious[k]))
        for j,a in enumerate(image_gt):
            v = a["visibility"]
            bucket = "0_25" if v<.25 else "25_50" if v<.5 else "50_75" if v<.75 else "75_100"
            bins[bucket]["gt"] += 1
            bins[bucket]["matched"] += int(j in matched)
    for item in bins.values():
        item["recall"] = item["matched"]/item["gt"] if item["gt"] else None
    matched_count = sum(v["matched"] for v in bins.values())
    confidence_predictions = sum(d["score"]>=0.5 for d in detections)
    precision50 = matched_count/confidence_predictions if confidence_predictions else 0.
    recall50 = matched_count/len(annotations) if annotations else 0.
    result = {"frame_count":len(frames),"gt_instances":len(annotations),"prediction_count":len(detections),
              "frame_keys":[f["key"] for f in frames],"metrics":results,"visibility_mask_recall_iou50_score50":bins,
              "instance_mask_iou50_score50":{"precision":precision50,"recall":recall50,
                  "F1":2*precision50*recall50/(precision50+recall50) if precision50+recall50 else 0.,
                  "mean_matched_mask_IoU":float(np.mean(matched_ious)) if matched_ious else None},
              "latency_seconds":{"mean_including_first_frame":float(np.mean(elapsed)),
                                  "median_excluding_first_frame":float(np.median(elapsed[1:] or elapsed)),
                                  "p95":float(np.percentile(elapsed,95))},
              "checkpoint":detector.checkpoint,"ground_truth_masks_used_for_predictions":False}
    if output:
        write_json(output,result)
    return result


def mask_iou(a,b):
    return float(np.logical_and(a,b).sum()/max(np.logical_or(a,b).sum(),1))


def pose_errors(mesh,pred,reference):
    from .geometry import transform_points
    from .pose import proper_rotation
    vertices = np.asarray(mesh.vertices)
    total_vertices = len(vertices)
    if len(vertices)>10000:
        vertices = vertices[np.linspace(0,len(vertices)-1,10000,dtype=int)]
    p,g = transform_points(vertices,pred),transform_points(vertices,reference)
    error = np.linalg.norm(reference[:3,:3].T@reference[:3,:3]-np.eye(3))
    return {"ADD_m":float(np.linalg.norm(p-g,axis=1).mean()),
            "ADD_S_m":float(cKDTree(g).query(p)[0].mean()),
            "translation_error_mm":float(np.linalg.norm(pred[:3,3]-reference[:3,3])*1000),
            "rotation_error_degrees":float(np.degrees(Rotation.from_matrix(pred[:3,:3]@proper_rotation(reference[:3,:3]).T).magnitude())),
            "reference_rotation_orthogonality_error":float(error),
            "metric_model_vertices_used":len(vertices),"metric_model_vertices_total":total_vertices,
            "metric_model_subsampling": "deterministic_vertex_linspace_max_10000",
            "rotation_metric_uses_projected_reference":True,
            "symmetry_metadata_available":False}
