"""YCB-V instance segmentation, visible point clouds and known-object 6D pose.

Reference masks/poses appear only in explicit oracle diagnostics or evaluation
after predictions. Grasp planning is outside the current project scope.
"""
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from .common import resolve,write_json,read_json
from .data import load_rgbd
from .geometry import backproject,instance_cloud,voxel_downsample,write_ply,transform_points
from .pose import GeometricPoseEstimator
from .evaluation import mask_iou,pose_errors
from .visualization import overlay,plot_scene,instance_color


def reference_detections(frame):
    masks,boxes,labels = [],[],[]
    for obj in frame.get('instances',[]):
        mask = np.asarray(Image.open(obj['mask_visib']))>0
        y,x = np.nonzero(mask)
        if len(x):
            masks.append(mask); boxes.append([x.min(),y.min(),x.max()+1,y.max()+1]); labels.append(obj['obj_id'])
    h,w = np.asarray(Image.open(frame['rgb'])).shape[:2]
    return {'boxes':np.asarray(boxes).reshape(-1,4),'labels':np.asarray(labels),'scores':np.ones(len(masks)),
            'masks':np.stack(masks) if masks else np.zeros((0,h,w),dtype=bool)}


def match_references(items,frame,minimum_iou=.5):
    references = [(obj,np.asarray(Image.open(obj['mask_visib']))>0) for obj in frame.get('instances',[])]
    if not items or not references:
        return {}
    costs = np.ones((len(items),len(references)))
    for i,item in enumerate(items):
        for j,(obj,mask) in enumerate(references):
            if item['obj_id']==obj['obj_id']:
                costs[i,j] = 1-mask_iou(item['mask'],mask)
    first,second = linear_sum_assignment(costs)
    return {int(i):(references[j][0],1-costs[i,j]) for i,j in zip(first,second) if costs[i,j]<=1-minimum_iou}


def get_estimator(config,backend,runtime,device):
    if backend in {'cosypose','cosypose_rgbd'}:
        if 'cosypose' not in runtime:
            from .cosypose_backend import CosyPoseEstimator
            runtime['cosypose'] = CosyPoseEstimator(config,device)
        return runtime['cosypose']
    if backend=='geometric':
        if 'geometric' not in runtime:
            runtime['geometric'] = GeometricPoseEstimator(config)
        return runtime['geometric']
    raise ValueError(f'Unsupported YCB-V pose backend: {backend}')


def run_frame(frame,config,detector=None,output='artifacts/ycbv/inference',oracle_masks=False,
              pose_backend=None,plots=True,depth_extraction='mask',pose_checkpoint=None,runtime=None,export_clouds=True):
    if not oracle_masks and detector is None:
        raise ValueError('Prediction mode requires a detector')
    directory = resolve(output); directory.mkdir(parents=True,exist_ok=True)
    (directory/'masks').mkdir(exist_ok=True)
    started = time.perf_counter()
    rgb,depth,k = load_rgbd(frame)
    timing = {'preprocessing_s':time.perf_counter()-started}
    begun = time.perf_counter()
    prediction = reference_detections(frame) if oracle_masks else (detector.predict(rgb,depth=depth,k=k)
                 if getattr(detector,'requires_depth',False) else detector.predict(rgb))
    timing['detection_s'] = time.perf_counter()-begun
    begun = time.perf_counter()
    need_scene=export_clouds or plots or not config.get('runtime',{}).get('lazy_scene_geometry',False)
    if need_scene:
        scene,pixels = backproject(depth,k,None,config['geometry']['depth_min_m'],config['geometry']['depth_max_m'])
        scene,colors = voxel_downsample(scene,config['geometry']['voxel_size_m'],rgb[pixels[:,0],pixels[:,1]])
    else:
        scene,colors=np.zeros((0,3)),np.zeros((0,3),dtype=np.uint8)
    timing['scene_geometry_s'] = time.perf_counter()-begun
    if export_clouds:
        write_ply(directory/'pointcloud.ply',scene,colors)
    backend = pose_backend or config['pose']['backend']
    cache = runtime if runtime is not None else {}
    estimator = get_estimator(config,backend,cache,str(detector.device) if detector is not None else 'auto')
    cleaning=config.get('point_refiner',{})
    cleaner=None
    if cleaning.get('enabled',False):
        if 'point_cleaner' not in cache:
            from .point_network import PointCleaner
            cache['point_cleaner']=PointCleaner(cleaning['checkpoint'],str(detector.device) if detector is not None else 'auto')
        cleaner=cache['point_cleaner']
    geometry = estimator.geometry if hasattr(estimator,'geometry') else estimator
    items,poses = [],[]
    batch_size=config.get('runtime',{}).get('pose_batch_size',1)
    batched=backend.startswith('cosypose') and batch_size>1 and hasattr(estimator,'estimate_rgbd_batch')
    batch_masks,batch_colors=[],[]
    timing.update(instance_geometry_s=0.,pose_s=0.)
    for i,(box,label,score,mask) in enumerate(zip(prediction['boxes'],prediction['labels'],prediction['scores'],prediction['masks'])):
        instance_id = f"{frame['key']}/pred_{i:03d}"
        Image.fromarray(mask.astype(np.uint8)*255).save(directory/'masks'/f'instance_{i:03d}.png')
        selected = mask
        if depth_extraction=='bbox':
            selected = np.zeros_like(mask)
            x1,y1,x2,y2 = np.asarray(box).astype(int)
            selected[max(0,y1):min(mask.shape[0],y2),max(0,x1):min(mask.shape[1],x2)] = True
        elif depth_extraction!='mask':
            raise ValueError('Depth extraction must be mask or bbox')
        begun = time.perf_counter()
        points,point_colors,diagnostics = instance_cloud(depth,k,selected,rgb,config)
        if cleaner is not None:
            uvh=points@k.T
            uv=np.rint(uvh[:,:2]/uvh[:,2:]).astype(int)
            uv[:,0]=np.clip(uv[:,0],0,depth.shape[1]-1); uv[:,1]=np.clip(uv[:,1],0,depth.shape[0]-1)
            membership=mask[uv[:,1],uv[:,0]]
            keep,cleaning_diagnostics=cleaner.filter(points,point_colors,int(label),membership=membership,min_points=config['geometry']['min_points'])
            diagnostics['retained_points_before_learned']=len(points)
            points,point_colors=points[keep],point_colors[keep]
            diagnostics['retained_points']=len(points)
            diagnostics['learned_3d_refinement']=cleaning_diagnostics
        timing['instance_geometry_s'] += time.perf_counter()-begun
        cloud_path = f'pointclouds/instance_{i:03d}.ply'
        if export_clouds:
            write_ply(directory/cloud_path,points,point_colors)
        item = {'instance_id':instance_id,'obj_id':int(label),'score':float(score),'box':box,'mask':mask,
                '_points':points,'geometry':diagnostics,'mask_path':f'masks/instance_{i:03d}.png',
                'pointcloud_path':cloud_path if export_clouds else None,'instance_color':instance_color(i)}
        items.append(item)
        if batched:
            batch_masks.append(selected); batch_colors.append(point_colors)
            continue
        begun = time.perf_counter()
        if backend.startswith('cosypose'):
            pose = estimator.estimate_rgbd(rgb,depth,k,box,selected,int(label),points,
                                          depth_refinement=backend=='cosypose_rgbd')
        else:
            pose = estimator.estimate(points,int(label))
        timing['pose_s'] += time.perf_counter()-begun
        poses.append({'instance_id':instance_id,'obj_id':int(label),'score':float(score),**pose})
        if export_clouds and 'T_camera_object' in pose:
            transform = np.asarray(pose['T_camera_object'])
            local = (points-transform[:3,3])@transform[:3,:3]
            write_ply(directory/'object_pointclouds'/f'instance_{i:03d}.ply',local,point_colors)
            _,cad,_ = geometry.mesh(int(label))
            write_ply(directory/'aligned_cad'/f'instance_{i:03d}.ply',transform_points(cad,transform))
    if batched:
        begun=time.perf_counter()
        results=estimator.estimate_rgbd_batch(rgb,depth,k,[item['box'] for item in items],batch_masks,
            [item['obj_id'] for item in items],[item['_points'] for item in items],
            depth_refinement=backend=='cosypose_rgbd',batch_size=batch_size)
        if len(results)!=len(items): raise RuntimeError('Pose batch output lost instances')
        timing['pose_s']+=time.perf_counter()-begun
        for i,(item,pose) in enumerate(zip(items,results)):
            poses.append({'instance_id':item['instance_id'],'obj_id':item['obj_id'],'score':item['score'],**pose})
            if export_clouds and 'T_camera_object' in pose:
                transform=np.asarray(pose['T_camera_object']); points=item['_points']
                write_ply(directory/'object_pointclouds'/f'instance_{i:03d}.ply',(points-transform[:3,3])@transform[:3,:3],batch_colors[i])
                _,cad,_=geometry.mesh(item['obj_id'])
                write_ply(directory/'aligned_cad'/f'instance_{i:03d}.ply',transform_points(cad,transform))
    if export_clouds:
        uvh = scene@k.T
        uv = np.rint(uvh[:,:2]/uvh[:,2:]).astype(int)
        uv[:,0] = np.clip(uv[:,0],0,depth.shape[1]-1); uv[:,1] = np.clip(uv[:,1],0,depth.shape[0]-1)
        labels = np.zeros(len(scene),dtype=np.int32)
        instance_colors = np.full((len(scene),3),120,dtype=np.uint8)
        for i,item in reversed(list(enumerate(items))):
            keep = item['mask'][uv[:,1],uv[:,0]]
            labels[keep] = i+1; instance_colors[keep] = instance_color(i)
        write_ply(directory/'instances_pointcloud.ply',scene,instance_colors)
        np.save(directory/'point_instance_ids.npy',labels)
    timing['inference_s_before_reference_evaluation'] = time.perf_counter()-started
    # Evaluation starts here, after segmentation, clouds and all estimated poses.
    matches = match_references(items,frame,config.get('evaluation',{}).get('minimum_mask_match_iou',.5))
    model_info = read_json(resolve(config['data']['models'])/'models_info.json')
    pose_diagnostics,cloud_diagnostics = [],[]
    for position,(reference,iou) in matches.items():
        item,pose = items[position],poses[position]
        reference_mask = np.asarray(Image.open(reference['mask_visib']))>0
        points = item['_points']
        contamination = None
        if len(points):
            uvh = points@k.T
            uv = np.rint(uvh[:,:2]/uvh[:,2:]).astype(int)
            uv[:,0] = np.clip(uv[:,0],0,depth.shape[1]-1); uv[:,1] = np.clip(uv[:,1],0,depth.shape[0]-1)
            contamination = float((~reference_mask[uv[:,1],uv[:,0]]).mean())
        cloud_diagnostics.append({'instance_id':item['instance_id'],'reference_instance_id':reference['instance_id'],
                                  'retained_points':len(points),'point_contamination_fraction':contamination,'mask_IoU':float(iou)})
        if 'T_camera_object' not in pose:
            continue
        truth = np.eye(4)
        truth[:3,:3] = np.asarray(reference['R_m2c']).reshape(3,3)
        truth[:3,3] = np.asarray(reference['t_m2c_mm'])/1000
        info = model_info[str(reference['obj_id'])]
        mesh,_,_ = geometry.mesh(reference['obj_id'])
        errors = pose_errors(mesh,np.asarray(pose['T_camera_object']),truth)
        threshold = info['diameter']/1000*config.get('evaluation',{}).get('pose_success_diameter_fraction',.1)
        symmetric = bool(info.get('symmetries_discrete') or info.get('symmetries_continuous'))
        errors.update(symmetry_metadata_available=True,symmetric_according_to_bop_models_info=symmetric)
        pose_diagnostics.append({'instance_id':item['instance_id'],'obj_id':item['obj_id'],
            'reference_instance_id':reference['instance_id'],'pose_status':pose['status'],'mask_iou':float(iou),
            'ADD_success_0_1d':errors['ADD_m']<threshold,'ADD_S_success_0_1d':errors['ADD_S_m']<threshold,
            'ADD_or_S_success_0_1d':errors['ADD_S_m' if symmetric else 'ADD_m']<threshold,**errors})
    provenance = {'dataset':config['data'].get('name'),'dataset_revision':config['data']['revision'],
        'frame_key':frame['key'],'units':'m','coordinate_frame':'camera_BOP_x_right_y_down_z_forward',
        'uses_ground_truth_masks':oracle_masks,'uses_ground_truth_pose_for_inference':False,
        'ground_truth_pose_used_only_for_evaluation':bool(frame.get('instances')),
        'detector_checkpoint':None if oracle_masks else detector.checkpoint,'pose_backend':backend,
        'pose_checkpoints':getattr(estimator,'checkpoints',None),'point_clouds_exported':export_clouds,
        'point_instance_labels_source':('reference' if oracle_masks else 'predicted')+' visible 2D masks backprojected through measured depth',
        'learned_point_refinement_enabled':cleaner is not None,
        'learned_point_checkpoint':getattr(cleaner,'checkpoint',None),
        'scene_instance_labels_before_learned_filter':True,
        'grasp_planning_performed':False,'config':config}
    write_json(directory/'detection.json',{**provenance,'instances':[{key:value for key,value in item.items() if key not in {'mask','_points'}} for item in items]})
    write_json(directory/'poses.json',{**provenance,'instances':poses})
    quality = [{'instance_id':pose['instance_id'],'obj_id':pose['obj_id'],
                'diagnostic':pose.get('pose_quality')} for pose in poses]
    write_json(directory/'pose_quality.json',{'instances':quality,'uses_gt':False,
               'scope':'geometric support diagnostic; not calibrated correctness or robot safety'})
    write_json(directory/'pose_diagnostics.json',{'matched_instances':pose_diagnostics,'gt_instance_count':len(frame.get('instances',[])),
        'scope':'diagnostic ADD/ADD-S/ADD(-S) at 0.1 diameter; not official BOP AR'})
    write_json(directory/'geometry_diagnostics.json',{'instances':cloud_diagnostics,'scope':'reference evaluation after inference'})
    write_json(directory/'timing.json',timing)
    overlay(rgb,items,k,directory/'final_visualization.png',oracle_masks,poses)
    if plots:
        plot_scene(scene,colors,directory/'scene_3d.png',poses,oracle_masks)
    if runtime is None and hasattr(estimator,'close'):
        estimator.close()
    return {'frame_key':frame['key'],'scene_id':frame['scene_id'],'image_id':frame['image_id'],
            'detections':len(items),'poses_ok':sum(pose['status']=='ok' for pose in poses),
            'gt_instances':len(frame.get('instances',[])),'timing':timing,'output':str(directory),
            'oracle_masks':oracle_masks,'pose_backend':backend,'pose_diagnostics':pose_diagnostics}
