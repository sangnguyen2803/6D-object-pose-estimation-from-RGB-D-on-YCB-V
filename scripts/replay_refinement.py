"""Replay refinement from saved predictions. CNN outputs remain fixed.

All refined predictions are computed before reading reference poses for scoring.
Timings cover geometry/refinement only, not end-to-end neural inference.
"""
import argparse,copy,time,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from clutterpick.common import load_config,resolve,read_json,write_json
from clutterpick.data import load_index,load_rgbd
from clutterpick.geometry import instance_cloud
from clutterpick.pose import GeometricPoseEstimator
from clutterpick.refinement import refine
from clutterpick.evaluation import pose_errors
from clutterpick.reporting import summarize,per_object
from clutterpick.cli import export_bop

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/enhanced.yaml')
    parser.add_argument('--source',default='artifacts/ycbv/benchmark_full')
    parser.add_argument('--output',default='artifacts/ycbv/extension/replay')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    cfg=load_config(args.config); index,_=load_index(cfg)
    lookup={frame['key']:frame for frame in index['frames']}
    source=resolve(args.source); source_rows=read_json(source/'run_summary.json')
    if args.limit: source_rows=source_rows[:args.limit]
    geometry=GeometricPoseEstimator(cfg)
    model_info=read_json(resolve(cfg['data']['models'])/'models_info.json')
    rows=[]; changes=[]; quality_counts={}
    for position,baseline in enumerate(source_rows):
        frame=lookup[baseline['frame_key']]; original=Path(baseline['output'])
        detector=read_json(original/'detection.json')
        neural=read_json(original/'poses.json')
        assert not neural['uses_ground_truth_pose_for_inference'] and not neural['uses_ground_truth_masks']
        rgb,depth,k=load_rgbd(frame)
        updated=[]; started=time.perf_counter()
        for item,pose in zip(detector['instances'],neural['instances']):
            prediction=copy.deepcopy(pose)
            if 'T_camera_object_before_depth_refinement' not in pose:
                raise ValueError('Replay requires stored RGB pose; cannot substitute a GT or refined pose')
            mask=np.asarray(Image.open(original/item['mask_path']))>0
            points,_,cloud=instance_cloud(depth,k,mask,rgb,cfg)
            _,model,normals=geometry.mesh(item['obj_id'])
            matrix,diagnostic=refine(model,normals,points,np.asarray(pose['T_camera_object_before_depth_refinement']),
                                     cfg['pose'],cfg['seed']+item['obj_id'],cfg['geometry']['min_points'])
            prediction.update(T_camera_object=matrix,refinement_diagnostics=diagnostic,
                              pose_quality=diagnostic['quality'],depth_refinement_accepted=diagnostic['accepted'],
                              observed_trimmed_rmse_m=diagnostic['after']['rmse_m'],
                              observed_coverage_10mm=diagnostic['after']['coverage_10mm'],
                              uses_depth_for_pose=diagnostic['accepted'])
            updated.append(prediction)
        elapsed=time.perf_counter()-started
        # Reference evaluation begins only after all per-frame predictions are complete.
        by_id={pose['instance_id']:pose for pose in updated}
        truth_lookup={obj['instance_id']:obj for obj in frame['instances']}
        diagnostics=[]
        for old in baseline['pose_diagnostics']:
            obj=truth_lookup[old['reference_instance_id']]
            pose=by_id[old['instance_id']]
            truth=np.eye(4); truth[:3,:3]=np.asarray(obj['R_m2c']).reshape(3,3)
            truth[:3,3]=np.asarray(obj['t_m2c_mm'])/1000
            mesh,_,_=geometry.mesh(obj['obj_id'])
            errors=pose_errors(mesh,np.asarray(pose['T_camera_object']),truth)
            info=model_info[str(obj['obj_id'])]
            symmetric=bool(info.get('symmetries_discrete') or info.get('symmetries_continuous'))
            threshold=info['diameter']/1000*.1
            new={**old,**errors,'ADD_success_0_1d':errors['ADD_m']<threshold,
                 'ADD_S_success_0_1d':errors['ADD_S_m']<threshold,
                 'ADD_or_S_success_0_1d':errors['ADD_S_m' if symmetric else 'ADD_m']<threshold,
                 'symmetry_metadata_available':True,'symmetric_according_to_bop_models_info':symmetric}
            diagnostics.append(new)
            before=old['ADD_or_S_success_0_1d']; after=new['ADD_or_S_success_0_1d']
            changes.append({'frame_key':frame['key'],'instance_id':pose['instance_id'],'obj_id':obj['obj_id'],
                            'baseline_success':before,'enhanced_success':after,
                            'quality_state':pose['pose_quality']['state'],
                            'selected_candidate':pose['refinement_diagnostics']['selected_candidate']})
        matched={item['instance_id']:item for item in diagnostics}
        for pose in updated:
            state=pose['pose_quality']['state']
            counts=quality_counts.setdefault(state,{'predictions':0,'matched_gt':0,'correct_ADD_or_S':0})
            counts['predictions']+=1
            if pose['instance_id'] in matched:
                counts['matched_gt']+=1
                counts['correct_ADD_or_S']+=int(matched[pose['instance_id']]['ADD_or_S_success_0_1d'])
        directory=resolve(args.output)/frame['key'].replace('/','_')
        write_json(directory/'poses.json',{**neural,'config':cfg,'instances':updated,
                   'inference_mode':'cached detector and RGB pose; fresh robust depth refinement',
                   'cached_source':str(original),'timing_scope':'geometry/refinement only'})
        write_json(directory/'pose_diagnostics.json',{'gt_instance_count':baseline['gt_instances'],'matched_instances':diagnostics})
        row={**baseline,'output':str(directory),'pose_diagnostics':diagnostics,
             'timing':{'inference_s_before_reference_evaluation':elapsed,'scope':'geometry/refinement replay only'}}
        rows.append(row)
        if (position+1)%25==0:
            write_json(resolve(args.output)/'run_summary.json',rows)
            print('Replay',position+1,'/',len(source_rows),flush=True)
    write_json(resolve(args.output)/'run_summary.json',rows)
    export_bop(rows,args.output)
    enhanced=summarize(rows); baseline_summary=summarize(source_rows)
    enhanced['median_refinement_geometry_s']=enhanced.pop('median_inference_s_excluding_first')
    enhanced['per_object']=per_object(rows,index)
    for counts in quality_counts.values():
        counts['diagnostic_precision']=counts['correct_ADD_or_S']/counts['predictions'] if counts['predictions'] else None
    result={'frames':len(rows),'baseline':baseline_summary,'enhanced':enhanced,
            'gains':sum(not item['baseline_success'] and item['enhanced_success'] for item in changes),
            'regressions':sum(item['baseline_success'] and not item['enhanced_success'] for item in changes),
            'quality_groups':quality_counts,'paired_instances':changes,'config':cfg,
            'scope':'same real test images and cached neural predictions; exploratory, not fresh independent validation',
            'latency_comparison_valid':False,'csv_time_scope':'geometry/refinement replay only',
            'uses_gt_for_refinement':False}
    write_json(resolve('reports/ycbv/extension/refinement_comparison.json'),result)
    print({key:value for key,value in result.items() if key in ['frames','gains','regressions','quality_groups']})
if __name__=='__main__': main()

