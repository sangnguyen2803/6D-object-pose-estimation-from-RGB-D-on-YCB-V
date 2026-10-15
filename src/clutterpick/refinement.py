"""Robust local RGB-D registration and geometric support diagnostics.

No GT pose, mask or class correction enters this module. Support is a geometric
diagnostic, never a calibrated probability of correct 6D pose.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .pose import trimmed_icp
from .geometry import transform_points


def fit_metrics(model,observed,pose,trim=.8,tree=None):
    if not len(observed):
        return {'rmse_m':None,'coverage_10mm':0.,'support_40mm':0.,'points':0}
    tree=tree if tree is not None else cKDTree(model)
    distances,_=tree.query((observed-pose[:3,3])@pose[:3,:3])
    selected=np.sort(distances)[:max(1,int(len(distances)*trim))]
    return {'rmse_m':float(np.sqrt(np.mean(selected**2))),
            'coverage_10mm':float(np.mean(distances<.01)),
            'support_40mm':float(np.mean(distances<.04)),'points':len(observed)}


def point_plane(model,normals,observed,initial,cfg,tree=None):
    """Huber IRLS with a damped, centered SE(3) update; measured CAD normals."""
    tree=cKDTree(model) if tree is None else tree
    pose=initial.copy()
    radius=max(float(np.sqrt(np.mean(np.sum((model-model.mean(0))**2,axis=1)))),.02)
    eigenvalues=np.zeros(6)
    iterations=0
    for iteration in range(cfg.get('robust_iterations',20)):
        local=(observed-pose[:3,3])@pose[:3,:3]
        distances,ids=tree.query(local)
        cutoff=min(cfg['correspondence_limit_m'],float(np.quantile(distances,.85)))
        keep=distances<=cutoff
        if keep.sum()<30: break
        targets=observed[keep]
        source=transform_points(model[ids[keep]],pose)
        normal=normals[ids[keep]]@pose[:3,:3].T
        length=np.linalg.norm(normal,axis=1)
        valid=length>1e-8
        targets,source,normal=targets[valid],source[valid],normal[valid]/length[valid,None]
        if len(source)<30: break
        residual=np.sum(normal*(targets-source),axis=1)
        delta=cfg.get('huber_delta_m',.004)
        weights=np.minimum(1.,delta/np.maximum(np.abs(residual),1e-12))
        jacobian=np.column_stack((np.cross(source-pose[:3,3],normal)/radius,normal))
        information=(jacobian.T*weights)@jacobian/max(weights.sum(),1)
        eigenvalues=np.linalg.eigvalsh(information)
        rhs=jacobian.T@(weights*residual)/max(weights.sum(),1)
        step=np.linalg.solve(information+np.eye(6)*cfg.get('damping',.01),rhs)
        angle=step[:3]/radius
        angle*=min(1.,np.deg2rad(2)/max(np.linalg.norm(angle),1e-12))
        translation=step[3:]
        translation*=min(1.,.01/max(np.linalg.norm(translation),1e-12))
        pose[:3,:3]=Rotation.from_rotvec(angle).as_matrix()@pose[:3,:3]
        pose[:3,3]+=translation
        iterations=iteration+1
        if np.linalg.norm(translation)+radius*np.linalg.norm(angle)<1e-5: break
    return pose,{'iterations':iterations,'information_eigenvalues':eigenvalues.tolist(),
                 'observable_directions':int(np.sum(eigenvalues>.01))}


def bounded(candidate,initial,cfg):
    movement=float(np.linalg.norm(candidate[:3,3]-initial[:3,3]))
    angle=float(np.degrees(Rotation.from_matrix(candidate[:3,:3]@initial[:3,:3].T).magnitude()))
    valid=(np.isfinite(candidate).all() and candidate[2,3]>0 and
           movement<cfg.get('depth_max_translation_update_m',.05) and
           angle<cfg.get('depth_max_rotation_update_degrees',10))
    return valid,movement,angle


def annotate_observability(diagnostic):
    """Conservative review flag from the robust candidate's measured constraints."""
    candidates=[candidate for candidate in diagnostic['candidates'] if candidate['method']=='huber_point_to_plane']
    rank=candidates[0].get('observable_directions') if candidates else None
    quality=diagnostic['quality']
    quality['robust_candidate_constraint_rank']=rank
    quality['review_reasons']=[]
    if quality['state']!='supported_geometry':
        quality['review_reasons'].append(quality['state'])
    if rank is not None and rank<6:
        quality['review_reasons'].append('robust_candidate_has_underconstrained_directions')
    quality['requires_review']=bool(quality['review_reasons'])
    return diagnostic


def refine(model,normals,points,initial,cfg,seed=42,min_points=40,tree=None):
    observed=np.asarray(points)
    finite=np.isfinite(observed).all(axis=1)
    observed=observed[finite]
    rng=np.random.default_rng(seed)
    observed=observed[rng.choice(len(observed),min(len(observed),cfg['observed_points']),replace=False)]
    reusable_tree=tree
    tree=cKDTree(model) if tree is None else tree
    before=fit_metrics(model,observed,initial,cfg['trim_fraction'],tree)
    candidates=[]
    selected=initial.copy(); selected_name='rgb'
    if len(observed)>=min_points:
        legacy,_,_=trimmed_icp(model,observed,initial,cfg['icp_iterations'],
                               cfg['correspondence_limit_m'],cfg['trim_fraction'],tree=reusable_tree)
        robust,details=point_plane(model,normals,observed,initial,cfg,tree=reusable_tree)
        for name,candidate,extra in [('trimmed_point_to_point',legacy,{}),('huber_point_to_plane',robust,details)]:
            metrics=fit_metrics(model,observed,candidate,cfg['trim_fraction'],tree)
            valid,translation,angle=bounded(candidate,initial,cfg)
            improvement=(before['rmse_m']-metrics['rmse_m'])/max(before['rmse_m'],1e-12)
            eligible=valid and improvement>cfg.get('depth_min_relative_fit_improvement',.1)
            if name=='huber_point_to_plane':
                eligible=eligible and metrics['support_40mm']>=.5 and metrics['coverage_10mm']>=before['coverage_10mm']-.02
            candidates.append({'method':name,'eligible':bool(eligible),'relative_rmse_improvement':float(improvement),
                               'translation_update_m':translation,'rotation_update_degrees':angle,**metrics,**extra})
        eligible=[(candidate['rmse_m'],index) for index,candidate in enumerate(candidates) if candidate['eligible']]
        if eligible:
            _,index=min(eligible)
            selected=[legacy,robust][index]; selected_name=candidates[index]['method']
    quality=fit_metrics(model,observed,selected,cfg['trim_fraction'],tree)
    supported=(quality['points']>=min_points and quality['coverage_10mm']>=cfg.get('min_coverage',.65)
               and quality['rmse_m']<=cfg.get('max_observed_rmse_m',.01))
    state='supported_geometry' if supported else ('insufficient_depth' if len(observed)<min_points else 'weak_geometry')
    diagnostic={'method':'robust_candidate_selection','selected_candidate':selected_name,
                     'accepted':selected_name!='rgb','before':before,'after':quality,'candidates':candidates,
                     'quality':{'state':state,'requires_review':not supported,**quality,
                                'calibrated_probability':False,
                                'limitation':'symmetry and wrong identity may fit geometry despite incorrect pose'}}
    return selected,annotate_observability(diagnostic)

