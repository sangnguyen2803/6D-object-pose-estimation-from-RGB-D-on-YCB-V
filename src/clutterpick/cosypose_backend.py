"""Published CosyPose networks with a Windows-compatible textured renderer.

The original crop, initialization and pose-update math are imported unchanged.
Pyrender replaces the original Bullet/EGL renderer; this port must be evaluated
locally and is not claimed to reproduce the published BOP scores exactly.
"""
from pathlib import Path
import sys
import types
import numpy as np
from .common import resolve
from .detection import choose_device
from .pose import GeometricPoseEstimator, trimmed_icp, proper_rotation
from .pretrained import read_published_config


def import_reference_code():
    source = resolve('third_party/cosypose')
    if not (source/'cosypose/models/pose.py').exists():
        raise RuntimeError('Run scripts/prepare_ycbv.py to fetch the pinned reference source')
    if str(source) not in sys.path:
        sys.path.insert(0,str(source))
    # Inference math does not need the original Conda/SLURM config or Pinocchio.
    # Provide only the config attribute consumed by PosePredictor, and load the
    # lib3d submodules without the robotics Transform import in its __init__.
    if 'cosypose.config' not in sys.modules:
        config = types.ModuleType('cosypose.config')
        config.DEBUG_DATA_DIR = resolve('artifacts/ycbv/debug')
        sys.modules['cosypose.config'] = config
    if 'cosypose.lib3d' not in sys.modules:
        package = types.ModuleType('cosypose.lib3d')
        package.__path__ = [str(source/'cosypose/lib3d')]
        sys.modules['cosypose.lib3d'] = package
    from cosypose.models.pose import PosePredictor
    from cosypose.models.efficientnet import EfficientNet
    from cosypose.lib3d.cosypose_ops import TCO_init_from_boxes_zup_autodepth
    return PosePredictor, EfficientNet, TCO_init_from_boxes_zup_autodepth


class InferenceMeshDatabase:
    """Same padded vertex and deterministic sampling convention as CosyPose."""
    def __init__(self,config,device):
        import torch
        import trimesh
        paths = sorted(resolve(config['data']['models']).glob('obj_*.ply'))
        vertices = {path.stem:np.asarray(trimesh.load(path,force='mesh').vertices)*.001 for path in paths}
        size = max(len(points) for points in vertices.values())
        chosen = np.random.RandomState(0).choice(size,2000,replace=False)
        padding = np.random.RandomState(0)
        self.points = {}
        for label,points in vertices.items():
            if len(points)<size:
                points = np.concatenate([points,points[padding.choice(len(points),size-len(points))]])
            self.points[label] = torch.as_tensor(points[chosen],dtype=torch.float32,device=device)

    def select(self,labels):
        import torch
        points = torch.stack([self.points[str(label)] for label in labels])
        return types.SimpleNamespace(sample_points=lambda count,deterministic=False: points[:,:count])


class TexturedRenderer:
    def __init__(self,config,device):
        self.config,self.device = config,device
        self.renderer = None
        self.meshes = {}
        self.scene = None
        self.scene_nodes={}
        self.camera=None
        self.cache_stats={'render_calls':0,'context_mesh_additions':0}

    def mesh(self,label):
        import trimesh
        import pyrender
        from PIL import Image
        if label not in self.meshes:
            path = resolve(self.config['data']['models'])/(label+'.ply')
            mesh = trimesh.load(path,force='mesh',process=False)
            mesh.apply_scale(.001)
            image = getattr(getattr(mesh.visual,'material',None),'image',None)
            if image is None:
                raise ValueError(f'Textured YCB-V mesh unavailable: {label}')
            image = image.copy().convert('RGB')
            image.thumbnail((1024,1024),Image.Resampling.LANCZOS)
            texture = pyrender.Texture(source=np.asarray(image),source_channels='RGB')
            material = pyrender.MetallicRoughnessMaterial(baseColorTexture=texture,
                metallicFactor=0.,roughnessFactor=1.,doubleSided=True)
            self.meshes[label] = pyrender.Mesh.from_trimesh(mesh,material=material,smooth=False)
        return self.meshes[label]

    def render(self,obj_infos,TCO,K,resolution=(240,320)):
        import torch
        import pyrender
        height,width = resolution
        if self.renderer is None:
            self.renderer = pyrender.OffscreenRenderer(width,height)
        self.renderer.viewport_width,self.renderer.viewport_height = width,height
        poses,intrinsics = TCO.detach().cpu().numpy(),K.detach().cpu().numpy()
        images = []
        conversion = np.diag([1.,-1.,-1.,1.])
        for obj,pose,k in zip(obj_infos,poses,intrinsics):
            if not np.isfinite(pose).all() or not np.isfinite(k).all():
                images.append(np.zeros((height,width,3),dtype=np.uint8))
                continue
            resident=self.config.get('pose',{}).get('resident_renderer',False)
            if resident:
                if self.scene is None:
                    self.scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.6,.6,.6])
                    self.camera=pyrender.IntrinsicsCamera(100,100,0,0,znear=.01,zfar=10.)
                    self.scene.add(self.camera,pose=np.eye(4))
                    self.scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=.6),pose=np.eye(4))
                scene=self.scene; label=obj['name']; mesh=self.mesh(label)
                if label not in self.scene_nodes:
                    mesh.is_visible=False
                    self.scene_nodes[label]=scene.add(mesh,pose=np.eye(4))
                self.camera.fx,self.camera.fy,self.camera.cx,self.camera.cy=map(float,[k[0,0],k[1,1],k[0,2],k[1,2]])
                scene.set_pose(self.scene_nodes[label],conversion@pose)
                mesh.is_visible=True
            else:
                scene = pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.6,.6,.6])
                scene.add(pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2],znear=.01,zfar=10.),pose=np.eye(4))
                scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=.6),pose=np.eye(4))
                scene.add(self.mesh(obj['name']),pose=conversion@pose)
            self.cache_stats['render_calls']+=1
            # Pinned Pyrender 0.1.45 uploads scene meshes absent from its current context.
            self.cache_stats['context_mesh_additions']+=len(scene.meshes-self.renderer._renderer._meshes)
            try:
                color,_ = self.renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
            finally:
                if resident: mesh.is_visible=False
            images.append(color[:,:,:3])
        return torch.as_tensor(np.stack(images),device=self.device).float().permute(0,3,1,2)/255

    def close(self):
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None
        self.scene=None; self.scene_nodes={}; self.camera=None


class CosyPoseEstimator:
    def __init__(self,config,device='auto'):
        import torch
        PosePredictor,EfficientNet,self.initialize = import_reference_code()
        torch.set_num_threads(4)
        self.config,self.device = config,choose_device(device)
        self.geometry = GeometricPoseEstimator(config)
        self.mesh_db = InferenceMeshDatabase(config,self.device)
        self.renderer = TexturedRenderer(config,self.device)
        self.networks = {}
        self.checkpoints = {}
        for role in ['coarse','refiner']:
            path = resolve(config['pose'][role+'_checkpoint'])
            published = read_published_config(path.parent/'config.yaml')
            if published['backbone_str']!='efficientnet-b3' or published['n_pose_dims']!=9:
                raise ValueError('Unsupported published pose network architecture')
            backbone = EfficientNet.from_name('efficientnet-b3',in_channels=6)
            backbone.n_features = 1536
            network = PosePredictor(backbone,self.renderer,self.mesh_db,render_size=(240,320),pose_dim=9)
            payload = torch.load(path,map_location='cpu',weights_only=False)
            network.load_state_dict(payload['state_dict'],strict=True)
            self.networks[role] = network.to(self.device).eval()
            self.checkpoints[role] = str(path)
        self.checkpoint = self.checkpoints['refiner']

    def estimate_rgbd(self,rgb,depth,k,box,mask,obj_id,points=None,depth_refinement=None):
        import torch
        from scipy.spatial.transform import Rotation
        from .geometry import instance_cloud
        label = f'obj_{obj_id:06d}'
        image = torch.as_tensor(rgb.copy(),device=self.device).float().permute(2,0,1)[None]/255
        intrinsic = torch.as_tensor(k,dtype=torch.float32,device=self.device)[None]
        bbox = torch.as_tensor(box,dtype=torch.float32,device=self.device)[None]
        sampled = self.mesh_db.select([label]).sample_points(2000,deterministic=True)
        with torch.inference_mode():
            pose = self.initialize(bbox,sampled,intrinsic)
            for role in ['coarse','refiner']:
                iterations = self.config['pose'][role+'_iterations']
                if iterations:
                    outputs = self.networks[role](image,intrinsic,[label],pose,n_iterations=iterations)
                    pose = outputs[f'iteration={iterations}']['TCO_output']
        transform = pose[0].cpu().numpy().astype(float)
        if points is None:
            points,_,_ = instance_cloud(depth,k,mask,rgb,self.config)
        return self.finalize_rgbd(transform,points,obj_id,depth_refinement)

    def estimate_rgbd_batch(self,rgb,depth,k,boxes,masks,obj_ids,point_clouds,depth_refinement=None,batch_size=4):
        """Batch independent objects through the published networks; keep per-object ICP."""
        import torch
        if batch_size<1: raise ValueError('Pose batch size must be positive')
        count=len(obj_ids)
        if not (len(boxes)==len(masks)==len(point_clouds)==count):
            raise ValueError('Batch inputs must have matching instance counts')
        if not count: return []
        image=torch.as_tensor(rgb.copy(),device=self.device).float().permute(2,0,1)[None]/255
        intrinsic=torch.as_tensor(k,dtype=torch.float32,device=self.device)[None]
        results=[]
        for start in range(0,count,batch_size):
            stop=min(start+batch_size,count); size=stop-start
            labels=[f'obj_{int(obj):06d}' for obj in obj_ids[start:stop]]
            images=image.expand(size,-1,-1,-1)
            intrinsics=intrinsic.expand(size,-1,-1)
            bbox=torch.as_tensor(np.asarray(boxes[start:stop]),dtype=torch.float32,device=self.device)
            sampled=self.mesh_db.select(labels).sample_points(2000,deterministic=True)
            with torch.inference_mode():
                pose=self.initialize(bbox,sampled,intrinsics)
                for role in ['coarse','refiner']:
                    iterations=self.config['pose'][role+'_iterations']
                    if iterations:
                        outputs=self.networks[role](images,intrinsics,labels,pose,n_iterations=iterations)
                        pose=outputs[f'iteration={iterations}']['TCO_output']
            transforms=pose.cpu().numpy().astype(float)
            for offset,transform in enumerate(transforms):
                index=start+offset
                result=self.finalize_rgbd(transform,point_clouds[index],int(obj_ids[index]),depth_refinement)
                result['neural_pose_batch_size']=size
                results.append(result)
        return results

    def finalize_rgbd(self,transform,points,obj_id,depth_refinement=None):
        """Shared geometry stage, with no reference masks or poses."""
        from scipy.spatial.transform import Rotation
        if not np.isfinite(transform).all() or transform[2,3]<=0:
            return {'status':'invalid_prediction','backend':'cosypose','uses_ground_truth_pose':False}
        transform[:3,:3] = proper_rotation(transform[:3,:3])
        neural = transform.copy()
        used = self.config['pose'].get('depth_refinement',False) if depth_refinement is None else depth_refinement
        accepted,rmse,coverage = False,None,None
        robust_details = None
        if used and self.config['pose'].get('refinement_method')=='robust':
            from .refinement import refine
            _,cad,normals = self.geometry.mesh(obj_id)
            transform,robust_details = refine(cad,normals,points,neural,self.config['pose'],
                                             self.config['seed']+obj_id,self.config['geometry']['min_points'],
                                             tree=self.geometry.registration_tree(obj_id) if self.config['pose'].get('cache_model_trees',False) else None)
            accepted = robust_details['accepted']
            rmse = robust_details['after']['rmse_m']
            coverage = robust_details['after']['coverage_10mm']
        elif len(points)>=self.config['geometry']['min_points']:
            _,cad,_ = self.geometry.mesh(obj_id)
            rng = np.random.default_rng(self.config['seed']+obj_id)
            observed = points[rng.choice(len(points),min(len(points),self.config['pose']['observed_points']),replace=False)]
            cfg = self.config['pose']
            limit,trim = cfg['correspondence_limit_m'],cfg['trim_fraction']
            _,before,coverage = trimmed_icp(cad,observed,transform,0,limit,trim)
            rmse = before
            if used:
                refined,after,new_coverage = trimmed_icp(cad,observed,transform,cfg['icp_iterations'],limit,trim)
                rotation_change = Rotation.from_matrix(refined[:3,:3]@transform[:3,:3].T).magnitude()
                accepted = (after<before*(1-cfg.get('depth_min_relative_fit_improvement',.1)) and
                            np.linalg.norm(refined[:3,3]-transform[:3,3])<cfg.get('depth_max_translation_update_m',.05) and
                            rotation_change<np.deg2rad(cfg.get('depth_max_rotation_update_degrees',10)))
                if accepted:
                    transform,rmse,coverage = refined,after,new_coverage
        return {'status':'ok','backend':'cosypose_pretrained_pyrender_port',
                'T_camera_object':transform,'T_camera_object_before_depth_refinement':neural,
                'observed_trimmed_rmse_m':rmse,'observed_coverage_10mm':coverage,
                'depth_refinement_requested':bool(used),'depth_refinement_accepted':bool(accepted),
                'uses_depth_for_pose':bool(accepted),'uses_ground_truth_pose':False,
                'pose_checkpoints':self.checkpoints,'renderer':'pyrender_textured_1024',
                'refinement_diagnostics':robust_details,
                'pose_quality':None if robust_details is None else robust_details['quality'],
                'limitations':'renderer port differs from published Bullet/EGL implementation; evaluated locally'}

    def close(self):
        self.renderer.close()
