"""CAD registration baseline: multiple proper-rotation initializations + trimmed ICP.

This baseline does not read reference poses. Scores describe geometric fit,
not calibrated confidence.
"""
import itertools
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .geometry import transform_points


def load_mesh(path):
    import trimesh
    mesh = trimesh.load(str(path),force="mesh",process=False)
    if len(mesh.vertices)==0 or len(mesh.faces)==0:
        raise ValueError(f"Empty CAD mesh: {path}")
    mesh.apply_scale(0.001)  # BOP CAD coordinates are millimeters.
    return mesh


def sample_mesh(mesh,count,seed=42):
    import trimesh
    points,face_ids = trimesh.sample.sample_surface(mesh,count,seed=seed)
    return points,np.asarray(mesh.face_normals)[face_ids]


def proper_rotation(matrix):
    u,_,vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1,-1] = np.linalg.det(u@vt)
    return u@correction@vt


def rigid_fit(source,target):
    a,b = source.mean(axis=0),target.mean(axis=0)
    r = proper_rotation((source-a).T@(target-b)).T
    t = b-r@a
    transform = np.eye(4)
    transform[:3,:3],transform[:3,3] = r,t
    return transform


def cube_rotations():
    matrices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product([-1,1],repeat=3):
            r = np.eye(3)[:,permutation] @ np.diag(signs)
            if np.linalg.det(r)>0:
                matrices.append(r)
    return matrices


def pca_basis(points):
    _,vectors = np.linalg.eigh(np.cov(points.T))
    return proper_rotation(vectors[:,::-1])


def fpfh_initializations(model,observed,model_normals,seed):
    """Optional Open3D feature-based global initializations before local ICP."""
    import open3d as o3d
    o3d.utility.random.seed(seed)
    source,target = o3d.geometry.PointCloud(),o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(model)
    source.normals = o3d.utility.Vector3dVector(model_normals)
    target.points = o3d.utility.Vector3dVector(observed)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.025,max_nn=40))
    target.orient_normals_towards_camera_location(np.zeros(3))
    source = source.voxel_down_sample(.005)
    target = target.voxel_down_sample(.005)
    registration = o3d.pipelines.registration
    search = o3d.geometry.KDTreeSearchParamHybrid(radius=.035,max_nn=100)
    source_features = registration.compute_fpfh_feature(source,search)
    target_features = registration.compute_fpfh_feature(target,search)
    fgr = registration.registration_fgr_based_on_feature_matching(source,target,source_features,target_features,
        registration.FastGlobalRegistrationOption(maximum_correspondence_distance=.02))
    ransac = registration.registration_ransac_based_on_feature_matching(source,target,source_features,target_features,
        True,.02,registration.TransformationEstimationPointToPoint(False),3,
        [registration.CorrespondenceCheckerBasedOnEdgeLength(.9),registration.CorrespondenceCheckerBasedOnDistance(.02)],
        registration.RANSACConvergenceCriteria(20000,.99))
    return [r.transformation for r in [fgr,ransac] if r.fitness>0]


def trimmed_icp(model,observed,initial,iterations,max_distance,trim_fraction,tree=None):
    tree = cKDTree(model) if tree is None else tree
    pose = initial.copy()
    for _ in range(iterations):
        local = (observed-pose[:3,3])@pose[:3,:3]
        distances,ids = tree.query(local)
        cutoff = min(max_distance,np.quantile(distances,trim_fraction))
        keep = distances<=cutoff
        if keep.sum()<12:
            break
        updated = rigid_fit(model[ids[keep]],observed[keep])
        change = np.linalg.norm(updated[:3,3]-pose[:3,3])+np.linalg.norm(updated[:3,:3]-pose[:3,:3])*0.01
        pose = updated
        if change<1e-5:
            break
    distances,_ = tree.query((observed-pose[:3,3])@pose[:3,:3])
    trimmed = np.sort(distances)[:max(1,int(len(distances)*trim_fraction))]
    rmse = float(np.sqrt(np.mean(trimmed**2)))
    coverage = float(np.mean(distances<0.01))
    return pose,rmse,coverage


class GeometricPoseEstimator:
    def __init__(self,config):
        self.config = config
        self.cache = {}
        self.registration_trees = {}

    def registration_tree(self,obj_id):
        """Reuse the immutable CAD index across instances and frames."""
        if obj_id not in self.registration_trees:
            self.registration_trees[obj_id]=cKDTree(self.mesh(obj_id)[1])
        return self.registration_trees[obj_id]

    def mesh(self,obj_id):
        from .common import resolve
        if obj_id not in self.cache:
            path = resolve(self.config["data"]["models"])/f"obj_{obj_id:06d}.ply"
            mesh = load_mesh(path)
            samples,normals = sample_mesh(mesh,self.config["pose"]["model_points"],self.config["seed"]+obj_id)
            self.cache[obj_id] = mesh,samples,normals
        return self.cache[obj_id]

    def estimate(self,points,obj_id):
        cfg = self.config["pose"]
        if len(points)<self.config["geometry"]["min_points"]:
            return {"status": "insufficient_points", "backend": "geometric"}
        _,model,model_normals = self.mesh(obj_id)
        rng = np.random.default_rng(self.config["seed"]+obj_id)
        observed = points[rng.choice(len(points),min(len(points),cfg["observed_points"]),replace=False)]
        a,b = pca_basis(model),pca_basis(observed)
        rotations = [b@r@a.T for r in cube_rotations()]
        rotations += list(Rotation.random(cfg["random_hypotheses"],random_state=rng).as_matrix())
        centroid = np.median(observed,axis=0)
        initial_results = []
        if cfg.get("fpfh_global_initialization",False):
            for initial in fpfh_initializations(model,observed,model_normals,self.config["seed"]):
                initial_results.append(trimmed_icp(model,observed,initial,6,cfg["correspondence_limit_m"],cfg["trim_fraction"]))
        for rotation in rotations:
            pose = np.eye(4)
            pose[:3,:3] = rotation
            pose[:3,3] = centroid-rotation@model.mean(axis=0)
            initial_results.append(trimmed_icp(model,observed,pose,6,cfg["correspondence_limit_m"],cfg["trim_fraction"]))
        initial_results.sort(key=lambda p: p[1]+0.01*(1-p[2]))
        self.last_coarse_hypotheses = [p[0] for p in initial_results]
        final = [trimmed_icp(model,observed,p[0],cfg["icp_iterations"],cfg["correspondence_limit_m"],cfg["trim_fraction"])
                 for p in initial_results[:cfg["refine_hypotheses"]]]
        pose,rmse,coverage = min(final,key=lambda p: p[1]+0.01*(1-p[2]))
        self.last_refined_hypotheses = [p[0] for p in final]
        reliable = coverage>=cfg["min_coverage"] and rmse<=cfg["max_observed_rmse_m"]
        return {"status": "ok" if reliable else "poor_fit", "backend": "geometric_multi_init_trimmed_icp",
                "T_camera_object": pose, "observed_trimmed_rmse_m": rmse,
                "observed_coverage_10mm": coverage, "hypotheses": len(rotations),
                "uses_ground_truth_pose": False, "limitations": "partial geometry and symmetry can yield an incorrect orientation despite low residual"}


