import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def backproject(depth, k, mask=None, depth_min=0, depth_max=np.inf):
    """Backproject metric depth with upper-triangular pinhole intrinsics, including skew."""
    depth = np.asarray(depth)
    k = np.asarray(k,dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("Depth must be a two-dimensional image")
    if (k.shape != (3,3) or not np.isfinite(k).all()
            or k[0,0] <= 0 or k[1,1] <= 0
            or not np.allclose(k[2],[0,0,1]) or not np.isclose(k[1,0],0)):
        raise ValueError("Camera intrinsics must be finite upper-triangular 3x3 with positive focal lengths and bottom row [0, 0, 1]")
    valid = np.isfinite(depth) & (depth>depth_min) & (depth<depth_max)
    if mask is not None:
        if mask.shape != depth.shape:
            raise ValueError("Mask/depth size mismatch")
        valid &= mask.astype(bool)
    v,u = np.nonzero(valid)
    z = depth[v,u]
    y = (v-k[1,2])*z/k[1,1]
    x = ((u-k[0,2])*z-k[0,1]*y)/k[0,0]
    points = np.column_stack((x,y,z))
    return points, np.column_stack((v,u))


def voxel_downsample(points, voxel_size, colors=None):
    if len(points)==0 or voxel_size<=0:
        return (points,colors) if colors is not None else points
    _, indices = np.unique(np.floor(points/voxel_size).astype(np.int64),axis=0,return_index=True)
    return (points[indices],colors[indices]) if colors is not None else points[indices]


def instance_cloud(depth,k,mask,rgb,config):
    mask = np.asarray(mask,dtype=bool)
    g = config["geometry"]
    clean = binary_erosion(mask, iterations=g["boundary_erosion_pixels"]) if g["boundary_erosion_pixels"]>0 else mask
    points, pixels = backproject(depth,k,clean,g["depth_min_m"],g["depth_max_m"])
    colors = rgb[pixels[:,0],pixels[:,1]]
    original = len(points)
    if len(points)>20:
        distances,_ = cKDTree(points).query(points,k=8)
        means = distances[:,1:].mean(axis=1)
        keep = means <= means.mean()+2*means.std()
        points, colors = points[keep], colors[keep]
    points,colors = voxel_downsample(points,g["voxel_size_m"],colors)
    missing = ~np.isfinite(depth) | (depth<=0)
    return points,colors,{"visible_mask_pixels": int(mask.sum()), "valid_points_before_filter": original,
                         "retained_points": len(points), "depth_missing_fraction": float((mask & missing).sum()/max(mask.sum(),1))}


def write_ply(path,points,colors=None):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    points = np.asarray(points).reshape(-1,3)
    if colors is None:
        colors = np.full(points.shape,180,dtype=np.uint8)
    with path.open("w",encoding="ascii") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p,c in zip(points,colors):
            f.write("%.6f %.6f %.6f %d %d %d\n" % (*p,*c))


def transform_points(points,transform):
    return np.asarray(points) @ transform[:3,:3].T + transform[:3,3]
