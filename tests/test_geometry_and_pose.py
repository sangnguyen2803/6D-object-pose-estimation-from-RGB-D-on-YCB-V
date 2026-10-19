import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from clutterpick.geometry import backproject,voxel_downsample,transform_points,instance_cloud,write_ply
from clutterpick.pose import rigid_fit,cube_rotations


def test_backprojection_known_metric_depth_and_invalid_pixels():
    depth = np.array([[0.,1.],[2.,np.nan]])
    k = np.array([[2.,0.,0.],[0.,4.,0.],[0.,0.,1.]])
    points,pixels = backproject(depth,k)
    np.testing.assert_allclose(points,[[.5,0.,1.],[0.,.5,2.]])
    np.testing.assert_array_equal(pixels,[[0,1],[1,0]])


def test_mask_prevents_background_depth_contamination():
    points,_ = backproject(np.array([[.5,3.]]),np.eye(3),np.array([[True,False]]))
    np.testing.assert_allclose(points,[[0,0,.5]])


def test_backprojection_with_skew_reprojects_to_original_pixels():
    depth = np.array([[1.,2.],[3.,4.]])
    k = np.array([[200.,25.,.5],[0.,150.,.5],[0.,0.,1.]])
    points,pixels = backproject(depth,k)
    projected = points@k.T
    np.testing.assert_allclose(projected[:,:2]/projected[:,2,None],pixels[:,::-1],atol=1e-12)
    np.testing.assert_array_equal(points[:,2],depth[pixels[:,0],pixels[:,1]])


@pytest.mark.parametrize("k",[
    np.eye(2),np.diag([0.,1.,1.]),np.diag([1.,-1.,1.]),
    np.diag([np.nan,1.,1.]),np.diag([1.,np.inf,1.]),
    np.array([[1.,0.,0.],[.1,1.,0.],[0.,0.,1.]]),
    np.diag([1.,1.,2.]),
])
def test_backprojection_rejects_invalid_camera_intrinsics(k):
    with pytest.raises(ValueError,match="intrinsics"):
        backproject(np.ones((2,2)),k)


def test_backprojection_rejects_multichannel_depth():
    with pytest.raises(ValueError,match="two-dimensional"):
        backproject(np.ones((2,2,3)),np.eye(3))


def test_voxel_keeps_colors_aligned():
    points = np.array([[.001,0,0],[.002,0,0],[.020,0,0]])
    colors = np.array([[255,0,0],[0,255,0],[0,0,255]])
    p,c = voxel_downsample(points,.01,colors)
    np.testing.assert_array_equal(c,[[255,0,0],[0,0,255]])


@pytest.mark.parametrize("mask_dtype",[bool,np.uint8])
def test_instance_depth_diagnostics_count_nonfinite_and_nonpositive_pixels(mask_dtype):
    depth = np.array([[0.,np.nan,np.inf],[-np.inf,-1.,1.]])
    mask = np.ones(depth.shape,dtype=mask_dtype)
    if mask_dtype==np.uint8:
        mask *= 255
    rgb = np.full((*depth.shape,3),120,dtype=np.uint8)
    config = {"geometry":{"boundary_erosion_pixels":0,"depth_min_m":.1,"depth_max_m":3.,"voxel_size_m":0}}
    points,colors,diagnostics = instance_cloud(depth,np.eye(3),mask,rgb,config)
    np.testing.assert_allclose(points,[[2.,1.,1.]])
    np.testing.assert_array_equal(colors,[[120,120,120]])
    assert diagnostics=={"visible_mask_pixels":6,"valid_points_before_filter":1,
                         "retained_points":1,"depth_missing_fraction":5/6}


def test_instance_depth_diagnostics_ignore_missing_pixels_outside_empty_mask():
    depth = np.full((2,2),np.nan)
    config = {"geometry":{"boundary_erosion_pixels":0,"depth_min_m":.1,"depth_max_m":3.,"voxel_size_m":0}}
    points,colors,diagnostics = instance_cloud(depth,np.eye(3),np.zeros((2,2),dtype=bool),
                                               np.zeros((2,2,3),dtype=np.uint8),config)
    assert points.shape==colors.shape==(0,3)
    assert diagnostics["visible_mask_pixels"]==diagnostics["depth_missing_fraction"]==0


def test_ply_roundtrip_preserves_vertices_and_colors(tmp_path):
    import trimesh
    points = np.array([[.1,-.2,.3],[1.,2.,3.]])
    colors = np.array([[255,0,30],[10,120,240]],dtype=np.uint8)
    path = tmp_path/"nested"/"cloud.ply"
    write_ply(path,points,colors.tolist())
    cloud = trimesh.load(path,process=False)
    np.testing.assert_allclose(cloud.vertices,points,atol=1e-6)
    np.testing.assert_array_equal(cloud.colors[:,:3],colors)


@pytest.mark.parametrize("points,colors",[
    ([[0,0,1],[1,1,1]],[[255,0,0]]),
    ([[0,0,1]],[[255,0,0],[0,255,0]]),
    ([[0,0,1]],[[255,0]]),
    ([[0,0,1]],[[256,0,0]]),
    ([[0,0,1]],[[-1,0,0]]),
    ([[0,0,1]],[[np.nan,0,0]]),
    ([[np.inf,0,1]],None),
    ([[0,np.nan,1]],None),
])
def test_invalid_ply_data_does_not_overwrite_existing_cloud(tmp_path,points,colors):
    path = tmp_path/"cloud.ply"
    path.write_text("existing cloud",encoding="ascii")
    with pytest.raises(ValueError):
        write_ply(path,points,colors)
    assert path.read_text(encoding="ascii")=="existing cloud"


def test_empty_ply_contains_no_vertex_rows(tmp_path):
    path = tmp_path/"empty.ply"
    write_ply(path,np.empty((0,3)))
    header,body = path.read_text(encoding="ascii").split("end_header\n")
    assert "element vertex 0\n" in header
    assert body==""


def test_rigid_registration_recovers_known_transform_without_reflection():
    rng = np.random.default_rng(17)
    source = rng.normal(size=(50,3))
    rotation = Rotation.from_euler("xyz",[.3,-.6,.8]).as_matrix()
    target = source@rotation.T+np.array([.2,-.1,.7])
    pose = rigid_fit(source,target)
    np.testing.assert_allclose(transform_points(source,pose),target,atol=1e-10)
    assert np.linalg.det(pose[:3,:3])>0.999


def test_pose_hypotheses_are_proper_rotations():
    rotations = cube_rotations()
    assert len(rotations)==24
    assert all(np.linalg.det(r)==1 for r in rotations)
