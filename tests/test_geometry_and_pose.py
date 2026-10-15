import numpy as np
from scipy.spatial.transform import Rotation
from clutterpick.geometry import backproject,voxel_downsample,transform_points
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


def test_voxel_keeps_colors_aligned():
    points = np.array([[.001,0,0],[.002,0,0],[.020,0,0]])
    colors = np.array([[255,0,0],[0,255,0],[0,0,255]])
    p,c = voxel_downsample(points,.01,colors)
    np.testing.assert_array_equal(c,[[255,0,0],[0,0,255]])


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
