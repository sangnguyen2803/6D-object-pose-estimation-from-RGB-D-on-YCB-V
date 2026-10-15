import numpy as np
from PIL import Image
from clutterpick.data import BOPDetectionDataset,select_frames,load_rgbd


def test_multiple_instances_same_class_remain_separate(tmp_path):
    rgb = tmp_path/"rgb.png"
    Image.fromarray(np.zeros((10,20,3),dtype=np.uint8)).save(rgb)
    objects = []
    for i,offset in enumerate([1,11]):
        mask = np.zeros((10,20),dtype=np.uint8)
        mask[2:8,offset:offset+5]=255
        path = tmp_path/f"{i}.png"; Image.fromarray(mask).save(path)
        objects.append({"obj_id":3,"mask_visib":str(path),"visib_fract":1})
    _,target = BOPDetectionDataset([{"id":0,"rgb":str(rgb),"instances":objects}])[0]
    assert target["labels"].tolist()==[3,3]
    assert target["masks"].shape==(2,10,20)
    assert target["boxes"].tolist()==[[1,2,6,8],[11,2,16,8]]


def test_depth_scale_is_applied_per_frame(tmp_path):
    rgb,depth = tmp_path/"rgb.png",tmp_path/"depth.png"
    Image.fromarray(np.zeros((2,2,3),dtype=np.uint8)).save(rgb)
    Image.fromarray(np.full((2,2),5000,dtype=np.uint16)).save(depth)
    _,z,_ = load_rgbd({"rgb":str(rgb),"depth":str(depth),"depth_scale_mm":.1,"K":np.eye(3).ravel().tolist()})
    np.testing.assert_allclose(z,.5)


def test_pilot_sampling_covers_scenes_instead_of_first_class_only():
    frames = [{"id":i,"group":"train","scene_id":str(i//10)} for i in range(30)]
    selected = select_frames({"frames":frames},list(range(30)),6)
    assert {f["scene_id"] for f in selected}=={"0","1","2"}


def test_pbr_holdout_scene_is_excluded_from_training_and_overlap_rejected(tmp_path):
    import pytest
    from clutterpick.data import build_index
    from clutterpick.common import write_json
    for group,scene in [('train_pbr','000000'),('train_pbr','000040'),('train_pbr','000045'),('test','000048')]:
        directory=tmp_path/group/scene
        for folder,name in [('rgb','000000.png'),('depth','000000.png'),('mask','000000_000000.png'),('mask_visib','000000_000000.png')]:
            path=directory/folder/name; path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.ones((2,2),dtype=np.uint8)).save(path)
        write_json(directory/'scene_gt.json',{'0':[{'obj_id':1,'cam_R_m2c':np.eye(3).ravel().tolist(),'cam_t_m2c':[0,0,1000]}]})
        write_json(directory/'scene_camera.json',{'0':{'cam_K':np.eye(3).ravel().tolist(),'depth_scale':.1}})
        write_json(directory/'scene_gt_info.json',{'0':[{'visib_fract':1.}]})
    cfg={'data':{'name':'ycbv','root':str(tmp_path),'index':str(tmp_path/'index.json'),
                 'splits':str(tmp_path/'splits.json'),'index_report':str(tmp_path/'report.json'),
                 'revision':'fixture','object_ids':[1],'validation_scenes':[],
                 'pbr_validation_scenes':['000045'],'pbr_holdout_scenes':['000040']}}
    index,splits=build_index(cfg)
    assert {name:len(splits[name]) for name in ['train','val','holdout','test']}==dict.fromkeys(['train','val','holdout','test'],1)
    assert index['frames'][splits['holdout'][0]]['scene_id']=='000040'
    cfg['data']['pbr_holdout_scenes']=['000045']
    with pytest.raises(ValueError,match='overlap'):
        build_index(cfg)


def test_copy_paste_updates_visible_masks_and_boxes(tmp_path):
    import torch
    np.random.seed(9); torch.manual_seed(9)
    rgb = np.zeros((80,100,3),dtype=np.uint8)
    objects = []
    for i in range(5):
        mask = np.zeros((80,100),dtype=np.uint8)
        mask[25:40,5+i*18:18+i*18] = 255
        rgb[mask>0] = [40+i*30,120,150]
        path = tmp_path/f'mask_{i}.png'; Image.fromarray(mask).save(path)
        objects.append({'obj_id':i+1,'mask_visib':str(path),'visib_fract':1.})
    path = tmp_path/'rgb.png'; Image.fromarray(rgb).save(path)
    frame = {'id':0,'rgb':str(path),'instances':objects}
    image,target = BOPDetectionDataset([frame],training=True,augmentation={'copy_paste_probability':1.})[0]
    assert image.shape==(3,80,100)
    assert target['masks'].sum(dim=0).max()<=1
    for mask,box in zip(target['masks'].numpy(),target['boxes'].numpy()):
        y,x = np.nonzero(mask)
        assert len(x)>=10
        np.testing.assert_array_equal(box,[x.min(),y.min(),x.max()+1,y.max()+1])
