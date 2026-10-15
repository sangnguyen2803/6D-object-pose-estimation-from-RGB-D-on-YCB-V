"""Real OpenGL equivalence check for visible objects and hidden resident meshes."""
import copy,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from clutterpick.common import load_config,resolve,write_json
from clutterpick.cosypose_backend import TexturedRenderer

def main():
    base=load_config('configs/enhanced.yaml'); optimized=copy.deepcopy(base)
    optimized['pose']['resident_renderer']=True
    base['pose']['renderer_validation_depth']=True
    optimized['pose']['renderer_validation_depth']=True
    labels=[14,6,19,1,19,14,6]
    poses=np.tile(np.eye(4),(len(labels),1,1))
    poses[:,:3,:3]=Rotation.from_euler('xyz',[[20,10,i*13] for i in range(len(labels))],degrees=True).as_matrix()
    poses[:,2,3]=.8
    poses[:,0,3]=np.linspace(-.03,.03,len(labels))
    k=np.tile([[600,0,160],[0,600,120],[0,0,1]],(len(labels),1,1))
    reference=TexturedRenderer(base,torch.device('cpu')); resident=TexturedRenderer(optimized,torch.device('cpu'))
    try:
        info=[{'name':f'obj_{obj:06d}'} for obj in labels]
        for label in sorted(set(labels)):
            name=f'obj_{label:06d}'
            x,y=reference.mesh(name),resident.mesh(name)
            print('CPU geometry',name,[(bool(np.array_equal(p.normals,q.normals)),float(np.max(np.abs(p.normals-q.normals))))
                  for p,q in zip(x.primitives,y.primitives)],flush=True)
        a=reference.render(info,torch.tensor(poses),torch.tensor(k))
        depth_a=reference.last_depths.copy()
        control=reference.render(info,torch.tensor(poses),torch.tensor(k))
        print('Legacy repeat max/nonzero',float((a-control).abs().max()),int(((a-control).abs()>0).sum()),flush=True)
        b=resident.render(info,torch.tensor(poses),torch.tensor(k))
        difference=(a-b).abs().numpy()
        from PIL import Image
        folder=resolve('artifacts/ycbv/architecture/render_equivalence'); folder.mkdir(parents=True,exist_ok=True)
        for i in range(len(labels)):
            for name,value in [('legacy',a[i]),('resident',b[i])]:
                Image.fromarray(np.rint(value.numpy().transpose(1,2,0)*255).astype(np.uint8)).save(folder/f'{i}_{name}.png')
        print({'max_difference':float(difference.max()),'mean_difference':float(difference.mean()),
               'per_case_max':difference.reshape(len(labels),-1).max(1).tolist(),
               'per_case_nonzero':(difference.reshape(len(labels),-1)>0).sum(1).tolist()},flush=True)
        depth_b=resident.last_depths
        np.testing.assert_array_equal(depth_a>0,depth_b>0)
        np.testing.assert_allclose(depth_a,depth_b,rtol=0,atol=1e-6)
        np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=0,atol=1/255+1e-7)
        assert float((difference>0).mean())<.001
        assert resident.cache_stats['context_mesh_additions']==len(set(labels))
        assert all(not mesh.is_visible for mesh in resident.meshes.values())
        result={'passed':True,'cases':len(labels),'max_rgb_difference':float(difference.max()),
                'rgb_different_channel_fraction':float((difference>0).mean()),
                'legacy_repeat_max_rgb_difference':float((a-control).abs().max()),
                'silhouettes_exactly_equal':True,'max_depth_difference_m':float(np.max(np.abs(depth_a-depth_b))),
                'rgb_tolerance':'one uint8 intensity level; <=0.1% channels differ; legacy repeat has the same quantization variation',
                'legacy':reference.cache_stats,'resident':resident.cache_stats,
                'scope':'same CAD, lights, poses and camera; hidden cached meshes must not affect pixels'}
        write_json(resolve('reports/ycbv/architecture/renderer_validation.json'),result)
        print(result)
    finally:
        reference.close(); resident.close()

if __name__=='__main__': main()
