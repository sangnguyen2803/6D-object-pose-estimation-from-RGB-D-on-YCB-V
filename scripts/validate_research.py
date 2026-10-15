import hashlib,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from clutterpick.common import resolve,read_json,write_json,load_config
from validate_ycbv import validate

def main():
    cfg=load_config('configs/research.yaml'); splits=read_json(resolve(cfg['data']['splits']))
    index=read_json(resolve(cfg['data']['index']))
    assert {name:len(splits[name]) for name in ['train','val','holdout','test']}=={'train':300,'val':60,'holdout':60,'test':900}
    scenes={name:{(index['frames'][i]['group'],index['frames'][i]['scene_id']) for i in splits[name]}
            for name in ['train','val','holdout','test']}
    for first in ['train','val','holdout','test']:
        for second in ['train','val','holdout','test']:
            if first!=second:
                assert not set(splits[first])&set(splits[second])
                assert not scenes[first]&scenes[second]
    audit=read_json(resolve('reports/ycbv/research/data_audit.json')); assert not audit['errors']
    data=read_json(resolve('reports/ycbv/research/pbr_subset_manifest.json'))
    assert data['member_crc_verified'] and not data['full_archive_sha256_verified']
    ablations=read_json(resolve('reports/ycbv/research/ablation.json'))
    assert set(ablations)==set(cfg['research']['variants'])
    cohorts=[value['test']['frame_keys'] for value in ablations.values()]
    assert all(keys==cohorts[0] for keys in cohorts)
    checks={}
    for name in ablations:
        payload=torch.load(resolve(f'artifacts/ycbv/research/fusion/{name}/best.pt'),map_location='cpu',weights_only=False)
        history=read_json(resolve(f'artifacts/ycbv/research/fusion/{name}/history.json'))
        assert payload['epoch']==max(history,key=lambda row:row['val_mask_AP'])['epoch']
        assert hashlib.sha256(resolve(f'artifacts/ycbv/research/fusion/{name}/best.pt').read_bytes()).hexdigest()==ablations[name]['checkpoint_sha256']
        assert not payload['trained_on_test']
        assert not set(payload['train_keys'])&set(payload['val_keys'])
        assert not set(payload['train_keys'])&set(cohorts[0])
        checks[name]=validate(f'artifacts/ycbv/research/evaluation/{name}/pose',24)
    point=torch.load(resolve('artifacts/ycbv/research/point_network/best.pt'),map_location='cpu',weights_only=False)
    assert not point['trained_on_test']
    assert not set(point['train_frame_keys'])&set(point['validation_frame_keys'])
    point_cache=read_json(resolve('data/ycbv/point_training/index.json'))
    for row in point_cache['records']:
        assert row['split'] in ['train','val','holdout']
        assert row['frame_key'] in {index['frames'][i]['key'] for i in splits[row['split']]}
    holdout=read_json(resolve('reports/ycbv/research/point_holdout.json'))
    assert holdout['fixed_validation_threshold']==point['threshold']
    real=read_json(resolve('reports/ycbv/research/point_real_comparison.json'))
    assert real['paired']['fail_to_pass']-real['paired']['pass_to_fail']==(
        real['learned_3d']['success_counts']['ADD_or_S_success_0_1d']-real['original_2d_mask']['success_counts']['ADD_or_S_success_0_1d'])
    combined=read_json(resolve('reports/ycbv/research/combined_pose.json'))
    paired=combined['paired_vs_gated_query_without_point_filter']
    assert paired['fail_to_pass']-paired['pass_to_fail']==combined['success_counts']['ADD_or_S_success_0_1d']-ablations['gated_query']['pose']['success_counts']['ADD_or_S_success_0_1d']
    checks['point_baseline']=validate('artifacts/ycbv/research/evaluation/point_baseline',24)
    checks['learned_3d']=validate('artifacts/ycbv/research/evaluation/point_network/pose',24)
    checks['combined']=validate('artifacts/ycbv/research/evaluation/combined/pose',24)
    checks['demo']=validate('artifacts/ycbv/research/demo',1)
    cli=resolve('artifacts/ycbv/research/cli_demo')
    if (cli/'run_summary.json').exists(): checks['cli_demo']=validate(cli,1)
    external=read_json(resolve('reports/ycbv/research/external_input_validation.json'))
    assert external['passed'] and external['gt_instances']==0 and external['masks_and_filtered_clouds_equal_without_GT']
    checks['external_demo']=validate('artifacts/ycbv/research/external_demo',1)
    report=resolve('reports/ycbv/research')
    write_json(report/'delivery_validation.json',{'passed':True,'checks':checks,'train':300,'val':60,'holdout':60})
    files=[]
    for name in ['src/clutterpick','scripts','configs','tests','docs','reports/ycbv/research']:
        files.extend(path for path in resolve(name).rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.name!='artifact_manifest.json')
    files.extend(resolve(name) for name in ['README.md','PROJECT_SUMMARY_VI.md','pyproject.toml'])
    files.extend(resolve(f'artifacts/ycbv/research/fusion/{name}/best.pt') for name in ablations)
    files.append(resolve('artifacts/ycbv/research/point_network/best.pt'))
    files.extend(resolve(name) for name in ['data/ycbv/research_splits.json','data/ycbv/research_index.json','data/ycbv/point_training/index.json'])
    for name in list(ablations)+['point_baseline','point_network','combined']:
        directory=resolve(f'artifacts/ycbv/research/evaluation/{name}')
        if name!='point_baseline': directory=directory/'pose'
        files.extend(directory/file for file in ['run_summary.json','bop_predictions.csv'])
    demo=resolve('artifacts/ycbv/research/demo/ycbv_test_000048_1074')
    files.extend(demo/file for file in ['poses.json','detection.json','final_visualization.png','instances_3d.png','instances_3d.html'] if (demo/file).exists())
    write_json(report/'artifact_manifest.json',{str(path.relative_to(resolve('.'))):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    print(checks)
if __name__=='__main__': main()

