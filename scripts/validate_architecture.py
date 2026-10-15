"""Verify comparable cohorts, unchanged weights, rendering, outputs, and delivery hashes."""
import hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from clutterpick.common import resolve,read_json,write_json
from validate_ycbv import validate

def main():
    root=resolve('reports/ycbv/architecture'); summary=read_json(root/'measured_summary.json')
    report=read_json(root/summary['source_report'])
    variants=report['variants']; baseline=variants['baseline']; optimized=variants['optimized']
    assert baseline['frames']==optimized['frames'] and len(baseline['frames'])==96
    assert baseline['config']['detector']==optimized['config']['detector']
    for key in ['coarse_checkpoint','refiner_checkpoint','coarse_iterations','refiner_iterations','trim_fraction','huber_delta_m']:
        assert baseline['config']['pose'][key]==optimized['config']['pose'][key]
    assert optimized['config']['pose']['resident_renderer'] and optimized['config']['pose']['cache_model_trees']
    assert optimized['renderer']['context_mesh_additions']==0, 'Meshes must remain resident after cohort warm-up'
    assert read_json(root/'renderer_validation.json')['passed']
    checks={}
    for name,value in variants.items():
        for repeat in range(report['repetitions']):
            directory=Path(value['output'])/f'repeat_{repeat}'
            checks[f'{name}_{repeat}']=validate(directory,96)
            rows=read_json(directory/'run_summary.json')
            assert [row['frame_key'] for row in rows]==value['frames']
            assert sum(row['gt_instances'] for row in rows)==value['repeats'][repeat]['gt_instances']
    checks['demo']=validate('artifacts/ycbv/architecture/demo',1)
    write_json(root/'delivery_validation.json',{'passed':True,'checks':checks,'cohort_frames':96,
               'scope':'checks consistency and provenance; does not assert SOTA or improvement in every metric'})
    files=[]
    for directory in ['src/clutterpick','scripts','configs','tests','docs','reports/ycbv/architecture']:
        files.extend(path for path in resolve(directory).rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.name!='artifact_manifest.json')
    files.extend(resolve(name) for name in ['README.md','PROJECT_SUMMARY_VI.md','pyproject.toml'])
    for value in variants.values():
        for repeat in range(report['repetitions']):
            files.extend(Path(value['output'])/f'repeat_{repeat}'/name for name in ['run_summary.json','bop_predictions.csv'])
    write_json(root/'artifact_manifest.json',{str(path.relative_to(resolve('.'))):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    print(checks)

if __name__=='__main__': main()
