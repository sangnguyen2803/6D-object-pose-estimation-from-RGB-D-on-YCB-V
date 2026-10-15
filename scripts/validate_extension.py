"""Check paired extension cohort, GT-free provenance, bounded updates and hashes."""
import sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from clutterpick.common import resolve,read_json,write_json
from clutterpick.refinement import bounded
from clutterpick.reporting import summarize
from validate_ycbv import validate

def main():
    root=resolve('reports/ycbv/extension')
    result=read_json(root/'refinement_comparison.json')
    assert result['frames']==900 and result['enhanced']['gt_instances']==4125
    base=read_json(resolve('artifacts/ycbv/benchmark_full/run_summary.json'))
    rows=read_json(resolve('artifacts/ycbv/extension/replay/run_summary.json'))
    assert [row['frame_key'] for row in rows]==[row['frame_key'] for row in base]
    checks={'replay':validate('artifacts/ycbv/extension/replay',900),
            'demo':validate('artifacts/ycbv/extension/demo',1)}
    assert summarize(rows)['success_counts']==result['enhanced']['success_counts']
    assert result['enhanced']['success_counts']['ADD_or_S_success_0_1d']-result['baseline']['success_counts']['ADD_or_S_success_0_1d']==result['gains']-result['regressions']
    for row in rows:
        payload=read_json(Path(row['output'])/'poses.json')
        assert payload['timing_scope']=='geometry/refinement only'
        for pose in payload['instances']:
            assert pose['pose_quality']['calibrated_probability'] is False
            if pose['depth_refinement_accepted']:
                assert bounded(np.asarray(pose['T_camera_object']),np.asarray(pose['T_camera_object_before_depth_refinement']),result['config']['pose'])[0]
            fit=pose['refinement_diagnostics']['after']
            assert pose['observed_trimmed_rmse_m']==fit['rmse_m']
    first=read_json(root/'synthetic_validation.json'); second=read_json(root/'synthetic_holdout.json')
    assert first['seed']!=second['seed'] and first['cases']==second['cases']==63
    assert not first['uses_real_test_gt'] and not second['uses_real_test_gt']
    write_json(root/'delivery_validation.json',{'passed':True,'checks':checks,'synthetic_cases':126,
               'timing_scope':'cached replay is not end-to-end latency'})
    files=[]
    for directory in ['src/clutterpick','scripts','configs','tests','docs','reports/ycbv/extension']:
        files.extend(path for path in resolve(directory).rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.name!='artifact_manifest.json')
    files.extend(resolve(name) for name in ['README.md','PROJECT_SUMMARY_VI.md','pyproject.toml',
         'artifacts/ycbv/extension/replay/run_summary.json','artifacts/ycbv/extension/replay/bop_predictions.csv'])
    write_json(root/'artifact_manifest.json',{str(path.relative_to(resolve('.'))):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    print(checks)
if __name__=='__main__': main()

