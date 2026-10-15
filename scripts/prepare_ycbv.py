"""Fetch pinned BOP19 data and published CosyPose weights with provenance."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = '5c2c4aa229800355648cd268040aa814f8dc94f0'
INRIA = 'https://www.rocq.inria.fr/archive_ylabbeprojectsdata/cosypose'
MODELS = {'detector': 'detector-bop-ycbv-synt+real--292971',
          'coarse': 'coarse-bop-ycbv-synt+real--822463',
          'refiner': 'refiner-bop-ycbv-synt+real--631598'}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(url, path, expected_size=None, expected_sha=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        if expected_sha is None or sha256(path) == expected_sha:
            return path
    partial = path.with_name(path.name + '.part')
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                              params={'resume_offset': offset}, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Invalid response range')
                received = offset if append else 0
                total = expected_size or (received + int(response.headers.get('Content-Length', 0)))
                progress = received
                with partial.open('ab' if append else 'wb') as stream:
                    for block in response.iter_content(4 * 1024 * 1024):
                        stream.write(block)
                        received += len(block)
                        if received - progress >= 64 * 1024 * 1024:
                            print(f'{path.name}: {received/1e6:.0f}/{total/1e6:.0f} MB', flush=True)
                            progress = received
                if total and partial.stat().st_size != total:
                    raise IOError('Incomplete transfer')
            if expected_sha and sha256(partial) != expected_sha:
                raise ValueError('SHA256 mismatch; refusing extraction')
            partial.replace(path)
            return path
        except (requests.RequestException, IOError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def extract(archive, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as payload:
        for entry in payload.infolist():
            target = (output / entry.filename).resolve()
            if not target.is_relative_to(output) or ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Unsafe ZIP member')
        payload.extractall(output)


def dataset_archive(entry):
    name = entry['path']
    archive = ROOT / 'data/ycbv/downloads' / name
    url = f'https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{REVISION}/{name}'
    fetch(url, archive, entry['size'], entry['lfs']['oid'])
    output = ROOT / 'data/ycbv/raw'
    if name != 'ycbv_base.zip':
        output /= 'ycbv'
    marker = archive.with_name(archive.name + '.extracted.json')
    if not marker.exists():
        extract(archive, output)
        marker.write_text(json.dumps({'revision': REVISION, 'sha256': entry['lfs']['oid']}), encoding='utf-8')
    print(f'Verified/extracted {name}', flush=True)
    return {'source': url, 'path': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size,
            'sha256': entry['lfs']['oid'], 'checksum_source': 'Hugging Face LFS'}


def model_file(item):
    role, model, filename = item
    url = f'{INRIA}/experiments/{model}/{filename}'
    path = ROOT / 'artifacts/pretrained/ycbv' / role / filename
    recorded = ROOT / 'reports/ycbv/download_manifest.json'
    expected_sha = None
    if recorded.exists():
        matches = [entry for entry in json.loads(recorded.read_text(encoding='utf-8'))['pretrained_models']
                   if entry['source'] == url]
        if matches:
            expected_sha = matches[0]['sha256']
    fetch(url, path, expected_sha=expected_sha)
    print(f'Downloaded {role}/{filename}', flush=True)
    return {'role': role, 'model_id': model, 'source': url, 'path': str(path.relative_to(ROOT)),
            'bytes': path.stat().st_size, 'sha256': sha256(path), 'checksum_source': 'recorded after HTTP transfer'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', nargs='+', choices=['base', 'models', 'test', 'train-pbr'],
                        default=['base', 'models', 'test'])
    args = parser.parse_args()
    response = requests.get(f'https://huggingface.co/api/datasets/bop-benchmark/ycbv/tree/{REVISION}', timeout=30)
    response.raise_for_status()
    wanted = {'base': 'ycbv_base.zip', 'models': 'ycbv_models.zip',
              'test': 'ycbv_test_bop19.zip', 'train-pbr': 'ycbv_train_pbr.zip'}
    entries = [entry for entry in response.json() if entry['path'] in {wanted[group] for group in args.groups}]
    if len(entries) != len(args.groups):
        raise ValueError('Requested archives unavailable at pinned revision')
    if 'train-pbr' in args.groups:
        import shutil
        pbr = next(entry for entry in entries if entry['path'] == wanted['train-pbr'])
        archive = ROOT / 'data/ycbv/downloads' / pbr['path']
        estimated_required = pbr['size'] * 2.2 - (archive.stat().st_size if archive.exists() else 0)
        marker = archive.with_name(archive.name + '.extracted.json')
        if not marker.exists() and shutil.disk_usage(ROOT).free < estimated_required:
            raise RuntimeError(f'Train PBR needs approximately {estimated_required/1e9:.1f} GB more free space including extraction')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        data = list(pool.map(dataset_archive, entries))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        models = list(pool.map(model_file, [(role, model, name) for role, model in MODELS.items()
                                           for name in ['config.yaml', 'checkpoint.pth.tar']]))
    source_revision = 'c90a04f434b1e89f02341cc03899eb63ea8facba'
    source_zip = ROOT / 'artifacts/cache/cosypose-source.zip'
    fetch(f'https://codeload.github.com/yannlabb/cosypose/zip/{source_revision}', source_zip,
          expected_sha='fed5f208acdfb178a79a06fa520c1aa452a45aea3d4c18d30f1c7934a0e258f0')
    source_root = ROOT / 'third_party'
    marker = source_root / 'cosypose_source.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))['revision'] != source_revision:
        raise ValueError('Existing CosyPose source differs from the pinned revision')
    if not marker.exists():
        extract(source_zip, source_root)
        source_directory = source_root / f'cosypose-{source_revision}'
        source_directory.rename(source_root / 'cosypose')
        (source_root / 'cosypose/local_data').mkdir(exist_ok=True)
        marker.write_text(json.dumps({'revision': source_revision, 'source': 'https://github.com/yannlabb/cosypose',
                                     'archive_sha256': sha256(source_zip)}, indent=2), encoding='utf-8')
    report = ROOT / 'reports/ycbv/download_manifest.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        previous = json.loads(report.read_text(encoding='utf-8'))
        if previous['dataset_revision'] != REVISION:
            raise ValueError('Existing data provenance differs from the pinned revision')
        archives = {record['path']:record for record in previous['archives']}
        archives.update({record['path']:record for record in data})
        data = list(archives.values())
    report.write_text(json.dumps({'dataset': 'YCB-V BOP', 'dataset_revision': REVISION,
                                  'archives': data, 'pretrained_models': models,
                                  'cosypose_source_revision': source_revision}, indent=2), encoding='utf-8')
    import torch
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    from clutterpick.common import load_config
    config = load_config()
    config['detector']['architecture'] = 'maskrcnn'
    detector_root = ROOT / 'artifacts/pretrained/ycbv/detector'
    raw = torch.load(detector_root / 'checkpoint.pth.tar', map_location='cpu', weights_only=False)
    torch.save({'model': raw['state_dict'], 'config': config,
                'source': 'CosyPose BOP20 YCB-V SYNT+REAL (unchanged weights)'}, detector_root / 'converted.pt')
    print(report, flush=True)


if __name__ == '__main__':
    main()
