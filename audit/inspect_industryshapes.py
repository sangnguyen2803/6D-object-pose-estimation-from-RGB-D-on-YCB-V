"""Read remote BOP ZIP metadata and selected samples without full downloads.

Writes evidence for a feasibility audit; does not train or estimate poses.
Dependencies: requests, numpy, Pillow. Run from any directory.
"""
import collections
import concurrent.futures
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parent
BASE = 'https://huggingface.co/datasets/POSE-Lab/IndustryShapes'


def get(url, **kwargs):
    r = requests.get(url, timeout=60, **kwargs)
    r.raise_for_status()
    return r


class RemoteFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.blocks = []
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        start = self.pos
        end = self.size if n < 0 else min(self.size, start + n)
        if start >= end:
            return b''
        for a, b, payload in self.blocks:
            if a <= start and end <= b:
                self.pos = end
                return payload[start-a:end-a]
        # Coalesce ZIP local headers and following compressed data.
        stop = min(self.size, max(end, start + 65536))
        if stop - start > 20_000_000:
            parts = []
            while self.pos < end:
                parts.append(self.read(min(8_000_000, end-self.pos)))
            return b''.join(parts)
        r = get(self.url, headers={'Range': f'bytes={start}-{stop-1}'},
                params={'audit_range': f'{start}-{stop-1}'}, stream=True)
        expected = f'bytes {start}-{stop-1}/{self.size}'
        if r.status_code != 206 or r.headers.get('Content-Range') != expected:
            r.close()
            raise ValueError('Server did not honor requested byte range')
        payload = r.content
        if len(payload) != stop-start:
            raise ValueError('Truncated range')
        self.bytes_read += len(payload)
        self.blocks.append((start, stop, payload))
        self.pos = end
        return payload[:end-start]


def read_json(z, name):
    return json.loads(z.read(name))


def sample(z, prefix, image_id, gt, camera, info, tag):
    out = ROOT / 'samples' / tag
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{int(image_id):06d}'
    names = z.namelist()
    paths = {}
    for kind in ['rgb', 'depth', 'mask', 'mask_visib']:
        suffix = stem if kind in ['rgb', 'depth'] else stem + '_000000'
        path = next((n for n in names if n.startswith(prefix + kind + '/')
                     and Path(n).stem == suffix), None)
        if path is None:
            raise ValueError(f'Missing {kind} in {tag}')
        payload = z.read(path)
        (out / (kind + Path(path).suffix)).write_bytes(payload)
        paths[kind] = Image.open(io.BytesIO(payload))
    depth = np.asarray(paths['depth'])
    visible = np.asarray(paths['mask_visib']) > 0
    full = np.asarray(paths['mask']) > 0
    valid = visible & (depth > 0)
    c = camera[image_id]
    scale = c.get('depth_scale', 1)
    k = np.array(c['cam_K']).reshape(3, 3)
    v, u = np.nonzero(valid)
    zz = depth[v, u].astype(float) * scale / 1000
    points = np.column_stack(((u-k[0, 2])*zz/k[0, 0], (v-k[1, 2])*zz/k[1, 1], zz))
    colors = np.asarray(paths['rgb'].convert('RGB'))[v, u]
    with (out / 'visible_points_gt_mask.ply').open('w', encoding='ascii') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n' % len(points))
        for p, rgb in zip(points, colors):
            f.write('%.6f %.6f %.6f %d %d %d\n' % (*p, *rgb))
    rgb_image = paths['rgb'].convert('RGB')
    overlay = np.array(rgb_image).copy()
    overlay[visible] = (overlay[visible] * 0.45 + np.array([255, 40, 40]) * 0.55).astype('uint8')
    Image.fromarray(overlay).save(out / 'mask_overlay.png')
    result = {
        'tag': tag, 'source_prefix': prefix, 'image_id': image_id,
        'image_dimensions': {n: list(im.size) for n, im in paths.items()},
        'depth_mode': paths['depth'].mode, 'depth_dtype': str(depth.dtype),
        'depth_scale_mm': scale, 'visible_pixels': int(visible.sum()),
        'full_mask_pixels': int(full.sum()), 'visible_outside_full_pixels': int((visible & ~full).sum()),
        'valid_depth_visible_pixels': int(valid.sum()),
        'invalid_depth_visible_fraction': float((visible & (depth == 0)).sum()/max(visible.sum(), 1)),
        'visible_depth_m_percentiles': np.percentile(zz, [5, 50, 95]).tolist() if len(zz) else [],
        'camera': c, 'first_instance_gt': gt[image_id][0],
        'first_instance_gt_info': info[image_id][0] if info else None,
        'pointcloud_uses_ground_truth_mask': True,
    }
    (out / 'sample_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def inspect_archive(entry):
    path = entry['path']
    label = Path(path).stem
    remote = RemoteFile(BASE + '/resolve/main/' + path, entry['size'])
    with zipfile.ZipFile(remote) as z:
        names = z.namelist()
        listing = [{'path': x.filename, 'bytes': x.file_size, 'compressed_bytes': x.compress_size} for x in z.infolist()]
        (ROOT / (label + '_manifest.json')).write_text(json.dumps(listing), encoding='utf-8')
        summary = {'archive': path, 'compressed_bytes': entry['size'],
                   'uncompressed_bytes': sum(x.file_size for x in z.infolist()),
                   'entries': len(names), 'scenes': [], 'samples': []}
        gt_paths = sorted(n for n in names if n.endswith('/scene_gt.json'))
        summary['scene_count'] = len(gt_paths)
        summary['image_file_counts'] = {kind: sum('/' + kind + '/' in n and not n.endswith('/') for n in names)
                                        for kind in ['rgb', 'depth', 'mask', 'mask_visib']}
        if gt_paths or 'cad_models' in label:
            for gt_path in (gt_paths if 'PBR' not in label else gt_paths[:1]):
                prefix = gt_path[:-len('scene_gt.json')]
                gt = read_json(z, gt_path)
                cp = prefix + 'scene_camera.json'
                ip = prefix + 'scene_gt_info.json'
                camera = read_json(z, cp) if cp in names else {}
                info = read_json(z, ip) if ip in names else {}
                objcounts = collections.Counter(x['obj_id'] for objects in gt.values() for x in objects)
                visibility = [x.get('visib_fract') for objects in info.values() for x in objects if 'visib_fract' in x]
                rotations = [np.array(x['cam_R_m2c']).reshape(3,3) for objects in gt.values() for x in objects]
                frame_sets = {kind: {str(int(Path(n).stem)) for n in names
                                    if n.startswith(prefix+kind+'/') and Path(n).stem.isdigit()}
                              for kind in ['rgb', 'depth']}
                expected_masks = {f'{int(i):06d}_{j:06d}' for i, objects in gt.items() for j in range(len(objects))}
                mask_sets = {kind: {Path(n).stem for n in names if n.startswith(prefix+kind+'/') and not n.endswith('/')}
                             for kind in ['mask', 'mask_visib']}
                rotation_errors = [float(np.linalg.norm(r.T@r-np.eye(3))) for r in rotations]
                scene = {'prefix': prefix, 'frames': len(gt), 'instances': sum(map(len, gt.values())),
                         'per_object_instances': dict(objcounts), 'camera_frames': len(camera),
                         'depth_scales': sorted(set(c.get('depth_scale', 1) for c in camera.values())),
                         'multi_instance_frames': sum(len(a)>1 for a in gt.values()),
                         'repeated_class_frames': sum(len(set(x['obj_id'] for x in a))<len(a) for a in gt.values()),
                         'visibility_min_median_max': np.percentile(visibility,[0,50,100]).tolist() if visibility else [],
                         'max_rotation_orthogonality_error': max(rotation_errors, default=0),
                         'rotation_orthogonality_error_gt_0_001_count': sum(e>0.001 for e in rotation_errors),
                         'rotation_determinant_min_max': [min(map(np.linalg.det, rotations)), max(map(np.linalg.det, rotations))] if rotations else [],
                         'missing_rgb_for_gt': sorted(set(gt)-frame_sets['rgb']),
                         'missing_depth_for_gt': sorted(set(gt)-frame_sets['depth']),
                         'extra_rgb_without_gt_count': len(frame_sets['rgb']-set(gt)),
                         'extra_depth_without_gt_count': len(frame_sets['depth']-set(gt)),
                         'missing_camera_for_gt': sorted(set(gt)-set(camera)),
                         'missing_mask_count': len(expected_masks-mask_sets['mask']),
                         'missing_mask_visib_count': len(expected_masks-mask_sets['mask_visib']),
                         'extra_mask_count': len(mask_sets['mask']-expected_masks),
                         'visibility_bins': {str((lo,hi)): sum(lo <= v < hi for v in visibility) for lo,hi in [(0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.00001)]}}
                summary['scenes'].append(scene)
                # One decoded frame per real archive, preferably with multiple instances.
                if len(summary['samples']) == 0 and camera and info:
                    candidate = next((i for i,a in gt.items() if len(a)>1), next(iter(gt)))
                    summary['samples'].append(sample(z, prefix, candidate, gt, camera, info, label))
            for model_info in [n for n in names if n.endswith('models_info.json')]:
                summary['models_info'] = read_json(z, model_info)
            if 'cad_models' in label:
                cad = ROOT / 'cad' / label
                cad.mkdir(parents=True, exist_ok=True)
                for name in names:
                    if Path(name).suffix.lower() in ['.ply', '.obj', '.json', '.mtl', '.png', '.jpg', '.jpeg']:
                        (cad / Path(name).name).write_bytes(z.read(name))
        summary['downloaded_bytes'] = remote.bytes_read
    print(json.dumps({'archive': label, 'scenes': summary['scene_count'],
                      'frames': sum(x['frames'] for x in summary['scenes']),
                      'instances': sum(x['instances'] for x in summary['scenes']),
                      'downloaded_MB': round(remote.bytes_read/1e6, 2)}), flush=True)
    return summary


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    api = 'https://huggingface.co/api/datasets/POSE-Lab/IndustryShapes'
    metadata = get(api).json()
    entries = get(api + '/tree/main', params={'recursive': 'true', 'expand': 'false'}).json()
    (ROOT / 'hf_repository_manifest.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
    (ROOT / 'dataset_card.md').write_text(get(BASE + '/raw/main/README.md').text, encoding='utf-8')
    summary = {'dataset': BASE, 'dataset_revision': metadata.get('sha'),
               'audit_date_user_timezone': '2026-10-12',
               'scope': 'All archive directories; all GT/camera/info JSON for non-PBR archives; first-scene JSON for each PBR archive; one decoded frame per scene archive; CAD assets. Full pixel corpus and remaining PBR scene labels not validated.',
               'archives': [], 'errors': []}
    archives = [x for x in entries if x['type']=='file' and x['path'].startswith('bop_format/')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(inspect_archive, x): x['path'] for x in archives}
        for job in concurrent.futures.as_completed(jobs):
            try:
                summary['archives'].append(job.result())
            except Exception as e:
                error = {'archive': jobs[job], 'error': repr(e)}
                summary['errors'].append(error)
                print(json.dumps(error), flush=True)
            (ROOT / 'audit_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('DONE', ROOT / 'audit_summary.json', flush=True)
    return bool(summary['errors'])


if __name__ == '__main__':
    sys.exit(main())
