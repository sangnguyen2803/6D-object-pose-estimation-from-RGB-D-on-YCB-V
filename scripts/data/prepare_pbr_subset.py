"""Read selected PBR frames through HTTP byte ranges; verify per-member ZIP CRC.

Full archive LFS SHA256 is recorded, not claimed verified without full download.
"""

import argparse
import concurrent.futures
import hashlib
import json
import struct
import sys
import threading
import time
import zipfile
import zlib
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.utils.common import resolve, write_json

REVISION = "5c2c4aa229800355648cd268040aa814f8dc94f0"
URL = f"https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{REVISION}/ycbv_train_pbr.zip"
DOWNLOAD_URL = None
LOCAL = threading.local()


def get_range(start, end):
    global DOWNLOAD_URL
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(5):
        try:
            response = LOCAL.session.get(
                DOWNLOAD_URL or URL,
                headers={"Range": f"bytes={start}-{end}"},
                params=None
                if DOWNLOAD_URL
                else {"range_start": start, "range_end": end, "attempt": attempt},
                timeout=(30, 120),
            )
            response.raise_for_status()
            if response.status_code != 206 or not response.headers.get(
                "Content-Range", ""
            ).startswith(f"bytes {start}-"):
                raise IOError("Server did not return the requested byte range")
            if len(response.content) != end - start + 1:
                raise IOError("Incomplete byte range")
            DOWNLOAD_URL = response.url
            return response.content
        except (requests.RequestException, IOError):
            DOWNLOAD_URL = None
            if attempt == 4:
                raise
            time.sleep(2**attempt)


class RangeFile:
    def __init__(self, size):
        self.size = size
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = (
            offset
            if whence == 0
            else (self.position + offset if whence == 1 else self.size + offset)
        )
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(
            self.size - self.position, size if size >= 0 else self.size - self.position
        )
        if size <= 0:
            return b""
        data = get_range(self.position, self.position + size - 1)
        self.position += len(data)
        return data

    def seekable(self):
        return True


def read_member(info):
    buffer = get_range(
        info.header_offset,
        info.header_offset
        + info.compress_size
        + 30
        + len(info.filename.encode("utf-8"))
        + len(info.extra)
        + 255,
    )
    values = struct.unpack("<4s5H3I2H", buffer[:30])
    if values[0] != b"PK\x03\x04":
        raise IOError("Invalid local ZIP header")
    offset = info.header_offset + 30 + values[-2] + values[-1]
    begin = offset - info.header_offset
    compressed = buffer[begin : begin + info.compress_size]
    if len(compressed) != info.compress_size:
        compressed = get_range(offset, offset + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError("Unsupported ZIP compression")
    if len(data) != info.file_size or zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise IOError("Member CRC mismatch")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-scene", type=int, default=30)
    parser.add_argument("--val-per-scene", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    response = requests.get(
        f"https://huggingface.co/api/datasets/bop-benchmark/ycbv/tree/{REVISION}",
        timeout=30,
    )
    response.raise_for_status()
    entry = next(e for e in response.json() if e["path"] == "ycbv_train_pbr.zip")
    print("Reading ZIP central directory through ranges", flush=True)
    archive = zipfile.ZipFile(RangeFile(entry["size"]))
    files = {item.filename: item for item in archive.infolist() if not item.is_dir()}
    root = resolve("data/ycbv/raw/ycbv").resolve()
    records = []
    groups = {
        "train": [f"{i:06d}" for i in range(10)],
        "val": ["000045", "000046", "000047"],
        "holdout": ["000040", "000041", "000042"],
    }
    selected = {}
    for split, scenes in groups.items():
        for scene in scenes:
            directory = root / "train_pbr" / scene
            metadata = {}
            for name in ["scene_camera.json", "scene_gt.json", "scene_gt_info.json"]:
                key = f"train_pbr/{scene}/{name}"
                if key not in files:
                    matches = [
                        name
                        for name in files
                        if name.endswith("/" + scene + "/" + key.split("/")[-1])
                    ]
                    if len(matches) != 1:
                        raise ValueError(f"Cannot locate {key}")
                    key = matches[0]
                data = read_member(files[key])
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                metadata[name] = json.loads(data)
            count = args.train_per_scene if split == "train" else args.val_per_scene
            keys = sorted(metadata["scene_gt.json"], key=int)
            chosen = [
                keys[int(i)]
                for i in __import__("numpy").linspace(
                    0, len(keys) - 1, min(count, len(keys)), dtype=int
                )
            ]
            selected[scene] = set(int(key) for key in chosen)
            for name in metadata:
                write_json(
                    directory / name, {key: metadata[name][key] for key in chosen}
                )
    wanted = []
    for name, info in files.items():
        parts = Path(name).parts
        try:
            index = parts.index("train_pbr")
        except ValueError:
            continue
        if len(parts) < index + 4:
            continue
        scene, kind, filename = parts[index + 1 : index + 4]
        if scene not in selected or kind not in ["rgb", "depth", "mask", "mask_visib"]:
            continue
        image = int(Path(filename).stem.split("_")[0])
        if image in selected[scene]:
            wanted.append((info, root / Path(*parts[index:])))
    print(
        "Selected",
        sum(map(len, selected.values())),
        "frames;",
        len(wanted),
        "assets",
        flush=True,
    )
    for _, target in wanted:
        if not target.absolute().is_relative_to(root):
            raise ValueError("Unsafe archive member path")
        target.parent.mkdir(parents=True, exist_ok=True)

    def save(item):
        info, target = item
        canonical = target.resolve()
        if not canonical.is_relative_to(root):
            raise ValueError(f"Unsafe ZIP path: {canonical} outside {root}")
        if target.exists() and target.stat().st_size == info.file_size:
            data = target.read_bytes()
            if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
                data = read_member(info)
                target.write_bytes(data)
        else:
            data = read_member(info)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return {
            "path": str(target.relative_to(resolve("."))),
            "bytes": len(data),
            "crc32": info.CRC,
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, record in enumerate(pool.map(save, wanted)):
            records.append(record)
            if (index + 1) % 100 == 0:
                print("Assets", index + 1, "/", len(wanted), flush=True)
    write_json(
        resolve("reports/ycbv/research/pbr_subset_manifest.json"),
        {
            "revision": REVISION,
            "url": URL,
            "archive_bytes": entry["size"],
            "archive_expected_sha256": entry["lfs"]["oid"],
            "full_archive_sha256_verified": False,
            "member_crc_verified": True,
            "groups": groups,
            "selected_images": {key: sorted(value) for key, value in selected.items()},
            "files": records,
        },
    )
    print("PBR subset complete", flush=True)


if __name__ == "__main__":
    main()
