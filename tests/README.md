# Kiểm thử

Tests được chia theo chức năng của mã nguồn:

| Folder | Nội dung |
|---|---|
| `data/` | Dataset, depth scale, sampling và chia train/val/test |
| `models/` | RGB-D fusion, mask head và point network |
| `geometry/` | Backprojection, PLY, rigid fit và ICP refinement |
| `inference/` | Pipeline, batching, provenance và CSV BOP |
| `evaluation/` | COCO AP và boundary AP |

Chạy từ thư mục gốc:

```powershell
python -m pytest -q
python -m pytest -q tests/geometry
python -m pytest -q tests/inference
```

Các test dùng dữ liệu nhỏ được tạo trong lúc chạy. File `test_entrypoints.py` kiểm tra đường dẫn gốc và việc mở CLI từ một thư mục khác.
