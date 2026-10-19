# Chạy các công cụ

Mở terminal ở thư mục gốc dự án. Trên Windows, dùng `.\.venv\Scripts\python.exe` thay cho `python` nếu chưa kích hoạt môi trường.

Lệnh chính nằm ở `main.py`:

```powershell
python main.py --help
python main.py index
python main.py audit
python main.py infer --frame-key ycbv_test/000048/1074
python main.py evaluate
python main.py train --epochs 8
```

## Folder và công việc

| Folder | Dùng khi nào | Script tiêu biểu |
|---|---|---|
| `data/` | Tải dataset, checkpoint và kiểm tra dữ liệu | `prepare_ycbv.py`, `prepare_pbr_subset.py`, `audit_local.py` |
| `training/` | Tạo cache và train các module bổ sung | `cache_point_training.py`, `train_fusion.py`, `train_point_network.py`, `point_cache_baselines.py` |
| `inference/` | Chạy RGB-D riêng, refinement replay và xem kết quả 3D | `infer_rgbd.py`, `replay_refinement.py`, `visualize_instances.py` |
| `evaluation/` | Chạy thí nghiệm và đo tốc độ/chất lượng | `run_experiments.py`, `evaluate_research.py`, `benchmark_architecture.py` |
| `reports/` | Tổng hợp artifact thành báo cáo | `build_report.py`, `build_extension_report.py`, `build_research_report.py`, `build_architecture_report.py`, `update_hybrid_docs.py` |
| `validation/` | Kiểm tra kết quả và các biến thể pipeline | Các file `validate_*.py` |
| `workflows/` | Chạy chuỗi bước bằng PowerShell | `run_all.ps1`, `run_extension.ps1`, `run_research.ps1`, `run_architecture.ps1`, `finalize_*.ps1` |
| `setup/` | Cài môi trường Python | `bootstrap.ps1` |
| `legacy/` | Audit dữ liệu IndustryShapes của hướng cũ | `inspect_industryshapes.py`, `check_geometry.py` |

Ví dụ:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup/bootstrap.ps1
python scripts/data/prepare_ycbv.py --help
python scripts/inference/infer_rgbd.py --help
python scripts/inference/visualize_instances.py artifacts/ycbv/demo_rgbd/ycbv_test_000048_1074
python scripts/evaluation/benchmark_architecture.py --help
python scripts/validation/validate_refinement_synthetic.py --help
powershell -ExecutionPolicy Bypass -File scripts/workflows/run_all.ps1
```

Các workflow có thể tải dữ liệu, train hoặc chạy lại benchmark; dùng lệnh riêng khi chỉ cần một tác vụ. `legacy/` tiếp tục đọc/ghi dữ liệu audit tại thư mục `audit/` ở gốc dự án.
