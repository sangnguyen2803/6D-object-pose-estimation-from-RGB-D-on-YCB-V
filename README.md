# YCBPose — RGB-D instance segmentation và 6D pose estimation

Dự án thực hiện **RGB-D → instance segmentation → point cloud từng instance → pose 6D của vật thể YCB đã biết**. Dùng pretrained Mask R-CNN và CosyPose, bổ sung đăng ký CAD với depth bằng ICP có giới hạn cập nhật. Tên package Python `clutterpick` được giữ để tương thích thư mục; tên dự án hiện tại là YCBPose.

Kết quả detector trên **900 ảnh / 4.125 GT instances / 21 lớp** của YCB-V BOP19: bbox AP@[.50:.95] **79,33%**, mask AP@[.50:.95] **74,27%**. Kết quả pose đầy đủ và thí nghiệm nằm trong [báo cáo đo thực tế](reports/ycbv/RESULTS_VI.md). Chỉ coi benchmark hoàn tất khi `complete_pose_test=true` trong [measured_summary.json](reports/ycbv/measured_summary.json) và [delivery_validation.json](reports/ycbv/delivery_validation.json) có `passed=true`.

Pose toàn bộ 900 ảnh: **ADD(-S)@0.1d 86,59%**, ADD-S 91,08%, ADD 71,98%; median inference 1,41 s/ảnh (bỏ frame đầu). Đây là recall chẩn đoán với 4.125 GT làm mẫu số, chưa phải BOP official AR. Có 3.572 GT đạt ADD(-S), 3.950 GT ghép được pose. Demo và thí nghiệm không thay mẫu số full benchmark.

## Cài đặt trên Windows

Máy đã kiểm tra: Python 3.11, Windows, RTX 4060 Laptop 8 GB, PyTorch 2.10.0 CUDA 12.8. Renderer Pyrender cần driver OpenGL hoạt động. CPU được hỗ trợ bởi CLI nhưng chưa benchmark tốc độ.

```powershell
cd D:\Github\clutterpick-rgb-d-perception-and-robotic-grasping
powershell -ExecutionPolicy Bypass -File scripts/setup/bootstrap.ps1
.\.venv\Scripts\python.exe scripts/data/prepare_ycbv.py
.\.venv\Scripts\python.exe main.py index
.\.venv\Scripts\python.exe main.py audit
.\.venv\Scripts\python.exe -m pytest -q
```

Downloader tải base, CAD, test BOP19 và ba checkpoint công khai. Dataset được kiểm SHA256 theo Hugging Face LFS và ghim revision; source CosyPose được ghim commit. Checksum checkpoint là hash ghi sau tải, không phải chữ ký do tác giả công bố. Dữ liệu nằm tại `data/ycbv/raw/ycbv`, mô hình tại `artifacts/pretrained/ycbv`.

## Chạy demo và đánh giá

```powershell
.\.venv\Scripts\python.exe main.py infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/demo_rgbd
.\.venv\Scripts\python.exe scripts/inference/visualize_instances.py artifacts/ycbv/demo_rgbd/ycbv_test_000048_1074
.\.venv\Scripts\python.exe main.py evaluate
.\.venv\Scripts\python.exe main.py benchmark --limit 900 --output artifacts/ycbv/benchmark_full
.\.venv\Scripts\python.exe scripts/evaluation/run_experiments.py --limit 24
.\.venv\Scripts\python.exe scripts/reports/build_report.py
.\.venv\Scripts\python.exe scripts/validation/validate_ycbv.py
```

Chạy toàn bộ theo thứ tự: `powershell -ExecutionPolicy Bypass -File scripts/workflows/run_all.ps1`. Lệnh này sẽ chạy lại đánh giá; không cần chạy để xem artifact đã có.

Demo xuất ảnh overlay segmentation và trục pose; PLY scene, PLY theo instance, PLY object coordinates, CAD đã căn chỉnh; JSON pose, diagnostics, timings; CSV BOP. Viewer `instances_3d.html` cho phép xoay, zoom và bật/tắt từng instance; PNG dùng đưa vào báo cáo. Các điểm chỉ thể hiện bề mặt nhìn thấy, không khôi phục toàn bộ vật thể bị che khuất.

## Phương pháp và cách đọc kết quả

- Mask R-CNN ResNet50-FPN pretrained YCB-V: 21 class foreground + background, mask/score threshold 0.5.
- Backproject depth uint16 bằng K và depth scale từng frame; YCB-V hiện có scale 0.1 mm. Tất cả hình học nội bộ dùng mét.
- CosyPose coarse + 4 vòng refiner, EfficientNet-b3 sáu kênh. Giữ checkpoint/crop/cập nhật pose từ source gốc, dùng renderer Pyrender trên Windows.
- ICP refinement từ CAD và visible depth: giới hạn tương ứng 40 mm, tối đa dịch chuyển 50 mm và xoay 10°, yêu cầu giảm RMSE ít nhất 10%. Những giá trị này là cấu hình kỹ thuật cố định; chưa chọn bằng validation trong benchmark baseline ban đầu; training mới không retune ICP trên test.
- ADD, ADD-S và ADD(-S)@0.1 diameter được tính **sau inference**, Hungarian cùng lớp với mask IoU ≥0.5. Mẫu số gồm tất cả GT instances, kể cả bỏ sót. ADD(-S) chọn ADD-S cho lớp có metadata đối xứng. Tối đa 10.000 đỉnh CAD.
- Đây là **metric chẩn đoán nội bộ**, không phải BOP AR VSD/MSSD/MSPD chính thức. Port renderer khác bản công bố; không khẳng định tái lập nguyên điểm paper.
- `status=ok` chỉ xác nhận pose hữu hạn/hợp lệ, không đồng nghĩa pose chính xác. Oracle mask chỉ dùng trong thí nghiệm ghi rõ `uses_ground_truth_masks=true`.

Pipeline không dùng GT pose để dự đoán. GT mask được dùng cho supervision train/validation của các module mới, oracle ablation và đánh giá; inference thực không dùng GT. Thí nghiệm cố định cùng 24 ảnh gồm mask+RGBD, bbox+RGBD, reference mask+RGBD và mask+RGB.

Pilot test đã được dùng debug renderer và xác định cấu hình ICP; chưa có validation độc lập. Điểm số là đánh giá implementation mang tính thăm dò, không phải holdout hoàn toàn chưa quan sát; không có training trọng số trên test.

## Mở rộng robust depth refinement

Cấu hình `configs/enhanced.yaml` bổ sung Huber point-to-plane ICP, so sánh nhiều candidates với giới hạn cập nhật, geometric support diagnostics và cờ review. Có validation CAD tổng hợp hai seed riêng và paired replay trên predictions của baseline. Xem [thiết kế](docs/ROBUST_REFINEMENT_VI.md) và [kết quả mở rộng](reports/ycbv/extension/RESULTS_VI.md).

Đã đo đủ 900 ảnh với cùng neural predictions: **ADD(-S) 86,59% → 88,80%** (+2,21 điểm phần trăm), **91 GT instances chuyển fail→pass, 0 pass→fail** tại ngưỡng 0.1 diameter. Đây là paired exploratory comparison. Synthetic holdout seed 271: mean ADD **10,70 → 6,09 mm**. **19 tests pass** và delivery validation của extension đạt. Nhóm supported geometry có diagnostic precision 98,27% trên tập này; giá trị đó không phải confidence đã calibration.

Chạy toàn bộ mở rộng: `powershell -ExecutionPolicy Bypass -File scripts/workflows/run_extension.ps1`. Demo: `main.py --config configs/enhanced.yaml infer --output artifacts/ycbv/extension/demo`. Replay tận dụng cache để giữ detector/RGB poses cố định; thời gian replay chỉ gồm geometry/refinement.


## Ba module học mới (v0.4)

Đã triển khai bốn chế độ RGB/RGB-D fusion, query Transformer mask head 56×56 và local point-attention foreground refinement trước robust ICP. Protocol mới dùng 300 PBR train / 60 validation / 60 holdout theo scene, index riêng; checkpoint và threshold chọn bằng validation. Published pretrained có thể đã gặp các PBR scenes này.

Xem [thiết kế và pipeline đầy đủ](docs/LEARNED_HYBRID_VI.md), [kết quả thí nghiệm](reports/ycbv/research/RESULTS_VI.md) và [kiểm tra bàn giao](reports/ycbv/research/delivery_validation.json). Các thí nghiệm mới trên 96 test frames / 24 pose frames là pilot, tách biệt với baseline full 900 ảnh. Kết quả chỉ hoàn tất khi report và delivery validation tồn tại với `passed=true`.

`configs/hybrid.yaml` kích hoạt đủ ba module (gated_query + learned point filter + robust ICP). Đây là cấu hình thử nghiệm, chưa mặc định rằng tốt hơn baseline. Chạy sau khi checkpoints train xong:

```powershell
.\.venv\Scripts\python.exe main.py --config configs/hybrid.yaml infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/hybrid_demo
```

Lệnh toàn bộ: `powershell -ExecutionPolicy Bypass -File scripts/workflows/run_research.ps1`. Downloader subset `scripts/data/prepare_pbr_subset.py` chỉ tải các ZIP members cần thiết; không cần tải toàn archive train_pbr 21 GB.

## Fine-tune và ảnh RGB-D riêng

Ảnh riêng đã đăng ký RGB-D có thể chạy bằng `scripts/inference/infer_rgbd.py --rgb ... --depth ... --intrinsics fx 0 cx 0 fy cy 0 0 1 --depth-scale-mm ...`. Depth phải uint16; scale là mm mỗi đơn vị raw. Công cụ không cần GT labels, nhưng model vẫn chỉ nhận 21 lớp YCB đã biết; chưa có benchmark trên camera riêng.

Index baseline ghi **train=0, val=0, test=900**; kết quả baseline dùng pretrained. Index research riêng có **train=300, val=60, holdout=60, test=900** cho các module mới. Không lấy test làm train/val. Để train:

```powershell
.\.venv\Scripts\python.exe scripts/data/prepare_ycbv.py --groups base models test train-pbr
.\.venv\Scripts\python.exe main.py index
.\.venv\Scripts\python.exe main.py train --epochs 8
```

Train PBR chiếm nhiều dung lượng (archive khoảng 21 GB, cộng dữ liệu giải nén); kiểm tra dung lượng trước tải. Scene PBR 000045–000049 làm validation, các scene PBR còn lại làm train. Script tạo `converted.pt` với trọng số detector công bố nguyên vẹn để warm-start. Chỉ checkpoint tốt nhất theo validation được dùng về sau. Train pose network chưa được cung cấp; pose sử dụng pretrained coarse/refiner.

## Tài liệu và cấu trúc

- [Tóm tắt chi tiết](PROJECT_SUMMARY_VI.md).
- [Thiết kế, thứ tự đọc code và hợp đồng dữ liệu](docs/IMPLEMENTATION_VI.md).
- [Kết quả](reports/ycbv/RESULTS_VI.md), [audit dataset](reports/ycbv/data_audit.json).
- [Nguồn và giấy phép](docs/THIRD_PARTY.md).
- `src/clutterpick`: mã nguồn được chia folder theo chức năng như sơ đồ dưới đây.
- `configs/default.yaml`: baseline; `enhanced.yaml`: robust ICP; `research.yaml`: protocol train/ablation; `hybrid.yaml`: ba module mới.
- [`scripts`](scripts/README.md): các lệnh riêng, chia folder theo công việc.
- [`tests`](tests/README.md): kiểm thử theo nhóm chức năng tương ứng với mã nguồn.

```text
main.py                      # Điểm bắt đầu: python main.py --help
src/clutterpick/
├── cli.py                   # Đọc tham số và chọn tác vụ
├── data/                    # Đọc dữ liệu, chia tập, kiểm tra dataset
├── models/                  # Detector, CosyPose, fusion, point network
├── geometry/                # Point cloud, pose hình học, hai thuật toán ICP
├── inference/               # Ghép các bước thành pipeline RGB-D
├── evaluation/              # Metric, ghép ground truth, tổng hợp kết quả
├── training/                # Vòng lặp huấn luyện
├── visualization/           # Vẽ ảnh 2D và point cloud 3D
└── utils/                   # Cấu hình, đọc/ghi file và xuất kết quả
scripts/
├── data/                    # Tải và kiểm tra dữ liệu
├── training/                # Tạo cache và train các module
├── inference/               # Chạy ảnh riêng, replay, viewer
├── evaluation/              # Benchmark và thí nghiệm
├── reports/                 # Tạo báo cáo
├── validation/              # Kiểm tra kết quả bàn giao
├── workflows/               # PowerShell chạy nhiều bước
├── setup/                   # Cài môi trường
└── legacy/                  # Công cụ audit IndustryShapes cũ
tests/                       # data/, models/, geometry/, inference/, evaluation/
configs/                     # Cấu hình YAML
docs/                        # Thiết kế và hướng dẫn
data/                        # Dữ liệu thực tế
artifacts/                   # Checkpoint, cache và kết quả chạy
reports/                     # Báo cáo và số liệu đo
third_party/                 # Source thư viện bên ngoài
```

Chạy từ thư mục gốc: `python main.py --help`. Sau khi cài package, có thể dùng `python -m clutterpick --help` hoặc `clutterpick --help`.

Source IndustryShapes cũ được lưu trong `legacy/industryshapes_source_snapshot.zip`. Artifact và báo cáo cũ ngoài `reports/ycbv` thuộc hướng trước, không phải kết quả YCBPose. Grasp planning và robot execution nằm ngoài phạm vi hiện tại.


<!-- learned-results-start -->
## Kết quả learned modules trên cohort riêng

Đo trên 96 test frames / 440 GT cho segmentation và 24 frames / 110 GT cho pose. Các số này không thay benchmark full 900 ảnh.

| Variant | Mask AP | Boundary AP | ADD(-S) |
|---|---:|---:|---:|
| Published RGB + robust ICP | 74.24% | 47.24% | 88.18% |
| rgb | 72.34% | 45.74% | 87.27% |
| early | 72.36% | 43.89% | 89.09% |
| late | 70.92% | 43.63% | 88.18% |
| gated | 70.11% | 41.93% | 89.09% |
| rgb_query | 69.91% | 41.40% | 89.09% |
| gated_query | 69.66% | 40.83% | 87.27% |

Learned 3D filter với cùng published detector: contamination 3.94% → 3.16%; ADD(-S) 88.18% → 88.18%. Pipeline đủ ba module đạt 88.18% ADD(-S).

Train là pilot 300 PBR frames, checkpoint/threshold chọn validation; pretrained có thể đã gặp PBR holdout. Chi tiết và các kết quả giảm: [research report](reports/ycbv/research/RESULTS_VI.md).
<!-- learned-results-end -->

<!-- architecture-results-start -->
## Kiến trúc inference đa vật thể

Benchmark 96 ảnh / 440 GT: median latency 1.901 → 1.190 s (37.40% giảm); p95 3.122 → 2.002 s.

Bản mới: batching, GPU-resident renderer, CAD KD-tree cache và lazy scene geometry; weights giữ nguyên. Chất lượng từng repeat và giới hạn: [architecture results](reports/ycbv/architecture/RESULTS_VI.md). Thiết kế/cách chạy: [inference architecture](docs/INFERENCE_ARCHITECTURE_VI.md).
<!-- architecture-results-end -->
