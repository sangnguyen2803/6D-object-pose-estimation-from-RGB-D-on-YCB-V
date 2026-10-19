# Thiết kế YCBPose

Luồng chính: CLI → BOP index/load_rgbd → pretrained detector → mask/depth point cloud → CosyPose coarse/refiner → bounded depth ICP → artifact writer → evaluator.

Các đường dẫn dưới đây tính từ `src/clutterpick/`.

| Module | Trách nhiệm |
|---|---|
| data/dataset.py | Index BOP, chia scene train/val/test, sampling, dataset Mask R-CNN |
| data/audit.py | Kiểm tra dữ liệu/masks/calib/duplicates/test targets |
| models/detector.py | Mask R-CNN pretrained, checkpoint/class contracts |
| models/fusion.py | RGB/early/late/gated fusion, RGB-D dataset, spatial-query mask head |
| models/point_network.py | Point features, local attention, foreground cleaner trước ICP |
| models/cosypose.py | Windows rendering adapter, strict published networks, RGB-D refinement |
| models/pretrained.py | Đọc cấu hình checkpoint công khai |
| models/yolo.py | Nhánh detector YOLO tùy chọn |
| geometry/point_cloud.py | Backprojection, mask filtering, voxel, PLY |
| geometry/icp.py | SVD rigid fit, trimmed point-to-point và Huber point-to-plane ICP |
| geometry/pose_estimation.py | Đọc CAD, tạo pose khởi tạo và geometric pose estimator |
| geometry/refinement.py | So sánh hai kết quả ICP, kiểm tra giới hạn cập nhật, chọn pose |
| inference/pipeline.py | Điều phối các bước xử lý một frame RGB-D |
| evaluation/metrics.py | COCO AP, ghép mask và đánh giá ADD/ADD-S sau inference |
| evaluation/boundary.py | Boundary COCO AP theo min(mask IoU, boundary IoU) |
| evaluation/reporting.py | Recall với tất cả GT trong mẫu số |
| training/train_detector.py | Vòng lặp huấn luyện và chọn checkpoint |
| visualization/draw.py | Vẽ overlay 2D, trục pose và point cloud 3D |
| utils/common.py | Đọc cấu hình, tìm thư mục dự án, JSON và random seed |
| utils/outputs.py | Ghi JSON, PLY theo instance và CSV BOP |
| cli.py | Đọc tham số và chạy index/audit/evaluate/train/infer/benchmark |

## Cách đọc code

Nên bắt đầu từ các hàm sau để nắm luồng xử lý, rồi mới đọc các module mạng:

1. `main.py` ở thư mục gốc gọi `cli.py`: `build_parser()` khai báo lệnh, `main()` chọn tác vụ, `run_inference()` chạy qua các frame.
2. `inference/pipeline.py`: `run_frame()` gọi các bước đọc RGB-D → detector → point cloud → pose → đánh giá → lưu kết quả. Các bước được đánh số trong code.
3. `geometry/point_cloud.py`: `backproject()` chuyển pixel và depth sang điểm 3D; `instance_cloud()` lọc điểm theo mask; `project_pixels()` chiếu điểm 3D về ảnh.
4. `geometry/pose_estimation.py`: `GeometricPoseEstimator` quản lý CAD và thử nhiều pose ban đầu. Nhánh pretrained dùng `models/cosypose.py`.
5. `geometry/icp.py`: `trimmed_icp()` ghép điểm rồi dùng SVD; `point_plane()` tính sai số theo normal rồi giải hệ tuyến tính. Chú thích trong hai vòng lặp giải thích từng bước.
6. `geometry/refinement.py`: `refine()` thử cả hai ICP; `evaluate_candidate()` kiểm tra điều kiện nhận kết quả. Nếu không có candidate đạt điều kiện thì giữ pose RGB.
7. `evaluation/metrics.py` và `utils/outputs.py`: một file tính metric, một file ghi kết quả ra đĩa.

Code được chia thành các package theo chức năng. Mỗi folder có `__init__.py`; các hàm dùng import đầy đủ, ví dụ `from clutterpick.geometry.icp import trimmed_icp`. Các bước xử lý dùng hàm và dictionary thông thường.

Scripts chạy trực tiếp nằm theo nhóm trong `scripts/`; xem [hướng dẫn scripts](../scripts/README.md). Tests được nhóm theo các chức năng tương ứng trong `tests/`. Các đường dẫn import và lệnh chạy trong dự án đã chuyển sang cấu trúc mới.

## Hợp đồng pose

`T_camera_object` là ma trận 4×4, transform p_camera=R*p_object+t. Rotation SO(3), translation mét. BOP raw CAD/GT translation mm được chuyển sang m. JSON/PLY theo hệ camera X right/Y down/Z forward. Renderer GL đổi basis bằng diag(1,-1,-1,1); dữ liệu pose xuất ra vẫn là BOP.

CSV BOP xuất R row-major 9 giá trị, t 3 giá trị mm, score detector, time tổng inference của frame. Reference evaluation và visualization không nằm trong latency benchmark; khởi tạo renderer/model có thể làm frame đầu chậm, report median bỏ frame đầu.

## Runtime và tài nguyên

Network/renderer/CAD được cache xuyên frames. Benchmark không xuất PLY/3D plots. Demo xuất point clouds và visualization đầy đủ. Pyrender cache CAD textured PLY; texture giới hạn 1024 để tránh tốn RAM. Package CosyPose gốc được vendor ghim commit; adapter tránh import dependency Pinocchio/Bullet không cần cho đường inference này. EfficientNet và pose math giữ từ source gốc.

Không cần WSL/Docker/MSVC cho đường inference Windows đang dùng. OpenGL driver bắt buộc cho renderer. CUDA model sử dụng float32; không huấn luyện song song benchmark.

## Quy tắc đánh giá

Prediction hoàn tất rồi mới truy cập GT pose ở nhánh evaluator. Oracle mask là tùy chọn rõ ràng trong ablation. Test `test_changing_gt_pose_changes_evaluation_but_not_prediction` kiểm chứng hợp đồng này. Validator kiểm tra SO(3), translation finite/z-positive, provenance, instance counts, cohorts và CSV units.

Đọc `RESULTS_VI.md` để phân biệt ADD diagnostics với BOP official AR. Thư mục IndustryShapes cũ và artifact cũ không được dùng để tạo báo cáo YCB-V.

## Training

Train detector tùy chọn chỉ khi train/val không rỗng. TrainPBR holdout theo scene, không lấy test làm validation. Warm-start weights published được chuyển format thành model/config mà không đổi tensor. Training pose chưa nằm trong implementation. Eval checkpoint fine-tuned cần dùng `--architecture maskrcnn --checkpoint ...`; mặc định detector vẫn là checkpoint published.

Phiên bản v0.4 có subset PBR 300 train / 60 validation / 60 holdout với index riêng. `scripts/training/train_fusion.py` train sáu biến thể; `train_point_network.py` train foreground point network. Kiến trúc learned fusion dùng `--architecture fusion`, hoặc `configs/hybrid.yaml` để bật đủ ba module. Xem [thiết kế learned hybrid](LEARNED_HYBRID_VI.md) và [thí nghiệm](../reports/ycbv/research/RESULTS_VI.md).

Phiên bản v0.5 giữ pretrained detector, thêm multi-instance pose batching, resident renderer, cached CAD spatial indices, lazy scene geometry và optional scalar recovery cho weak geometry. `configs/optimized.yaml` bật quality gate; `optimized_fast.yaml` ưu tiên latency. Xem [inference architecture](INFERENCE_ARCHITECTURE_VI.md) và [benchmark có đối chứng](../reports/ycbv/architecture/RESULTS_VI.md). Kết quả recovery được ghi riêng, không mặc định rằng chạy lại luôn cải thiện pose.

