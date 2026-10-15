# Thiết kế YCBPose

Luồng chính: CLI → BOP index/load_rgbd → pretrained detector → mask/depth point cloud → CosyPose coarse/refiner → bounded depth ICP → artifact writer → evaluator.

| Module | Trách nhiệm |
|---|---|
| data.py | Index BOP, chia scene train/val/test, sampling, dataset Mask R-CNN |
| audit.py | Kiểm tra dữ liệu/masks/calib/duplicates/test targets |
| detection.py | Mask R-CNN pretrained, checkpoint/class contracts |
| fusion.py | RGB/early/late/gated fusion, RGB-D dataset, spatial-query mask head |
| boundary.py | Boundary COCO AP theo min(mask IoU, boundary IoU) |
| point_network.py | Point features, local attention, foreground cleaner trước ICP |
| geometry.py | Backprojection, mask filtering, voxel, PLY |
| cosypose_backend.py | Windows rendering adapter, strict published networks, RGB-D refinement |
| pose.py | CAD geometry, rigid registration, geometric alternative |
| pipeline.py | Inference trước reference evaluation, JSON/PLY/PNG provenance |
| evaluation.py | COCO AP, mask matching, ADD/ADD-S diagnostic |
| reporting.py | Recall với tất cả GT trong mẫu số |
| cli.py | Index/audit/evaluate/train/infer/benchmark và BOP export |

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

Phiên bản v0.4 có subset PBR 300 train / 60 validation / 60 holdout với index riêng. `scripts/train_fusion.py` train sáu biến thể; `train_point_network.py` train foreground point network. Kiến trúc learned fusion dùng `--architecture fusion`, hoặc `configs/hybrid.yaml` để bật đủ ba module. Xem [thiết kế learned hybrid](LEARNED_HYBRID_VI.md) và [thí nghiệm](../reports/ycbv/research/RESULTS_VI.md).

Phiên bản v0.5 giữ pretrained detector, thêm multi-instance pose batching, resident renderer, cached CAD spatial indices, lazy scene geometry và optional scalar recovery cho weak geometry. `configs/optimized.yaml` bật quality gate; `optimized_fast.yaml` ưu tiên latency. Xem [inference architecture](INFERENCE_ARCHITECTURE_VI.md) và [benchmark có đối chứng](../reports/ycbv/architecture/RESULTS_VI.md). Kết quả recovery được ghi riêng, không mặc định rằng chạy lại luôn cải thiện pose.

