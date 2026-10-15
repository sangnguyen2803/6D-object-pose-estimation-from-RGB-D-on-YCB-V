# Tóm tắt dự án YCBPose

## 1. Mục tiêu

Chuyển toàn bộ pipeline đang hoạt động sang YCB-V: nhận RGB-D đã đăng ký, tách từng instance, tạo point cloud riêng và ước lượng pose 6D của vật thể đã biết. Đầu ra pose là ma trận 4×4 object-to-camera gồm rotation SO(3) và translation 3D.

Bài toán có CAD và identity đã biết, không phải pose estimation cho vật thể bất kỳ không có model. Kết quả thực tế tổng hợp trong [RESULTS_VI.md](reports/ycbv/RESULTS_VI.md), bảng JSON máy đọc được trong [measured_summary.json](reports/ycbv/measured_summary.json).

## 2. Dataset và tính phù hợp

YCB-V BOP19 có RGB, depth uint16, K, depth scale, visible masks, full masks, class ID, GT pose và CAD. Đủ nhãn trực tiếp cho segmentation và đánh giá 6D pose; không cần nhãn grasp ở phạm vi này.

Phiên bản tải hiện tại có 900 ảnh test, 12 scene 000048–000059, 75 ảnh/scene, 4.125 GT instances và 21 lớp. Audit giải mã toàn bộ ảnh/masks, kiểm tra depth scale, shape, calib, containment mask, RGB duplicate, frame overlap và danh sách ảnh test chính thức. Xem [data_audit.json](reports/ycbv/data_audit.json). Mean invalid depth khoảng 17,06%; pixel depth 0 bị loại.

Dataset ghim HF revision `5c2c4aa229800355648cd268040aa814f8dc94f0`. Source, SHA256 và checkpoint IDs nằm trong [download_manifest.json](reports/ycbv/download_manifest.json).

Baseline ban đầu chưa tải train_real/train_pbr. Phiên bản v0.4 đã tải subset 420 PBR frames và có index research riêng; train_real vẫn chưa tải. Train/val trống là trạng thái inference-only hợp lệ, không phải lấy test làm validation. Train PBR tùy chọn chia theo scene 45–49 làm validation. Checkpoint pretrained tác giả đã train trên SYNT+REAL; chúng tôi không báo cáo đó là quá trình train do dự án mới thực hiện.

## 3. Kiến trúc

### Instance segmentation

Published Mask R-CNN ResNet50-FPN pretrained YCB-V, 21 foreground classes + background. Input native 640×480, score/mask threshold 0.5. Checkpoint được load strict; class mapping kiểm tra object ID 1–21. Instance cùng lớp vẫn tách riêng.

### Point cloud

Z = raw_depth × depth_scale_mm / 1000. X=(u-cx)Z/fx, Y=(v-cy)Z/fy. Hệ camera BOP: X sang phải, Y xuống, Z phía trước. Chỉ giữ depth 0.1–3 m. Predicted visible masks chọn điểm, erosion một pixel và voxel 3 mm giảm nhiễu biên; diagnostics ghi số điểm được giữ.

Xuất scene point cloud, từng instance trong camera frame và object frame, instance-colored scene cùng point_instance_ids. Label 3D lấy từ mask 2D backproject qua depth; baseline không có mạng 3D riêng; cấu hình hybrid v0.4 có local point-attention foreground network, xem phần nâng cấp bên dưới. PLY CAD căn chỉnh là model đã biết, không phải bề mặt do hệ thống quan sát hoàn toàn.

### Pose 6D

CosyPose coarse/refiner BOP20 YCB-V SYNT+REAL, EfficientNet-b3 nhận sáu kênh gồm ảnh crop và CAD render. Coarse chạy một vòng, refiner bốn vòng. Giữ nguyên checkpoint, crop và pose-update math từ source CosyPose ghim commit; port renderer sang Pyrender để chạy Windows.

RGB-D refinement dùng CAD-to-observed ICP, correspondence limit 40 mm, 25 vòng, trim 80%. Chỉ chấp nhận khi relative RMSE cải thiện ít nhất 10%, translation update <=50 mm và rotation update <=10°. Nếu không đạt, giữ RGB pose. Cấu hình là lựa chọn kỹ thuật cố định, chưa tối ưu trên validation riêng.

Checkpoint có provenance trong poses.json. `status=ok` chỉ xác nhận pose hợp lệ, không chứng nhận độ đúng. Có backend geometric riêng để thử đăng ký CAD; baseline chính vẫn là CosyPose RGB-D.

## 4. Chống rò rỉ nhãn và đơn vị

Inference chỉ nhận RGB-D, K, predicted mask/box/class và CAD. Tất cả inference xong mới chạy evaluator GT. Test kiểm tra thay GT translation làm đổi lỗi ADD nhưng pose dự đoán không đổi.

Oracle-mask ablation dùng mask GT và được đánh dấu rõ, không tính là pipeline chạy tự động. JSON/PLY dùng mét; BOP CSV dùng mm cho translation, rotation row-major, scene_id/im_id/obj_id/score/time theo format BOP. Kiểm tra bàn giao đối chiếu CSV với JSON từng pose.

## 5. Đánh giá và kết quả

Detector đã đánh giá đủ 900 ảnh / 4.125 instances: bbox COCO AP@[.50:.95] **79,33%**, mask AP@[.50:.95] **74,27%**; mask AP50 **96,03%**. Kéo/scissors và clamps là các lớp segmentation còn khó hơn các hộp/chai lớn.

Pose và thí nghiệm được đọc trực tiếp từ artifact trong [RESULTS_VI.md](reports/ycbv/RESULTS_VI.md), không chép số pilot vào kết quả toàn bộ. Báo cáo có số frame, số GT, recall ADD/ADD-S/ADD(-S), median latency và từng lớp.

Benchmark pose đã chạy đủ **900 ảnh / 4.125 GT instances**: ADD **71,98%** (2.969/4.125), ADD-S **91,08%** (3.757/4.125), ADD(-S) **86,59%** (3.572/4.125). Có 4.128 detections và 3.950 pose ghép với GT. Median inference bỏ frame đầu **1,414 s/ảnh** trên RTX 4060 Laptop. Các trường hợp không ghép được vẫn là thất bại trong recall. Đây là kết quả implementation hiện tại với pretrained và ICP, không phải training mới.

GT denominator gồm mọi instance, cả bị detector bỏ sót. Hungarian ghép cùng class với mask IoU >=0.5. ADD so đỉnh tương ứng, ADD-S so nearest neighbor, ADD(-S) chọn ADD-S khi CAD có symmetry metadata. Ngưỡng 0.1 diameter từng CAD. Lấy tối đa 10.000 đỉnh xác định cho runtime.

**Không phải metric BOP AR chính thức**: chưa chạy VSD/MSSD/MSPD evaluator với target instance-count protocol và visibility tolerance chuẩn. Dùng chung danh sách 900 ảnh không đồng nghĩa đã tái lập mọi quy tắc scoring BOP.

Pilot lấy từ test đã được dùng kiểm tra port renderer và xác định cấu hình ICP trước benchmark toàn tập. Chưa có validation độc lập, nên đây là đánh giá implementation mang tính thăm dò; không tuyên bố một holdout hoàn toàn chưa quan sát. Trọng số pretrained không được train thêm trên test.

## 6. Thí nghiệm có kiểm soát

Cố định 24 ảnh trải 12 scene bằng seed 42, cùng cohort cho cả bốn phương án:

1. Predicted mask + RGB-D pose (pipeline mặc định).
2. Bounding box + RGB-D pose (đo ảnh hưởng background depth).
3. Reference mask + RGB-D pose (oracle diagnostic).
4. Predicted mask + RGB-only pose (đo tác dụng depth refinement).

Báo cáo cả pose recall và tỷ lệ điểm nằm ngoài visible GT mask cho những instance đã ghép. Contamination là proxy dựa mask/điểm chiếu, không phải metric mọi point cloud GT 3D; chưa ghép không được tính contamination và cần đọc cùng detection recall. Cohort nhỏ phục vụ giải thích, không thay full benchmark.

Contamination được lấy trung bình không trọng số theo instance đã ghép, không phải tỷ lệ của tổng số điểm toàn dataset.

Kết quả đo trên 24 ảnh / 110 GT instances:

| Phương án | ADD(-S) recall | Mean contamination |
|---|---:|---:|
| Predicted mask + RGB-D | 87,27% | 3,94% |
| Bounding box + RGB-D | 85,45% | 32,64% |
| Reference mask + RGB-D (oracle) | 90,00% | 0,00% |
| Predicted mask + RGB pose | 82,73% | 3,94% |

Trong cohort nhỏ này, mask giảm nhiễm background và depth refinement cải thiện 4,55 điểm phần trăm ADD(-S) so với RGB pose. Oracle cho thấy vẫn có lỗi pose ngay cả khi mask đúng. Những khác biệt này mang tính mô tả, chưa có kiểm định thống kê hoặc validation độc lập.

## 7. Đầu ra bàn giao

Demo: overlay PNG, scene_3d.png, instances_3d.png, viewer HTML offline; masks theo instance; pointcloud.ply, instances_pointcloud.ply, point_instance_ids.npy; per-instance/object-frame/aligned-CAD PLY; detection.json/poses.json; geometry/pose diagnostics và timings. Benchmark có JSON/PNG/CSV, bỏ PLY để tiết kiệm dung lượng.

Reports: dataset audit, split counts, download/source provenance, measured summary, RESULTS_VI, chart pose recall, validation và source artifact SHA256 manifest. Source IndustryShapes cũ được lưu ZIP riêng, không còn được import vào pipeline hiện tại.

## 8. Giới hạn thực tế và hướng phát triển

- v0.4 fine-tune segmentation trên subset PBR và train point network; pose networks vẫn dùng pretrained.
- Renderer Windows khác renderer original; pretrained tải strict nhưng điểm số có thể khác.
- CAD identity phải thuộc 21 lớp YCB đã biết.
- Occlusion mạnh, depth holes, reflective/black material và vật thể mảnh gây lỗi.
- Visible point cloud không phải full object reconstruction.
- RGB-D ICP chỉ sửa cục bộ và có thể bị depth/background contamination.
- Chưa có grasp planning, controller, calibration camera–robot, mô phỏng grasp hay success robot thực.
- Với đồ án perception offline, pipeline có dữ liệu, pretrained, kết quả, demo và quy trình đánh giá. Để triển khai robot cần thêm kiểm tra riêng theo ứng dụng; không suy ra an toàn robot từ ADD.

Hướng tiếp: tải train PBR khi đủ dung lượng, fine-tune theo validation độc lập; chạy official BOP evaluator; kiểm tra renderer port; bổ sung multi-view temporal consistency và real camera input.

## 9. Tái lập

Kiểm tra bàn giao đã vượt qua: full benchmark 900 ảnh / 4.128 pose xuất hợp lệ, cả bốn cohort 24 ảnh đồng nhất, demo 5 pose; 15 tests vượt qua. Xem [delivery_validation.json](reports/ycbv/delivery_validation.json) và [artifact_manifest.json](reports/ycbv/artifact_manifest.json).

Làm theo [README](README.md). `scripts/run_all.ps1` thực hiện chuẩn bị → index → audit → tests → detector eval → pose benchmark → demo → ablations → report → delivery validation. Không tự train trên test. Config hiện tại nằm tại `configs/default.yaml`; xem `requirements-measured.txt` cho môi trường đã đo.

## 10. Mở rộng robust depth refinement

Bổ sung Huber point-to-plane ICP, candidate selection, geometric quality và observability review. Trên 900 ảnh giữ nguyên neural predictions: ADD(-S) 86.59% → 88.80%, 91 GT fail→pass và 0 pass→fail tại ngưỡng 0.1 diameter; không khẳng định mọi numerical pose error đều giảm. Synthetic holdout seed 271, 63 cases: mean ADD 10.70 → 6.09 mm.

Cấu hình enhanced riêng; không thay pretrained weights. Quality không phải xác suất correctness. Real comparison vẫn là exploratory test evaluation; replay latency không phải end-to-end. Chi tiết: [kết quả mở rộng](reports/ycbv/extension/RESULTS_VI.md), [phương pháp](docs/ROBUST_REFINEMENT_VI.md).


## Ba module học mới (v0.4)

Đã triển khai bốn chế độ RGB/RGB-D fusion, query Transformer mask head 56×56 và local point-attention foreground refinement trước robust ICP. Protocol mới dùng 300 PBR train / 60 validation / 60 holdout theo scene, index riêng; checkpoint và threshold chọn bằng validation. Published pretrained có thể đã gặp các PBR scenes này.

Xem [thiết kế và pipeline đầy đủ](docs/LEARNED_HYBRID_VI.md), [kết quả thí nghiệm](reports/ycbv/research/RESULTS_VI.md) và [kiểm tra bàn giao](reports/ycbv/research/delivery_validation.json). Các thí nghiệm mới trên 96 test frames / 24 pose frames là pilot, tách biệt với baseline full 900 ảnh. Kết quả chỉ hoàn tất khi report và delivery validation tồn tại với `passed=true`.

`configs/hybrid.yaml` kích hoạt đủ ba module (gated_query + learned point filter + robust ICP). Đây là cấu hình thử nghiệm, chưa mặc định rằng tốt hơn baseline. Chạy sau khi checkpoints train xong:

```powershell
.\.venv\Scripts\python.exe scripts/run.py --config configs/hybrid.yaml infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/hybrid_demo
```

Lệnh toàn bộ: `powershell -ExecutionPolicy Bypass -File scripts/run_research.ps1`. Downloader subset `scripts/prepare_pbr_subset.py` chỉ tải các ZIP members cần thiết; không cần tải toàn archive train_pbr 21 GB.

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
