# Đọc kết quả inference

Với `main.py infer --output artifacts/ycbv/demo`, mỗi frame được lưu trong
folder có tên là frame key đổi `/` thành `_`, ví dụ `ycbv_test_000048_1074`.
Folder output chung chứa `run_summary.json` và `bop_predictions.csv`.

| File trong folder frame | Nội dung |
|---|---|
| `detection.json` | Class, score, bbox, đường dẫn mask/cloud và provenance |
| `poses.json` | Pose từng instance và provenance |
| `timing.json` | Thời gian từng bước |
| `pose_quality.json` | Diagnostics hỗ trợ hình học |
| `pose_diagnostics.json` | ADD/ADD-S/ADD(-S) khi có GT để đánh giá |
| `geometry_diagnostics.json` | Đánh giá geometry sau inference |
| `final_visualization.png` | Overlay segmentation và trục pose |
| `masks/` | Mask 2D từng instance |
| `pointcloud.ply` | Visible scene cloud trong hệ camera |
| `instances_pointcloud.ply` | Scene cloud tô màu theo instance |
| `point_instance_ids.npy` | Nhãn theo thứ tự vertex của scene cloud: 0 là background |
| `object_pointclouds/` | Observed points đổi sang hệ tọa độ object |
| `aligned_cad/` | CAD points đổi sang hệ camera theo pose |

`T_camera_object` biến đổi điểm từ object sang camera. JSON và PLY dùng mét;
hệ camera theo BOP: x sang phải, y xuống dưới, z hướng ra trước.
BOP CSV ghi rotation theo hàng và translation bằng milimét. Pose không có
`T_camera_object` được bỏ qua khi xuất CSV.

`status=ok` cho biết pose hợp lệ theo pipeline, không chứng minh pose đúng.
`pose_quality` là diagnostic hình học, chưa phải confidence đã calibration.
Kiểm tra `uses_ground_truth_masks` trước khi so sánh kết quả; oracle dùng GT
mask chỉ phục vụ thí nghiệm. ADD(-S) nội bộ không phải official BOP AR.

`infer --no-plots` bỏ `scene_3d.png`; overlay vẫn được xuất.
`infer --skip-cloud-export` bỏ các cloud và scene labels; viewer cần cloud
nên hãy chạy infer với export mặc định nếu muốn xem 3D. Lệnh `benchmark`
không xuất cloud và scene 3D plot theo mặc định.

Point cloud chỉ chứa bề mặt có depth quan sát được. Không coi phần bị che
khuất là geometry đã được khôi phục. Xem thêm [hướng dẫn ảnh riêng](EXTERNAL_RGBD_VI.md).
