# Kết quả cải thiện baseline

Các số dưới đây được đọc từ artifact đo thực tế. Mục thiếu artifact được ghi chưa hoàn tất, không suy ra từ training loss.

Detector đã hoàn tất 11/12 epoch; dense pose 0/24 epoch. Train dùng resampling: 4.096 image draws/epoch cho detector, 8.192 instance draws/epoch cho dense pose.

## Detection trên Classic Test

| Model | Frames | Box AP50:95 | Mask AP50:95 | Mask AP50 |
|---|---:|---:|---:|---:|
| Baseline Mask R-CNN | 923 | 0.4237 | 0.2763 | 0.4802 |
| Improved Mask R-CNN | 923 | 0.5453 | 0.3209 | 0.5986 |

Theo yêu cầu, dừng epoch 12 đang dở sau 11 epoch hoàn tất; dùng best checkpoint epoch 6 với weighted validation AP 0.5737. Không resume detector để hoàn tất epoch 12.

| Model | Obj 1 | Obj 2 | Obj 3 | Obj 4 | Obj 5 |
|---|---:|---:|---:|---:|---:|
| Baseline Mask R-CNN | 0.6606 | 0.3324 | 0.3439 | 0.0444 | 0.0001 |
| Improved Mask R-CNN | 0.5983 | 0.3496 | 0.3949 | 0.2268 | 0.0350 |

## Pose và candidate availability

| Pipeline | Frames | ADD recall @0.1d | ADD-S recall @0.1d | Frame có candidate | Median inference (s) |
|---|---:|---:|---:|---:|---:|
| Baseline RGB corners + ICP | 923 | 0.133 | 0.325 | 192/923 | 0.768 |

Missed detections và failed/poor-fit poses là failures trong mẫu số tất cả GT instances. ADD/ADD-S diagnostic dùng tối đa 10.000 CAD vertices, threshold 10% diameter; chưa phải official BOP AR. Timing bỏ reference evaluation, visualization và PLY export, vẫn gồm input/mask IO. Candidate availability không phải grasp success.

## Cùng reference mesh evaluator

| Candidate set | Frames | Evaluable top-k | Reference collisions | Rate |
|---|---:|---:|---:|---:|

Evaluator dùng GT CAD target/neighbors, FCL triangle/box checks trên swept approach volumes và observed depth, sau planning. Baseline candidate cũ cũng được kiểm tra lại bằng evaluator này; không trộn tỷ lệ từ hai evaluator khác nhau. N/A nghĩa không có candidate để đối chiếu, không có nghĩa collision rate bằng 0.

## Thay đổi và giới hạn

- Bổ sung onboarding/PBR, exact RGB dedup, balanced sampling, overlap augmentation và PBR scene holdout.
- Dense correspondence pose/RANSAC thay cho 8-corner regression; ICP update kiểm tra bằng correspondence residual.
- Foreground decoder dùng measured RGB-D context; refinement chỉ bật theo per-class validation AP.
- Sampler ray exit/diverse anchors, CAD-contact support, gripper opening clearance và swept-volume mesh collision checks.
- Dataset real clutter validation độc lập vẫn còn thiếu; PBR clutter là proxy. Classic holdout có cả synthetic object 3.
- Scan mesh hở có thể thiếu geometry. Contact/closing/lifting, dynamics, friction thực tế và IK chưa được xác minh.
- Không có simulation hoặc physical grasp success. Pose residual là gate heuristic, chưa phải uncertainty đã hiệu chuẩn.

Xem [phương pháp và lệnh tái lập](../../docs/IMPROVEMENT_VI.md). Các artifacts mới nằm trong `artifacts/training/improved_maskrcnn`, `artifacts/training/dense_pose`, `artifacts/benchmark/improved_full` và `artifacts/experiments/improved_ablations`.
