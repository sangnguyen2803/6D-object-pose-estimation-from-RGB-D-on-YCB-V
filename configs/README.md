# Chọn cấu hình

Mỗi YAML là một cấu hình đầy đủ, không kế thừa tự động từ `default.yaml`.
Đặt `--config` trước tên lệnh:

```powershell
python main.py --config configs/enhanced.yaml infer --limit 1 --output artifacts/ycbv/enhanced_demo
```

| File | Mục đích | Điều kiện |
|---|---|---|
| `default.yaml` | Baseline pretrained CosyPose và depth refinement | Dataset, CAD và checkpoint công khai |
| `enhanced.yaml` | Robust depth refinement | Cùng checkpoint với baseline |
| `optimized.yaml` | Robust refinement, batching, cache và resident renderer | Môi trường renderer hoạt động |
| `optimized_fast.yaml` | Như optimized, tắt quality retry | Đọc giới hạn trong báo cáo architecture |
| `research.yaml` | Protocol train/validation/holdout cho các module mới | Research index và PBR subset |
| `point_refined.yaml` | Published detector và learned point filter | Checkpoint point network đã train |
| `hybrid.yaml` | Fusion detector, learned point filter và robust refinement | Cả checkpoint fusion và point network |

Đường dẫn tương đối trong cấu hình được tính từ gốc dự án, không từ folder chứa
YAML. Có thể dùng đường dẫn tuyệt đối khi dataset hoặc checkpoint ở ổ khác.

Hình học nội bộ dùng mét: `geometry.depth_min_m`, `depth_max_m`, `voxel_size_m`
và các giới hạn pose có hậu tố `_m`. `depth_scale_mm` trong metadata frame là
milimét trên mỗi đơn vị depth raw; không thay bằng giá trị tính theo mét.

Sau khi đổi các đường dẫn index hoặc cách chia scene, chạy lại `index` với cùng
cấu hình. Dùng output riêng khi so sánh các variant để tránh ghi đè kết quả.
Không chọn threshold bằng test set.

Chi tiết: [robust refinement](../docs/ROBUST_REFINEMENT_VI.md),
[learned modules](../docs/LEARNED_HYBRID_VI.md),
[inference architecture](../docs/INFERENCE_ARCHITECTURE_VI.md).
