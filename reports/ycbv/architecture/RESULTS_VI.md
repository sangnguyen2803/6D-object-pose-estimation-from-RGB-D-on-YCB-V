# YCBPose: cải thiện kiến trúc inference

Giữ detector/CosyPose weights và robust ICP. Kiến trúc mới gom instances thành batch, giữ CAD meshes thường trú trong renderer, cache KD-tree theo class và dựng scene clouds khi cần. Có optional scalar recovery cho weak geometry. Không fine-tune trên test.

## Component ablation: 24 ảnh

Phase đầu batch/cache/lazy chưa có resident renderer. Resident-only được đo ở phase sau. Các phase chạy tuần tự nên không quy mọi chênh lệch thời gian thành tác dụng thuật toán.

| Variant | p50 (s) | p95 (s) | ADD(-S), từng repeat |
|---|---:|---:|---|
| baseline | 1.825 | 2.743 | 88.18% / 88.18% |
| batch_only | 1.896 | 2.453 | 88.18% / 88.18% |
| cache_only | 1.792 | 2.705 | 88.18% / 88.18% |
| batch_cache_lazy_v1 | 1.467 | 1.969 | 88.18% / 88.18% |
| resident_only | 1.732 | 2.851 | 88.18% / 88.18% |

## Đối chứng mở rộng

96 ảnh / 440 GT, 2 measured repeats mỗi cấu hình, warm-up 96 frames mỗi cấu hình. GPU: NVIDIA GeForce RTX 4060 Laptop GPU.

| Metric | Sequential robust baseline | Optimized |
|---|---:|---:|
| p50 latency (s) | 1.901 | 1.190 |
| p95 latency (s) | 3.122 | 2.002 |
| Throughput (frames/s) | 0.508 | 0.815 |
| ADD(-S) repeat 0 | 88.64% | 88.64% |
| ADD(-S) repeat 1 | 88.64% | 88.64% |

Median latency giảm **37.40%**, p95 giảm **35.88%**; throughput **1.60×** so với baseline đo trong cùng benchmark.

## Renderer và bộ nhớ

| Variant | Render calls sau warm-up | Mesh context additions | Peak Torch MiB |
|---|---:|---:|---:|
| baseline | 4360 | 864 | 411.6 |
| optimized | 4720 | 0 | 417.3 |

Peak Torch memory không bao gồm OpenGL/driver. Resident mesh cache dùng thêm GPU graphics memory; không tuyên bố giảm tổng VRAM. Pixel equivalence test có trong `renderer_validation.json`.

## Paired pose cases

| Repeat | Fail→pass | Pass→fail |
|---|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 0 |

Quality-aware scalar recovery: 72 object attempts, 0 replacements qua hai repeats. Chỉ thay khi geometry supported, RMSE thấp hơn và coverage không giảm; cùng weights và bounds.

Baseline được tái dùng từ lượt đo sequential trước đó trên cùng máy/cohort; bản quality-recovery được đo sau, trong process riêng. Raw report ghi nguồn baseline. Đây không phải phép đo interleaved loại bỏ hoàn toàn thermal/load drift.

## Hai runtime profiles

| Profile | p50 (s) | p95 (s) | ADD(-S) repeats |
|---|---:|---:|---|
| Fast, no recovery | 0.759 | 1.027 | 88.64% / 88.41% |
| Quality gate enabled | 1.190 | 2.002 | 88.64% / 88.64% |

`configs/optimized_fast.yaml` không chạy scalar recovery; `configs/optimized.yaml` bật gate. Hai profile đo ở processes/lượt khác nhau nên không tách toàn bộ overhead khỏi load drift. Nếu recovery không thay pose nào, không kết luận nó là nguyên nhân recall tốt hơn. Fast có một GT pass→fail ở repeat 1; case này và các lần chạy lại không phải bằng chứng không regression.

Đây là crossing tại ADD(-S)@0.1 diameter, không phải mọi numerical pose error đều giảm.

## Giới hạn

Cohort này không thay benchmark full 900-frame cũ; ADD(-S) không phải official BOP AR. Cùng weights nhưng batching/renderer có thể tạo chênh lệch số học, nên quality được chấm lại. Variants chạy tuần tự; thermal/background-load drift chưa bị loại hoàn toàn. Latency là inference trước reference scoring, sau warm-up, không bao gồm initialization và report plots. Không tuyên bố real-time hoặc robot success.

## Achievement có số đo

> Re-architected multi-object RGB-D 6D pose inference with batched estimation, GPU-resident CAD rendering, cached spatial indices, and on-demand scene reconstruction; reduced median latency by 37.4% and p95 latency by 35.9% on 96 YCB-V frames (440 GT instances), with ADD(-S) recall of 88.64–88.64% across two repeats (baseline 88.64%).

Kèm bảng ADD(-S) khi trình bày CV/phỏng vấn; không ghi “preserved accuracy” nếu bảng có giảm.

## Tái lập

`powershell -ExecutionPolicy Bypass -File scripts/workflows/run_architecture.ps1`

Demo: `artifacts/ycbv/architecture/demo/ycbv_test_000048_1074/instances_3d.html`. Thiết kế: [INFERENCE_ARCHITECTURE_VI.md](../../../docs/INFERENCE_ARCHITECTURE_VI.md).
