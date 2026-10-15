# Dàn ý trình bày đồ án

1. **Bài toán:** Từ RGB-D phân biệt từng instance và xác định pose 6D của 21 vật thể YCB có CAD. Minh họa RGB input và overlay pose.
2. **Dataset:** YCB-V BOP19, 900 test frames, 4.125 GT instances; research subset PBR 300 train / 60 validation / 60 holdout, tách scene. Phân biệt pretraining công khai với training mới và khả năng pretrained đã gặp PBR holdout.
3. **Pipeline:** Mask R-CNN → backproject visible instance clouds → CosyPose coarse/refiner → bounded ICP. Nêu camera frame và thống nhất mét/mm.
4. **Demo:** Mở instances_3d.html, bật/tắt instance/CAD, xoay scene. Phân biệt depth quan sát và CAD căn chỉnh.
5. **Đánh giá:** COCO bbox/mask AP; ADD/ADD-S/ADD(-S)@0.1d, đếm misses trong GT denominator. Đưa chart pose_recall.png và bảng đầy đủ từ RESULTS_VI.
6. **Ablation:** Cùng 24 ảnh cho mask, bbox, reference-mask, RGB-only. Reference-mask là oracle, không phải pipeline độc lập.
7. **Thảo luận:** Vật thể mảnh/clamps, occlusion, depth holes, local ICP. Windows renderer khác original. Không dùng metric chẩn đoán thay BOP official AR.
8. **Đóng góp mới v0.4:** RGB/early/late/gated fusion, query Transformer mask head 56×56, local point-attention foreground filter trước robust ICP. Trình bày ablations và dropout/noise; đọc contamination cùng foreground retention và pose recall.
9. **Giới hạn/hướng tiếp:** Fine-tune với validation riêng, official BOP metrics, real camera/multi-view. Không tuyên bố grasp hoặc robot success.

Nguồn số liệu duy nhất cho slide kết quả là reports/ycbv/RESULTS_VI.md và measured_summary.json. Không dùng pilot hoặc kết quả IndustryShapes để mô tả benchmark YCB-V đầy đủ.

Kết quả learned modules ở [research/RESULTS_VI.md](../reports/ycbv/research/RESULTS_VI.md), trên 96 test frames / 24 pose frames. Không thay thế full 900-frame baseline. Query head thay cả resolution và decoder nên chưa tách attention contribution. Pipeline đầy đủ: [LEARNED_HYBRID_VI.md](LEARNED_HYBRID_VI.md).

