"""Summarize measured ablations and limitations; never fill missing results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.utils.common import read_json, resolve, write_json


def main():
    root = resolve("reports/ycbv/research")
    ablations = read_json(root / "ablation.json")
    point = read_json(root / "point_holdout.json")
    real = read_json(root / "point_real_comparison.json")
    combined = read_json(root / "combined_pose.json")
    # Derived paired diagnostics can also be rebuilt from an already-running evaluation.
    if (
        "paired" not in real
        or "paired_vs_gated_query_without_point_filter" not in combined
    ):
        from evaluate_research import paired_pose

        from clutterpick.data.dataset import load_index
        from clutterpick.utils.common import load_config

        index, _ = load_index(load_config("configs/research.yaml"))
        evaluation = resolve("artifacts/ycbv/research/evaluation")
        real["paired"] = paired_pose(
            read_json(evaluation / "point_baseline" / "run_summary.json"),
            read_json(evaluation / "point_network" / "pose" / "run_summary.json"),
            index,
        )
        combined["paired_vs_gated_query_without_point_filter"] = paired_pose(
            read_json(evaluation / "gated_query" / "pose" / "run_summary.json"),
            read_json(evaluation / "combined" / "pose" / "run_summary.json"),
            index,
        )
        write_json(root / "point_real_comparison.json", real)
        write_json(root / "combined_pose.json", combined)
    published = read_json(root / "published_detector.json")
    point_baselines = read_json(root / "point_holdout_baselines.json")
    training = read_json(root / "fusion_training.json")
    rows = []
    for name, value in ablations.items():
        rows.append(
            {
                "variant": name,
                "mask_AP": value["test"]["metrics"]["segm"]["AP_50_95"],
                "boundary_AP": value["test"]["metrics"]["boundary"]["AP_50_95"],
                "pose_ADD_or_S": value["pose"]["recall"]["ADD_or_S_success_0_1d"],
                "contamination": value["pose"]["mean_contamination"],
            }
        )
    write_json(
        root / "measured_summary.json",
        {
            "ablations": rows,
            "published_detector": published,
            "point_holdout": point,
            "point_real": real,
            "combined": combined,
            "test_cohort_frames": 96,
            "pose_cohort_frames": 24,
            "trained_on_real_test": False,
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [row["mask_AP"] * 100 for row in rows], 0.36, label="Mask AP")
    ax.bar(
        x + 0.18, [row["boundary_AP"] * 100 for row in rows], 0.36, label="Boundary AP"
    )
    ax.set_xticks(x, [row["variant"] for row in rows], rotation=20)
    ax.set_ylabel("AP (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "fusion_boundary.png", dpi=180)
    plt.close(fig)
    lines = [
        "# YCBPose: ba nhánh học mới",
        "",
        "## Phạm vi triển khai",
        "",
        "1. RGB-only, early fusion 4 channels, late two-branch FPN fusion và gated mid-level fusion. "
        "Depth là cùng scalar metric depth, nhánh depth ba kênh lặp scalar để tái dùng CNN RGB initialization. "
        "RGB trunk frozen, depth layer4/head/adapters được train; early fusion adapt stem bốn kênh.",
        "2. CNN mask pooling 28×28 + lightweight spatial-query transformer decoder, output masks 56×56. "
        "Đây là query-based head tự triển khai, không phải bản PointRend chính thức. "
        "Comparison thay đồng thời resolution và decoder, chưa tách riêng tác dụng attention với resolution.",
        "3. Class-conditioned local point-attention network, predict foreground/background từng điểm "
        "trong ROI; filter trước robust ICP. Nhãn point lấy từ visible GT masks backproject vào RGB-D, không có full-volume 3D labels.",
        "",
        "## Dữ liệu và protocol",
        "",
        "300 PBR train frames, 60 validation frames, 60 held-out PBR frames, tách scene. "
        "420 frames subset được tải HTTP ranges, verified ZIP CRC/member SHA256. "
        "SHA256 archive toàn bộ được ghi expected nhưng không verified vì không tải archive 21 GB. "
        "Published CosyPose pretraining có thể đã dùng các PBR scenes; holdout chỉ độc lập với "
        "fine-tuning/module training mới, không phải holdout chưa từng được toàn bộ pretrained pipeline nhìn thấy.",
        "",
        "Checkpoint/threshold chọn bằng validation; real test chỉ dùng scoring. "
        "Real test cohort cố định 96 ảnh, pose cohort 24 ảnh trên 12 scenes. Không trộn với kết quả full 900 ảnh baseline. "
        "Train budget giống nhau: bốn epochs, 300 frames mỗi variant. Depth modality dropout 20%, pixel dropout 10%, noise sigma 2 mm khi train.",
        "",
        "## Checkpoint chọn trên validation",
        "",
        "| Variant | Best epoch | Val mask AP | Val boundary AP |",
        "|---|---:|---:|---:|",
    ]
    for name, history in training.items():
        best = max(history, key=lambda row: row["val_mask_AP"])
        lines.append(
            f"| {name} | {best['epoch']} | {best['val_mask_AP'] * 100:.2f}% | {best['val_boundary_AP'] * 100:.2f}% |"
        )
    lines += [
        "",
        "## Mask và pose trên test cohort",
        "",
        "| Variant | Mask AP | Boundary AP | ADD(-S), 24 ảnh | Contamination |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.append(
        f"| Published RGB detector | {published['test']['metrics']['segm']['AP_50_95'] * 100:.2f}% | "
        f"{published['test']['metrics']['boundary']['AP_50_95'] * 100:.2f}% | "
        f"{real['original_2d_mask']['recall']['ADD_or_S_success_0_1d'] * 100:.2f}% | "
        f"{real['original_2d_mask']['mean_contamination'] * 100:.2f}% |"
    )
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['mask_AP'] * 100:.2f}% | {row['boundary_AP'] * 100:.2f}% | "
            f"{row['pose_ADD_or_S'] * 100:.2f}% | {row['contamination'] * 100:.2f}% |"
        )
    lines += [
        "",
        "![Fusion và boundary](fusion_boundary.png)",
        "",
        "## Robustness depth",
        "",
        "| Variant | Clean AP | All missing | Drop 30% | Drop 80% | Noise 5 mm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in ablations.items():
        if not value["robustness"]:
            continue
        values = [value["test"]["metrics"]["segm"]["AP_50_95"]] + [
            value["robustness"][key]["metrics"]["segm"]["AP_50_95"]
            for key in ["missing_all", "dropout_0.3", "dropout_0.8", "noise_5mm"]
        ]
        lines.append(
            "| " + name + " | " + " | ".join(f"{v * 100:.2f}%" for v in values) + " |"
        )
    lines += [
        "",
        "## Các lớp mảnh / nhỏ",
        "",
        "Subset cho power drill (15), wood block (16), scissors (17), markers (18), clamps (19/20). "
        "Mỗi ô là Mask AP / Boundary AP / pose ADD(-S) / contamination. Số GT pose nhỏ cần đọc như pilot.",
        "",
        "| Object ID | GT pose | " + " | ".join(ablations) + " |",
        "|---|---:|" + "|".join(["---:"] * len(ablations)) + "|",
    ]
    for object_id in [15, 16, 17, 18, 19, 20]:
        key = str(object_id)
        cells = []
        count = 0
        for value in ablations.values():
            pose = value["pose"]["per_object"].get(key)
            count = pose["gt"] if pose else count
            metrics = [
                value["test"]["metrics"][kind]["per_class_AP"].get(key)
                for kind in ["segm", "boundary"]
            ]
            metrics.append(pose["ADD_or_S_success_0_1d"] if pose else None)
            metrics.append(pose["mean_contamination"] if pose else None)
            cells.append(
                " / ".join(
                    f"{number * 100:.1f}%" if number is not None else "N/A"
                    for number in metrics
                )
            )
        lines.append(
            "| "
            + str(object_id)
            + " | "
            + str(count)
            + " | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Mạng 3D",
        "",
        f"- PBR holdout: {point['cases']} matched ROI examples; fixed threshold {point['fixed_validation_threshold']}.",
        f"- Foreground precision {point['metrics']['precision'] * 100:.2f}%, recall {point['metrics']['recall'] * 100:.2f}%, F1 {point['metrics']['F1'] * 100:.2f}%.",
        "- Các chỉ số point là trên proposals ghép được, không phải scene-level segmentation accuracy.",
        "",
        "| PBR held-out ROI classifier | Precision | Recall | F1 | Contamination |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in list(point_baselines["metrics"].items()) + [
        ("learned point network", point["metrics"])
    ]:
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                f"{metrics[key] * 100:.2f}%"
                for key in ["precision", "recall", "F1", "contamination"]
            )
            + " |"
        )
    lines += [
        "",
        "| Pipeline (cùng RGB detector và robust ICP) | Contamination | ADD(-S), 24 ảnh |",
        "|---|---:|---:|",
    ]
    for name, key in [
        ("Không learned filter", "original_2d_mask"),
        ("Có learned 3D filter", "learned_3d"),
    ]:
        value = real[key]
        lines.append(
            f"| {name} | {value['mean_contamination'] * 100:.2f}% | {value['recall']['ADD_or_S_success_0_1d'] * 100:.2f}% |"
        )
    if real["learned_3d"].get("mean_foreground_retention") is not None:
        lines += [
            "",
            f"Mean foreground retention theo GT projection: {real['learned_3d']['mean_foreground_retention'] * 100:.2f}%. "
            "Tính trên matched baseline instances, trường hợp mất matching được tính 0.",
        ]
    lines += [
        "",
        f"Paired ADD(-S)@0.1d: {real['paired']['fail_to_pass']} fail→pass, "
        f"{real['paired']['pass_to_fail']} pass→fail. Đây là crossing tại ngưỡng, không phải mọi numerical pose error.",
    ]
    lines += [
        "",
        "## Pipeline đủ cả ba module",
        "",
        f"Gated RGB-D + query mask head + learned point filter + robust ICP: ADD(-S) "
        f"{combined['recall']['ADD_or_S_success_0_1d'] * 100:.2f}% trên {combined['frames']} ảnh / "
        f"{combined['gt_instances']} GT; contamination {combined['mean_contamination'] * 100:.2f}%.",
        "",
        "Demo tại `artifacts/ycbv/research/demo/ycbv_test_000048_1074`: RGB overlays, "
        "PLY instance clouds/CAD, JSON pose và BOP CSV. Viewer HTML được tạo bằng `visualize_instances.py`.",
    ]
    lines += [
        "",
        "## Cách đọc và giới hạn",
        "",
        "Không khẳng định mọi module đều cải thiện mọi metric: bảng giữ cả kết quả giảm. "
        "Boundary AP dùng min(mask IoU, inner-boundary IoU) với boundary width 2% image diagonal theo protocol "
        "[Boundary IoU](https://github.com/bowenc0221/boundary-iou-api). "
        "Contamination là trung bình per matched instance từ point projection vào visible GT mask; "
        "chưa phản ánh mọi depth misregistration. Pose là ADD(-S) chẩn đoán, không phải official BOP AR.",
        "",
        "Point cleaner có fallback nếu giữ dưới 40 points hoặc dưới 25% cloud. "
        "Tỷ lệ contamination thấp cần đọc cùng foreground retention/pose recall; filter mạnh có thể bỏ mất điểm đúng. "
        "Chưa có benchmark camera mới hoặc đánh giá robot.",
        "",
        "## Chạy lại",
        "",
        "```powershell",
        "powershell -ExecutionPolicy Bypass -File scripts/workflows/run_research.ps1",
        "```",
    ]
    rgb = next(row for row in rows if row["variant"] == "rgb")
    query = next(row for row in rows if row["variant"] == "rgb_query")
    published_ap = published["test"]["metrics"]["segm"]["AP_50_95"]
    strongest = max(rows, key=lambda row: row["mask_AP"])
    comments = [
        "",
        "## Nhận xét từ số đo",
        "",
        f"- Mask AP cao nhất trong sáu biến thể: {strongest['variant']} {strongest['mask_AP'] * 100:.2f}%; "
        f"published detector cùng cohort {published_ap * 100:.2f}%. Đây là mô tả test, không dùng để chọn epoch.",
        f"- RGB query head so với RGB head thường: mask AP {(query['mask_AP'] - rgb['mask_AP']) * 100:+.2f} pp, "
        f"boundary AP {(query['boundary_AP'] - rgb['boundary_AP']) * 100:+.2f} pp.",
        f"- Learned filter với cùng published detector: ADD(-S) "
        f"{(real['learned_3d']['recall']['ADD_or_S_success_0_1d'] - real['original_2d_mask']['recall']['ADD_or_S_success_0_1d']) * 100:+.2f} pp.",
        "- Train budget nhỏ và chỉ PBR có thể gây domain gap / forgetting với ảnh thật; thí nghiệm này chưa xác định nguyên nhân riêng biệt. "
        "Kiến trúc đã triển khai và đo được; chưa đủ chứng minh mọi nhánh nâng cấp tốt hơn pretrained baseline.",
    ]
    lines += comments
    external_path = root / "external_input_validation.json"
    if external_path.exists():
        external = read_json(external_path)
        lines += [
            "",
            "## Registered RGB-D không nhãn",
            "",
            "External CLI đã chạy đủ ba module với GT count = 0, xuất pose hợp lệ. "
            "Masks và per-instance filtered clouds trùng với lần chạy có reference labels; "
            "GT không được cung cấp vào external CLI.",
            "",
            f"Pose giữa hai processes có chênh lệch tối đa "
            f"{max(row['translation_mm'] for row in external['pose_cross_process_deviations']):.3f} mm / "
            f"{max(row['rotation_degrees'] for row in external['pose_cross_process_deviations']):.3f}°. "
            "Không khẳng định pose bitwise-equivalent hoặc quy chênh lệch này thành tác động của GT. "
            "Xem `external_input_validation.json`.",
        ]
        if external.get("initial_strict_comparison"):
            trial = external["initial_strict_comparison"]
            lines += [
                "",
                f"Lần kiểm tra đầu với tolerance elementwise 1e-5 không đạt: "
                f"đã quan sát tối đa {trial['max_translation_mm']:.3f} mm / {trial['max_rotation_degrees']:.3f}°. "
                "Các giá trị nhỏ hơn ở lần chạy lại không thay thế chênh lệch đã quan sát ban đầu.",
            ]
    (root / "RESULTS_VI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    snapshot = [
        "<!-- learned-results-start -->",
        "## Kết quả learned modules trên cohort riêng",
        "",
        "Đo trên 96 test frames / 440 GT cho segmentation và 24 frames / 110 GT cho pose. "
        "Các số này không thay benchmark full 900 ảnh.",
        "",
        "| Variant | Mask AP | Boundary AP | ADD(-S) |",
        "|---|---:|---:|---:|",
    ]
    snapshot.append(
        f"| Published RGB + robust ICP | {published_ap * 100:.2f}% | "
        f"{published['test']['metrics']['boundary']['AP_50_95'] * 100:.2f}% | "
        f"{real['original_2d_mask']['recall']['ADD_or_S_success_0_1d'] * 100:.2f}% |"
    )
    for row in rows:
        snapshot.append(
            f"| {row['variant']} | {row['mask_AP'] * 100:.2f}% | {row['boundary_AP'] * 100:.2f}% | {row['pose_ADD_or_S'] * 100:.2f}% |"
        )
    snapshot += [
        "",
        f"Learned 3D filter với cùng published detector: contamination "
        f"{real['original_2d_mask']['mean_contamination'] * 100:.2f}% → {real['learned_3d']['mean_contamination'] * 100:.2f}%; "
        f"ADD(-S) {real['original_2d_mask']['recall']['ADD_or_S_success_0_1d'] * 100:.2f}% → "
        f"{real['learned_3d']['recall']['ADD_or_S_success_0_1d'] * 100:.2f}%. "
        f"Pipeline đủ ba module đạt {combined['recall']['ADD_or_S_success_0_1d'] * 100:.2f}% ADD(-S).",
        "",
        "Train là pilot 300 PBR frames, checkpoint/threshold chọn validation; pretrained có thể đã gặp PBR holdout. "
        "Chi tiết và các kết quả giảm: [research report](reports/ycbv/research/RESULTS_VI.md).",
        "<!-- learned-results-end -->",
    ]
    import re

    for name in ["README.md", "PROJECT_SUMMARY_VI.md"]:
        path = resolve(name)
        content = path.read_text(encoding="utf-8")
        block = "\n".join(snapshot) + "\n"
        if "<!-- learned-results-start -->" in content:
            content = re.sub(
                r"<!-- learned-results-start -->.*?<!-- learned-results-end -->\n?",
                lambda _: block,
                content,
                flags=re.S,
            )
        else:
            content += "\n" + block
        path.write_text(content, encoding="utf-8")
    print(rows)


if __name__ == "__main__":
    main()
