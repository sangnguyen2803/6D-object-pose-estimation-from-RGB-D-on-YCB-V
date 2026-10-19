"""Measured paired results and reproducible extension report."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from PIL import Image

from clutterpick.data.dataset import load_rgbd
from clutterpick.evaluation.reporting import per_object
from clutterpick.geometry.refinement import annotate_observability
from clutterpick.utils.common import read_json, resolve, write_json
from clutterpick.visualization.draw import overlay


def main():
    root = resolve("reports/ycbv/extension")
    result = read_json(root / "refinement_comparison.json")
    development = read_json(root / "synthetic_validation.json")
    holdout = read_json(root / "synthetic_holdout.json")
    rows = read_json(resolve("artifacts/ycbv/extension/replay/run_summary.json"))
    methods = Counter()
    review_groups = {}
    # Normalize top-level fit fields for earlier replay rows, using recorded diagnostics.
    for row in rows:
        path = Path(row["output"]) / "poses.json"
        payload = read_json(path)
        for pose in payload["instances"]:
            diagnostic = annotate_observability(pose["refinement_diagnostics"])
            pose["pose_quality"] = diagnostic["quality"]
            methods[diagnostic["selected_candidate"]] += 1
            pose["observed_trimmed_rmse_m"] = diagnostic["after"]["rmse_m"]
            pose["observed_coverage_10mm"] = diagnostic["after"]["coverage_10mm"]
            group = (
                "review"
                if pose["pose_quality"]["requires_review"]
                else "supported_and_constrained"
            )
            counts = review_groups.setdefault(
                group, {"predictions": 0, "correct_ADD_or_S": 0}
            )
            counts["predictions"] += 1
            evaluation = next(
                (
                    item
                    for item in row["pose_diagnostics"]
                    if item["instance_id"] == pose["instance_id"]
                ),
                None,
            )
            counts["correct_ADD_or_S"] += int(
                bool(evaluation and evaluation["ADD_or_S_success_0_1d"])
            )
        write_json(path, payload)
    result["candidate_counts"] = dict(methods)
    result["review_groups"] = review_groups
    baseline_rows = read_json(
        resolve("artifacts/ycbv/benchmark_full/run_summary.json")
    )[: len(rows)]
    baseline_classes = per_object(
        baseline_rows, read_json(resolve("data/ycbv/index.json"))
    )
    before = result["baseline"]["recall"]["ADD_or_S_success_0_1d"] * 100
    after = result["enhanced"]["recall"]["ADD_or_S_success_0_1d"] * 100
    gt = result["enhanced"]["gt_instances"]
    supported = result["quality_groups"].get("supported_geometry", {})
    result["geometry_support_policy"] = {
        "retained_predictions": supported.get("predictions", 0),
        "gt_recall": supported.get("correct_ADD_or_S", 0) / gt,
        "diagnostic_precision": supported.get("diagnostic_precision"),
        "calibrated_probability": False,
    }
    constrained = review_groups.get(
        "supported_and_constrained", {"predictions": 0, "correct_ADD_or_S": 0}
    )
    result["review_policy"] = {
        "retained_predictions": constrained["predictions"],
        "gt_recall": constrained["correct_ADD_or_S"] / gt,
        "diagnostic_precision": constrained["correct_ADD_or_S"]
        / max(constrained["predictions"], 1),
        "calibrated_probability": False,
    }
    write_json(root / "refinement_comparison.json", result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bars = axes[0].bar(
        ["Baseline", "Robust candidates"], [before, after], color=["#4477aa", "#228833"]
    )
    axes[0].bar_label(bars, fmt="%.2f%%")
    axes[0].set_ylim(0, 105)
    axes[0].set_title(f"Real YCB-V: {len(rows)} frames")
    values = [
        holdout["summary"][name]["mean_ADD_mm"] for name in ["baseline", "enhanced"]
    ]
    bars = axes[1].bar(
        ["Baseline", "Robust candidates"], values, color=["#4477aa", "#228833"]
    )
    axes[1].bar_label(bars, fmt="%.2f mm")
    axes[1].set_ylabel("Mean ADD (mm)")
    axes[1].set_title("Independent synthetic seed 271")
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Mở rộng YCBPose: robust depth refinement",
        "",
        "## Thay đổi thực hiện",
        "",
        "Bổ sung Huber point-to-plane ICP dùng CAD normals và damping cho cập nhật SE(3). "
        "So sánh candidate RGB, trimmed point-to-point và robust point-to-plane trên cùng raw observed points; "
        "chỉ chấp nhận candidate cải thiện fit và trong giới hạn cập nhật. "
        "Xuất quality diagnostics để phân biệt geometry được hỗ trợ, yếu hoặc thiếu depth. "
        "Đây là mở rộng implementation dựa phương pháp ICP đã biết, chưa tuyên bố thuật toán nghiên cứu hoàn toàn mới.",
        "",
        "## Kết quả đo",
        "",
        f"- Real YCB-V: {result['frames']} ảnh, {gt} GT instances, cùng detector và RGB neural poses đã lưu.",
        f"- ADD(-S)@0.1d: baseline **{before:.2f}%**, enhanced **{after:.2f}%**, chênh **{after - before:+.2f} điểm phần trăm**.",
        f"- Paired gains: {result['gains']}; regressions: {result['regressions']}.",
        f"- Candidate counts: {dict(methods)}.",
        "",
        "![So sánh](comparison.png)",
        "",
        "| Dữ liệu tổng hợp | Cases | Baseline mean ADD | Enhanced mean ADD |",
        "|---|---:|---:|---:|",
    ]
    for name, report in [("Seed 137", development), ("Holdout seed 271", holdout)]:
        lines.append(
            f"| {name} | {report['cases']} | {report['summary']['baseline']['mean_ADD_mm']:.2f} mm | "
            f"{report['summary']['enhanced']['mean_ADD_mm']:.2f} mm |"
        )
    lines += [
        "",
        "Hai seed dùng cùng cấu hình kỹ thuật cố định, 21 CAD, ba mức crop/outliers, depth noise 1 mm, "
        "pose khởi tạo bị nhiễu. Không dùng real test GT trong refinement hoặc synthetic generation. "
        "Synthetic holdout không thay thế validation RGB-D thực độc lập.",
        "",
        "## Quality diagnostics",
        "",
        "| Nhóm | Predictions | Correct ADD(-S) | Diagnostic precision |",
        "|---|---:|---:|---:|",
    ]
    for name, counts in result["quality_groups"].items():
        lines.append(
            f"| {name} | {counts['predictions']} | {counts['correct_ADD_or_S']} | {counts['diagnostic_precision'] * 100:.2f}% |"
        )
    lines += ["", "| Cờ review | Predictions | Correct ADD(-S) |", "|---|---:|---:|"]
    for name, counts in review_groups.items():
        lines.append(
            f"| {name} | {counts['predictions']} | {counts['correct_ADD_or_S']} |"
        )
    lines += [
        "",
        "Diagnostic precision tính correct/retained predictions, bao gồm false positives là lỗi. "
        "Không phải confidence đã calibration. Symmetry hoặc sai object identity vẫn có thể khớp geometry. "
        "Pipeline giữ các pose và gắn requires_review; chưa tự loại khỏi CSV.",
        "",
        "## Theo lớp",
        "",
        "| Object | GT | Baseline ADD(-S) | Enhanced ADD(-S) |",
        "|---|---:|---:|---:|",
    ]
    for obj, values in result["enhanced"]["per_object"].items():
        lines.append(
            f"| {obj} | {values['gt']} | {baseline_classes[obj]['ADD_or_S_success_0_1d'] * 100:.2f}% | {values['ADD_or_S_success_0_1d'] * 100:.2f}% |"
        )
    lines += [
        "",
        "## Chạy lại",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe main.py --config configs/enhanced.yaml infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/extension/demo",
        "powershell -ExecutionPolicy Bypass -File scripts/workflows/run_extension.ps1",
        "```",
        "",
        "## Giới hạn đánh giá",
        "",
        "Real comparison dùng cache predictions của benchmark đã quan sát, không phải validation riêng. "
        "GT chỉ dùng scoring sau prediction. Giữ metric ADD/ADD-S/ADD(-S) chẩn đoán, chưa có official BOP AR. "
        "CSV time của replay chỉ là geometry/refinement CPU, không so trực tiếp với latency end-to-end baseline. "
        "Cấu hình mới nằm configs/enhanced.yaml; configs/default.yaml tái lập baseline.",
    ]
    (root / "RESULTS_VI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    demo_rows = read_json(resolve("artifacts/ycbv/extension/demo/run_summary.json"))
    lookup = {
        frame["key"]: frame
        for frame in read_json(resolve("data/ycbv/index.json"))["frames"]
    }
    for row in demo_rows:
        directory = Path(row["output"])
        payload = read_json(directory / "poses.json")
        for pose in payload["instances"]:
            pose["pose_quality"] = annotate_observability(
                pose["refinement_diagnostics"]
            )["quality"]
        write_json(directory / "poses.json", payload)
        write_json(
            directory / "pose_quality.json",
            {
                "uses_gt": False,
                "scope": "geometric support and candidate observability; not calibrated probability",
                "instances": [
                    {
                        "instance_id": pose["instance_id"],
                        "obj_id": pose["obj_id"],
                        "diagnostic": pose["pose_quality"],
                    }
                    for pose in payload["instances"]
                ],
            },
        )
        items = read_json(directory / "detection.json")["instances"]
        for item in items:
            item["mask"] = np.asarray(Image.open(directory / item["mask_path"])) > 0
        rgb, _, k = load_rgbd(lookup[row["frame_key"]])
        overlay(
            rgb,
            items,
            k,
            directory / "final_visualization.png",
            poses=payload["instances"],
        )
    summary_path = resolve("PROJECT_SUMMARY_VI.md")
    summary = (
        summary_path.read_text(encoding="utf-8")
        .split("## 10. Mở rộng robust depth refinement")[0]
        .rstrip()
    )
    summary += "\n\n## 10. Mở rộng robust depth refinement\n\nBổ sung Huber point-to-plane ICP, candidate selection, geometric quality và observability review. "
    summary += f"Trên {result['frames']} ảnh giữ nguyên neural predictions: ADD(-S) {before:.2f}% → {after:.2f}%, "
    summary += f"{result['gains']} trường hợp cải thiện và {result['regressions']} trường hợp giảm chất lượng. "
    summary += f"Synthetic holdout seed 271, 63 cases: mean ADD {holdout['summary']['baseline']['mean_ADD_mm']:.2f} → {holdout['summary']['enhanced']['mean_ADD_mm']:.2f} mm.\n\n"
    summary += "Cấu hình enhanced riêng; không thay pretrained weights. Quality không phải xác suất correctness. "
    summary += "Real comparison vẫn là exploratory test evaluation; replay latency không phải end-to-end. "
    summary += "Chi tiết: [kết quả mở rộng](reports/ycbv/extension/RESULTS_VI.md), [phương pháp](docs/ROBUST_REFINEMENT_VI.md).\n"
    summary_path.write_text(summary, encoding="utf-8")
    print(
        {
            "baseline_ADD_or_S": before,
            "enhanced_ADD_or_S": after,
            "gains": result["gains"],
            "regressions": result["regressions"],
        }
    )


if __name__ == "__main__":
    main()
