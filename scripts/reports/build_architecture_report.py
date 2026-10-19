"""Measured architecture results and CV wording; no assumed gains."""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.utils.common import read_json, resolve, write_json


def successes(rows):
    return {
        item["reference_instance_id"]
        for row in rows
        for item in row["pose_diagnostics"]
        if item["ADD_or_S_success_0_1d"]
    }


def main():
    root = resolve("reports/ycbv/architecture")
    component = root / "component_ablation.json"
    pilot = read_json(component if component.exists() else root / "pilot.json")
    resident = read_json(root / "resident_ablation.json")
    source = max(
        [
            path
            for path in [root / "benchmark96.json", root / "recovery96.json"]
            if path.exists()
        ],
        key=lambda path: path.stat().st_mtime,
    )
    benchmark = read_json(source)
    variants = benchmark["variants"]
    base, opt = variants["baseline"], variants["optimized"]
    p50_reduction = 1 - opt["timing"]["p50_s"] / base["timing"]["p50_s"]
    p95_reduction = 1 - opt["timing"]["p95_s"] / base["timing"]["p95_s"]
    speedup = opt["timing"]["frames_per_s"] / base["timing"]["frames_per_s"]
    paired = []
    for repeat in range(benchmark["repetitions"]):
        first = read_json(
            Path(base["output"]) / f"repeat_{repeat}" / "run_summary.json"
        )
        second = read_json(
            Path(opt["output"]) / f"repeat_{repeat}" / "run_summary.json"
        )
        a, b = successes(first), successes(second)
        paired.append(
            {
                "repeat": repeat,
                "fail_to_pass": len(b - a),
                "pass_to_fail": len(a - b),
                "baseline_recall": base["repeats"][repeat]["recall"][
                    "ADD_or_S_success_0_1d"
                ],
                "optimized_recall": opt["repeats"][repeat]["recall"][
                    "ADD_or_S_success_0_1d"
                ],
            }
        )
    summary = {
        "p50_latency_reduction_fraction": p50_reduction,
        "p95_latency_reduction_fraction": p95_reduction,
        "throughput_speedup": speedup,
        "frames": benchmark["cohort_size"],
        "gt_instances": base["repeats"][0]["gt_instances"],
        "paired": paired,
        "baseline_timing": base["timing"],
        "optimized_timing": opt["timing"],
        "scope": benchmark["scope"],
        "hardware": benchmark["hardware"],
    }
    summary["source_report"] = source.name
    summary["reused_baseline_report"] = benchmark.get("reused_baseline_report")
    write_json(root / "measured_summary.json", summary)
    lines = [
        "# YCBPose: cải thiện kiến trúc inference",
        "",
        "Giữ detector/CosyPose weights và robust ICP. Kiến trúc mới gom instances thành batch, giữ CAD meshes "
        "thường trú trong renderer, cache KD-tree theo class và dựng scene clouds khi cần. Có optional scalar recovery "
        "cho weak geometry. Không fine-tune trên test.",
        "",
        "## Component ablation: 24 ảnh",
        "",
        "Phase đầu batch/cache/lazy chưa có resident renderer. Resident-only được đo ở phase sau. "
        "Các phase chạy tuần tự nên không quy mọi chênh lệch thời gian thành tác dụng thuật toán.",
        "",
        "| Variant | p50 (s) | p95 (s) | ADD(-S), từng repeat |",
        "|---|---:|---:|---|",
    ]
    for name, value in list(pilot["variants"].items()) + list(
        resident["variants"].items()
    ):
        label = (
            "batch_cache_lazy_v1"
            if name == "optimized"
            and not value["config"]["pose"].get("resident_renderer")
            else name
        )
        recalls = " / ".join(
            f"{part['recall']['ADD_or_S_success_0_1d'] * 100:.2f}%"
            for part in value["repeats"]
        )
        lines.append(
            f"| {label} | {value['timing']['p50_s']:.3f} | {value['timing']['p95_s']:.3f} | {recalls} |"
        )
    lines += [
        "",
        "## Đối chứng mở rộng",
        "",
        f"{summary['frames']} ảnh / {summary['gt_instances']} GT, {benchmark['repetitions']} measured repeats mỗi cấu hình, "
        f"warm-up {summary['frames']} frames mỗi cấu hình. GPU: {summary['hardware']}.",
        "",
        "| Metric | Sequential robust baseline | Optimized |",
        "|---|---:|---:|",
    ]
    for name, key in [
        ("p50 latency (s)", "p50_s"),
        ("p95 latency (s)", "p95_s"),
        ("Throughput (frames/s)", "frames_per_s"),
    ]:
        lines.append(
            f"| {name} | {base['timing'][key]:.3f} | {opt['timing'][key]:.3f} |"
        )
    for repeat, value in enumerate(paired):
        lines.append(
            f"| ADD(-S) repeat {repeat} | {value['baseline_recall'] * 100:.2f}% | {value['optimized_recall'] * 100:.2f}% |"
        )
    lines += [
        "",
        f"Median latency giảm **{p50_reduction * 100:.2f}%**, p95 giảm **{p95_reduction * 100:.2f}%**; "
        f"throughput **{speedup:.2f}×** so với baseline đo trong cùng benchmark.",
        "",
        "## Renderer và bộ nhớ",
        "",
        "| Variant | Render calls sau warm-up | Mesh context additions | Peak Torch MiB |",
        "|---|---:|---:|---:|",
    ]
    for name, value in variants.items():
        lines.append(
            f"| {name} | {value['renderer']['render_calls']} | {value['renderer']['context_mesh_additions']} | "
            f"{value['memory']['peak_torch_allocated_MiB']:.1f} |"
        )
    lines += [
        "",
        "Peak Torch memory không bao gồm OpenGL/driver. Resident mesh cache dùng thêm GPU graphics memory; "
        "không tuyên bố giảm tổng VRAM. Pixel equivalence test có trong `renderer_validation.json`.",
        "",
        "## Paired pose cases",
        "",
        "| Repeat | Fail→pass | Pass→fail |",
        "|---|---:|---:|",
    ]
    for value in paired:
        lines.append(
            f"| {value['repeat']} | {value['fail_to_pass']} | {value['pass_to_fail']} |"
        )
    if opt["config"]["pose"].get("quality_retry"):
        attempts = selected = 0
        for repeat in range(benchmark["repetitions"]):
            for row in read_json(
                Path(opt["output"]) / f"repeat_{repeat}" / "run_summary.json"
            ):
                for pose in read_json(Path(row["output"]) / "poses.json")["instances"]:
                    recovery = pose.get("quality_recovery", {})
                    attempts += int(recovery.get("attempted", False))
                    selected += int(recovery.get("selected", False))
        summary["quality_recovery"] = {
            "attempted_objects": attempts,
            "selected_objects": selected,
        }
        lines += [
            "",
            f"Quality-aware scalar recovery: {attempts} object attempts, {selected} replacements qua hai repeats. "
            "Chỉ thay khi geometry supported, RMSE thấp hơn và coverage không giảm; cùng weights và bounds.",
        ]
        write_json(root / "measured_summary.json", summary)
    if summary["reused_baseline_report"]:
        lines += [
            "",
            "Baseline được tái dùng từ lượt đo sequential trước đó trên cùng máy/cohort; "
            "bản quality-recovery được đo sau, trong process riêng. Raw report ghi nguồn baseline. "
            "Đây không phải phép đo interleaved loại bỏ hoàn toàn thermal/load drift.",
        ]
    original_variants = read_json(root / "benchmark96.json")["variants"]
    fast = original_variants.get("optimized_fast", original_variants["optimized"])
    if not fast["config"]["pose"].get("quality_retry"):
        import yaml

        fast_cfg = fast["config"].copy()
        fast_cfg["pose"] = fast_cfg["pose"].copy()
        fast_cfg["pose"]["quality_retry"] = False
        resolve("configs/optimized_fast.yaml").write_text(
            yaml.safe_dump(fast_cfg, sort_keys=False), encoding="utf-8"
        )
        lines += [
            "",
            "## Hai runtime profiles",
            "",
            "| Profile | p50 (s) | p95 (s) | ADD(-S) repeats |",
            "|---|---:|---:|---|",
        ]
        for name, value in [("Fast, no recovery", fast), ("Quality gate enabled", opt)]:
            recalls = " / ".join(
                f"{part['recall']['ADD_or_S_success_0_1d'] * 100:.2f}%"
                for part in value["repeats"]
            )
            lines.append(
                f"| {name} | {value['timing']['p50_s']:.3f} | {value['timing']['p95_s']:.3f} | {recalls} |"
            )
        lines += [
            "",
            "`configs/optimized_fast.yaml` không chạy scalar recovery; `configs/optimized.yaml` bật gate. "
            "Hai profile đo ở processes/lượt khác nhau nên không tách toàn bộ overhead khỏi load drift. "
            "Nếu recovery không thay pose nào, không kết luận nó là nguyên nhân recall tốt hơn. "
            "Fast có một GT pass→fail ở repeat 1; case này và các lần chạy lại không phải bằng chứng không regression.",
        ]
    lines += [
        "",
        "Đây là crossing tại ADD(-S)@0.1 diameter, không phải mọi numerical pose error đều giảm.",
        "",
        "## Giới hạn",
        "",
        "Cohort này không thay benchmark full 900-frame cũ; ADD(-S) không phải official BOP AR. "
        "Cùng weights nhưng batching/renderer có thể tạo chênh lệch số học, nên quality được chấm lại. "
        "Variants chạy tuần tự; thermal/background-load drift chưa bị loại hoàn toàn. "
        "Latency là inference trước reference scoring, sau warm-up, không bao gồm initialization và report plots. "
        "Không tuyên bố real-time hoặc robot success.",
        "",
        "## Achievement có số đo",
        "",
        f"> Re-architected multi-object RGB-D 6D pose inference with batched estimation, GPU-resident CAD rendering, "
        f"cached spatial indices, and on-demand scene reconstruction; reduced median latency by {p50_reduction * 100:.1f}% "
        f"and p95 latency by {p95_reduction * 100:.1f}% on {summary['frames']} YCB-V frames ({summary['gt_instances']} GT instances), "
        f"with ADD(-S) recall of {min(value['optimized_recall'] for value in paired) * 100:.2f}–"
        f"{max(value['optimized_recall'] for value in paired) * 100:.2f}% across two repeats "
        f"(baseline {paired[0]['baseline_recall'] * 100:.2f}%).",
        "",
        "Kèm bảng ADD(-S) khi trình bày CV/phỏng vấn; không ghi “preserved accuracy” nếu bảng có giảm.",
        "",
        "## Tái lập",
        "",
        "`powershell -ExecutionPolicy Bypass -File scripts/workflows/run_architecture.ps1`",
        "",
        "Demo: `artifacts/ycbv/architecture/demo/ycbv_test_000048_1074/instances_3d.html`. "
        "Thiết kế: [INFERENCE_ARCHITECTURE_VI.md](../../../docs/INFERENCE_ARCHITECTURE_VI.md).",
    ]
    (root / "RESULTS_VI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    ax.bar(
        x - 0.18, [base["timing"]["p50_s"], opt["timing"]["p50_s"]], 0.36, label="p50"
    )
    ax.bar(
        x + 0.18, [base["timing"]["p95_s"], opt["timing"]["p95_s"]], 0.36, label="p95"
    )
    ax.set_xticks(x, ["Sequential baseline", "Optimized architecture"])
    ax.set_ylabel("Seconds / frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "latency.png", dpi=160)
    plt.close(fig)
    block = (
        "\n".join(
            [
                "<!-- architecture-results-start -->",
                "## Kiến trúc inference đa vật thể",
                "",
                f"Benchmark {summary['frames']} ảnh / {summary['gt_instances']} GT: median latency "
                f"{base['timing']['p50_s']:.3f} → {opt['timing']['p50_s']:.3f} s ({p50_reduction * 100:.2f}% giảm); "
                f"p95 {base['timing']['p95_s']:.3f} → {opt['timing']['p95_s']:.3f} s.",
                "",
                "Bản mới: batching, GPU-resident renderer, CAD KD-tree cache và lazy scene geometry; "
                "weights giữ nguyên. Chất lượng từng repeat và giới hạn: "
                "[architecture results](reports/ycbv/architecture/RESULTS_VI.md). "
                "Thiết kế/cách chạy: [inference architecture](docs/INFERENCE_ARCHITECTURE_VI.md).",
                "<!-- architecture-results-end -->",
            ]
        )
        + "\n"
    )
    for name in ["README.md", "PROJECT_SUMMARY_VI.md"]:
        path = resolve(name)
        content = path.read_text(encoding="utf-8")
        if "<!-- architecture-results-start -->" in content:
            content = re.sub(
                r"<!-- architecture-results-start -->.*?<!-- architecture-results-end -->\n?",
                lambda _: block,
                content,
                flags=re.S,
            )
        else:
            content += "\n" + block
        path.write_text(content, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
