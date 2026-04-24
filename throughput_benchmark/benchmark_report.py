import csv
from pathlib import Path

from throughput_benchmark.benchmark_types import BenchmarkResult


def write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "batch",
                "in_features",
                "out_features",
                "copies",
                "method",
                "latency_ms",
                "tflops",
                "memory_mb",
                "block_m",
                "block_n",
                "block_k",
                "num_warps",
                "num_stages",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "layer": row.layer,
                    "batch": row.batch,
                    "in_features": row.in_features,
                    "out_features": row.out_features,
                    "copies": row.copies,
                    "method": row.method,
                    "latency_ms": f"{row.latency_ms:.6f}",
                    "tflops": f"{row.tflops:.6f}",
                    "memory_mb": f"{row.memory_mb:.6f}",
                    "block_m": format_optional(row.kernel_config.block_m),
                    "block_n": format_optional(row.kernel_config.block_n),
                    "block_k": format_optional(row.kernel_config.block_k),
                    "num_warps": format_optional(row.kernel_config.num_warps),
                    "num_stages": format_optional(row.kernel_config.num_stages),
                }
            )


def format_optional(value: int | None) -> str:
    return "" if value is None else str(value)


def format_kernel_config(row: BenchmarkResult) -> str:
    kernel_config = row.kernel_config
    if kernel_config.block_m is None:
        return ""

    parts = [
        f"BM={kernel_config.block_m}",
        f"BN={kernel_config.block_n}",
        f"BK={kernel_config.block_k}",
    ]
    if kernel_config.num_warps is not None:
        parts.append(f"warps={kernel_config.num_warps}")
    if kernel_config.num_stages is not None:
        parts.append(f"stages={kernel_config.num_stages}")

    return "(" + ", ".join(parts) + ")"


def write_markdown(path: Path, results: list[BenchmarkResult]) -> None:
    md_path = path.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "layer",
        "batch",
        "IN",
        "OUT",
        "method",
        "latency_ms",
        "TFLOP/s",
        "weight_MB",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "warps",
        "stages",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.layer,
                    str(row.batch),
                    str(row.in_features),
                    str(row.out_features),
                    row.method,
                    f"{row.latency_ms:.3f}",
                    f"{row.tflops:.3f}",
                    f"{row.memory_mb:.2f}",
                    format_optional(row.kernel_config.block_m),
                    format_optional(row.kernel_config.block_n),
                    format_optional(row.kernel_config.block_k),
                    format_optional(row.kernel_config.num_warps),
                    format_optional(row.kernel_config.num_stages),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(results: list[BenchmarkResult]) -> None:
    for row in results:
        kernel_config = format_kernel_config(row)
        print(
            f"{row.layer:9s} B={row.batch:<4d} "
            f"IN={row.in_features:<5d} OUT={row.out_features:<6d} "
            f"{row.method:19s} {row.latency_ms:8.3f} ms "
            f"{row.tflops:8.3f} TFLOP/s {kernel_config}"
        )
