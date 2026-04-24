from typing import Callable

import torch

from throughput_benchmark import benchmark_config as config
from throughput_benchmark.benchmark_report import print_summary, write_csv, write_markdown
from throughput_benchmark.benchmark_types import (
    BenchmarkCase,
    BenchmarkMethod,
    BenchmarkResult,
    KernelConfig,
)
from triton_kernels.quantization_kernels import QuantizedKernel, get_quantized_kernel
from utils.cuda import synchronize
from utils.memory import tensor_memory_mib


def make_cases(skip_lm_head: bool) -> list[BenchmarkCase]:
    layers = config.LLAMA_3_2_1B_LINEAR_LAYERS
    if skip_lm_head:
        layers = [layer for layer in layers if layer[0] != "lm_head"]
    batches = config.BATCH_SIZES
    return [
        BenchmarkCase(layer, batch, in_features, out_features, copies)
        for layer, in_features, out_features, copies in layers
        for batch in batches
    ]


def measure_latency_ms(fn: Callable[[], torch.Tensor], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    synchronize()
    start_event.record()
    for _ in range(repeat):
        fn()
    end_event.record()
    synchronize()
    return start_event.elapsed_time(end_event) / repeat


def matmul_tflops(batch: int, in_features: int, out_features: int, latency_ms: float) -> float:
    ops = 2 * batch * in_features * out_features
    latency_s = latency_ms / 1000.0
    return ops / latency_s / 1e12


def fixed_kernel_config(block_m: int, block_n: int, block_k: int) -> KernelConfig:
    return KernelConfig(block_m=block_m, block_n=block_n, block_k=block_k)


def autotuned_kernel_config(kernel: QuantizedKernel) -> KernelConfig:
    if kernel.get_autotuned_config is None:
        return KernelConfig()

    best_config = kernel.get_autotuned_config()
    if best_config is None:
        return KernelConfig()

    return KernelConfig(
        block_m=best_config.get("block_m"),
        block_n=best_config.get("block_n"),
        block_k=best_config.get("block_k"),
        num_warps=best_config.get("num_warps"),
        num_stages=best_config.get("num_stages"),
    )


def make_methods(
    x: torch.Tensor,
    w: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    kernel: QuantizedKernel,
    in_features: int,
    block_m: int,
    block_n: int,
    block_k: int,
    skip_autotuned: bool,
) -> list[BenchmarkMethod]:
    fp16_weight_memory = tensor_memory_mib(w)
    w4_weight_memory = tensor_memory_mib(w_packed, w_scales)

    methods = [
        BenchmarkMethod(
            name="torch_fp16",
            run=lambda: x @ w.T,
            weight_memory_mb=fp16_weight_memory,
        ),
        BenchmarkMethod(
            name="triton_fixed",
            run=lambda: kernel.matmul(
                x,
                w_packed,
                w_scales,
                in_features,
                block_m,
                block_n,
                block_k,
            ),
            weight_memory_mb=w4_weight_memory,
            get_kernel_config=lambda: fixed_kernel_config(block_m, block_n, block_k),
        ),
    ]

    if kernel.reference_matmul is not None:
        methods.insert(
            1,
            BenchmarkMethod(
                name="torch_quantized_ref",
                run=lambda: kernel.reference_matmul(x, w_packed, w_scales, in_features),
                weight_memory_mb=w4_weight_memory,
            ),
        )

    if not skip_autotuned and kernel.matmul_autotuned is not None:
        methods.append(
            BenchmarkMethod(
                name="triton_autotuned",
                run=lambda: kernel.matmul_autotuned(
                    x,
                    w_packed,
                    w_scales,
                    in_features,
                ),
                weight_memory_mb=w4_weight_memory,
                get_kernel_config=lambda: autotuned_kernel_config(kernel),
            )
        )

    return methods


def benchmark_case(
    case: BenchmarkCase,
    warmup: int,
    repeat: int,
    block_m: int,
    block_n: int,
    block_k: int,
    skip_autotuned: bool,
    kernel_name: str,
) -> list[BenchmarkResult]:
    kernel = get_quantized_kernel(kernel_name)
    x = torch.randn(case.batch, case.in_features, device="cuda", dtype=torch.float16)
    w = torch.randn(case.out_features, case.in_features, device="cuda", dtype=torch.float16)
    w_packed, w_scales = kernel.quantize(w)
    methods = make_methods(
        x=x,
        w=w,
        w_packed=w_packed,
        w_scales=w_scales,
        kernel=kernel,
        in_features=case.in_features,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        skip_autotuned=skip_autotuned,
    )

    results = []
    for method in methods:
        latency_ms = measure_latency_ms(method.run, warmup=warmup, repeat=repeat)
        kernel_config = method.get_kernel_config()
        results.append(
            BenchmarkResult(
                kernel_name=kernel.name,
                layer=case.layer,
                batch=case.batch,
                in_features=case.in_features,
                out_features=case.out_features,
                copies=case.copies,
                method=method.name,
                latency_ms=latency_ms,
                tflops=matmul_tflops(
                    case.batch,
                    case.in_features,
                    case.out_features,
                    latency_ms,
                ),
                memory_mb=method.weight_memory_mb,
                kernel_config=kernel_config,
            )
        )
    return results


def main() -> None:
    torch.manual_seed(config.SEED)
    cases = make_cases(config.SKIP_LM_HEAD)
    all_results = []

    for i, case in enumerate(cases, start=1):
        print(
            f"[{i}/{len(cases)}] {case.layer}: "
            f"B={case.batch}, IN={case.in_features}, OUT={case.out_features}"
        )
        case_results = benchmark_case(
            case=case,
            warmup=config.WARMUP,
            repeat=config.REPEAT,
            block_m=config.BLOCK_M,
            block_n=config.BLOCK_N,
            block_k=config.BLOCK_K,
            skip_autotuned=config.SKIP_AUTOTUNED,
            kernel_name=config.KERNEL_NAME,
        )
        all_results.extend(case_results)
        print_summary(case_results)

    write_csv(config.OUTPUT, all_results)
    write_markdown(config.OUTPUT, all_results)
    print(f"\nSaved CSV results to {config.OUTPUT}")
    print(f"Saved Markdown results to {config.OUTPUT.with_suffix('.md')}")


if __name__ == "__main__":
    main()
