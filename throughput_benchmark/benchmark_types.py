from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BenchmarkCase:
    layer: str
    batch: int
    in_features: int
    out_features: int
    copies: int


@dataclass(frozen=True)
class KernelConfig:
    block_m: int | None = None
    block_n: int | None = None
    block_k: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    kernel_name: str
    layer: str
    batch: int
    in_features: int
    out_features: int
    copies: int
    method: str
    latency_ms: float
    tflops: float
    memory_mb: float
    kernel_config: KernelConfig


def empty_kernel_config() -> KernelConfig:
    return KernelConfig()


@dataclass(frozen=True)
class BenchmarkMethod:
    name: str
    run: Callable[[], object]
    weight_memory_mb: float
    get_kernel_config: Callable[[], KernelConfig] = empty_kernel_config
