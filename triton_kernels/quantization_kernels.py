from dataclasses import dataclass
from typing import Callable

import torch

from triton_kernels.matmul_int4 import (
    _matmul_x16_w4_kernel_autotuned,
    matmul_x16_w4,
    matmul_x16_w4_autotuned,
    matmul_x16_w4_ref,
)
from triton_kernels.quantize_kernel import quantize_rowwise_int4


def rowwise_int4_autotuned_config() -> dict[str, int] | None:
    best_config = _matmul_x16_w4_kernel_autotuned.best_config
    if best_config is None:
        return None

    return {
        "block_m": best_config.kwargs.get("BLOCK_M"),
        "block_n": best_config.kwargs.get("BLOCK_N"),
        "block_k": best_config.kwargs.get("BLOCK_K"),
        "num_warps": best_config.num_warps,
        "num_stages": best_config.num_stages,
    }


@dataclass(frozen=True)
class QuantizedKernel:
    name: str
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    matmul: Callable[..., torch.Tensor]
    matmul_autotuned: Callable[..., torch.Tensor] | None
    reference_matmul: Callable[..., torch.Tensor] | None
    get_autotuned_config: Callable[[], dict[str, int] | None] | None = None


KERNELS = {
    "rowwise_int4": QuantizedKernel(
        name="rowwise_int4",
        quantize=quantize_rowwise_int4,
        matmul=matmul_x16_w4,
        matmul_autotuned=matmul_x16_w4_autotuned,
        reference_matmul=matmul_x16_w4_ref,
        get_autotuned_config=rowwise_int4_autotuned_config,
    )
}


def get_quantized_kernel(name: str) -> QuantizedKernel:
    try:
        return KERNELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(KERNELS))
        raise ValueError(f"Unknown quantized kernel '{name}'. Available: {available}") from exc
