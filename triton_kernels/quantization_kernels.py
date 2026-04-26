from dataclasses import dataclass
from typing import Any, Callable

import torch

from triton_kernels.matmul_int4_sym import (
    _matmul_x16_w4_kernel_autotuned,
    matmul_x16_w4,
    matmul_x16_w4_autotuned,
    matmul_x16_w4_ref,
)
from triton_kernels.matmul_int4_asym import (
    _matmul_x16_w4_asym_kernel_autotuned,
    matmul_x16_w4_asym,
    matmul_x16_w4_asym_autotuned,
    matmul_x16_w4_asym_ref,
)
from triton_kernels.matmul_int4_gptq import (
    _matmul_x16_w4_gw_kernel,
    matmul_x16_w4_groupwise,
    matmul_x16_w4_groupwise_plain,
)
from triton_kernels.quantize_kernel_sym import dequantize_rowwise_int4, quantize_rowwise_int4
from triton_kernels.quantize_kernel_asym import (
    dequantize_rowwise_int4_asym_ref,
    quantize_rowwise_int4_asym,
)
from triton_kernels.quantize_kernel_gptq import dequantize_groupwise_int4, pack_int4_groupwise


@dataclass(frozen=True)
class QuantizedWeights:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def rowwise_int4_asym_autotuned_config() -> dict[str, int] | None:
    best_config = _matmul_x16_w4_asym_kernel_autotuned.best_config
    if best_config is None:
        return None

    return {
        "block_m": best_config.kwargs.get("BLOCK_M"),
        "block_n": best_config.kwargs.get("BLOCK_N"),
        "block_k": best_config.kwargs.get("BLOCK_K"),
        "num_warps": best_config.num_warps,
        "num_stages": best_config.num_stages,
    }


def quantize_rowwise_int4_asym_weights(weight: torch.Tensor) -> QuantizedWeights:
    packed, scales, zp = quantize_rowwise_int4_asym(weight)
    return QuantizedWeights(
        tensors={
            "packed": packed,
            "scales": scales,
            "zero_points": zp,
        },
        metadata={
            "format": "rowwise_int4_asym",
        },
    )


def rowwise_int4_asym_matmul(
    x: torch.Tensor,
    weights: QuantizedWeights,
    in_features: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    return matmul_x16_w4_asym(
        x,
        weights.tensors["packed"],
        weights.tensors["scales"],
        weights.tensors["zero_points"],
        in_features,
        block_m,
        block_n,
        block_k,
    )


def rowwise_int4_asym_matmul_autotuned(
    x: torch.Tensor,
    weights: QuantizedWeights,
    in_features: int,
) -> torch.Tensor:
    return matmul_x16_w4_asym_autotuned(
        x,
        weights.tensors["packed"],
        weights.tensors["scales"],
        weights.tensors["zero_points"],
        in_features,
    )


def rowwise_int4_asym_reference_matmul(
    x: torch.Tensor,
    weights: QuantizedWeights,
    in_features: int,
) -> torch.Tensor:
    return matmul_x16_w4_asym_ref(
        x,
        weights.tensors["packed"],
        weights.tensors["scales"],
        weights.tensors["zero_points"],
        in_features,
    )


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


def quantize_rowwise_int4_weights(weight: torch.Tensor) -> QuantizedWeights:
    packed, scales = quantize_rowwise_int4(weight)
    return QuantizedWeights(
        tensors={
            "packed": packed,
            "scales": scales,
        },
        metadata={
            "format": "rowwise_int4",
        },
    )


def rowwise_int4_matmul(
    x: torch.Tensor,
    weights: QuantizedWeights,
    in_features: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    return matmul_x16_w4(
        x,
        weights.tensors["packed"],
        weights.tensors["scales"],
        in_features,
        block_m,
        block_n,
        block_k,
    )


def rowwise_int4_matmul_autotuned(
    x: torch.Tensor,
    weights: QuantizedWeights,
    in_features: int,
) -> torch.Tensor:
    return matmul_x16_w4_autotuned(
        x,
        weights.tensors["packed"],
        weights.tensors["scales"],
        in_features,
    )


def rowwise_int4_reference_matmul(
    x: torch.Tensor,
    weights: QuantizedWeights,
    in_features: int,
) -> torch.Tensor:
    return matmul_x16_w4_ref(
        x,
        weights.tensors["packed"],
        weights.tensors["scales"],
        in_features,
    )


@dataclass(frozen=True)
class QuantizedKernel:
    name: str
    quantize: Callable[[torch.Tensor], QuantizedWeights]
    matmul: Callable[..., torch.Tensor]
    matmul_autotuned: Callable[..., torch.Tensor] | None
    reference_matmul: Callable[..., torch.Tensor] | None
    get_autotuned_config: Callable[[], dict[str, int] | None] | None = None
    dequantize: Callable[["QuantizedWeights", int], torch.Tensor] | None = None


def rowwise_int4_gptq_autotuned_config() -> dict[str, int] | None:
    best_config = _matmul_x16_w4_gw_kernel.best_config
    if best_config is None:
        return None

    return {
        "block_m": best_config.kwargs.get("BLOCK_M"),
        "block_n": best_config.kwargs.get("BLOCK_N"),
        "block_k": best_config.kwargs.get("BLOCK_K"),
        "num_warps": best_config.num_warps,
        "num_stages": best_config.num_stages,
    }


def quantize_groupwise_int4_weights(weight: torch.Tensor) -> QuantizedWeights:
    """Наивная group-wise упаковка без GPTQ-коррекции. Используется в бенчмарке."""
    weight_f32 = weight.float().cuda()
    OUT, IN = weight_f32.shape
    group_size = 128
    assert IN % group_size == 0
    n_groups = IN // group_size
    scales = (weight_f32.reshape(OUT, n_groups, group_size).abs().amax(dim=-1) / 7.0).clamp(min=1e-8)
    packed, w_scales = pack_int4_groupwise(weight_f32, scales, group_size)
    return QuantizedWeights(
        tensors={"packed": packed, "scales": w_scales},
        metadata={"format": "rowwise_int4_gptq", "group_size": group_size},
    )


def rowwise_int4_gptq_matmul(
    x: torch.Tensor, weights: QuantizedWeights, in_features: int,
    block_m: int, block_n: int, block_k: int,
) -> torch.Tensor:
    gs = weights.metadata.get("group_size", 128)
    return matmul_x16_w4_groupwise_plain(
        x, weights.tensors["packed"], weights.tensors["scales"], in_features, gs,
        block_m=block_m, block_n=block_n, block_k=block_k,
    )


def rowwise_int4_gptq_matmul_autotuned(
    x: torch.Tensor, weights: QuantizedWeights, in_features: int,
) -> torch.Tensor:
    gs = weights.metadata.get("group_size", 128)
    return matmul_x16_w4_groupwise(x, weights.tensors["packed"], weights.tensors["scales"], in_features, gs)


def rowwise_int4_gptq_reference_matmul(
    x: torch.Tensor, weights: QuantizedWeights, in_features: int,
) -> torch.Tensor:
    gs = weights.metadata.get("group_size", 128)
    w_fp = dequantize_groupwise_int4(weights.tensors["packed"], weights.tensors["scales"], in_features, gs).to(x.dtype)
    return x @ w_fp.T


def rowwise_int4_dequantize(weights: QuantizedWeights, in_features: int) -> torch.Tensor:
    return dequantize_rowwise_int4(weights.tensors["packed"], weights.tensors["scales"], in_features)


def rowwise_int4_asym_dequantize(weights: QuantizedWeights, in_features: int) -> torch.Tensor:
    return dequantize_rowwise_int4_asym_ref(
        weights.tensors["packed"],
        weights.tensors["scales"],
        weights.tensors["zero_points"],
        in_features,
    )


def rowwise_int4_gptq_dequantize(weights: QuantizedWeights, in_features: int) -> torch.Tensor:
    gs = weights.metadata.get("group_size", 128)
    return dequantize_groupwise_int4(weights.tensors["packed"], weights.tensors["scales"], in_features, gs)


KERNELS = {
    "rowwise_int4": QuantizedKernel(
        name="rowwise_int4",
        quantize=quantize_rowwise_int4_weights,
        matmul=rowwise_int4_matmul,
        matmul_autotuned=rowwise_int4_matmul_autotuned,
        reference_matmul=rowwise_int4_reference_matmul,
        get_autotuned_config=rowwise_int4_autotuned_config,
        dequantize=rowwise_int4_dequantize,
    ),
    "rowwise_int4_asym": QuantizedKernel(
        name="rowwise_int4_asym",
        quantize=quantize_rowwise_int4_asym_weights,
        matmul=rowwise_int4_asym_matmul,
        matmul_autotuned=rowwise_int4_asym_matmul_autotuned,
        reference_matmul=rowwise_int4_asym_reference_matmul,
        get_autotuned_config=rowwise_int4_asym_autotuned_config,
        dequantize=rowwise_int4_asym_dequantize,
    ),
    "rowwise_int4_gptq": QuantizedKernel(
        name="rowwise_int4_gptq",
        quantize=quantize_groupwise_int4_weights,
        matmul=rowwise_int4_gptq_matmul,
        matmul_autotuned=rowwise_int4_gptq_matmul_autotuned,
        reference_matmul=rowwise_int4_gptq_reference_matmul,
        get_autotuned_config=rowwise_int4_gptq_autotuned_config,
        dequantize=rowwise_int4_gptq_dequantize,
    ),
}


def get_quantized_kernel(name: str) -> QuantizedKernel:
    try:
        return KERNELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(KERNELS))
        raise ValueError(f"Unknown quantized kernel '{name}'. Available: {available}") from exc
