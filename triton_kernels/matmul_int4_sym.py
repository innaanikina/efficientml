import torch
import triton
import triton.language as tl

from triton_kernels.quantize_kernel_sym import dequantize_rowwise_int4


def matmul_x16_w4_ref(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    w = dequantize_rowwise_int4(w_packed, w_scales, in_features).to(x.dtype)
    return x @ w.T


@triton.jit
def _matmul_x16_w4_kernel(
    x_ptr, w_ptr, scale_ptr, y_ptr,
    B, IN, OUT, PACK_IN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    PACK_K: tl.constexpr = BLOCK_K // 8
    shifts = (tl.arange(0, 8) * 4).to(tl.uint32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, IN, BLOCK_K):
        k_idx = k + tl.arange(0, BLOCK_K)
        pk_idx = (k // 8) + tl.arange(0, PACK_K)

        off_x = row_idx[:, None] * IN + k_idx[None, :]
        mask_x = (row_idx < B)[:, None] & (k_idx < IN)[None, :]
        x = tl.load(x_ptr + off_x, mask_x, 0.0)

        off_w = col_idx[:, None] * PACK_IN + pk_idx[None, :]
        mask_w = (col_idx < OUT)[:, None] & (pk_idx < PACK_IN)[None, :]
        wp = tl.load(w_ptr + off_w, mask_w, 0)

        wu = (wp[:, :, None].to(tl.uint32) >> shifts[None, None, :]) & 0xF
        wu = wu.to(tl.int32)
        wu = tl.where(wu >= 8, wu - 16, wu)
        w = tl.reshape(wu, (BLOCK_N, BLOCK_K)).to(tl.float16)
        w = tl.where(k_idx[None, :] < IN, w, tl.zeros_like(w))

        acc += tl.dot(x, tl.trans(w))

    scale_mask = col_idx < OUT
    scales = tl.load(scale_ptr + col_idx, mask=scale_mask, other=0.0).to(tl.float32)
    acc = acc * (scales / 7.0)[None, :]

    off_y = row_idx[:, None] * OUT + col_idx[None, :]
    mask_y = (row_idx < B)[:, None] & (col_idx < OUT)[None, :]
    tl.store(y_ptr + off_y, acc.to(tl.float16), mask_y)


def matmul_x16_w4(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    in_features: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
):
    assert x.is_cuda and w_packed.is_cuda and w_scales.is_cuda
    assert x.dtype == torch.float16
    assert w_packed.dtype == torch.int32
    B, IN = x.shape
    OUT, PACK_IN = w_packed.shape
    assert IN == in_features
    y = torch.empty((B, OUT), device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(OUT, BLOCK_N))
    _matmul_x16_w4_kernel[grid](
        x_ptr=x, w_ptr=w_packed, scale_ptr=w_scales, y_ptr=y,
        B=B, IN=IN, OUT=OUT, PACK_IN=PACK_IN,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return y


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32,  "BLOCK_K": 32},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 32},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32,  "BLOCK_K": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    ],
    key=["B", "IN", "OUT"],
)
@triton.jit
def _matmul_x16_w4_kernel_autotuned(
    x_ptr, w_ptr, scale_ptr, y_ptr,
    B, IN, OUT, PACK_IN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    PACK_K: tl.constexpr = BLOCK_K // 8
    shifts = (tl.arange(0, 8) * 4).to(tl.uint32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, IN, BLOCK_K):
        k_idx = k + tl.arange(0, BLOCK_K)
        pk_idx = (k // 8) + tl.arange(0, PACK_K)

        off_x = row_idx[:, None] * IN + k_idx[None, :]
        mask_x = (row_idx < B)[:, None] & (k_idx < IN)[None, :]
        x = tl.load(x_ptr + off_x, mask_x, 0.0)

        off_w = col_idx[:, None] * PACK_IN + pk_idx[None, :]
        mask_w = (col_idx < OUT)[:, None] & (pk_idx < PACK_IN)[None, :]
        wp = tl.load(w_ptr + off_w, mask_w, 0)

        wu = (wp[:, :, None].to(tl.uint32) >> shifts[None, None, :]) & 0xF
        wu = wu.to(tl.int32)
        wu = tl.where(wu >= 8, wu - 16, wu)
        w = tl.reshape(wu, (BLOCK_N, BLOCK_K)).to(tl.float16)
        w = tl.where(k_idx[None, :] < IN, w, tl.zeros_like(w))

        acc += tl.dot(x, tl.trans(w))

    scale_mask = col_idx < OUT
    scales = tl.load(scale_ptr + col_idx, mask=scale_mask, other=0.0).to(tl.float32)
    acc = acc * (scales / 7.0)[None, :]

    off_y = row_idx[:, None] * OUT + col_idx[None, :]
    mask_y = (row_idx < B)[:, None] & (col_idx < OUT)[None, :]
    tl.store(y_ptr + off_y, acc.to(tl.float16), mask_y)


def matmul_x16_w4_autotuned(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    in_features: int,
):
    assert x.is_cuda and w_packed.is_cuda and w_scales.is_cuda
    assert x.dtype == torch.float16
    assert w_packed.dtype == torch.int32
    B, IN = x.shape
    OUT, PACK_IN = w_packed.shape
    assert IN == in_features
    y = torch.empty((B, OUT), device=x.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_M"]),
        triton.cdiv(OUT, meta["BLOCK_N"]),
    )
    _matmul_x16_w4_kernel_autotuned[grid](
        x, w_packed, w_scales, y,
        B, IN, OUT, PACK_IN,
    )
    return y
