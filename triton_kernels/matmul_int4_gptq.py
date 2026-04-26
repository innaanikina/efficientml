import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_x16_w4_gw_kernel_plain(
    x_ptr, w_ptr, scale_ptr, y_ptr,
    B, IN, OUT, PACK_IN, N_GROUPS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
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
        x_block = tl.load(x_ptr + off_x, mask_x, 0.0)

        off_w = col_idx[:, None] * PACK_IN + pk_idx[None, :]
        mask_w = (col_idx < OUT)[:, None] & (pk_idx < PACK_IN)[None, :]
        wp = tl.load(w_ptr + off_w, mask_w, 0)

        wu = (wp[:, :, None].to(tl.uint32) >> shifts[None, None, :]) & 0xF
        wu = wu.to(tl.int32)
        wu = tl.where(wu >= 8, wu - 16, wu)
        w_block = tl.reshape(wu, (BLOCK_N, BLOCK_K)).to(tl.float16)
        w_block = tl.where(k_idx[None, :] < IN, w_block, tl.zeros_like(w_block))

        g_idx = k // GROUP_SIZE
        g_off = col_idx * N_GROUPS + g_idx
        g_mask = col_idx < OUT
        g_scales = tl.load(scale_ptr + g_off, mask=g_mask, other=0.0).to(tl.float32)

        partial = tl.dot(x_block, tl.trans(w_block))
        acc += partial * (g_scales / 7.0)[None, :]

    off_y = row_idx[:, None] * OUT + col_idx[None, :]
    mask_y = (row_idx < B)[:, None] & (col_idx < OUT)[None, :]
    tl.store(y_ptr + off_y, acc.to(tl.float16), mask_y)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["B", "IN", "OUT"],
)
@triton.jit
def _matmul_x16_w4_gw_kernel(
    x_ptr, w_ptr, scale_ptr, y_ptr,
    B, IN, OUT, PACK_IN, N_GROUPS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
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
        x_block = tl.load(x_ptr + off_x, mask_x, 0.0)

        off_w = col_idx[:, None] * PACK_IN + pk_idx[None, :]
        mask_w = (col_idx < OUT)[:, None] & (pk_idx < PACK_IN)[None, :]
        wp = tl.load(w_ptr + off_w, mask_w, 0)

        wu = (wp[:, :, None].to(tl.uint32) >> shifts[None, None, :]) & 0xF
        wu = wu.to(tl.int32)
        wu = tl.where(wu >= 8, wu - 16, wu)
        w_block = tl.reshape(wu, (BLOCK_N, BLOCK_K)).to(tl.float16)
        w_block = tl.where(k_idx[None, :] < IN, w_block, tl.zeros_like(w_block))

        g_idx = k // GROUP_SIZE
        g_off = col_idx * N_GROUPS + g_idx
        g_mask = col_idx < OUT
        g_scales = tl.load(scale_ptr + g_off, mask=g_mask, other=0.0).to(tl.float32)

        partial = tl.dot(x_block, tl.trans(w_block))
        acc += partial * (g_scales / 7.0)[None, :]

    off_y = row_idx[:, None] * OUT + col_idx[None, :]
    mask_y = (row_idx < B)[:, None] & (col_idx < OUT)[None, :]
    tl.store(y_ptr + off_y, acc.to(tl.float16), mask_y)


def matmul_x16_w4_groupwise_plain(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    in_features: int,
    group_size: int = 128,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    assert x.is_cuda and w_packed.is_cuda and w_scales.is_cuda
    assert x.dtype == torch.float16
    assert w_packed.dtype == torch.int32

    B, IN = x.shape
    OUT, PACK_IN = w_packed.shape
    assert IN == in_features
    assert IN % group_size == 0
    n_groups = IN // group_size

    y = torch.empty((B, OUT), device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(B, block_m), triton.cdiv(OUT, block_n))
    _matmul_x16_w4_gw_kernel_plain[grid](
        x, w_packed, w_scales, y,
        B, IN, OUT, PACK_IN, n_groups,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_SIZE=group_size,
    )
    return y


def matmul_x16_w4_groupwise(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    in_features: int,
    group_size: int = 128,
) -> torch.Tensor:
    assert x.is_cuda and w_packed.is_cuda and w_scales.is_cuda
    assert x.dtype == torch.float16
    assert w_packed.dtype == torch.int32

    B, IN = x.shape
    OUT, PACK_IN = w_packed.shape
    assert IN == in_features
    assert IN % group_size == 0, f"IN ({IN}) must be divisible by group_size ({group_size})"
    n_groups = IN // group_size
    assert w_scales.shape == (OUT, n_groups), (
        f"Expected w_scales shape ({OUT}, {n_groups}), got {w_scales.shape}"
    )

    y = torch.empty((B, OUT), device=x.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_M"]), triton.cdiv(OUT, meta["BLOCK_N"]))
    _matmul_x16_w4_gw_kernel[grid](
        x, w_packed, w_scales, y,
        B, IN, OUT, PACK_IN, n_groups,
        GROUP_SIZE=group_size,
    )
    return y
