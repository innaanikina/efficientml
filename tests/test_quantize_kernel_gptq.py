import pytest
import torch

from triton_kernels.quantize_kernel_gptq import (
    dequantize_groupwise_int4,
    gptq_pack,
    pack_int4_groupwise,
)

GROUP_SIZE = 8


def _naive_pack(w: torch.Tensor, group_size: int):
    OUT, IN = w.shape
    n_groups = IN // group_size
    scales = (w.reshape(OUT, n_groups, group_size).abs().amax(dim=-1) / 7.0).clamp(min=1e-8)
    return pack_int4_groupwise(w, scales, group_size)


def test_output_shapes():
    w = torch.randn(4, 32, device="cuda")
    packed, w_scales = _naive_pack(w, GROUP_SIZE)

    assert packed.shape == (4, 4)
    assert packed.dtype == torch.int32
    assert w_scales.shape == (4, 4)
    assert w_scales.dtype == torch.float16


def test_roundtrip():
    torch.manual_seed(0)
    OUT, IN = 8, 64
    w = torch.randn(OUT, IN, device="cuda")
    packed, w_scales = _naive_pack(w, GROUP_SIZE)
    w_hat = dequantize_groupwise_int4(packed, w_scales, IN, GROUP_SIZE)

    assert w_hat.shape == (OUT, IN)

    err = (w - w_hat).abs()
    half_step = (w_scales.float() / 7.0 / 2.0).repeat_interleave(GROUP_SIZE, dim=1)
    assert torch.all(err <= half_step + 1e-3)


def test_dequantize_shape():
    OUT, IN = 3, 24
    packed = torch.zeros(OUT, IN // 8, dtype=torch.int32, device="cuda")
    absmaxs = torch.ones(OUT, IN // GROUP_SIZE, dtype=torch.float16, device="cuda") * 7.0
    out = dequantize_groupwise_int4(packed, absmaxs, IN, GROUP_SIZE)

    assert out.shape == (OUT, IN)


def test_decoded_values_in_range():
    torch.manual_seed(1)
    w = torch.randn(4, 32, device="cuda")
    packed, w_scales = _naive_pack(w, GROUP_SIZE)
    w_hat = dequantize_groupwise_int4(packed, w_scales, 32, GROUP_SIZE)

    absmax_per_elem = (w_scales.float() / 7.0 * 7.0).repeat_interleave(GROUP_SIZE, dim=1)
    assert torch.all(w_hat.abs() <= absmax_per_elem + 1e-4)


def test_zero_input():
    OUT, IN = 2, 16
    w = torch.zeros(OUT, IN, device="cuda")
    scales = torch.ones(OUT, IN // GROUP_SIZE, device="cuda") * (1.0 / 7.0)
    packed, w_scales = pack_int4_groupwise(w, scales, GROUP_SIZE)
    w_hat = dequantize_groupwise_int4(packed, w_scales, IN, GROUP_SIZE)

    assert torch.allclose(w_hat, torch.zeros_like(w_hat), atol=1e-4)


def test_gptq_pack_output_shapes():
    torch.manual_seed(42)
    OUT, IN = 16, 128
    W = torch.randn(OUT, IN, device="cuda")
    H = torch.eye(IN, device="cuda")

    packed, w_scales, perm = gptq_pack(W, H, group_size=128, act_order=False)

    assert packed.shape == (OUT, IN // 8)
    assert w_scales.shape == (OUT, 1)
    assert perm is None


def test_gptq_pack_act_order_perm():
    torch.manual_seed(0)
    OUT, IN = 8, 128
    W = torch.randn(OUT, IN, device="cuda")
    X = torch.randn(64, IN, device="cuda")
    H = X.T @ X

    _, _, perm = gptq_pack(W, H, group_size=128, act_order=True)

    assert perm is not None
    assert perm.shape == (IN,)
    assert perm.dtype == torch.int64
    assert torch.equal(perm.sort().values, torch.arange(IN, device="cuda"))


def test_gptq_pack_error_bounded():
    torch.manual_seed(7)
    OUT, IN = 16, 256
    W = torch.randn(OUT, IN, device="cuda", dtype=torch.float16)
    H = torch.eye(IN, device="cuda")

    packed, w_scales, _ = gptq_pack(W, H, group_size=128, act_order=False)
    W_hat = dequantize_groupwise_int4(packed, w_scales, IN, 128).to(torch.float16)

    err = (W.float() - W_hat.float()).abs().mean()
    assert err < 0.1
