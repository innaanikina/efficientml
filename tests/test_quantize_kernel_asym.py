import pytest
import torch

from triton_kernels.quantize_kernel_asym import (
    quantize_rowwise_int4_asym,
    dequantize_rowwise_int4_asym_ref,
)


def test_output_shapes():
    x = torch.randn(4, 17, device="cuda", dtype=torch.float16)
    packed, scales, zp = quantize_rowwise_int4_asym(x)

    assert packed.shape == (4, 3)   # ceil(17/8) = 3
    assert packed.dtype == torch.int32
    assert scales.shape == (4,)
    assert scales.dtype == torch.float16
    assert zp.shape == (4,)
    assert zp.dtype == torch.int32


def test_roundtrip_known_values():
    x = torch.arange(16, dtype=torch.float16, device="cuda").unsqueeze(0)
    packed, scales, zp = quantize_rowwise_int4_asym(x)
    x_hat = dequantize_rowwise_int4_asym_ref(packed, scales, zp, x.shape[1])

    assert scales[0].item() == pytest.approx(1.0, abs=1e-2)
    assert zp[0].item() == 0
    assert torch.allclose(x.float(), x_hat, atol=0.1)


def test_error_bounded():
    torch.manual_seed(42)
    x = torch.randn(8, 32, device="cuda", dtype=torch.float16)
    packed, scales, zp = quantize_rowwise_int4_asym(x)
    x_hat = dequantize_rowwise_int4_asym_ref(packed, scales, zp, x.shape[1])

    err = (x.float() - x_hat).abs()
    max_allowed = (scales.float() / 2.0).unsqueeze(1)
    assert torch.all(err <= max_allowed + 1e-3)


def test_zero_points_in_range():
    torch.manual_seed(0)
    x = torch.randn(16, 32, device="cuda", dtype=torch.float16)
    _, _, zp = quantize_rowwise_int4_asym(x)

    assert zp.min().item() >= 0
    assert zp.max().item() <= 15


def test_zero_row():
    x = torch.zeros(1, 16, device="cuda", dtype=torch.float16)
    packed, scales, zp = quantize_rowwise_int4_asym(x)
    x_hat = dequantize_rowwise_int4_asym_ref(packed, scales, zp, x.shape[1])

    assert torch.allclose(x_hat, torch.zeros_like(x_hat), atol=1e-3)


def test_constant_row():
    x = torch.full((1, 16), 3.0, device="cuda", dtype=torch.float16)
    packed, scales, zp = quantize_rowwise_int4_asym(x)
    x_hat = dequantize_rowwise_int4_asym_ref(packed, scales, zp, x.shape[1])

    assert torch.allclose(x_hat, x.float(), atol=1e-3)


def test_non_multiple_of_8():
    torch.manual_seed(7)
    x = torch.randn(3, 13, device="cuda", dtype=torch.float16)
    packed, scales, zp = quantize_rowwise_int4_asym(x)
    x_hat = dequantize_rowwise_int4_asym_ref(packed, scales, zp, x.shape[1])

    assert x_hat.shape == x.shape
    err = (x.float() - x_hat).abs()
    max_allowed = (scales.float() / 2.0).unsqueeze(1)
    assert torch.all(err <= max_allowed + 1e-3)
