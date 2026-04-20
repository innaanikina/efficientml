import pytest
import torch

from triton_kernels.matmul_int4 import matmul_x16_w4, matmul_x16_w4_ref
from tests.fixtures import SHAPES, make_case


@pytest.mark.parametrize("shape", SHAPES)
def test_triton_matches_reference(shape):
    B, IN, OUT = shape
    x, _, w_packed, w_scales = make_case(B, IN, OUT)

    y_ref = matmul_x16_w4_ref(x, w_packed, w_scales, IN)
    y_triton = matmul_x16_w4(x, w_packed, w_scales, IN)

    assert y_triton.shape == (B, OUT)
    assert y_triton.dtype == torch.float16
    torch.testing.assert_close(y_triton, y_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("shape", SHAPES)
def test_reference_bounded_vs_fp16(shape):
    B, IN, OUT = shape
    x, w_fp16, w_packed, w_scales = make_case(B, IN, OUT)

    y_ref = matmul_x16_w4_ref(x, w_packed, w_scales, IN).float()
    y_fp16 = (x @ w_fp16.T).float()

    err = (y_ref - y_fp16).abs()
    bound = (x.abs().sum(dim=1, keepdim=True).float()
             * (w_scales.float() / 7.0)[None, :]) + 1e-2
    assert torch.all(err <= bound)


def test_zero_x():
    B, IN, OUT = 4, 32, 16
    _, _, w_packed, w_scales = make_case(B, IN, OUT)
    x = torch.zeros(B, IN, device="cuda", dtype=torch.float16)

    y = matmul_x16_w4(x, w_packed, w_scales, IN)
    assert torch.all(y == 0)


def test_batch_one():
    B, IN, OUT = 1, 64, 32
    x, _, w_packed, w_scales = make_case(B, IN, OUT)

    y_ref = matmul_x16_w4_ref(x, w_packed, w_scales, IN)
    y_triton = matmul_x16_w4(x, w_packed, w_scales, IN)
    torch.testing.assert_close(y_triton, y_ref, atol=5e-2, rtol=5e-2)


def test_non_multiple_of_8_in():
    B, IN, OUT = 4, 40, 24
    x, _, w_packed, w_scales = make_case(B, IN, OUT)

    y_ref = matmul_x16_w4_ref(x, w_packed, w_scales, IN)
    y_triton = matmul_x16_w4(x, w_packed, w_scales, IN)
    torch.testing.assert_close(y_triton, y_ref, atol=5e-2, rtol=5e-2)
