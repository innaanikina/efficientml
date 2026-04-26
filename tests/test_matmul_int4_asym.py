import math

import pytest
import torch

from triton_kernels.matmul_int4_asym import (
    matmul_x16_w4_asym,
    matmul_x16_w4_asym_autotuned,
    matmul_x16_w4_asym_ref,
)
from tests.fixtures import SHAPES, make_case_asym


IMPLS = {
    "plain": lambda x, wp, ws, wz, IN: matmul_x16_w4_asym(x, wp, ws, wz, IN, 32, 32, 32),
    # "autotuned": lambda x, wp, ws, wz, IN: matmul_x16_w4_asym_autotuned(x, wp, ws, wz, IN),
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("impl", IMPLS.values(), ids=IMPLS.keys())
def test_triton_matches_reference(shape, impl):
    B, IN, OUT = shape
    x, _, w_packed, w_scales, w_zp = make_case_asym(B, IN, OUT)

    y_ref = matmul_x16_w4_asym_ref(x, w_packed, w_scales, w_zp, IN)
    y_triton = impl(x, w_packed, w_scales, w_zp, IN)

    assert y_triton.shape == (B, OUT)
    assert y_triton.dtype == torch.float16
    torch.testing.assert_close(y_triton, y_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("shape", SHAPES)
def test_reference_bounded_vs_fp16(shape):
    B, IN, OUT = shape
    x, w_fp16, w_packed, w_scales, w_zp = make_case_asym(B, IN, OUT)

    y_ref = matmul_x16_w4_asym_ref(x, w_packed, w_scales, w_zp, IN).float()
    y_fp16 = (x @ w_fp16.T).float()

    err = (y_ref - y_fp16).abs().mean()

    bound = 0.15 * math.sqrt(IN)
    assert err < bound


@pytest.mark.parametrize("impl", IMPLS.values(), ids=IMPLS.keys())
def test_zero_x(impl):
    B, IN, OUT = 4, 32, 16
    _, _, w_packed, w_scales, w_zp = make_case_asym(B, IN, OUT)
    x = torch.zeros(B, IN, device="cuda", dtype=torch.float16)

    y = impl(x, w_packed, w_scales, w_zp, IN)
    assert torch.all(y == 0)


@pytest.mark.parametrize("impl", IMPLS.values(), ids=IMPLS.keys())
def test_batch_one(impl):
    B, IN, OUT = 1, 64, 32
    x, _, w_packed, w_scales, w_zp = make_case_asym(B, IN, OUT)

    y_ref = matmul_x16_w4_asym_ref(x, w_packed, w_scales, w_zp, IN)
    y_triton = impl(x, w_packed, w_scales, w_zp, IN)
    torch.testing.assert_close(y_triton, y_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("impl", IMPLS.values(), ids=IMPLS.keys())
def test_non_multiple_of_8_in(impl):
    B, IN, OUT = 4, 40, 24
    x, _, w_packed, w_scales, w_zp = make_case_asym(B, IN, OUT)

    y_ref = matmul_x16_w4_asym_ref(x, w_packed, w_scales, w_zp, IN)
    y_triton = impl(x, w_packed, w_scales, w_zp, IN)
    torch.testing.assert_close(y_triton, y_ref, atol=5e-2, rtol=5e-2)
