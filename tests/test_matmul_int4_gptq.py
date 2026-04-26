import math

import pytest
import torch

from triton_kernels.matmul_int4_gptq import matmul_x16_w4_groupwise, matmul_x16_w4_groupwise_plain
from triton_kernels.quantize_kernel_gptq import dequantize_groupwise_int4
from tests.fixtures import SHAPES_GPTQ, make_case_gptq


GROUP_SIZE = 128

IMPLS = {
    "plain":     lambda x, wp, ws, IN: matmul_x16_w4_groupwise_plain(x, wp, ws, IN, GROUP_SIZE),
    "autotuned": lambda x, wp, ws, IN: matmul_x16_w4_groupwise(x, wp, ws, IN, GROUP_SIZE),
}


def ref_matmul(x, w_packed, w_scales, in_features):
    w_fp = dequantize_groupwise_int4(w_packed, w_scales, in_features, GROUP_SIZE).to(x.dtype)
    return x @ w_fp.T


@pytest.mark.parametrize("shape", SHAPES_GPTQ)
@pytest.mark.parametrize("impl", IMPLS.values(), ids=IMPLS.keys())
def test_triton_matches_reference(shape, impl):
    B, IN, OUT = shape
    x, _, w_packed, w_scales = make_case_gptq(B, IN, OUT)

    y_ref = ref_matmul(x, w_packed, w_scales, IN)
    y_triton = impl(x, w_packed, w_scales, IN)

    assert y_triton.shape == (B, OUT)
    assert y_triton.dtype == torch.float16
    torch.testing.assert_close(y_triton, y_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("shape", SHAPES_GPTQ)
def test_reference_bounded_vs_fp16(shape):
    B, IN, OUT = shape
    x, w_fp16, w_packed, w_scales = make_case_gptq(B, IN, OUT)

    y_ref = ref_matmul(x, w_packed, w_scales, IN).float()
    y_fp16 = (x @ w_fp16.T).float()

    err = (y_ref - y_fp16).abs().mean()
    bound = 0.15 * math.sqrt(IN)
    assert err < bound


@pytest.mark.parametrize("impl", IMPLS.values(), ids=IMPLS.keys())
def test_zero_x(impl):
    B, IN, OUT = 4, 128, 64
    _, _, w_packed, w_scales = make_case_gptq(B, IN, OUT)
    x = torch.zeros(B, IN, device="cuda", dtype=torch.float16)

    y = impl(x, w_packed, w_scales, IN)
    assert torch.all(y == 0)


@pytest.mark.parametrize("impl", IMPLS.values(), ids=IMPLS.keys())
def test_batch_one(impl):
    B, IN, OUT = 1, 256, 128
    x, _, w_packed, w_scales = make_case_gptq(B, IN, OUT)

    y_ref = ref_matmul(x, w_packed, w_scales, IN)
    y_triton = impl(x, w_packed, w_scales, IN)
    torch.testing.assert_close(y_triton, y_ref, atol=5e-2, rtol=5e-2)
