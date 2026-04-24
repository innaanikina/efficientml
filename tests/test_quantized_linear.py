import torch
import torch.nn as nn

from triton_kernels.matmul_int4 import matmul_x16_w4_ref
from triton_kernels.quantized_linear import QuantizedLinear, replace_linear_layers


def test_quantized_linear_matches_reference_without_bias():
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=False, device="cuda", dtype=torch.float16)
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)

    quantized = QuantizedLinear.from_linear(linear, use_autotuned=False)
    assert quantized.kernel_name == "rowwise_int4"
    y = quantized(x)
    y_ref = matmul_x16_w4_ref(
        x,
        quantized.w_packed,
        quantized.w_scales,
        quantized.in_features,
    )

    torch.testing.assert_close(y, y_ref, atol=5e-2, rtol=5e-2)


def test_quantized_linear_matches_reference_with_bias():
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=True, device="cuda", dtype=torch.float16)
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)

    quantized = QuantizedLinear.from_linear(linear, use_autotuned=False)
    y = quantized(x)
    y_ref = matmul_x16_w4_ref(
        x,
        quantized.w_packed,
        quantized.w_scales,
        quantized.in_features,
    ) + linear.bias

    torch.testing.assert_close(y, y_ref, atol=5e-2, rtol=5e-2)


def test_quantized_linear_preserves_leading_dimensions():
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=False, device="cuda", dtype=torch.float16)
    x = torch.randn(2, 4, 64, device="cuda", dtype=torch.float16)

    quantized = QuantizedLinear.from_linear(linear, use_autotuned=False)
    y = quantized(x)

    assert y.shape == (2, 4, 32)


def test_replace_linear_layers_recursively():
    model = nn.Sequential(
        nn.Linear(16, 32, bias=False, device="cuda", dtype=torch.float16),
        nn.Sequential(
            nn.Linear(32, 8, bias=True, device="cuda", dtype=torch.float16),
        ),
    )

    replace_linear_layers(model, use_autotuned=False)

    assert isinstance(model[0], QuantizedLinear)
    assert isinstance(model[1][0], QuantizedLinear)


def test_replace_linear_layers_can_skip_by_name():
    model = nn.Sequential(
        nn.Linear(16, 32, bias=False, device="cuda", dtype=torch.float16),
        nn.Sequential(
            nn.Linear(32, 8, bias=True, device="cuda", dtype=torch.float16),
        ),
    )

    replace_linear_layers(model, skip_module_names={"1.0"}, use_autotuned=False)

    assert isinstance(model[0], QuantizedLinear)
    assert isinstance(model[1][0], nn.Linear)
