import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from triton_kernels.quantized_linear import QuantizedLinear, replace_linear_layers
from utils.cuda import synchronize
from utils.memory import (
    bytes_to_mib,
    linear_and_quantized_weight_bytes,
    linear_weight_bytes,
)


def count_modules(model: nn.Module, module_type: type[nn.Module]) -> int:
    return sum(1 for module in model.modules() if isinstance(module, module_type))


def load_model_and_tokenizer(model_id: str, use_fast_tokenizer: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def load_tokenizer(model_id: str, use_fast_tokenizer: bool = True):
    return AutoTokenizer.from_pretrained(model_id, use_fast=use_fast_tokenizer)


def compute_error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    diff = actual.float() - expected.float()
    return {
        "mse": (diff**2).mean().item(),
        "mae": diff.abs().mean().item(),
        "max_abs_error": diff.abs().max().item(),
    }


def quantize_model(
    model: nn.Module,
    skip_module_names: set[str],
    use_autotuned: bool,
    kernel_name: str,
    block_m: int,
    block_n: int,
    block_k: int,
) -> dict:
    linear_weight_bytes_before = linear_weight_bytes(model)
    linear_count_before = count_modules(model, nn.Linear)

    synchronize()
    start = time.perf_counter()
    replace_linear_layers(
        model,
        skip_module_names=skip_module_names,
        use_autotuned=use_autotuned,
        kernel_name=kernel_name,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    synchronize()
    quantization_time_s = time.perf_counter() - start

    linear_weight_bytes_after = linear_and_quantized_weight_bytes(model)

    return {
        "linear_before": linear_count_before,
        "linear_after": count_modules(model, nn.Linear),
        "quantized_linear_after": count_modules(model, QuantizedLinear),
        "kernel_name": kernel_name,
        "quantization_time_s": quantization_time_s,
        "linear_weight_mib_before": bytes_to_mib(linear_weight_bytes_before),
        "linear_weight_mib_after": bytes_to_mib(linear_weight_bytes_after),
        "linear_weight_compression": linear_weight_bytes_before / linear_weight_bytes_after,
    }
