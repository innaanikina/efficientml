import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_inference import llama_inference_config as config
from triton_kernels.quantized_linear import replace_linear_layers


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LLAMA_INTEGRATION_TESTS") != "1",
    reason="Set RUN_LLAMA_INTEGRATION_TESTS=1 to run Llama integration tests.",
)


def compute_error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    diff = actual.float() - expected.float()
    return {
        "mse": (diff**2).mean().item(),
        "mae": diff.abs().mean().item(),
        "max_abs_error": diff.abs().max().item(),
    }


@torch.no_grad()
def test_quantized_llama_logits_close_to_fp16_baseline():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    inputs = tokenizer(config.PROMPT, return_tensors="pt").to("cuda")
    baseline_logits = model(**inputs).logits.detach().cpu()

    replace_linear_layers(
        model,
        skip_module_names=config.SKIP_MODULE_NAMES,
        use_autotuned=config.USE_AUTOTUNED,
        block_m=config.BLOCK_M,
        block_n=config.BLOCK_N,
        block_k=config.BLOCK_K,
    )

    quantized_logits = model(**inputs).logits.detach().cpu()
    assert quantized_logits.shape == baseline_logits.shape
    assert torch.isfinite(quantized_logits).all()

    errors = compute_error(quantized_logits, baseline_logits)
    assert errors["mae"] < 1.0, f"MAE is too high, threshold set to 1.0, actual: {errors['mae']:.4f}"
    assert errors["max_abs_error"] < 20.0, f"Max abs error is too high, threshold set to 20.0, actual: {errors['max_abs_error']:.4f}"
