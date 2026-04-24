import gc
import json
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_inference import llama_inference_config as config
from triton_kernels.quantized_linear import QuantizedLinear, replace_linear_layers


def count_modules(model: nn.Module, module_type: type[nn.Module]) -> int:
    return sum(1 for module in model.modules() if isinstance(module, module_type))


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.numel()


def bytes_to_mib(nbytes: int) -> float:
    return nbytes / 1024**2


def linear_weight_bytes(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += tensor_nbytes(module.weight)
            if module.bias is not None:
                total += tensor_nbytes(module.bias)
    return total


def quantized_linear_weight_bytes(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            total += tensor_nbytes(module.w_packed)
            total += tensor_nbytes(module.w_scales)
            if module.bias is not None:
                total += tensor_nbytes(module.bias)
    return total


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def quantize_model(model: nn.Module) -> dict:
    linear_before = count_modules(model, nn.Linear)
    linear_weight_bytes_before = linear_weight_bytes(model)

    start = time.perf_counter()
    replace_linear_layers(
        model,
        skip_module_names=config.SKIP_MODULE_NAMES,
        use_autotuned=config.USE_AUTOTUNED,
        block_m=config.BLOCK_M,
        block_n=config.BLOCK_N,
        block_k=config.BLOCK_K,
    )
    synchronize()
    elapsed_s = time.perf_counter() - start

    gc.collect()
    torch.cuda.empty_cache()
    linear_weight_bytes_after = linear_weight_bytes(model) + quantized_linear_weight_bytes(model)

    return {
        "linear_before": linear_before,
        "linear_after": count_modules(model, nn.Linear),
        "quantized_linear_after": count_modules(model, QuantizedLinear),
        "quantization_time_s": elapsed_s,
        "linear_weight_mib_before": bytes_to_mib(linear_weight_bytes_before),
        "linear_weight_mib_after": bytes_to_mib(linear_weight_bytes_after),
        "quantized_weight_mib_after": bytes_to_mib(quantized_linear_weight_bytes(model)),
        "linear_weight_compression": linear_weight_bytes_before / linear_weight_bytes_after,
    }


@torch.no_grad()
def run_forward_check(model, tokenizer) -> dict:
    inputs = tokenizer(config.PROMPT, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    logits = outputs.logits

    return {
        "input_shape": list(inputs["input_ids"].shape),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
    }


@torch.no_grad()
def run_generation(model, tokenizer) -> dict:
    inputs = tokenizer(config.PROMPT, return_tensors="pt").to("cuda")

    synchronize()
    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    synchronize()
    elapsed_s = time.perf_counter() - start

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_tokens = output_ids.shape[-1] - inputs["input_ids"].shape[-1]

    return {
        "generated_tokens": int(generated_tokens),
        "generation_time_s": elapsed_s,
        "tokens_per_second": generated_tokens / elapsed_s,
        "text": decoded,
    }


def main() -> None:
    model, tokenizer = load_model_and_tokenizer()
    quantization = quantize_model(model)
    forward = run_forward_check(model, tokenizer)
    generation = run_generation(model, tokenizer)

    report = {
        "model_id": config.MODEL_ID,
        "skip_module_names": sorted(config.SKIP_MODULE_NAMES),
        "use_autotuned": config.USE_AUTOTUNED,
        "block_m": config.BLOCK_M,
        "block_n": config.BLOCK_N,
        "block_k": config.BLOCK_K,
        "prompt": config.PROMPT,
        "quantization": quantization,
        "forward": forward,
        "generation": generation,
    }

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nОтчет сохранен: {config.OUTPUT_JSON}")


if __name__ == "__main__":
    main()
