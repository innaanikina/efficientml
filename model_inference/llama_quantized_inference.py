import json
import time

import torch

from model_inference import llama_inference_config as config
from utils.cuda import synchronize
from utils.model import compute_error, load_model_and_tokenizer, quantize_model


@torch.no_grad()
def run_forward_check(model, tokenizer, baseline_logits=None) -> dict:
    inputs = tokenizer(config.PROMPT, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    logits = outputs.logits

    result = {
        "input_shape": list(inputs["input_ids"].shape),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
    }

    if baseline_logits is not None:
        result["logits_error"] = compute_error(
            logits.detach().cpu(),
            baseline_logits,
        )

    return result


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


def _do_quantize(model, tokenizer) -> dict:
    """Dispatch to the correct quantization path based on config."""
    if config.QUANTIZATION_METHOD == "gptq":
        from utils.gptq_pipeline import gptq_quantize_model, load_calibration_data
        calib = load_calibration_data(tokenizer, n_samples=config.CALIB_N_SAMPLES)
        return gptq_quantize_model(
            model,
            calib,
            group_size=config.GROUP_SIZE,
            act_order=config.ACT_ORDER,
        )
    else:
        return quantize_model(
            model,
            skip_module_names=config.SKIP_MODULE_NAMES,
            use_autotuned=config.USE_AUTOTUNED,
            kernel_name=config.KERNEL_NAME,
            block_m=config.BLOCK_M,
            block_n=config.BLOCK_N,
            block_k=config.BLOCK_K,
        )


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(config.MODEL_ID, use_fast_tokenizer=False)

    inputs = tokenizer(config.PROMPT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        baseline_logits = model(**inputs).logits.detach().cpu()

    quantization = _do_quantize(model, tokenizer)
    forward = run_forward_check(model, tokenizer, baseline_logits=baseline_logits)
    generation = run_generation(model, tokenizer)

    report = {
        "model_id": config.MODEL_ID,
        "quantization_method": config.QUANTIZATION_METHOD,
        # simple kernel params
        "kernel_name": config.KERNEL_NAME,
        "skip_module_names": sorted(config.SKIP_MODULE_NAMES),
        "use_autotuned": config.USE_AUTOTUNED,
        "block_m": config.BLOCK_M,
        "block_n": config.BLOCK_N,
        "block_k": config.BLOCK_K,
        # gptq params
        "group_size": config.GROUP_SIZE,
        "act_order": config.ACT_ORDER,
        "calib_n_samples": config.CALIB_N_SAMPLES,
        "prompt": config.PROMPT,
        "quantization": quantization,
        "forward": forward,
        "generation": generation,
    }

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nОтчет сохранен: {config.OUTPUT_JSON}")


if __name__ == "__main__":
    main()
