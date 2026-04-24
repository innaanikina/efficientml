import gc
import json
import math
import time

import torch
from datasets import load_dataset

from evaluation.evaluation_report import build_comparison, write_markdown
from evaluation import wikitext2_config as config
from utils.cuda import synchronize
from utils.model import load_model_and_tokenizer, load_tokenizer, quantize_model


def load_wikitext2_input_ids(tokenizer) -> torch.Tensor:
    dataset = load_dataset(
        config.DATASET_NAME,
        config.DATASET_CONFIG,
        split=config.DATASET_SPLIT,
    )
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    return input_ids[:, :config.MAX_EVAL_TOKENS]


@torch.no_grad()
def evaluate_perplexity(model, input_ids: torch.Tensor) -> dict:
    nll_sum = 0.0
    token_count = 0
    total_tokens = input_ids.shape[1]

    synchronize()
    start = time.perf_counter()
    for begin in range(0, total_tokens, config.SEQUENCE_LENGTH):
        end = begin + config.SEQUENCE_LENGTH
        if end > total_tokens:
            break

        input_chunk = input_ids[:, begin:end].to("cuda")
        outputs = model(input_chunk, labels=input_chunk)
        valid_tokens = input_chunk.numel()
        nll_sum += outputs.loss.item() * valid_tokens
        token_count += valid_tokens

    synchronize()
    elapsed_s = time.perf_counter() - start
    mean_nll = nll_sum / token_count

    return {
        "tokens": token_count,
        "sequence_length": config.SEQUENCE_LENGTH,
        "eval_time_s": elapsed_s,
        "eval_tokens_per_second": token_count / elapsed_s,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }


@torch.no_grad()
def evaluate_generation_speed(model, tokenizer) -> dict:
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

    generated_tokens = output_ids.shape[-1] - inputs["input_ids"].shape[-1]
    return {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_tokens": int(generated_tokens),
        "generation_time_s": elapsed_s,
        "generation_tokens_per_second": generated_tokens / elapsed_s,
    }


def evaluate_model(model_kind: str, input_ids: torch.Tensor) -> dict:
    model, tokenizer = load_model_and_tokenizer(config.MODEL_ID)
    quantization = None
    if model_kind == "quantized":
        quantization = quantize_model(
            model,
            skip_module_names=config.SKIP_MODULE_NAMES,
            use_autotuned=config.USE_AUTOTUNED,
            kernel_name=config.KERNEL_NAME,
            block_m=config.BLOCK_M,
            block_n=config.BLOCK_N,
            block_k=config.BLOCK_K,
        )

    perplexity = evaluate_perplexity(model, input_ids)
    generation = evaluate_generation_speed(model, tokenizer)

    return {
        "quantization": quantization,
        "perplexity": perplexity,
        "generation": generation,
    }


def main() -> None:
    tokenizer = load_tokenizer(config.MODEL_ID)
    input_ids = load_wikitext2_input_ids(tokenizer)

    baseline = evaluate_model("baseline", input_ids)
    quantized = evaluate_model("quantized", input_ids)
    comparison = build_comparison(baseline, quantized)

    report = {
        "model_id": config.MODEL_ID,
        "dataset": {
            "name": config.DATASET_NAME,
            "config": config.DATASET_CONFIG,
            "split": config.DATASET_SPLIT,
        },
        "eval_tokens": int(input_ids.shape[1]),
        "quantized_config": {
            "skip_module_names": sorted(config.SKIP_MODULE_NAMES),
            "use_autotuned": config.USE_AUTOTUNED,
            "kernel_name": config.KERNEL_NAME,
            "block_m": config.BLOCK_M,
            "block_n": config.BLOCK_N,
            "block_k": config.BLOCK_K,
        },
        "baseline": baseline,
        "quantized": quantized,
        "comparison": comparison,
    }

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(report, config.OUTPUT_MD)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nОтчет в JSON сохранен: {config.OUTPUT_JSON}")
    print(f"Отчет в Markdown сохранен: {config.OUTPUT_MD}")


if __name__ == "__main__":
    main()
