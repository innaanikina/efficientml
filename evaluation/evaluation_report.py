from pathlib import Path


def build_comparison(baseline: dict, quantized: dict) -> dict:
    return {
        "perplexity_delta": quantized["perplexity"]["perplexity"]
        - baseline["perplexity"]["perplexity"],
        "perplexity_ratio": quantized["perplexity"]["perplexity"]
        / baseline["perplexity"]["perplexity"],
        "eval_speed_ratio": quantized["perplexity"]["eval_tokens_per_second"]
        / baseline["perplexity"]["eval_tokens_per_second"],
        "generation_speed_ratio": quantized["generation"]["generation_tokens_per_second"]
        / baseline["generation"]["generation_tokens_per_second"],
    }


def write_markdown(report: dict, path: Path) -> None:
    baseline = report["baseline"]
    quantized = report["quantized"]
    comparison = report["comparison"]

    lines = [
        "# WikiText-2 evaluation",
        "",
        f"- Model: `{report['model_id']}`",
        f"- Eval tokens: `{report['eval_tokens']}`",
        f"- Quantized skip modules: `{report['quantized_config']['skip_module_names']}`",
        f"- Quantized autotuned: `{report['quantized_config']['use_autotuned']}`",
        "",
        "| metric | baseline fp16 | quantized |",
        "| --- | ---: | ---: |",
        f"| perplexity | {baseline['perplexity']['perplexity']:.4f} | {quantized['perplexity']['perplexity']:.4f} |",
        f"| eval tokens/s | {baseline['perplexity']['eval_tokens_per_second']:.2f} | {quantized['perplexity']['eval_tokens_per_second']:.2f} |",
        f"| generation tokens/s | {baseline['generation']['generation_tokens_per_second']:.2f} | {quantized['generation']['generation_tokens_per_second']:.2f} |",
        "",
        "| comparison | value |",
        "| --- | ---: |",
        f"| perplexity delta | {comparison['perplexity_delta']:.4f} |",
        f"| perplexity ratio | {comparison['perplexity_ratio']:.4f} |",
        f"| eval speed ratio | {comparison['eval_speed_ratio']:.4f} |",
        f"| generation speed ratio | {comparison['generation_speed_ratio']:.4f} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
