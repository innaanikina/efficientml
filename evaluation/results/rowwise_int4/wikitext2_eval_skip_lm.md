# WikiText-2 evaluation

- Model: `unsloth/Llama-3.2-1B-Instruct`
- Eval tokens: `8192`
- Quantized skip modules: `['lm_head']`
- Quantized autotuned: `False`

| metric | baseline fp16 | quantized |
| --- | ---: | ---: |
| perplexity | 20.5000 | 52.5700 |
| eval tokens/s | 15930.95 | 22432.11 |
| generation tokens/s | 73.27 | 55.08 |

| comparison | value |
| --- | ---: |
| perplexity delta | 32.0700 |
| perplexity ratio | 2.5644 |
| eval speed ratio | 1.4081 |
| generation speed ratio | 0.7518 |
