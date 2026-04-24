# WikiText-2 evaluation

- Model: `unsloth/Llama-3.2-1B-Instruct`
- Eval tokens: `8192`
- Quantized skip modules: `['lm_head']`
- Quantized autotuned: `False`

| metric | baseline fp16 | quantized |
| --- | ---: | ---: |
| perplexity | 20.4999 | 52.5700 |
| eval tokens/s | 9032.90 | 8274.35 |
| generation tokens/s | 55.34 | 30.74 |

| comparison | value |
| --- | ---: |
| perplexity delta | 32.0701 |
| perplexity ratio | 2.5644 |
| eval speed ratio | 0.9160 |
| generation speed ratio | 0.5555 |
