# WikiText-2 evaluation

- Model: `unsloth/Llama-3.2-1B-Instruct`
- Eval tokens: `8192`
- Quantized skip modules: `['lm_head']`
- Quantized autotuned: `False`

| metric | baseline fp16 | quantized |
| --- | ---: | ---: |
| perplexity | 20.4999 | 34.3861 |
| eval tokens/s | 8801.40 | 5233.70 |
| generation tokens/s | 51.59 | 35.40 |

| comparison | value |
| --- | ---: |
| perplexity delta | 13.8861 |
| perplexity ratio | 1.6774 |
| eval speed ratio | 0.5946 |
| generation speed ratio | 0.6861 |
