# WikiText-2 evaluation

- Model: `unsloth/Llama-3.2-1B-Instruct`
- Eval tokens: `8192`
- Quantized skip modules: `['lm_head']`
- Quantized autotuned: `False`

| metric | baseline fp16 | quantized |
| --- | ---: | ---: |
| perplexity | 20.5000 | 34.3861 |
| eval tokens/s | 18696.20 | 22093.83 |
| generation tokens/s | 70.61 | 52.19 |

| comparison | value |
| --- | ---: |
| perplexity delta | 13.8860 |
| perplexity ratio | 1.6774 |
| eval speed ratio | 1.1817 |
| generation speed ratio | 0.7391 |
