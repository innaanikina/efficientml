# WikiText-2 evaluation

- Model: `unsloth/Llama-3.2-1B-Instruct`
- Eval tokens: `8192`
- Quantized skip modules: `[]`
- Quantized autotuned: `False`

| metric | baseline fp16 | quantized |
| --- | ---: | ---: |
| perplexity | 20.5000 | 22.9118 |
| eval tokens/s | 12424.97 | 14996.65 |
| generation tokens/s | 57.89 | 48.35 |

| comparison | value |
| --- | ---: |
| perplexity delta | 2.4118 |
| perplexity ratio | 1.1176 |
| eval speed ratio | 1.2070 |
| generation speed ratio | 0.8352 |
