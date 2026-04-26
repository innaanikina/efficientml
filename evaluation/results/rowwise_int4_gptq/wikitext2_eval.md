# WikiText-2 evaluation

- Model: `unsloth/Llama-3.2-1B-Instruct`
- Eval tokens: `8192`
- Quantized skip modules: `[]`
- Quantized autotuned: `False`

| metric | baseline fp16 | quantized |
| --- | ---: | ---: |
| perplexity | 20.5000 | 23.1378 |
| eval tokens/s | 14486.91 | 1207.61 |
| generation tokens/s | 63.78 | 2.82 |

| comparison | value |
| --- | ---: |
| perplexity delta | 2.6377 |
| perplexity ratio | 1.1287 |
| eval speed ratio | 0.0834 |
| generation speed ratio | 0.0442 |
