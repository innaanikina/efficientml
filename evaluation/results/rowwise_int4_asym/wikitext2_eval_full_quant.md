# WikiText-2 evaluation

- Model: `unsloth/Llama-3.2-1B-Instruct`
- Eval tokens: `8192`
- Quantized skip modules: `[]`
- Quantized autotuned: `False`

| metric | baseline fp16 | quantized |
| --- | ---: | ---: |
| perplexity | 20.5000 | 36.4100 |
| eval tokens/s | 17854.29 | 20681.97 |
| generation tokens/s | 72.05 | 52.13 |

| comparison | value |
| --- | ---: |
| perplexity delta | 15.9100 |
| perplexity ratio | 1.7761 |
| eval speed ratio | 1.1584 |
| generation speed ratio | 0.7235 |
