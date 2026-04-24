# WikiText-2 evaluation

- Model: `unsloth/Llama-3.2-1B-Instruct`
- Eval tokens: `8192`
- Quantized skip modules: `[]`
- Quantized autotuned: `False`

| metric | baseline fp16 | quantized |
| --- | ---: | ---: |
| perplexity | 20.4999 | 54.2952 |
| eval tokens/s | 8351.74 | 5022.50 |
| generation tokens/s | 46.02 | 30.57 |

| comparison | value |
| --- | ---: |
| perplexity delta | 33.7953 |
| perplexity ratio | 2.6486 |
| eval speed ratio | 0.6014 |
| generation speed ratio | 0.6643 |
