# WikiText-2 evaluation

- Model: `unsloth/Llama-3.2-1B-Instruct`
- Eval tokens: `8192`
- Quantized skip modules: `[]`
- Quantized autotuned: `False`

| metric | baseline fp16 | quantized |
| --- | ---: | ---: |
| perplexity | 20.4999 | 35.9559 |
| eval tokens/s | 6932.23 | 9759.21 |
| generation tokens/s | 53.14 | 38.02 |

| comparison | value |
| --- | ---: |
| perplexity delta | 15.4560 |
| perplexity ratio | 1.7540 |
| eval speed ratio | 1.4078 |
| generation speed ratio | 0.7154 |
