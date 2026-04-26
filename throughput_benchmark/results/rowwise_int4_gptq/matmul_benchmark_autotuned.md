| layer | kernel | batch | IN | OUT | method | latency_ms | TFLOP/s | weight_MB | BLOCK_M | BLOCK_N | BLOCK_K | warps | stages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | torch_fp16 | 0.017 | 62.800 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | torch_quantized_ref | 0.218 | 4.918 | 2.06 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | triton_fixed | 0.053 | 20.362 | 2.06 | 64 | 64 | 64 |  |  |
| q_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | triton_autotuned | 0.062 | 17.371 | 2.06 | 64 | 64 | 64 | 4 | 2 |
| q_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | torch_fp16 | 0.032 | 136.168 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | torch_quantized_ref | 0.212 | 20.294 | 2.06 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | triton_fixed | 0.099 | 43.167 | 2.06 | 64 | 64 | 64 |  |  |
| q_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | triton_autotuned | 0.097 | 44.407 | 2.06 | 64 | 64 | 32 | 4 | 2 |
| q_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | torch_fp16 | 0.081 | 212.514 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | torch_quantized_ref | 0.277 | 62.048 | 2.06 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | triton_fixed | 0.329 | 52.149 | 2.06 | 64 | 64 | 64 |  |  |
| q_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | triton_autotuned | 0.245 | 70.139 | 2.06 | 128 | 64 | 64 | 4 | 2 |
| k_proj | rowwise_int4_gptq | 128 | 2048 | 512 | torch_fp16 | 0.017 | 15.439 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 128 | 2048 | 512 | torch_quantized_ref | 0.191 | 1.404 | 0.52 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 128 | 2048 | 512 | triton_fixed | 0.044 | 6.087 | 0.52 | 64 | 64 | 64 |  |  |
| k_proj | rowwise_int4_gptq | 128 | 2048 | 512 | triton_autotuned | 0.062 | 4.352 | 0.52 | 64 | 64 | 64 | 4 | 2 |
| k_proj | rowwise_int4_gptq | 512 | 2048 | 512 | torch_fp16 | 0.022 | 49.387 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 512 | 2048 | 512 | torch_quantized_ref | 0.201 | 5.340 | 0.52 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 512 | 2048 | 512 | triton_fixed | 0.045 | 23.967 | 0.52 | 64 | 64 | 64 |  |  |
| k_proj | rowwise_int4_gptq | 512 | 2048 | 512 | triton_autotuned | 0.062 | 17.309 | 0.52 | 64 | 64 | 64 | 4 | 2 |
| k_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | torch_fp16 | 0.032 | 136.070 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | torch_quantized_ref | 0.184 | 23.319 | 0.52 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | triton_fixed | 0.099 | 43.169 | 0.52 | 64 | 64 | 64 |  |  |
| k_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | triton_autotuned | 0.097 | 44.322 | 0.52 | 64 | 64 | 32 | 4 | 2 |
| v_proj | rowwise_int4_gptq | 128 | 2048 | 512 | torch_fp16 | 0.017 | 15.566 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 128 | 2048 | 512 | torch_quantized_ref | 0.191 | 1.403 | 0.52 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 128 | 2048 | 512 | triton_fixed | 0.043 | 6.255 | 0.52 | 64 | 64 | 64 |  |  |
| v_proj | rowwise_int4_gptq | 128 | 2048 | 512 | triton_autotuned | 0.062 | 4.300 | 0.52 | 64 | 64 | 64 | 4 | 2 |
| v_proj | rowwise_int4_gptq | 512 | 2048 | 512 | torch_fp16 | 0.014 | 78.283 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 512 | 2048 | 512 | torch_quantized_ref | 0.183 | 5.873 | 0.52 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 512 | 2048 | 512 | triton_fixed | 0.043 | 24.896 | 0.52 | 64 | 64 | 64 |  |  |
| v_proj | rowwise_int4_gptq | 512 | 2048 | 512 | triton_autotuned | 0.062 | 17.457 | 0.52 | 64 | 64 | 64 | 4 | 2 |
| v_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | torch_fp16 | 0.032 | 136.129 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | torch_quantized_ref | 0.182 | 23.613 | 0.52 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | triton_fixed | 0.099 | 43.180 | 0.52 | 64 | 64 | 64 |  |  |
| v_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | triton_autotuned | 0.097 | 44.336 | 0.52 | 64 | 64 | 32 | 4 | 2 |
| o_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | torch_fp16 | 0.014 | 77.889 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | torch_quantized_ref | 0.195 | 5.520 | 2.06 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | triton_fixed | 0.043 | 24.913 | 2.06 | 64 | 64 | 64 |  |  |
| o_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | triton_autotuned | 0.061 | 17.723 | 2.06 | 64 | 64 | 64 | 4 | 2 |
| o_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | torch_fp16 | 0.032 | 136.227 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | torch_quantized_ref | 0.212 | 20.284 | 2.06 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | triton_fixed | 0.101 | 42.733 | 2.06 | 64 | 64 | 64 |  |  |
| o_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | triton_autotuned | 0.098 | 43.964 | 2.06 | 64 | 64 | 32 | 4 | 2 |
| o_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | torch_fp16 | 0.081 | 211.785 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | torch_quantized_ref | 0.280 | 61.304 | 2.06 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | triton_fixed | 0.337 | 50.968 | 2.06 | 64 | 64 | 64 |  |  |
| o_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | triton_autotuned | 0.257 | 66.851 | 2.06 | 128 | 64 | 64 | 4 | 2 |
| gate_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | torch_fp16 | 0.036 | 118.797 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | torch_quantized_ref | 0.808 | 5.316 | 8.25 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | triton_fixed | 0.104 | 41.453 | 8.25 | 64 | 64 | 64 |  |  |
| gate_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | triton_autotuned | 0.097 | 44.359 | 8.25 | 64 | 64 | 32 | 4 | 2 |
| gate_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | torch_fp16 | 0.109 | 158.138 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | torch_quantized_ref | 0.882 | 19.476 | 8.25 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | triton_fixed | 0.321 | 53.448 | 8.25 | 64 | 64 | 64 |  |  |
| gate_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | triton_autotuned | 0.244 | 70.328 | 8.25 | 128 | 64 | 64 | 4 | 2 |
| gate_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | torch_fp16 | 0.319 | 215.597 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | torch_quantized_ref | 1.110 | 61.925 | 8.25 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | triton_fixed | 1.235 | 55.656 | 8.25 | 64 | 64 | 64 |  |  |
| gate_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | triton_autotuned | 0.967 | 71.052 | 8.25 | 128 | 64 | 64 | 4 | 2 |
| up_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | torch_fp16 | 0.037 | 116.751 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | torch_quantized_ref | 0.811 | 5.298 | 8.25 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | triton_fixed | 0.104 | 41.111 | 8.25 | 64 | 64 | 64 |  |  |
| up_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | triton_autotuned | 0.101 | 42.440 | 8.25 | 64 | 64 | 32 | 4 | 2 |
| up_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | torch_fp16 | 0.110 | 155.800 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | torch_quantized_ref | 0.888 | 19.346 | 8.25 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | triton_fixed | 0.325 | 52.930 | 8.25 | 64 | 64 | 64 |  |  |
| up_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | triton_autotuned | 0.259 | 66.440 | 8.25 | 128 | 64 | 64 | 4 | 2 |
| up_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | torch_fp16 | 0.355 | 193.498 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | torch_quantized_ref | 1.117 | 61.515 | 8.25 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | triton_fixed | 1.231 | 55.825 | 8.25 | 64 | 64 | 64 |  |  |
| up_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | triton_autotuned | 0.973 | 70.601 | 8.25 | 128 | 64 | 64 | 4 | 2 |
| down_proj | rowwise_int4_gptq | 128 | 8192 | 2048 | torch_fp16 | 0.041 | 105.787 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 128 | 8192 | 2048 | torch_quantized_ref | 0.809 | 5.311 | 8.25 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 128 | 8192 | 2048 | triton_fixed | 0.170 | 25.241 | 8.25 | 64 | 64 | 64 |  |  |
| down_proj | rowwise_int4_gptq | 128 | 8192 | 2048 | triton_autotuned | 0.191 | 22.454 | 8.25 | 64 | 64 | 64 | 4 | 2 |
| down_proj | rowwise_int4_gptq | 512 | 8192 | 2048 | torch_fp16 | 0.098 | 174.817 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 512 | 8192 | 2048 | torch_quantized_ref | 0.874 | 19.647 | 8.25 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 512 | 8192 | 2048 | triton_fixed | 0.410 | 41.856 | 8.25 | 64 | 64 | 64 |  |  |
| down_proj | rowwise_int4_gptq | 512 | 8192 | 2048 | triton_autotuned | 0.367 | 46.796 | 8.25 | 128 | 64 | 32 | 4 | 2 |
| down_proj | rowwise_int4_gptq | 2048 | 8192 | 2048 | torch_fp16 | 0.370 | 185.808 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 2048 | 8192 | 2048 | torch_quantized_ref | 1.134 | 60.622 | 8.25 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 2048 | 8192 | 2048 | triton_fixed | 1.347 | 50.998 | 8.25 | 64 | 64 | 64 |  |  |
| down_proj | rowwise_int4_gptq | 2048 | 8192 | 2048 | triton_autotuned | 1.079 | 63.675 | 8.25 | 128 | 64 | 64 | 4 | 2 |
| lm_head | rowwise_int4_gptq | 128 | 2048 | 128256 | torch_fp16 | 0.430 | 156.219 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 128 | 2048 | 128256 | torch_quantized_ref | 11.479 | 5.858 | 129.16 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 128 | 2048 | 128256 | triton_fixed | 1.258 | 53.450 | 129.16 | 64 | 64 | 64 |  |  |
| lm_head | rowwise_int4_gptq | 128 | 2048 | 128256 | triton_autotuned | 1.016 | 66.188 | 129.16 | 128 | 64 | 64 | 4 | 2 |
| lm_head | rowwise_int4_gptq | 512 | 2048 | 128256 | torch_fp16 | 1.758 | 152.966 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 512 | 2048 | 128256 | torch_quantized_ref | 12.340 | 21.796 | 129.16 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 512 | 2048 | 128256 | triton_fixed | 4.794 | 56.104 | 129.16 | 64 | 64 | 64 |  |  |
| lm_head | rowwise_int4_gptq | 512 | 2048 | 128256 | triton_autotuned | 3.856 | 69.758 | 129.16 | 128 | 64 | 32 | 4 | 2 |
| lm_head | rowwise_int4_gptq | 2048 | 2048 | 128256 | torch_fp16 | 6.202 | 173.463 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 2048 | 2048 | 128256 | torch_quantized_ref | 16.145 | 66.641 | 129.16 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 2048 | 2048 | 128256 | triton_fixed | 19.014 | 56.584 | 129.16 | 64 | 64 | 64 |  |  |
| lm_head | rowwise_int4_gptq | 2048 | 2048 | 128256 | triton_autotuned | 15.025 | 71.607 | 129.16 | 128 | 64 | 64 | 4 | 2 |
