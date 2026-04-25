| layer | kernel | batch | IN | OUT | method | latency_ms | TFLOP/s | weight_MB | BLOCK_M | BLOCK_N | BLOCK_K | warps | stages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_fp16 | 0.031 | 34.657 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_quantized_ref | 0.606 | 1.772 | 2.01 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_fixed | 0.106 | 10.156 | 2.01 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_autotuned | 0.175 | 6.139 | 2.01 | 128 | 32 | 32 | 4 | 2 |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_fp16 | 0.096 | 44.891 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_quantized_ref | 0.668 | 6.433 | 2.01 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_fixed | 0.374 | 11.497 | 2.01 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_autotuned | 0.157 | 27.421 | 2.01 | 256 | 64 | 32 | 4 | 3 |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_fp16 | 0.282 | 60.994 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_quantized_ref | 0.848 | 20.268 | 2.01 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_fixed | 1.452 | 11.833 | 2.01 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_autotuned | 0.471 | 36.465 | 2.01 | 256 | 128 | 32 | 8 | 3 |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_fp16 | 0.012 | 22.028 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_quantized_ref | 0.121 | 2.212 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_fixed | 0.037 | 7.215 | 0.50 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_autotuned | 0.059 | 4.564 | 0.50 | 64 | 64 | 64 | 4 | 3 |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_fp16 | 0.031 | 34.789 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_quantized_ref | 0.182 | 5.889 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_fixed | 0.107 | 10.057 | 0.50 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_autotuned | 0.062 | 17.453 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_fp16 | 0.079 | 54.381 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_quantized_ref | 0.180 | 23.817 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_fixed | 0.373 | 11.526 | 0.50 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_autotuned | 0.159 | 27.010 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_fp16 | 0.016 | 16.994 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_quantized_ref | 0.169 | 1.591 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_fixed | 0.047 | 5.685 | 0.50 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_autotuned | 0.052 | 5.181 | 0.50 | 64 | 64 | 64 | 4 | 3 |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_fp16 | 0.031 | 34.838 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_quantized_ref | 0.127 | 8.431 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_fixed | 0.106 | 10.126 | 0.50 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_autotuned | 0.062 | 17.389 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_fp16 | 0.079 | 54.393 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_quantized_ref | 0.180 | 23.830 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_fixed | 0.372 | 11.533 | 0.50 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_autotuned | 0.158 | 27.218 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_fp16 | 0.031 | 34.830 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_quantized_ref | 0.606 | 1.773 | 2.01 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_fixed | 0.106 | 10.139 | 2.01 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_autotuned | 0.059 | 18.074 | 2.01 | 128 | 32 | 32 | 4 | 2 |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_fp16 | 0.095 | 45.054 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_quantized_ref | 0.667 | 6.437 | 2.01 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_fixed | 0.373 | 11.502 | 2.01 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_autotuned | 0.156 | 27.604 | 2.01 | 256 | 64 | 32 | 4 | 3 |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_fp16 | 0.282 | 61.014 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_quantized_ref | 0.848 | 20.257 | 2.01 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_fixed | 1.452 | 11.832 | 2.01 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_autotuned | 0.471 | 36.462 | 2.01 | 256 | 128 | 32 | 8 | 3 |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_fp16 | 0.137 | 31.331 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_quantized_ref | 2.330 | 1.843 | 8.05 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_fixed | 0.374 | 11.472 | 8.05 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_autotuned | 0.183 | 23.451 | 8.05 | 128 | 32 | 32 | 4 | 2 |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_fp16 | 0.330 | 52.105 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_quantized_ref | 2.517 | 6.825 | 8.05 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_fixed | 1.463 | 11.740 | 8.05 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_autotuned | 0.470 | 36.559 | 8.05 | 256 | 128 | 32 | 8 | 3 |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_fp16 | 1.059 | 64.913 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_quantized_ref | 3.255 | 21.112 | 8.05 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_fixed | 5.764 | 11.922 | 8.05 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_autotuned | 1.770 | 38.832 | 8.05 | 256 | 64 | 32 | 4 | 3 |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_fp16 | 0.138 | 31.025 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_quantized_ref | 2.331 | 1.842 | 8.05 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_fixed | 0.374 | 11.471 | 8.05 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_autotuned | 0.183 | 23.445 | 8.05 | 128 | 32 | 32 | 4 | 2 |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_fp16 | 0.329 | 52.168 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_quantized_ref | 2.516 | 6.827 | 8.05 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_fixed | 1.463 | 11.743 | 8.05 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_autotuned | 0.470 | 36.573 | 8.05 | 256 | 128 | 32 | 8 | 3 |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_fp16 | 1.060 | 64.823 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_quantized_ref | 3.255 | 21.113 | 8.05 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_fixed | 5.764 | 11.922 | 8.05 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_autotuned | 1.771 | 38.793 | 8.05 | 256 | 64 | 32 | 4 | 3 |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | torch_fp16 | 0.149 | 28.836 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | torch_quantized_ref | 2.339 | 1.836 | 8.01 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | triton_fixed | 0.411 | 10.456 | 8.01 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | triton_autotuned | 0.247 | 17.359 | 8.01 | 128 | 32 | 32 | 4 | 2 |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | torch_fp16 | 0.384 | 44.747 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | torch_quantized_ref | 2.566 | 6.696 | 8.01 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | triton_fixed | 1.460 | 11.764 | 8.01 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | triton_autotuned | 0.598 | 28.706 | 8.01 | 256 | 64 | 32 | 4 | 3 |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | torch_fp16 | 1.044 | 65.812 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | torch_quantized_ref | 3.231 | 21.271 | 8.01 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | triton_fixed | 8.653 | 7.942 | 8.01 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | triton_autotuned | 1.745 | 39.382 | 8.01 | 256 | 128 | 32 | 8 | 3 |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | torch_fp16 | 1.831 | 36.734 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | torch_quantized_ref | 35.535 | 1.892 | 125.98 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | triton_fixed | 5.682 | 11.835 | 125.98 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | triton_autotuned | 2.430 | 27.670 | 125.98 | 128 | 64 | 32 | 4 | 2 |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | torch_fp16 | 4.637 | 58.000 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | torch_quantized_ref | 38.340 | 7.015 | 125.98 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | triton_fixed | 22.691 | 11.854 | 125.98 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | triton_autotuned | 6.688 | 40.216 | 125.98 | 256 | 64 | 32 | 4 | 3 |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | torch_fp16 | 15.905 | 67.645 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | torch_quantized_ref | 49.620 | 21.682 | 125.98 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | triton_fixed | 90.087 | 11.943 | 125.98 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | triton_autotuned | 26.809 | 40.131 | 125.98 | 256 | 64 | 32 | 4 | 3 |
