| layer | kernel | batch | IN | OUT | method | latency_ms | TFLOP/s | weight_MB | BLOCK_M | BLOCK_N | BLOCK_K | warps | stages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_fp16 | 0.017 | 63.450 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_quantized_ref | 0.167 | 6.441 | 2.01 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_fixed | 0.075 | 14.399 | 2.01 | 256 | 64 | 32 |  |  |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_autotuned | 0.062 | 17.230 | 2.01 | 64 | 64 | 64 | 4 | 3 |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_fp16 | 0.032 | 135.715 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_quantized_ref | 0.161 | 26.649 | 2.01 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_fixed | 0.063 | 68.719 | 2.01 | 256 | 64 | 32 |  |  |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_autotuned | 0.062 | 69.383 | 2.01 | 256 | 64 | 32 | 4 | 3 |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_fp16 | 0.081 | 210.995 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_quantized_ref | 0.230 | 74.745 | 2.01 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_fixed | 0.206 | 83.255 | 2.01 | 256 | 64 | 32 |  |  |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_autotuned | 0.175 | 98.136 | 2.01 | 256 | 64 | 32 | 4 | 3 |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_fp16 | 0.017 | 15.350 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_quantized_ref | 0.144 | 1.858 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_fixed | 0.062 | 4.313 | 0.50 | 256 | 64 | 32 |  |  |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_autotuned | 0.061 | 4.410 | 0.50 | 32 | 32 | 64 | 2 | 3 |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_fp16 | 0.014 | 78.556 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_quantized_ref | 0.135 | 7.945 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_fixed | 0.062 | 17.452 | 0.50 | 256 | 64 | 32 |  |  |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_autotuned | 0.061 | 17.610 | 0.50 | 64 | 64 | 64 | 4 | 3 |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_fp16 | 0.031 | 136.372 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_quantized_ref | 0.136 | 31.681 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_fixed | 0.062 | 69.637 | 0.50 | 256 | 64 | 32 |  |  |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_autotuned | 0.062 | 69.155 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_fp16 | 0.017 | 15.496 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_quantized_ref | 0.167 | 1.608 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_fixed | 0.062 | 4.346 | 0.50 | 256 | 64 | 32 |  |  |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_autotuned | 0.078 | 3.449 | 0.50 | 32 | 32 | 64 | 2 | 3 |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_fp16 | 0.014 | 78.807 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_quantized_ref | 0.148 | 7.270 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_fixed | 0.062 | 17.454 | 0.50 | 256 | 64 | 32 |  |  |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_autotuned | 0.060 | 17.887 | 0.50 | 64 | 64 | 64 | 4 | 3 |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_fp16 | 0.032 | 136.321 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_quantized_ref | 0.136 | 31.682 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_fixed | 0.062 | 69.631 | 0.50 | 256 | 64 | 32 |  |  |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_autotuned | 0.062 | 69.511 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_fp16 | 0.014 | 78.110 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_quantized_ref | 0.142 | 7.545 | 2.01 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_fixed | 0.062 | 17.427 | 2.01 | 256 | 64 | 32 |  |  |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_autotuned | 0.060 | 17.978 | 2.01 | 64 | 64 | 64 | 4 | 3 |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_fp16 | 0.032 | 135.955 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_quantized_ref | 0.161 | 26.695 | 2.01 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_fixed | 0.063 | 68.006 | 2.01 | 256 | 64 | 32 |  |  |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_autotuned | 0.063 | 67.847 | 2.01 | 256 | 64 | 32 | 4 | 3 |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_fp16 | 0.082 | 210.107 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_quantized_ref | 0.234 | 73.434 | 2.01 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_fixed | 0.209 | 82.313 | 2.01 | 256 | 64 | 32 |  |  |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_autotuned | 0.197 | 87.389 | 2.01 | 256 | 64 | 32 | 4 | 3 |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_fp16 | 0.037 | 116.565 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_quantized_ref | 0.559 | 7.690 | 8.05 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_fixed | 0.102 | 42.153 | 8.05 | 256 | 64 | 32 |  |  |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_autotuned | 0.073 | 58.838 | 8.05 | 64 | 64 | 64 | 4 | 3 |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_fp16 | 0.109 | 157.347 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_quantized_ref | 0.641 | 26.797 | 8.05 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_fixed | 0.198 | 86.682 | 8.05 | 256 | 64 | 32 |  |  |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_autotuned | 0.166 | 103.710 | 8.05 | 256 | 64 | 32 | 4 | 3 |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_fp16 | 0.354 | 193.859 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_quantized_ref | 0.887 | 77.465 | 8.05 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_fixed | 0.568 | 121.057 | 8.05 | 256 | 64 | 32 |  |  |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_autotuned | 0.531 | 129.416 | 8.05 | 256 | 128 | 32 | 8 | 3 |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_fp16 | 0.038 | 112.554 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_quantized_ref | 0.565 | 7.600 | 8.05 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_fixed | 0.104 | 41.418 | 8.05 | 256 | 64 | 32 |  |  |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_autotuned | 0.079 | 54.550 | 8.05 | 64 | 64 | 64 | 4 | 3 |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_fp16 | 0.112 | 153.814 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_quantized_ref | 0.649 | 26.479 | 8.05 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_fixed | 0.197 | 87.393 | 8.05 | 256 | 64 | 32 |  |  |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_autotuned | 0.191 | 90.139 | 8.05 | 256 | 64 | 32 | 4 | 3 |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_fp16 | 0.341 | 201.374 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_quantized_ref | 0.885 | 77.614 | 8.05 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_fixed | 0.573 | 119.988 | 8.05 | 256 | 64 | 32 |  |  |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_autotuned | 0.553 | 124.334 | 8.05 | 256 | 128 | 32 | 8 | 3 |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | torch_fp16 | 0.042 | 101.861 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | torch_quantized_ref | 0.562 | 7.645 | 8.01 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | triton_fixed | 0.243 | 17.679 | 8.01 | 256 | 64 | 32 |  |  |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | triton_autotuned | 0.115 | 37.450 | 8.01 | 64 | 64 | 64 | 4 | 3 |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | torch_fp16 | 0.105 | 163.218 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | torch_quantized_ref | 0.636 | 27.000 | 8.01 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | triton_fixed | 0.257 | 66.851 | 8.01 | 256 | 64 | 32 |  |  |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | triton_autotuned | 0.233 | 73.799 | 8.01 | 256 | 64 | 32 | 4 | 3 |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | torch_fp16 | 0.375 | 183.364 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | torch_quantized_ref | 0.913 | 75.259 | 8.01 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | triton_fixed | 0.829 | 82.874 | 8.01 | 256 | 64 | 32 |  |  |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | triton_autotuned | 0.752 | 91.360 | 8.01 | 256 | 128 | 32 | 8 | 3 |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | torch_fp16 | 0.423 | 158.845 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | torch_quantized_ref | 7.890 | 8.523 | 125.98 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | triton_fixed | 1.012 | 66.431 | 125.98 | 256 | 64 | 32 |  |  |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | triton_autotuned | 0.779 | 86.311 | 125.98 | 128 | 64 | 32 | 4 | 2 |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | torch_fp16 | 1.776 | 151.464 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | torch_quantized_ref | 8.884 | 30.275 | 125.98 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | triton_fixed | 2.168 | 124.048 | 125.98 | 256 | 64 | 32 |  |  |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | triton_autotuned | 2.084 | 129.057 | 125.98 | 256 | 128 | 32 | 8 | 3 |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | torch_fp16 | 6.307 | 170.597 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | torch_quantized_ref | 12.944 | 83.118 | 125.98 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | triton_fixed | 8.491 | 126.705 | 125.98 | 256 | 64 | 32 |  |  |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | triton_autotuned | 8.107 | 132.715 | 125.98 | 256 | 128 | 32 | 8 | 3 |
