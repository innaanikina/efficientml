| layer | kernel | batch | IN | OUT | method | latency_ms | TFLOP/s | weight_MB | BLOCK_M | BLOCK_N | BLOCK_K | warps | stages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_proj | rowwise_int4 | 128 | 2048 | 2048 | torch_fp16 | 0.017 | 62.975 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 128 | 2048 | 2048 | torch_quantized_ref | 0.196 | 5.472 | 2.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 128 | 2048 | 2048 | triton_fixed | 0.048 | 22.549 | 2.00 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4 | 128 | 2048 | 2048 | triton_autotuned | 0.059 | 18.204 | 2.00 | 64 | 64 | 64 | 4 | 3 |
| q_proj | rowwise_int4 | 512 | 2048 | 2048 | torch_fp16 | 0.032 | 135.468 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 512 | 2048 | 2048 | torch_quantized_ref | 0.190 | 22.550 | 2.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 512 | 2048 | 2048 | triton_fixed | 0.107 | 40.181 | 2.00 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4 | 512 | 2048 | 2048 | triton_autotuned | 0.067 | 63.670 | 2.00 | 256 | 64 | 32 | 4 | 3 |
| q_proj | rowwise_int4 | 2048 | 2048 | 2048 | torch_fp16 | 0.081 | 212.867 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 2048 | 2048 | 2048 | torch_quantized_ref | 0.253 | 67.924 | 2.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 2048 | 2048 | 2048 | triton_fixed | 0.430 | 39.985 | 2.00 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4 | 2048 | 2048 | 2048 | triton_autotuned | 0.161 | 106.603 | 2.00 | 256 | 64 | 32 | 4 | 3 |
| k_proj | rowwise_int4 | 128 | 2048 | 512 | torch_fp16 | 0.017 | 15.476 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4 | 128 | 2048 | 512 | torch_quantized_ref | 0.172 | 1.559 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4 | 128 | 2048 | 512 | triton_fixed | 0.039 | 6.885 | 0.50 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4 | 128 | 2048 | 512 | triton_autotuned | 0.067 | 4.026 | 0.50 | 32 | 64 | 64 | 4 | 3 |
| k_proj | rowwise_int4 | 512 | 2048 | 512 | torch_fp16 | 0.014 | 77.829 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4 | 512 | 2048 | 512 | torch_quantized_ref | 0.163 | 6.593 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4 | 512 | 2048 | 512 | triton_fixed | 0.040 | 27.084 | 0.50 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4 | 512 | 2048 | 512 | triton_autotuned | 0.061 | 17.699 | 0.50 | 64 | 64 | 64 | 4 | 3 |
| k_proj | rowwise_int4 | 2048 | 2048 | 512 | torch_fp16 | 0.032 | 132.668 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4 | 2048 | 2048 | 512 | torch_quantized_ref | 0.201 | 21.334 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4 | 2048 | 2048 | 512 | triton_fixed | 0.107 | 40.098 | 0.50 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4 | 2048 | 2048 | 512 | triton_autotuned | 0.067 | 63.666 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| v_proj | rowwise_int4 | 128 | 2048 | 512 | torch_fp16 | 0.018 | 15.189 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4 | 128 | 2048 | 512 | torch_quantized_ref | 0.172 | 1.562 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4 | 128 | 2048 | 512 | triton_fixed | 0.039 | 6.900 | 0.50 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4 | 128 | 2048 | 512 | triton_autotuned | 0.058 | 4.651 | 0.50 | 32 | 64 | 64 | 4 | 3 |
| v_proj | rowwise_int4 | 512 | 2048 | 512 | torch_fp16 | 0.014 | 78.032 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4 | 512 | 2048 | 512 | torch_quantized_ref | 0.163 | 6.588 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4 | 512 | 2048 | 512 | triton_fixed | 0.039 | 27.320 | 0.50 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4 | 512 | 2048 | 512 | triton_autotuned | 0.057 | 18.909 | 0.50 | 64 | 64 | 64 | 4 | 3 |
| v_proj | rowwise_int4 | 2048 | 2048 | 512 | torch_fp16 | 0.032 | 135.772 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4 | 2048 | 2048 | 512 | torch_quantized_ref | 0.163 | 26.358 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4 | 2048 | 2048 | 512 | triton_fixed | 0.107 | 40.147 | 0.50 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4 | 2048 | 2048 | 512 | triton_autotuned | 0.068 | 63.586 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| o_proj | rowwise_int4 | 128 | 2048 | 2048 | torch_fp16 | 0.014 | 77.477 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 128 | 2048 | 2048 | torch_quantized_ref | 0.172 | 6.242 | 2.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 128 | 2048 | 2048 | triton_fixed | 0.039 | 27.338 | 2.00 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4 | 128 | 2048 | 2048 | triton_autotuned | 0.057 | 18.841 | 2.00 | 64 | 64 | 64 | 4 | 3 |
| o_proj | rowwise_int4 | 512 | 2048 | 2048 | torch_fp16 | 0.032 | 135.606 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 512 | 2048 | 2048 | torch_quantized_ref | 0.190 | 22.595 | 2.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 512 | 2048 | 2048 | triton_fixed | 0.109 | 39.561 | 2.00 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4 | 512 | 2048 | 2048 | triton_autotuned | 0.069 | 62.333 | 2.00 | 256 | 64 | 32 | 4 | 3 |
| o_proj | rowwise_int4 | 2048 | 2048 | 2048 | torch_fp16 | 0.081 | 211.343 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 2048 | 2048 | 2048 | torch_quantized_ref | 0.262 | 65.669 | 2.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 2048 | 2048 | 2048 | triton_fixed | 0.427 | 40.248 | 2.00 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4 | 2048 | 2048 | 2048 | triton_autotuned | 0.179 | 95.968 | 2.00 | 256 | 64 | 32 | 4 | 3 |
| gate_proj | rowwise_int4 | 128 | 2048 | 8192 | torch_fp16 | 0.036 | 118.027 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 128 | 2048 | 8192 | torch_quantized_ref | 0.724 | 5.932 | 8.02 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 128 | 2048 | 8192 | triton_fixed | 0.113 | 37.952 | 8.02 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4 | 128 | 2048 | 8192 | triton_autotuned | 0.063 | 68.023 | 8.02 | 128 | 32 | 32 | 4 | 2 |
| gate_proj | rowwise_int4 | 512 | 2048 | 8192 | torch_fp16 | 0.109 | 158.204 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 512 | 2048 | 8192 | torch_quantized_ref | 0.800 | 21.469 | 8.02 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 512 | 2048 | 8192 | triton_fixed | 0.418 | 41.084 | 8.02 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4 | 512 | 2048 | 8192 | triton_autotuned | 0.163 | 105.673 | 8.02 | 256 | 64 | 32 | 4 | 3 |
| gate_proj | rowwise_int4 | 2048 | 2048 | 8192 | torch_fp16 | 0.351 | 195.817 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 2048 | 2048 | 8192 | torch_quantized_ref | 1.055 | 65.146 | 8.02 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 2048 | 2048 | 8192 | triton_fixed | 1.706 | 40.284 | 8.02 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4 | 2048 | 2048 | 8192 | triton_autotuned | 0.559 | 122.884 | 8.02 | 256 | 64 | 32 | 4 | 3 |
| up_proj | rowwise_int4 | 128 | 2048 | 8192 | torch_fp16 | 0.039 | 109.843 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4 | 128 | 2048 | 8192 | torch_quantized_ref | 0.727 | 5.905 | 8.02 |  |  |  |  |  |
| up_proj | rowwise_int4 | 128 | 2048 | 8192 | triton_fixed | 0.114 | 37.811 | 8.02 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4 | 128 | 2048 | 8192 | triton_autotuned | 0.073 | 58.610 | 8.02 | 128 | 32 | 32 | 4 | 2 |
| up_proj | rowwise_int4 | 512 | 2048 | 8192 | torch_fp16 | 0.113 | 152.697 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4 | 512 | 2048 | 8192 | torch_quantized_ref | 0.808 | 21.269 | 8.02 |  |  |  |  |  |
| up_proj | rowwise_int4 | 512 | 2048 | 8192 | triton_fixed | 0.429 | 40.084 | 8.02 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4 | 512 | 2048 | 8192 | triton_autotuned | 0.181 | 94.665 | 8.02 | 256 | 64 | 32 | 4 | 3 |
| up_proj | rowwise_int4 | 2048 | 2048 | 8192 | torch_fp16 | 0.337 | 203.942 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4 | 2048 | 2048 | 8192 | torch_quantized_ref | 1.026 | 66.953 | 8.02 |  |  |  |  |  |
| up_proj | rowwise_int4 | 2048 | 2048 | 8192 | triton_fixed | 1.706 | 40.289 | 8.02 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4 | 2048 | 2048 | 8192 | triton_autotuned | 0.567 | 121.258 | 8.02 | 256 | 64 | 32 | 4 | 3 |
| down_proj | rowwise_int4 | 128 | 8192 | 2048 | torch_fp16 | 0.042 | 101.456 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 128 | 8192 | 2048 | torch_quantized_ref | 0.723 | 5.940 | 8.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 128 | 8192 | 2048 | triton_fixed | 0.150 | 28.685 | 8.00 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4 | 128 | 8192 | 2048 | triton_autotuned | 0.132 | 32.441 | 8.00 | 64 | 64 | 64 | 4 | 3 |
| down_proj | rowwise_int4 | 512 | 8192 | 2048 | torch_fp16 | 0.104 | 165.370 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 512 | 8192 | 2048 | torch_quantized_ref | 0.793 | 21.670 | 8.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 512 | 8192 | 2048 | triton_fixed | 0.449 | 38.286 | 8.00 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4 | 512 | 8192 | 2048 | triton_autotuned | 0.254 | 67.535 | 8.00 | 256 | 64 | 32 | 4 | 3 |
| down_proj | rowwise_int4 | 2048 | 8192 | 2048 | torch_fp16 | 0.370 | 185.788 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 2048 | 8192 | 2048 | torch_quantized_ref | 1.056 | 65.066 | 8.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 2048 | 8192 | 2048 | triton_fixed | 1.881 | 36.534 | 8.00 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4 | 2048 | 8192 | 2048 | triton_autotuned | 0.682 | 100.731 | 8.00 | 256 | 64 | 32 | 4 | 3 |
| lm_head | rowwise_int4 | 128 | 2048 | 128256 | torch_fp16 | 0.435 | 154.640 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4 | 128 | 2048 | 128256 | torch_quantized_ref | 10.240 | 6.567 | 125.49 |  |  |  |  |  |
| lm_head | rowwise_int4 | 128 | 2048 | 128256 | triton_fixed | 1.706 | 39.407 | 125.49 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4 | 128 | 2048 | 128256 | triton_autotuned | 0.782 | 85.943 | 125.49 | 128 | 64 | 32 | 4 | 2 |
| lm_head | rowwise_int4 | 512 | 2048 | 128256 | torch_fp16 | 1.714 | 156.942 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4 | 512 | 2048 | 128256 | torch_quantized_ref | 11.109 | 24.212 | 125.49 |  |  |  |  |  |
| lm_head | rowwise_int4 | 512 | 2048 | 128256 | triton_fixed | 6.668 | 40.337 | 125.49 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4 | 512 | 2048 | 128256 | triton_autotuned | 2.173 | 123.752 | 125.49 | 256 | 64 | 32 | 4 | 3 |
| lm_head | rowwise_int4 | 2048 | 2048 | 128256 | torch_fp16 | 6.203 | 173.450 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4 | 2048 | 2048 | 128256 | torch_quantized_ref | 14.961 | 71.913 | 125.49 |  |  |  |  |  |
| lm_head | rowwise_int4 | 2048 | 2048 | 128256 | triton_fixed | 26.625 | 40.409 | 125.49 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4 | 2048 | 2048 | 128256 | triton_autotuned | 8.558 | 125.716 | 125.49 | 256 | 64 | 32 | 4 | 3 |
