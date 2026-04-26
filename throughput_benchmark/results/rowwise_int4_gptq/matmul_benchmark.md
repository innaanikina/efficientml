| layer | kernel | batch | IN | OUT | method | latency_ms | TFLOP/s | weight_MB | BLOCK_M | BLOCK_N | BLOCK_K | warps | stages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | torch_fp16 | 0.017 | 62.631 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | torch_quantized_ref | 0.219 | 4.896 | 2.06 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | triton_fixed | 0.055 | 19.485 | 2.06 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | triton_autotuned | 0.061 | 17.638 | 2.06 | 64 | 64 | 64 | 4 | 2 |
| q_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | torch_fp16 | 0.032 | 135.746 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | torch_quantized_ref | 0.212 | 20.242 | 2.06 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | triton_fixed | 0.128 | 33.527 | 2.06 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | triton_autotuned | 0.097 | 44.412 | 2.06 | 64 | 64 | 32 | 4 | 2 |
| q_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | torch_fp16 | 0.080 | 213.781 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | torch_quantized_ref | 0.272 | 63.235 | 2.06 |  |  |  |  |  |
| q_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | triton_fixed | 0.495 | 34.716 | 2.06 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | triton_autotuned | 0.241 | 71.424 | 2.06 | 128 | 64 | 64 | 4 | 2 |
| k_proj | rowwise_int4_gptq | 128 | 2048 | 512 | torch_fp16 | 0.018 | 15.261 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 128 | 2048 | 512 | torch_quantized_ref | 0.194 | 1.382 | 0.52 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 128 | 2048 | 512 | triton_fixed | 0.039 | 6.832 | 0.52 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4_gptq | 128 | 2048 | 512 | triton_autotuned | 0.062 | 4.351 | 0.52 | 64 | 64 | 64 | 4 | 2 |
| k_proj | rowwise_int4_gptq | 512 | 2048 | 512 | torch_fp16 | 0.014 | 77.568 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 512 | 2048 | 512 | torch_quantized_ref | 0.183 | 5.856 | 0.52 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 512 | 2048 | 512 | triton_fixed | 0.045 | 23.870 | 0.52 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4_gptq | 512 | 2048 | 512 | triton_autotuned | 0.061 | 17.606 | 0.52 | 64 | 64 | 64 | 4 | 2 |
| k_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | torch_fp16 | 0.032 | 135.785 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | torch_quantized_ref | 0.183 | 23.492 | 0.52 |  |  |  |  |  |
| k_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | triton_fixed | 0.125 | 34.353 | 0.52 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | triton_autotuned | 0.097 | 44.335 | 0.52 | 64 | 64 | 32 | 4 | 2 |
| v_proj | rowwise_int4_gptq | 128 | 2048 | 512 | torch_fp16 | 0.018 | 15.124 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 128 | 2048 | 512 | torch_quantized_ref | 0.193 | 1.393 | 0.52 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 128 | 2048 | 512 | triton_fixed | 0.040 | 6.762 | 0.52 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4_gptq | 128 | 2048 | 512 | triton_autotuned | 0.061 | 4.425 | 0.52 | 64 | 64 | 64 | 4 | 2 |
| v_proj | rowwise_int4_gptq | 512 | 2048 | 512 | torch_fp16 | 0.014 | 77.775 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 512 | 2048 | 512 | torch_quantized_ref | 0.182 | 5.914 | 0.52 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 512 | 2048 | 512 | triton_fixed | 0.045 | 23.894 | 0.52 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4_gptq | 512 | 2048 | 512 | triton_autotuned | 0.060 | 18.045 | 0.52 | 64 | 64 | 64 | 4 | 2 |
| v_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | torch_fp16 | 0.032 | 135.864 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | torch_quantized_ref | 0.182 | 23.543 | 0.52 |  |  |  |  |  |
| v_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | triton_fixed | 0.125 | 34.352 | 0.52 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4_gptq | 2048 | 2048 | 512 | triton_autotuned | 0.097 | 44.350 | 0.52 | 64 | 64 | 32 | 4 | 2 |
| o_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | torch_fp16 | 0.014 | 75.580 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | torch_quantized_ref | 0.197 | 5.455 | 2.06 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | triton_fixed | 0.045 | 23.647 | 2.06 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4_gptq | 128 | 2048 | 2048 | triton_autotuned | 0.059 | 18.143 | 2.06 | 64 | 64 | 64 | 4 | 2 |
| o_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | torch_fp16 | 0.032 | 134.137 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | torch_quantized_ref | 0.214 | 20.091 | 2.06 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | triton_fixed | 0.127 | 33.709 | 2.06 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4_gptq | 512 | 2048 | 2048 | triton_autotuned | 0.101 | 42.569 | 2.06 | 64 | 64 | 32 | 4 | 2 |
| o_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | torch_fp16 | 0.082 | 210.444 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | torch_quantized_ref | 0.283 | 60.619 | 2.06 |  |  |  |  |  |
| o_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | triton_fixed | 0.506 | 33.943 | 2.06 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4_gptq | 2048 | 2048 | 2048 | triton_autotuned | 0.267 | 64.428 | 2.06 | 128 | 64 | 64 | 4 | 2 |
| gate_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | torch_fp16 | 0.037 | 115.981 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | torch_quantized_ref | 0.807 | 5.321 | 8.25 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | triton_fixed | 0.128 | 33.437 | 8.25 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | triton_autotuned | 0.095 | 45.174 | 8.25 | 128 | 64 | 32 | 4 | 2 |
| gate_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | torch_fp16 | 0.108 | 158.484 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | torch_quantized_ref | 0.879 | 19.541 | 8.25 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | triton_fixed | 0.474 | 36.246 | 8.25 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | triton_autotuned | 0.244 | 70.459 | 8.25 | 128 | 64 | 64 | 4 | 2 |
| gate_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | torch_fp16 | 0.345 | 199.444 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | torch_quantized_ref | 1.108 | 61.994 | 8.25 |  |  |  |  |  |
| gate_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | triton_fixed | 2.008 | 34.216 | 8.25 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | triton_autotuned | 0.966 | 71.136 | 8.25 | 128 | 64 | 64 | 4 | 2 |
| up_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | torch_fp16 | 0.037 | 116.115 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | torch_quantized_ref | 0.811 | 5.297 | 8.25 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | triton_fixed | 0.131 | 32.773 | 8.25 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4_gptq | 128 | 2048 | 8192 | triton_autotuned | 0.101 | 42.479 | 8.25 | 128 | 64 | 32 | 4 | 2 |
| up_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | torch_fp16 | 0.111 | 154.148 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | torch_quantized_ref | 0.886 | 19.394 | 8.25 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | triton_fixed | 0.494 | 34.773 | 8.25 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4_gptq | 512 | 2048 | 8192 | triton_autotuned | 0.267 | 64.349 | 8.25 | 128 | 64 | 64 | 4 | 2 |
| up_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | torch_fp16 | 0.328 | 209.463 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | torch_quantized_ref | 1.107 | 62.093 | 8.25 |  |  |  |  |  |
| up_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | triton_fixed | 1.989 | 34.555 | 8.25 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4_gptq | 2048 | 2048 | 8192 | triton_autotuned | 0.972 | 70.735 | 8.25 | 128 | 64 | 64 | 4 | 2 |
| down_proj | rowwise_int4_gptq | 128 | 8192 | 2048 | torch_fp16 | 0.040 | 107.280 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 128 | 8192 | 2048 | torch_quantized_ref | 0.807 | 5.322 | 8.25 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 128 | 8192 | 2048 | triton_fixed | 0.182 | 23.591 | 8.25 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4_gptq | 128 | 8192 | 2048 | triton_autotuned | 0.192 | 22.398 | 8.25 | 64 | 64 | 64 | 4 | 2 |
| down_proj | rowwise_int4_gptq | 512 | 8192 | 2048 | torch_fp16 | 0.098 | 175.489 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 512 | 8192 | 2048 | torch_quantized_ref | 0.873 | 19.675 | 8.25 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 512 | 8192 | 2048 | triton_fixed | 0.571 | 30.076 | 8.25 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4_gptq | 512 | 8192 | 2048 | triton_autotuned | 0.368 | 46.693 | 8.25 | 128 | 64 | 32 | 4 | 2 |
| down_proj | rowwise_int4_gptq | 2048 | 8192 | 2048 | torch_fp16 | 0.366 | 187.742 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 2048 | 8192 | 2048 | torch_quantized_ref | 1.131 | 60.739 | 8.25 |  |  |  |  |  |
| down_proj | rowwise_int4_gptq | 2048 | 8192 | 2048 | triton_fixed | 2.327 | 29.528 | 8.25 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4_gptq | 2048 | 8192 | 2048 | triton_autotuned | 1.069 | 64.262 | 8.25 | 128 | 64 | 64 | 4 | 2 |
| lm_head | rowwise_int4_gptq | 128 | 2048 | 128256 | torch_fp16 | 0.420 | 160.193 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 128 | 2048 | 128256 | torch_quantized_ref | 11.445 | 5.875 | 129.16 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 128 | 2048 | 128256 | triton_fixed | 1.978 | 33.994 | 129.16 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4_gptq | 128 | 2048 | 128256 | triton_autotuned | 1.005 | 66.887 | 129.16 | 128 | 64 | 64 | 4 | 2 |
| lm_head | rowwise_int4_gptq | 512 | 2048 | 128256 | torch_fp16 | 1.716 | 156.745 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 512 | 2048 | 128256 | torch_quantized_ref | 12.300 | 21.867 | 129.16 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 512 | 2048 | 128256 | triton_fixed | 7.840 | 34.310 | 129.16 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4_gptq | 512 | 2048 | 128256 | triton_autotuned | 3.828 | 70.259 | 129.16 | 128 | 64 | 64 | 4 | 2 |
| lm_head | rowwise_int4_gptq | 2048 | 2048 | 128256 | torch_fp16 | 6.203 | 173.460 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 2048 | 2048 | 128256 | torch_quantized_ref | 16.110 | 66.785 | 129.16 |  |  |  |  |  |
| lm_head | rowwise_int4_gptq | 2048 | 2048 | 128256 | triton_fixed | 31.290 | 34.384 | 129.16 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4_gptq | 2048 | 2048 | 128256 | triton_autotuned | 15.000 | 71.725 | 129.16 | 128 | 64 | 32 | 4 | 2 |
