| layer | kernel | batch | IN | OUT | method | latency_ms | TFLOP/s | weight_MB | BLOCK_M | BLOCK_N | BLOCK_K | warps | stages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_proj | rowwise_int4 | 128 | 2048 | 2048 | torch_fp16 | 0.017 | 63.380 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 128 | 2048 | 2048 | torch_quantized_ref | 0.195 | 5.500 | 2.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 128 | 2048 | 2048 | triton_fixed | 0.081 | 13.308 | 2.00 | 256 | 64 | 32 |  |  |
| q_proj | rowwise_int4 | 128 | 2048 | 2048 | triton_autotuned | 0.058 | 18.394 | 2.00 | 64 | 64 | 64 | 4 | 3 |
| q_proj | rowwise_int4 | 512 | 2048 | 2048 | torch_fp16 | 0.032 | 135.759 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 512 | 2048 | 2048 | torch_quantized_ref | 0.189 | 22.683 | 2.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 512 | 2048 | 2048 | triton_fixed | 0.067 | 63.865 | 2.00 | 256 | 64 | 32 |  |  |
| q_proj | rowwise_int4 | 512 | 2048 | 2048 | triton_autotuned | 0.067 | 63.755 | 2.00 | 256 | 64 | 32 | 4 | 3 |
| q_proj | rowwise_int4 | 2048 | 2048 | 2048 | torch_fp16 | 0.081 | 211.370 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 2048 | 2048 | 2048 | torch_quantized_ref | 0.252 | 68.089 | 2.00 |  |  |  |  |  |
| q_proj | rowwise_int4 | 2048 | 2048 | 2048 | triton_fixed | 0.182 | 94.528 | 2.00 | 256 | 64 | 32 |  |  |
| q_proj | rowwise_int4 | 2048 | 2048 | 2048 | triton_autotuned | 0.164 | 105.060 | 2.00 | 256 | 64 | 32 | 4 | 3 |
| k_proj | rowwise_int4 | 128 | 2048 | 512 | torch_fp16 | 0.018 | 15.124 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4 | 128 | 2048 | 512 | torch_quantized_ref | 0.173 | 1.555 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4 | 128 | 2048 | 512 | triton_fixed | 0.067 | 4.002 | 0.50 | 256 | 64 | 32 |  |  |
| k_proj | rowwise_int4 | 128 | 2048 | 512 | triton_autotuned | 0.057 | 4.685 | 0.50 | 32 | 64 | 64 | 4 | 3 |
| k_proj | rowwise_int4 | 512 | 2048 | 512 | torch_fp16 | 0.014 | 78.256 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4 | 512 | 2048 | 512 | torch_quantized_ref | 0.163 | 6.576 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4 | 512 | 2048 | 512 | triton_fixed | 0.067 | 16.069 | 0.50 | 256 | 64 | 32 |  |  |
| k_proj | rowwise_int4 | 512 | 2048 | 512 | triton_autotuned | 0.058 | 18.610 | 0.50 | 64 | 64 | 64 | 4 | 3 |
| k_proj | rowwise_int4 | 2048 | 2048 | 512 | torch_fp16 | 0.032 | 136.026 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4 | 2048 | 2048 | 512 | torch_quantized_ref | 0.162 | 26.489 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4 | 2048 | 2048 | 512 | triton_fixed | 0.067 | 63.901 | 0.50 | 256 | 64 | 32 |  |  |
| k_proj | rowwise_int4 | 2048 | 2048 | 512 | triton_autotuned | 0.067 | 63.765 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| v_proj | rowwise_int4 | 128 | 2048 | 512 | torch_fp16 | 0.017 | 15.346 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4 | 128 | 2048 | 512 | torch_quantized_ref | 0.171 | 1.567 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4 | 128 | 2048 | 512 | triton_fixed | 0.066 | 4.053 | 0.50 | 256 | 64 | 32 |  |  |
| v_proj | rowwise_int4 | 128 | 2048 | 512 | triton_autotuned | 0.058 | 4.640 | 0.50 | 32 | 64 | 64 | 4 | 3 |
| v_proj | rowwise_int4 | 512 | 2048 | 512 | torch_fp16 | 0.014 | 78.252 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4 | 512 | 2048 | 512 | torch_quantized_ref | 0.162 | 6.645 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4 | 512 | 2048 | 512 | triton_fixed | 0.067 | 16.069 | 0.50 | 256 | 64 | 32 |  |  |
| v_proj | rowwise_int4 | 512 | 2048 | 512 | triton_autotuned | 0.056 | 19.043 | 0.50 | 64 | 64 | 64 | 4 | 3 |
| v_proj | rowwise_int4 | 2048 | 2048 | 512 | torch_fp16 | 0.032 | 136.070 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4 | 2048 | 2048 | 512 | torch_quantized_ref | 0.162 | 26.451 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4 | 2048 | 2048 | 512 | triton_fixed | 0.067 | 63.872 | 0.50 | 256 | 64 | 32 |  |  |
| v_proj | rowwise_int4 | 2048 | 2048 | 512 | triton_autotuned | 0.067 | 63.747 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| o_proj | rowwise_int4 | 128 | 2048 | 2048 | torch_fp16 | 0.014 | 77.925 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 128 | 2048 | 2048 | torch_quantized_ref | 0.171 | 6.275 | 2.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 128 | 2048 | 2048 | triton_fixed | 0.067 | 16.141 | 2.00 | 256 | 64 | 32 |  |  |
| o_proj | rowwise_int4 | 128 | 2048 | 2048 | triton_autotuned | 0.056 | 19.033 | 2.00 | 64 | 64 | 64 | 4 | 3 |
| o_proj | rowwise_int4 | 512 | 2048 | 2048 | torch_fp16 | 0.032 | 135.951 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 512 | 2048 | 2048 | torch_quantized_ref | 0.189 | 22.706 | 2.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 512 | 2048 | 2048 | triton_fixed | 0.068 | 63.199 | 2.00 | 256 | 64 | 32 |  |  |
| o_proj | rowwise_int4 | 512 | 2048 | 2048 | triton_autotuned | 0.068 | 63.103 | 2.00 | 256 | 64 | 32 | 4 | 3 |
| o_proj | rowwise_int4 | 2048 | 2048 | 2048 | torch_fp16 | 0.081 | 211.185 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 2048 | 2048 | 2048 | torch_quantized_ref | 0.260 | 66.071 | 2.00 |  |  |  |  |  |
| o_proj | rowwise_int4 | 2048 | 2048 | 2048 | triton_fixed | 0.185 | 92.949 | 2.00 | 256 | 64 | 32 |  |  |
| o_proj | rowwise_int4 | 2048 | 2048 | 2048 | triton_autotuned | 0.178 | 96.668 | 2.00 | 256 | 64 | 32 | 4 | 3 |
| gate_proj | rowwise_int4 | 128 | 2048 | 8192 | torch_fp16 | 0.036 | 119.643 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 128 | 2048 | 8192 | torch_quantized_ref | 0.721 | 5.953 | 8.02 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 128 | 2048 | 8192 | triton_fixed | 0.105 | 40.796 | 8.02 | 256 | 64 | 32 |  |  |
| gate_proj | rowwise_int4 | 128 | 2048 | 8192 | triton_autotuned | 0.063 | 68.064 | 8.02 | 128 | 32 | 32 | 4 | 2 |
| gate_proj | rowwise_int4 | 512 | 2048 | 8192 | torch_fp16 | 0.109 | 157.923 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 512 | 2048 | 8192 | torch_quantized_ref | 0.800 | 21.476 | 8.02 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 512 | 2048 | 8192 | triton_fixed | 0.175 | 98.282 | 8.02 | 256 | 64 | 32 |  |  |
| gate_proj | rowwise_int4 | 512 | 2048 | 8192 | triton_autotuned | 0.162 | 106.039 | 8.02 | 256 | 64 | 32 | 4 | 3 |
| gate_proj | rowwise_int4 | 2048 | 2048 | 8192 | torch_fp16 | 0.355 | 193.726 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 2048 | 2048 | 8192 | torch_quantized_ref | 1.030 | 66.722 | 8.02 |  |  |  |  |  |
| gate_proj | rowwise_int4 | 2048 | 2048 | 8192 | triton_fixed | 0.567 | 121.169 | 8.02 | 256 | 64 | 32 |  |  |
| gate_proj | rowwise_int4 | 2048 | 2048 | 8192 | triton_autotuned | 0.566 | 121.316 | 8.02 | 256 | 64 | 32 | 4 | 3 |
| up_proj | rowwise_int4 | 128 | 2048 | 8192 | torch_fp16 | 0.040 | 108.705 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4 | 128 | 2048 | 8192 | torch_quantized_ref | 0.729 | 5.894 | 8.02 |  |  |  |  |  |
| up_proj | rowwise_int4 | 128 | 2048 | 8192 | triton_fixed | 0.108 | 39.851 | 8.02 | 256 | 64 | 32 |  |  |
| up_proj | rowwise_int4 | 128 | 2048 | 8192 | triton_autotuned | 0.066 | 64.992 | 8.02 | 128 | 32 | 32 | 4 | 2 |
| up_proj | rowwise_int4 | 512 | 2048 | 8192 | torch_fp16 | 0.113 | 152.704 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4 | 512 | 2048 | 8192 | torch_quantized_ref | 0.803 | 21.393 | 8.02 |  |  |  |  |  |
| up_proj | rowwise_int4 | 512 | 2048 | 8192 | triton_fixed | 0.177 | 96.906 | 8.02 | 256 | 64 | 32 |  |  |
| up_proj | rowwise_int4 | 512 | 2048 | 8192 | triton_autotuned | 0.176 | 97.873 | 8.02 | 256 | 64 | 32 | 4 | 3 |
| up_proj | rowwise_int4 | 2048 | 2048 | 8192 | torch_fp16 | 0.352 | 195.048 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4 | 2048 | 2048 | 8192 | torch_quantized_ref | 1.029 | 66.761 | 8.02 |  |  |  |  |  |
| up_proj | rowwise_int4 | 2048 | 2048 | 8192 | triton_fixed | 0.567 | 121.163 | 8.02 | 256 | 64 | 32 |  |  |
| up_proj | rowwise_int4 | 2048 | 2048 | 8192 | triton_autotuned | 0.571 | 120.309 | 8.02 | 256 | 64 | 32 | 4 | 3 |
| down_proj | rowwise_int4 | 128 | 8192 | 2048 | torch_fp16 | 0.042 | 101.430 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 128 | 8192 | 2048 | torch_quantized_ref | 0.728 | 5.899 | 8.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 128 | 8192 | 2048 | triton_fixed | 0.264 | 16.280 | 8.00 | 256 | 64 | 32 |  |  |
| down_proj | rowwise_int4 | 128 | 8192 | 2048 | triton_autotuned | 0.132 | 32.483 | 8.00 | 64 | 64 | 64 | 4 | 3 |
| down_proj | rowwise_int4 | 512 | 8192 | 2048 | torch_fp16 | 0.104 | 165.383 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 512 | 8192 | 2048 | torch_quantized_ref | 0.794 | 21.644 | 8.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 512 | 8192 | 2048 | triton_fixed | 0.267 | 64.313 | 8.00 | 256 | 64 | 32 |  |  |
| down_proj | rowwise_int4 | 512 | 8192 | 2048 | triton_autotuned | 0.254 | 67.660 | 8.00 | 256 | 64 | 32 | 4 | 3 |
| down_proj | rowwise_int4 | 2048 | 8192 | 2048 | torch_fp16 | 0.369 | 186.388 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 2048 | 8192 | 2048 | torch_quantized_ref | 1.056 | 65.071 | 8.00 |  |  |  |  |  |
| down_proj | rowwise_int4 | 2048 | 8192 | 2048 | triton_fixed | 0.703 | 97.710 | 8.00 | 256 | 64 | 32 |  |  |
| down_proj | rowwise_int4 | 2048 | 8192 | 2048 | triton_autotuned | 0.689 | 99.745 | 8.00 | 256 | 64 | 32 | 4 | 3 |
| lm_head | rowwise_int4 | 128 | 2048 | 128256 | torch_fp16 | 0.444 | 151.506 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4 | 128 | 2048 | 128256 | torch_quantized_ref | 10.249 | 6.561 | 125.49 |  |  |  |  |  |
| lm_head | rowwise_int4 | 128 | 2048 | 128256 | triton_fixed | 1.017 | 66.110 | 125.49 | 256 | 64 | 32 |  |  |
| lm_head | rowwise_int4 | 128 | 2048 | 128256 | triton_autotuned | 0.789 | 85.200 | 125.49 | 128 | 64 | 32 | 4 | 2 |
| lm_head | rowwise_int4 | 512 | 2048 | 128256 | torch_fp16 | 1.782 | 150.966 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4 | 512 | 2048 | 128256 | torch_quantized_ref | 11.120 | 24.188 | 125.49 |  |  |  |  |  |
| lm_head | rowwise_int4 | 512 | 2048 | 128256 | triton_fixed | 2.193 | 122.643 | 125.49 | 256 | 64 | 32 |  |  |
| lm_head | rowwise_int4 | 512 | 2048 | 128256 | triton_autotuned | 2.201 | 122.179 | 125.49 | 256 | 64 | 32 | 4 | 3 |
| lm_head | rowwise_int4 | 2048 | 2048 | 128256 | torch_fp16 | 6.306 | 170.604 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4 | 2048 | 2048 | 128256 | torch_quantized_ref | 15.012 | 71.668 | 125.49 |  |  |  |  |  |
| lm_head | rowwise_int4 | 2048 | 2048 | 128256 | triton_fixed | 8.564 | 125.636 | 125.49 | 256 | 64 | 32 |  |  |
| lm_head | rowwise_int4 | 2048 | 2048 | 128256 | triton_autotuned | 8.558 | 125.715 | 125.49 | 256 | 64 | 32 | 4 | 3 |
