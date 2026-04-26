| layer | kernel | batch | IN | OUT | method | latency_ms | TFLOP/s | weight_MB | BLOCK_M | BLOCK_N | BLOCK_K | warps | stages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_fp16 | 0.017 | 63.235 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_quantized_ref | 0.167 | 6.439 | 2.01 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_fixed | 0.046 | 23.327 | 2.01 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_autotuned | 0.062 | 17.218 | 2.01 | 64 | 64 | 64 | 4 | 3 |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_fp16 | 0.032 | 135.904 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_quantized_ref | 0.161 | 26.608 | 2.01 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_fixed | 0.109 | 39.507 | 2.01 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_autotuned | 0.062 | 68.842 | 2.01 | 256 | 64 | 32 | 4 | 3 |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_fp16 | 0.081 | 210.983 | 8.00 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_quantized_ref | 0.227 | 75.628 | 2.01 |  |  |  |  |  |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_fixed | 0.442 | 38.878 | 2.01 | 32 | 32 | 32 |  |  |
| q_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_autotuned | 0.178 | 96.491 | 2.01 | 256 | 64 | 32 | 4 | 3 |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_fp16 | 0.018 | 15.261 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_quantized_ref | 0.147 | 1.829 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_fixed | 0.041 | 6.482 | 0.50 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_autotuned | 0.062 | 4.356 | 0.50 | 32 | 32 | 64 | 2 | 3 |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_fp16 | 0.014 | 78.398 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_quantized_ref | 0.137 | 7.846 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_fixed | 0.042 | 25.630 | 0.50 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_autotuned | 0.061 | 17.546 | 0.50 | 64 | 64 | 64 | 4 | 3 |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_fp16 | 0.031 | 136.438 | 2.00 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_quantized_ref | 0.138 | 31.131 | 0.50 |  |  |  |  |  |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_fixed | 0.107 | 40.168 | 0.50 | 32 | 32 | 32 |  |  |
| k_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_autotuned | 0.062 | 69.328 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_fp16 | 0.018 | 15.013 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | torch_quantized_ref | 0.147 | 1.829 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_fixed | 0.042 | 6.394 | 0.50 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4_asym | 128 | 2048 | 512 | triton_autotuned | 0.061 | 4.401 | 0.50 | 32 | 32 | 64 | 2 | 3 |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_fp16 | 0.014 | 78.705 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | torch_quantized_ref | 0.137 | 7.847 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_fixed | 0.042 | 25.769 | 0.50 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4_asym | 512 | 2048 | 512 | triton_autotuned | 0.060 | 17.831 | 0.50 | 64 | 64 | 64 | 4 | 3 |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_fp16 | 0.031 | 136.554 | 2.00 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | torch_quantized_ref | 0.138 | 31.222 | 0.50 |  |  |  |  |  |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_fixed | 0.107 | 39.957 | 0.50 | 32 | 32 | 32 |  |  |
| v_proj | rowwise_int4_asym | 2048 | 2048 | 512 | triton_autotuned | 0.064 | 66.985 | 0.50 | 256 | 64 | 32 | 4 | 3 |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_fp16 | 0.014 | 75.184 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | torch_quantized_ref | 0.146 | 7.352 | 2.01 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_fixed | 0.041 | 26.073 | 2.01 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4_asym | 128 | 2048 | 2048 | triton_autotuned | 0.061 | 17.730 | 2.01 | 64 | 64 | 64 | 4 | 3 |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_fp16 | 0.033 | 131.045 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | torch_quantized_ref | 0.164 | 26.228 | 2.01 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_fixed | 0.109 | 39.302 | 2.01 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4_asym | 512 | 2048 | 2048 | triton_autotuned | 0.067 | 64.391 | 2.01 | 256 | 64 | 32 | 4 | 3 |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_fp16 | 0.083 | 207.421 | 8.00 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | torch_quantized_ref | 0.231 | 74.314 | 2.01 |  |  |  |  |  |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_fixed | 0.439 | 39.091 | 2.01 | 32 | 32 | 32 |  |  |
| o_proj | rowwise_int4_asym | 2048 | 2048 | 2048 | triton_autotuned | 0.206 | 83.451 | 2.01 | 256 | 64 | 32 | 4 | 3 |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_fp16 | 0.038 | 113.187 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_quantized_ref | 0.559 | 7.684 | 8.05 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_fixed | 0.115 | 37.220 | 8.05 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_autotuned | 0.073 | 58.824 | 8.05 | 64 | 64 | 64 | 4 | 3 |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_fp16 | 0.110 | 156.013 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_quantized_ref | 0.642 | 26.740 | 8.05 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_fixed | 0.431 | 39.853 | 8.05 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_autotuned | 0.177 | 96.806 | 8.05 | 256 | 64 | 32 | 4 | 3 |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_fp16 | 0.352 | 195.359 | 32.00 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_quantized_ref | 0.879 | 78.141 | 8.05 |  |  |  |  |  |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_fixed | 1.729 | 39.754 | 8.05 | 32 | 32 | 32 |  |  |
| gate_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_autotuned | 0.532 | 129.142 | 8.05 | 256 | 128 | 32 | 8 | 3 |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_fp16 | 0.038 | 112.933 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | torch_quantized_ref | 0.563 | 7.633 | 8.05 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_fixed | 0.116 | 36.948 | 8.05 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4_asym | 128 | 2048 | 8192 | triton_autotuned | 0.080 | 53.851 | 8.05 | 64 | 64 | 64 | 4 | 3 |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_fp16 | 0.113 | 152.497 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | torch_quantized_ref | 0.650 | 26.449 | 8.05 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_fixed | 0.436 | 39.359 | 8.05 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4_asym | 512 | 2048 | 8192 | triton_autotuned | 0.198 | 86.652 | 8.05 | 256 | 64 | 32 | 4 | 3 |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_fp16 | 0.356 | 193.172 | 32.00 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | torch_quantized_ref | 0.894 | 76.893 | 8.05 |  |  |  |  |  |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_fixed | 1.729 | 39.749 | 8.05 | 32 | 32 | 32 |  |  |
| up_proj | rowwise_int4_asym | 2048 | 2048 | 8192 | triton_autotuned | 0.542 | 126.734 | 8.05 | 256 | 128 | 32 | 8 | 3 |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | torch_fp16 | 0.042 | 102.992 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | torch_quantized_ref | 0.558 | 7.699 | 8.01 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | triton_fixed | 0.149 | 28.807 | 8.01 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4_asym | 128 | 8192 | 2048 | triton_autotuned | 0.113 | 38.000 | 8.01 | 64 | 64 | 64 | 4 | 3 |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | torch_fp16 | 0.105 | 164.394 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | torch_quantized_ref | 0.636 | 26.993 | 8.01 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | triton_fixed | 0.457 | 37.586 | 8.01 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4_asym | 512 | 8192 | 2048 | triton_autotuned | 0.233 | 73.800 | 8.01 | 256 | 64 | 32 | 4 | 3 |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | torch_fp16 | 0.382 | 180.100 | 32.00 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | torch_quantized_ref | 0.908 | 75.693 | 8.01 |  |  |  |  |  |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | triton_fixed | 1.912 | 35.945 | 8.01 | 32 | 32 | 32 |  |  |
| down_proj | rowwise_int4_asym | 2048 | 8192 | 2048 | triton_autotuned | 0.748 | 91.823 | 8.01 | 256 | 128 | 32 | 8 | 3 |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | torch_fp16 | 0.424 | 158.460 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | torch_quantized_ref | 7.897 | 8.515 | 125.98 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | triton_fixed | 1.734 | 38.783 | 125.98 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4_asym | 128 | 2048 | 128256 | triton_autotuned | 0.775 | 86.740 | 125.98 | 128 | 64 | 32 | 4 | 2 |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | torch_fp16 | 1.787 | 150.506 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | torch_quantized_ref | 8.873 | 30.315 | 125.98 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | triton_fixed | 6.762 | 39.775 | 125.98 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4_asym | 512 | 2048 | 128256 | triton_autotuned | 2.085 | 129.001 | 125.98 | 256 | 128 | 32 | 8 | 3 |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | torch_fp16 | 6.300 | 170.783 | 501.00 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | torch_quantized_ref | 12.905 | 83.370 | 125.98 |  |  |  |  |  |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | triton_fixed | 27.013 | 39.829 | 125.98 | 32 | 32 | 32 |  |  |
| lm_head | rowwise_int4_asym | 2048 | 2048 | 128256 | triton_autotuned | 8.204 | 131.148 | 125.98 | 256 | 128 | 32 | 8 | 3 |
