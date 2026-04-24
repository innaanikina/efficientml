from pathlib import Path

LLAMA_3_2_1B_LINEAR_LAYERS = [
    ("q_proj", 2048, 2048, 16),
    ("k_proj", 2048, 512, 16),
    ("v_proj", 2048, 512, 16),
    ("o_proj", 2048, 2048, 16),
    ("gate_proj", 2048, 8192, 16),
    ("up_proj", 2048, 8192, 16),
    ("down_proj", 8192, 2048, 16),
    ("lm_head", 2048, 128256, 1),
]

BATCH_SIZES = [128, 512, 2048]

SKIP_LM_HEAD = False
SKIP_AUTOTUNED = False

OUTPUT = Path("throughput_benchmark/results/matmul_benchmark.csv")

REPEAT = 100
WARMUP = 50
SEED = 42

BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32
