from pathlib import Path


MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

SKIP_MODULE_NAMES = set()

USE_AUTOTUNED = False
KERNEL_NAME = "rowwise_int4"
BLOCK_M = 256
BLOCK_N = 64
BLOCK_K = 32

PROMPT = "Вкратце, квантизация в int4 - это"
MAX_NEW_TOKENS = 32

OUTPUT_DIR = Path("model_inference/results")
OUTPUT_JSON = OUTPUT_DIR / "llama_quantized_report_m256_n64_k32.json"
