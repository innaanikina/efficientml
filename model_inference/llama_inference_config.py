from pathlib import Path


MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

SKIP_MODULE_NAMES = {"lm_head"}

USE_AUTOTUNED = False
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32

PROMPT = "Объясни квантизацию в int4 в одном предложении"
MAX_NEW_TOKENS = 32

OUTPUT_DIR = Path("model_inference/results")
OUTPUT_JSON = OUTPUT_DIR / "llama_quantized_report.json"
