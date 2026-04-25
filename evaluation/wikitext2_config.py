from pathlib import Path

from model_inference import llama_inference_config


MODEL_ID = llama_inference_config.MODEL_ID

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"

MAX_EVAL_TOKENS = 8192
SEQUENCE_LENGTH = 512

PROMPT = llama_inference_config.PROMPT
MAX_NEW_TOKENS = llama_inference_config.MAX_NEW_TOKENS

SKIP_MODULE_NAMES = llama_inference_config.SKIP_MODULE_NAMES
USE_AUTOTUNED = llama_inference_config.USE_AUTOTUNED
KERNEL_NAME = llama_inference_config.KERNEL_NAME
BLOCK_M = llama_inference_config.BLOCK_M
BLOCK_N = llama_inference_config.BLOCK_N
BLOCK_K = llama_inference_config.BLOCK_K

OUTPUT_DIR = Path("evaluation/results") / KERNEL_NAME
OUTPUT_JSON = OUTPUT_DIR / "wikitext2_eval.json"
OUTPUT_MD = OUTPUT_DIR / "wikitext2_eval.md"
