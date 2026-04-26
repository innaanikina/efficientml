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

# Simple кернелы
SKIP_MODULE_NAMES = llama_inference_config.SKIP_MODULE_NAMES
USE_AUTOTUNED = llama_inference_config.USE_AUTOTUNED
KERNEL_NAME = llama_inference_config.KERNEL_NAME
BLOCK_M = llama_inference_config.BLOCK_M
BLOCK_N = llama_inference_config.BLOCK_N
BLOCK_K = llama_inference_config.BLOCK_K

# GPTQ 
QUANTIZATION_METHOD = llama_inference_config.QUANTIZATION_METHOD
GROUP_SIZE = llama_inference_config.GROUP_SIZE
CALIB_N_SAMPLES = llama_inference_config.CALIB_N_SAMPLES
ACT_ORDER = llama_inference_config.ACT_ORDER

OUTPUT_DIR = Path("evaluation/results") / llama_inference_config._output_label
OUTPUT_JSON = OUTPUT_DIR / "wikitext2_eval.json"
OUTPUT_MD = OUTPUT_DIR / "wikitext2_eval.md"
