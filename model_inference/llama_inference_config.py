from pathlib import Path


MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
QUANTIZATION_METHOD = "simple"  # доступно: "simple", "gptq"

# Параметры для кернелей "simple"
SKIP_MODULE_NAMES = set()

# общие параметры
KERNEL_NAME = "rowwise_int4"  # доступно "rowwise_int4", "rowwise_int4_asym", "rowwise_int4_gptq"
BLOCK_M = 256
BLOCK_N = 64
BLOCK_K = 32
USE_AUTOTUNED = False

# Параметры для "gptq"
GROUP_SIZE = 128       # columns per scale group
CALIB_N_SAMPLES = 32   # calibration sequences
ACT_ORDER = True       # sort columns by Hessian diagonal

PROMPT = "Вкратце, квантизация в int4 - это"
MAX_NEW_TOKENS = 32

_output_label = "rowwise_int4_gptq" if QUANTIZATION_METHOD == "gptq" else KERNEL_NAME
OUTPUT_DIR = Path("model_inference/results") / _output_label
OUTPUT_JSON = OUTPUT_DIR / "llama_quantized_report_skip_lm_head.json"
