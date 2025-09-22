CONTEXT_LENGTH = 7000 # for Qwen32B to fit
MAX_NUM_BATCHED_TOKENS = 4096
BLOCK_SIZE = 16
GPU_MEM_UTIL = 0.9
MODES = ["train", "test"]
MAX_NUM_SEQS = 4096
SEED = 42
GPU_TYPE = "NVIDIA-H100-80GB-HBM3"
GPU_MEMORY_MIN = 50000
NAMESPACE = "blis"
CHUNK_SIZES = [1024, 4096]
NUM_PROMPTS = 2000
REQUEST_RATES = [5, 50]
DATASET_PATH = "/mnt/data/ShareGPT_V3_unfiltered_cleaned_split.json"
MODELS = ["Qwen/Qwen3-32B"]
# MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "google/gemma-7b", "mistralai/Mistral-7B-Instruct-v0.1", "meta-llama/Llama-3.1-8B", "ibm-granite/granite-3.3-8b-instruct", "Qwen/Qwen3-14B", "mistralai/Mistral-Small-24B-Instruct-2501", "Qwen/Qwen3-32B"]
SPECS = ["LL", "LH", "HL", "HH"]
LL_SPECS = {
    "TYPE": "LL",
    "INPUT_LEN_MIN": 2,
    "INPUT_LEN_MAX": 100,
    "OUTPUT_LEN_MIN": 1,
    "OUTPUT_LEN_MAX": 100
}
LH_SPECS = {
    "TYPE": "LH",
    "INPUT_LEN_MIN": 2,
    "INPUT_LEN_MAX": 100,
    "OUTPUT_LEN_MIN": 100,
    "OUTPUT_LEN_MAX": 2000
}
HL_SPECS = {
    "TYPE": "HL",
    "INPUT_LEN_MIN": 100,
    "INPUT_LEN_MAX": 4000,
    "OUTPUT_LEN_MIN": 1,
    "OUTPUT_LEN_MAX": 100
}
HH_SPECS = {
    "TYPE": "HH",
    "INPUT_LEN_MIN": 100,
    "INPUT_LEN_MAX": 4000,
    "OUTPUT_LEN_MIN": 100,
    "OUTPUT_LEN_MAX": 2000
}