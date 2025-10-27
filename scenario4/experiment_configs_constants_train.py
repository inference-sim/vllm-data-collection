# vllm config
MAX_MODEL_LEN = 32768 # do not send with vllm serve
MAX_NUM_BATCHED_TOKENS = [1024, 2048, 4096]
GPU_MEM_UTIL = 0.9

# cluster config
GPU_TYPE = "NVIDIA-H100-80GB-HBM3"
GPU_MEMORY_MIN = 50000
NAMESPACE = "blis"

# workload config
NUM_PROMPTS = 2000
REQUEST_RATES = {
    "LL": [2, 4, 8, 12, 15], # max throughput is 21
    "LH": [1.0, 1.5, 2.0, 2.5, 3.0], # max throughput is 5.71
    "HL": [1.0, 1.5, 2.0, 2.5, 3.0], # max throughput is 5.99
    "HH": [0.25, 0.5, 0.75, 1.0] # max throughput is 2.49
}
DATASET_NAME = "random" # random/sharegpt
MODE = "train"
SEED = 42 # change for val
MODEL = "Qwen/Qwen2.5-7B"
SPECS = ["LL", "LH", "HL", "HH"]
HH_SPECS = {'TYPE': 'HH', 'INPUT_LEN_MEAN': 4000, 'OUTPUT_LEN_MEAN': 4000, "NUM_PREFIXES": 5, "PREFIX_HIT_RATIO_MEAN": 0.25}
HL_SPECS = {'TYPE': 'HL', 'INPUT_LEN_MEAN': 4000, 'OUTPUT_LEN_MEAN': 512, "NUM_PREFIXES": 5, "PREFIX_HIT_RATIO_MEAN": 0.25}
LH_SPECS = {'TYPE': 'LH', 'INPUT_LEN_MEAN': 512, 'OUTPUT_LEN_MEAN': 4000, "NUM_PREFIXES": 5, "PREFIX_HIT_RATIO_MEAN": 0.25}
LL_SPECS = {'TYPE': 'LL', 'INPUT_LEN_MEAN': 512, 'OUTPUT_LEN_MEAN': 512, "NUM_PREFIXES": 5, "PREFIX_HIT_RATIO_MEAN": 0.25}

# Saturation config
SATURATION_PERCENTAGE = 0.9
