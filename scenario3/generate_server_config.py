import numpy as np
import yaml

from experiment_configs_constants import *

for mode in MODES:
    config = {}
    config["name"] =  f"{mode}_server_config_for_scenario_3"
    config["description"] = f"Handcrafted {mode} server params"
    
    config["experiments"] = []
    for idx, lptt in enumerate(CHUNK_SIZES):
        experiment_specs = {}
        experiment_specs["name"] = f"benchmark-{idx+1}"
        experiment_specs["chunk_size"] = lptt
        experiment_specs["vllm"] = {
            "gpu_memory_utilization": GPU_MEM_UTIL,
            "block_size": BLOCK_SIZE,
            "max_model_len": CONTEXT_LENGTH,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": MAX_NUM_SEQS,
            "seed": SEED,
            "gpu_type": GPU_TYPE,
            "gpu_memory_min": GPU_MEMORY_MIN,
            "long_prefill_token_threshold": lptt
        }

        config["experiments"].append(experiment_specs)

    output_filename = f"scenario3_server_config_{mode}.yaml"
    with open(output_filename, 'w') as file:
        yaml.dump(config, file)