import yaml

long_prefill_token_thresholds = [256, 512, 1024, 2048, 4096]
context_length = 8192
max_num_batched_tokens = 2048
block_size = 16
gpu_mem_util = 0.9
modes = ["train", "test"]
deltas = {"train": 20, "test": 25}

for mode in modes:
    config = {}
    config["name"] =  f"simulator_{mode}ing_for_scenario_2"
    config["description"] = f"Handcrafted {mode} data for model"
    config["warmstart"] = {
        "prompt_count": 50,
        "prompt_len": 128
    }
    config["client"] = {
        "temperature": 0.0,
        "output_len": 1,
        "seed": 42
    }
    config["experiments"] = []
    for idx, lptt in enumerate(long_prefill_token_thresholds):
        experiment_specs = {}
        experiment_specs["name"] = f"benchmark-{idx+1}"
        experiment_specs["chunk_size"] = lptt
        experiment_specs["vllm"] = {
            "gpu_memory_utilization": gpu_mem_util,
            "block_size": block_size,
            "max_model_len": context_length,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": 1,
            "seed": 42,
            "gpu_type": "NVIDIA-H100-80GB-HBM3",
            "gpu_memory_min": 50000,
            "long_prefill_token_threshold": lptt
        }

        experiment_specs["data"] = {"workloads": []}
        for m in range(context_length//lptt):
            workload = {}
            workload["m"] = m
            workload["input_pairs"] = []
            J = deltas[mode]
            input_len = m*lptt
            while input_len + J < context_length and (input_len + J - m*lptt) < lptt:
                input_len += J
                workload["input_pairs"].append([m*lptt, input_len])
            experiment_specs["data"]["workloads"].append(workload)

        config["experiments"].append(experiment_specs)

    output_filename = f"scenario2_config_{mode}.yaml"
    with open(output_filename, 'w') as file:
        yaml.dump(config, file)