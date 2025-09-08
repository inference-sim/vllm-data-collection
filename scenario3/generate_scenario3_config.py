import numpy as np
import yaml

long_prefill_token_thresholds = [256, 512, 1024, 2048, 4096]
context_length = 8192
max_num_batched_tokens = 2048
block_size = 16
gpu_mem_util = 0.9
modes = ["train", "test"]
deltas = {"train": 20, "test": 25}
num_quads = 4
concentration_params = [1, 1, 1]
max_budget = context_length - 10

def sample_dirichlet_multinomial(concentration_params):
  """
  Generates one sample from a Dirichlet-multinomial distribution.

  Args:
    n: Budget
    concentration_params: etas

  Returns:
    3 integers representing the counts
  """
  n = np.random.uniform(low=context_length//2, high=max_budget)
  p = np.random.dirichlet(concentration_params)
  x = np.random.multinomial(n, p)
  x = x.tolist()
  return x


for mode in modes:
    config = {}
    config["name"] =  f"simulator_{mode}ing_for_scenario_3"
    config["description"] = f"Handcrafted {mode} data for model"
    config["client_template"] = {
            "temperature": 0.0,
            "output_len": 1,
            "seed": 42
        }
    config["warmstart"] = {
        "prompt_count": 50,
        "prompt_len": 128,
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
        for idx in range(num_quads):
            deltas = sample_dirichlet_multinomial(concentration_params)

            quad_info = {
                "deltas": deltas
            }
            
            experiment_specs["data"]["workloads"].append(quad_info)

        config["experiments"].append(experiment_specs)

    output_filename = f"scenario3_config_{mode}.yaml"
    with open(output_filename, 'w') as file:
        yaml.dump(config, file)