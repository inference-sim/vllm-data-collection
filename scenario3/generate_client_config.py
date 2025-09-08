import argparse
import numpy as np
import yaml

from experiment_configs_constants import *

def sample_dirichlet_multinomial(context_length_cap, concentration_params):
  """
  Generates one sample from a Dirichlet-multinomial distribution.

  Args:
    n: Budget
    concentration_params: etas

  Returns:
    3 integers representing the counts
  """
  max_budget = context_length_cap - 10
  n = np.random.uniform(low=context_length_cap//2, high=max_budget)
  p = np.random.dirichlet(concentration_params)
  x = np.random.multinomial(n, p)
  x = x.tolist()
  return x

def generate_config(context_length_cap):
    for mode in MODES:
        config = {}
        config["name"] =  f"{mode}_client_config_for_scenario_3"
        config["description"] = f"Handcrafted {mode} client params"
        config["client_template"] = {
                "temperature": 0.0,
                "output_len": 1,
                "seed": SEED
            }
        config["warmstart"] = {
            "prompt_count": WARMSTART_PROMPT_COUNT,
            "prompt_len": WARMSTART_PROMPT_LEN,
        }
        
        config["experiments"] = []
        for idx, lptt in enumerate(CHUNK_SIZES):
            experiment_specs = {}
            experiment_specs["name"] = f"benchmark-{idx+1}"
            experiment_specs["chunk_size"] = lptt
            experiment_specs["data"] = {"workloads": []}
            for idx in range(NUM_QUADS):
                deltas = sample_dirichlet_multinomial(context_length_cap, CONCENTRATION_PARAMS)

                quad_info = {
                    "deltas": deltas
                }
                
                experiment_specs["data"]["workloads"].append(quad_info)

            config["experiments"].append(experiment_specs)

        output_filename = f"scenario3_client_config_{mode}.yaml"
        with open(output_filename, 'w') as file:
            yaml.dump(config, file)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Simple vLLM Benchmark Runner')
    parser.add_argument('--model', help='LLM name',  default="facebook/opt-125m")
    args = parser.parse_args()
    context_length_cap = CONTEXT_LENGTH
    small_models = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]
    if args.model in small_models:
        context_length_cap = 2048
    generate_config(context_length_cap)
    