import argparse
import json
import os
import requests
import subprocess
import time
import yaml

def generate_request(prompt, client_config, model):
   payload = {
        "model": model,
        "prompt": prompt, # 1 prompt per request
        "max_tokens": client_config["output_len"],
        "temperature": client_config["temperature"],
        "seed": client_config["seed"]
    }
   
   return payload

def post_request(endpoint, model, prompt, client_config, e2e_logging = False):
    headers = {
        "Content-Type": "application/json"
        }
    payload = generate_request(prompt, client_config, model)
    if e2e_logging:
       start_time = time.time()
    response = requests.post(endpoint, headers=headers, json=payload, stream = False)
    output = json.loads(response.content)
    if e2e_logging:
      e2e = time.time() - start_time
      return e2e, output
    return None, output

def get_network_latency(models_endpoint):
   attempts = 50 
   all_latencies = []

   for i in range(attempts):
        try:
            start_time = time.time()
            response = requests.get(models_endpoint, timeout=5)
            all_latencies.append(time.time() - start_time)
        except requests.exceptions.RequestException:
            break
   return sum(all_latencies)/attempts

def main():
    parser = argparse.ArgumentParser(description='Simple vLLM Benchmark Runner')
    parser.add_argument('--mode', help='train/test',  default="train")
    parser.add_argument('--model', help='LLM name',  default="facebook/opt-125m")
    parser.add_argument('--results_folder', help='Result folder in PVC',  default="scenario1")
    args = parser.parse_args()
    config_file = f"scenario1_config_{args.mode}.yaml"

    with open(config_file, "r") as f:
       config = yaml.safe_load(f) # read necessary configs and seed files
    data_config = config["data"]
       
    endpoint = "http://localhost:8000/v1/completions"  # endpoint should end with /v1/completions
    models_endpoint = "http://localhost:8000/v1/models"

    avg_network_latency = get_network_latency(models_endpoint)

    # load handcrafted prompts
    experiment_prompts = []
    with open(f"prompts_unique_{args.mode}.txt", "r") as f:
       experiment_prompts = [line.strip() for line in f if line != '\n']
    
    client_config = config["client"]

    # warm start - discard some requests
    warmstart_prompts = experiment_prompts[:data_config["workload"]["num_exps"]]
    for prompt in warmstart_prompts:
       _, res = post_request(endpoint, args.model, prompt, client_config)

    # real requests
    train_prompts = experiment_prompts[data_config["workload"]["num_exps"]:]
    results = {}
    results["input_prompts"] = train_prompts
    results["e2es"] = []
    results["e2e - network_latency"] = []
    results["request_ids"] = []
    results["prompt_lens"] = []
    results["block_size"] = []
    for prompt in train_prompts:
      e2e, res = post_request(endpoint, args.model, prompt, client_config, e2e_logging = True)
      results["e2es"].append(e2e)
      results["e2e - network_latency"].append(e2e - avg_network_latency)
      results["request_ids"].append(res["id"])
      results["prompt_lens"].append(res["usage"]["prompt_tokens"])
      results["block_size"].append(config["vllm"]["block_size"])
      
    full_results_path = f"/mnt/{args.results_folder}/results"
    full_spec_path = f"/mnt/{args.results_folder}/spec"
    os.makedirs(full_results_path, exist_ok=True)
    os.makedirs(full_spec_path, exist_ok=True)
    result_filename = f"{full_results_path}/scenario1_output_{args.mode}.json"
    with open(result_filename, 'w', encoding='utf-8') as f:
       json.dump(results, f, indent=4)

    pip_command = ['pip', 'freeze']
    result = subprocess.run(pip_command, capture_output=True, text=True, check=True)

    # Write the captured output to the requirements file
    with open(f"{full_spec_path}/requirements.txt", 'w') as f:
        f.write(result.stdout)

    with open(f"{full_spec_path}/scenario1_config_{args.mode}.yaml", 'w') as f:
       yaml.dump(config, f, sort_keys=False)

    print ("Finished workload experiment")

if __name__=="__main__":
   main()