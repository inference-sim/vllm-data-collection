import argparse
import json
import os
import requests
import subprocess
import time
import yaml
from transformers import AutoTokenizer

def generate_request(prompt, client_config, model):
   payload = {
        "model": model,
        "prompt": prompt, # 1 prompt per request
        "max_tokens": client_config["output_len"],
        "temperature": client_config["temperature"],
        "seed": client_config["seed"]
    }
   
   return payload

def generate_unique_prefix_prompt_pairs(idx, prompt_len, model, extended_len):
   tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
   unique_prefix = f"{prompt_len}-{idx}"
   orig_prompt = unique_prefix
   
   token_id = tokenizer.encode(orig_prompt, add_special_tokens=False)
   orig_prompt += " the" * (prompt_len - len(token_id))
   encoded_prompt = tokenizer.encode(orig_prompt, add_special_tokens=False)
   while len(encoded_prompt) > (prompt_len - 1):
         orig_prompt = orig_prompt[:-4]
         encoded_prompt = tokenizer.encode(orig_prompt, add_special_tokens=False)
   extended_prompt = orig_prompt + " the" * (extended_len - prompt_len)
   encoded_extended_prompt = tokenizer.encode(extended_prompt, add_special_tokens=False)
   while len(encoded_extended_prompt) > (extended_len - 1):
         extended_prompt = extended_prompt[:-4]
         encoded_extended_prompt = tokenizer.encode(extended_prompt, add_special_tokens=False)
   return [orig_prompt, extended_prompt]

def post_request(endpoint, model, prompt, client_config, e2e_logging = False):
    headers = {
        "Content-Type": "application/json"
        }
    payload = generate_request(prompt, client_config, model)
    output = ""
    if e2e_logging:
       start_time = time.time()
    try:
        response = requests.post(endpoint, headers=headers, json=payload, stream = False)
        response.raise_for_status()
        output = json.loads(response.content)
        if e2e_logging:
            e2e = time.time() - start_time
            return e2e, output
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}, {prompt}")
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}, {prompt}")
    return None, output

def main():
    parser = argparse.ArgumentParser(description='Simple vLLM Benchmark Runner')
    parser.add_argument('--mode', help='train/test',  default="train")
    parser.add_argument('--model', help='LLM name',  default="facebook/opt-125m")
    parser.add_argument('--chunk_size', help='Chunk size',  default="256")
    parser.add_argument('--results_folder', help='Result folder in PVC',  default="scenario2")
    args = parser.parse_args()
    config_file = f"scenario2_config_{args.mode}.yaml"

    with open(config_file, "r") as f:
       config = yaml.safe_load(f) # read necessary configs
    warmstart_config = config["warmstart"]
       
    endpoint = "http://localhost:8000/v1/completions"  # endpoint should end with /v1/completions
    
    client_config = config["client"]

    # warm start - discard some requests
    for it in range(warmstart_config["prompt_count"]):
      warmstart_prompts = generate_unique_prefix_prompt_pairs(it, warmstart_config["prompt_len"], args.model, 20)
      for prompt in warmstart_prompts:
         _, res = post_request(endpoint, args.model, prompt, client_config)


    # real requests
    data_config = {}
    for experiment in config["experiments"]:
        if experiment["chunk_size"] == int(args.chunk_size):
            data_config = experiment["data"]
    results = {}
    results["workloads"] = []
    for workload in data_config["workloads"]:
        results_for_m = {"m": workload["m"]}
        results_for_m["input_prompt_pairs"] = []
        results_for_m["e2e_pairs"] = []
        results_for_m["request_id_pairs"] = []
        results_for_m["prompt_len_pairs"] = []
        for idx, pair in enumerate(workload["input_pairs"]):
            prompt_pair = generate_unique_prefix_prompt_pairs(idx, pair[0], args.model, pair[1])
            results_for_m["input_prompt_pairs"].append(prompt_pair)
            e2e1, res1 = post_request(endpoint, args.model, prompt_pair[0], client_config, e2e_logging = True)
            e2e2, res2 = post_request(endpoint, args.model, prompt_pair[1], client_config, e2e_logging = True)
            results_for_m["e2e_pairs"].append([e2e1, e2e2])
            results_for_m["request_id_pairs"].append([res1["id"], res2["id"]])
            results_for_m["prompt_len_pairs"].append([res1["id"], res2["id"]])
        results["workloads"].append(results_for_m)
      
    model_alias = args.model.split("/")[-1].replace(".", "_")
    full_results_path = f"/mnt/scenario2/results/{model_alias}/{args.results_folder}/chunk_size_{args.chunk_size}/results"
    full_spec_path = f"/mnt/scenario2/results/{model_alias}/{args.results_folder}/chunk_size_{args.chunk_size}/spec"
    os.makedirs(full_results_path, exist_ok=True)
    os.makedirs(full_spec_path, exist_ok=True)
    result_filename = f"{full_results_path}/scenario2_output_{args.mode}.json"
    with open(result_filename, 'w', encoding='utf-8') as f:
       json.dump(results, f, indent=4)

    pip_command = ['pip', 'freeze']
    result = subprocess.run(pip_command, capture_output=True, text=True, check=True)

    # Write the captured output to the requirements file
    with open(f"{full_spec_path}/requirements.txt", 'w') as f:
        f.write(result.stdout)

    with open(f"{full_spec_path}/scenario2_config_{args.mode}.yaml", 'w') as f:
       yaml.dump(config, f, sort_keys=False)

    print ("Finished workload experiment")

if __name__=="__main__":
   main()