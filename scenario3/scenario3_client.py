import argparse
import copy
import json
import os
import requests
import subprocess
import time
import yaml
from transformers import AutoTokenizer

def generate_request(prompt, client_config, model):
   # generate request payload with prompt, model and config params
   payload = {
        "model": model,
        "prompt": prompt, # 1 prompt per request
        "max_tokens": client_config["output_len"],
        "temperature": client_config["temperature"],
        "seed": client_config["seed"]
    }
   
   return payload

def generate_prompt_segment(prompt_len, model, seed_unique):
   tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
   prompt_segment = ""
   
   num_tokens_in_seed = len(tokenizer.encode(seed_unique, add_special_tokens=False))
   prompt_segment += seed_unique * (prompt_len//num_tokens_in_seed + 1)
   encoded_prompt = tokenizer.encode(prompt_segment, add_special_tokens=False)
   while len(encoded_prompt) > prompt_len:
         prompt_segment = prompt_segment[:-1]
         encoded_prompt = tokenizer.encode(prompt_segment, add_special_tokens=False)
   print (len(encoded_prompt))
   return prompt_segment

def post_request(endpoint, model, prompt, client_config, e2e_logging = False):
    # Posts a single request to the vllm server endpoint
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
    client_config_template = config["client_template"]
       
    endpoint = "http://localhost:8000/v1/completions"  # endpoint should end with /v1/completions
    
    # warm start - discard some requests
    for it in range(warmstart_config["prompt_count"]):
      warmstart_prompt = generate_prompt_segment(warmstart_config["prompt_len"], args.model, "*w")
      _, res = post_request(endpoint, args.model, warmstart_prompt, client_config_template)


    # real requests
    data_config = {}
    for experiment in config["experiments"]:
        if experiment["chunk_size"] == int(args.chunk_size):
            data_config = experiment["data"]
    results = {}
    results["workloads"] = []
    for idx, workload in enumerate(data_config["workloads"]):
        results_for_quad = {}
        results_for_quad["input_prompt_quads"] = []
        results_for_quad["e2e_quads"] = []
        results_for_quad["request_id_quads"] = []
        results_for_quad["prompt_len_quads"] = []
        deltas = workload["deltas"]
        prefix = generate_prompt_segment(4, args.model, f"{idx}-")
        segment1 = generate_prompt_segment(deltas[0], args.model, " 1")
        segment2 = generate_prompt_segment(1 + deltas[1],  args.model, " 2")
        segment3 = generate_prompt_segment(1 + deltas[1],  args.model, " 3")
        prompt1 = prefix + segment1
        special_models = ["mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b", "meta-llama/Llama-3.1-8B", "mistralai/Mistral-Small-24B-Instruct-2501"]
        if args.model in special_models:
            prompt1 = prompt1[:-1]
        prompts = [prompt1, prompt1 + segment2, prompt1, prompt1 + segment3]
        for idx, prompt in enumerate(prompts):
            client_config = copy.deepcopy(client_config_template)
            if idx <= 1:
                client_config["output_len"] = 1
            else:
                client_config["output_len"] = 1 + deltas[2]
            e2e, res = post_request(endpoint, args.model, prompt, client_config, e2e_logging = True)
            results_for_quad["e2e_quads"].append(e2e)
            results_for_quad["request_id_quads"].append(res["id"])
            results_for_quad["prompt_len_quads"].append(res["usage"]["prompt_tokens"])
        results["workloads"].append(results_for_quad)
      
    model_alias = args.model.split("/")[-1].replace(".", "_")
    full_results_path = f"/mnt/scenario3/results/{model_alias}/{args.results_folder}/chunk_size_{args.chunk_size}/results"
    full_spec_path = f"/mnt/scenario3/results/{model_alias}/{args.results_folder}/chunk_size_{args.chunk_size}/spec"
    os.makedirs(full_results_path, exist_ok=True)
    os.makedirs(full_spec_path, exist_ok=True)
    result_filename = f"{full_results_path}/scenario3_output_{args.mode}.json"
    with open(result_filename, 'w', encoding='utf-8') as f:
       json.dump(results, f, indent=4)

    pip_command = ['pip', 'freeze']
    result = subprocess.run(pip_command, capture_output=True, text=True, check=True)

    # Write the captured output to the requirements file
    with open(f"{full_spec_path}/requirements.txt", 'w') as f:
        f.write(result.stdout)

    with open(f"{full_spec_path}/scenario3_config_{args.mode}.yaml", 'w') as f:
       yaml.dump(config, f, sort_keys=False)

    print ("Finished workload experiment")

if __name__=="__main__":
#    main()
    special_models = ["ibm-granite/granite-3.3-8b-instruct", "Qwen/Qwen3-14B", "mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b", "meta-llama/Llama-3.1-8B", "mistralai/Mistral-Small-24B-Instruct-2501"]
    for model in special_models:
        print ("#############",model,"###############")
        for idx in [1, 5, 10, 50, 100, 200, 500]:
            prefix = generate_prompt_segment(4, model, f"{idx}-")
            print (prefix)
            segment1 = generate_prompt_segment(25, model, " 1")
            segment2 = generate_prompt_segment(1 + 20,  model, " 2")
            segment3 = generate_prompt_segment(1 + 20,  model, " 3")
            warmstart_prompt = generate_prompt_segment(128, model, "*w")