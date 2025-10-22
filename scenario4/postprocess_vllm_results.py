import os
import json
import argparse
import numpy as np

def get_server_side_metrics(model_name, mode, rr, spec, mbnt):
    e2e_server = []
    experiment_metrics = {}
    spec_small = spec.lower()
    total_active_steps = 0
    results_folder = f"../results_new/scenario4/{model_name}/{spec_small}/mbnt_{mbnt}/rr_{rr}/"
    if os.path.isdir(results_folder):
        for dirpath, _, filenames in os.walk(results_folder):
            for filename in filenames:
                if filename == f"detailed_results_{mode}.json":
                    full_path = os.path.join(dirpath, filename)
                    with open (full_path, "r") as f:
                        json_data = json.load(f)
                    experiment_metrics["duration"] = json_data["duration(s)"]
                    experiment_metrics["total_input_tokens"] = json_data["total_input_tokens"]
                    experiment_metrics["total_output_tokens"] = json_data["total_output_tokens"]
                    experiment_metrics["throughput"] = json_data["request_throughput"]
                    # check for saturation
                    if experiment_metrics["throughput"] < SATURATION_PERCENTAGE * rr:
                        print(f"Saturated scenario, rr={rr}, spec={spec}, mbnt={mbnt}, skipping...")
                        return
                    
                    experiment_metrics["completed_requests"] = json_data["completed"]
                    prompts = json_data["prompts"]
                    total_input_tokens = 0
                    total_output_tokens = 0
                    for prompt in prompts:
                        prompt_finished = False
                        server_left_time = 0
                        server_hit_time = 0
                        total_input_tokens += prompt["input_len"]
                        total_output_tokens += prompt["output_len"]
                        if prompt["error"]=="": #check for errors
                            for event in prompt["events"]:
                                if event["event_type"] == "SERVER_LEFT":
                                    server_left_time = event["timestamp"]
                                elif event["event_type"] == "SERVER_HIT":
                                    server_hit_time = event["timestamp"]
                                elif event["event_type"] == "SCHEDULED":
                                    scheduled_step = event["step"]
                                elif event["event_type"] == "FINISHED":
                                    finished_step = event["step"]
                                    prompt_finished = True
                            if prompt_finished:
                                total_active_steps += (finished_step - scheduled_step)
                                e2e_server.append(server_left_time - server_hit_time)
                    if total_input_tokens != experiment_metrics["total_input_tokens"]:
                        print("input token mismatch", results_folder, total_input_tokens, experiment_metrics["total_input_tokens"])
                    if total_output_tokens != experiment_metrics["total_output_tokens"]:
                        print("output token mismatch", results_folder, total_output_tokens, experiment_metrics["total_output_tokens"])
    mean_value = np.mean(e2e_server)
    median_value = np.median(e2e_server)
    p99_value = np.percentile(e2e_server, 99)
    mean_active_steps = total_active_steps/len(e2e_server)
    results_folder = f"results_server_side/{mode}/{model_name}"
    os.makedirs(results_folder, exist_ok=True)
    results_filename = f"vllm_{rr}r_{spec}_{mbnt}.json"
    full_results_filename = os.path.join(results_folder, results_filename)
    full_results = {}
    full_results["model"] = model_name
    full_results["request_rate"] = rr
    full_results["spec"] = spec
    full_results["mbnt"] = mbnt
    full_results["dataset"] = "random"
    full_results["Completed Requests"] = experiment_metrics["completed_requests"]
    full_results["Request Rate(req/s)"] = rr
    full_results["Total Input Tokens"] = experiment_metrics["total_input_tokens"]
    full_results["Total Output Tokens"] = experiment_metrics["total_output_tokens"] 
    full_results["vLLM duration(s)"] = experiment_metrics["duration"]
    full_results["Request throughput (req/s)"] = experiment_metrics["throughput"]
    full_results["Mean E2E(ms)"] = mean_value*1e3
    full_results["Median E2E(ms)"] = median_value*1e3
    full_results["P99 E2E(ms)"] = p99_value*1e3
    full_results["Mean Active Steps"] = mean_active_steps
    with open(full_results_filename, 'w+') as f:
        json.dump(full_results, f, indent=4)

parser = argparse.ArgumentParser(description="Postprocess vllm results to get aggregate metrics/saturation filtering")
parser.add_argument('-m', '--mode', type=str, required=True,
                    help="Specify the mode of operation (e.g., 'train', 'val', 'test').")

args = parser.parse_args()
if args.mode == "train":
    from experiment_configs_constants_train import *
elif args.mode == "val":
    from experiment_configs_constants_val import *
elif args.mode == "test":
    from experiment_configs_constants_test import *

for model in MODELS:
    model_name = model.split("/")[-1].replace(".", "_")

    # get server side final metrics for comparison against sim
    for spec in SPECS:
        for rr in REQUEST_RATES:
            for mbnt in MAX_NUM_BATCHED_TOKENS:
                    get_server_side_metrics(model_name, args.mode, rr, spec, mbnt)

