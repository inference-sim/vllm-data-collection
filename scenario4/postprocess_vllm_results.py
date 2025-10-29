import os
import json
import argparse
import numpy as np
import pandas as pd

import re

def extract_total_kv_blocks(model_name):
    """
    Finds the avg value of "Total KV blocks" for input to the simulator during testing. 
    Extracts the number following 'num_gpu_blocks is:' from vllm's server log file,
    and averages it over all experiments for a given model.

    Assumption: Given a model, the total KV blocks is nearly constant across experiments.
    Usage: ONLY use this function only for BLIS's "test" mode.

    Parameters:
        log_file_path (str): Path to the log file.

    Returns:
        int: Total KV Blocks for input to simulator (testing phase)
    """
    total_kv_blocks = 0
    count_exp = 0
    for spec in SPECS:
        spec_small = spec.lower()
        for mbnt in MAX_NUM_BATCHED_TOKENS:
            for rr in REQUEST_RATES[spec]:
                rr = f"{float(rr):.2f}"
                results_folder = f"../results_new/scenario4/{model_name}/test/{spec_small}/mbnt_{mbnt}/rr_{rr}"
                if os.path.isdir(results_folder):
                    for dirpath, _, filenames in os.walk(results_folder):
                        for filename in filenames:
                            if filename == f"scenario4_server_test.log":
                                log_file_path = os.path.join(dirpath, filename)
                                try:
                                    with open(log_file_path, 'r') as file:
                                        for line in file:
                                            match = re.search(r'num_gpu_blocks is:\s*(\d+)', line)
                                            if match:
                                                total_kv_blocks += int(match.group(1))
                                                count_exp += 1
                                except FileNotFoundError:
                                    print(f"File not found: {log_file_path}")
    return total_kv_blocks//count_exp

def saturation_RPS(request_throughput, request_arrival_rate):
    """
    Determines if avg request throughput (req/s) is less than 90% of request arrival rate.
    Parameters:
        request_throughput (float): Avg request output throughput (req/s)
        request_arrival_rate (float): Poisson distribution mean request arrival rate (req/s)

    Returns:
        bool: Whether the experiment under consideration is saturated or not
    """
    if request_throughput < SATURATION_PERCENTAGE * float(request_arrival_rate):
        return True
    return False

def saturation_latency(e2e_latency_list, moving_avg_window = 100, threshold=0.02):
    """
    Determines if the request-level e2e latency values have an inflection point,
    after which latencies grow bigger.

    Parameters:
        e2e_latency_list (list[float]): List of request-level e2e latencies

    Returns:
        bool: Whether the experiment under consideration is saturated or not
    """
    moving_avg = pd.Series(e2e_latency_list).rolling(window=moving_avg_window).mean()
    grad = np.gradient(moving_avg)
    sat_idx = np.where(pd.Series(grad).rolling(window=moving_avg_window).mean() > threshold)
    if sat_idx and len(sat_idx[0]) > 0:
        sat_idx = sat_idx[0][0]
        return True
    return False

def process_server_side_metrics(model_name, mode, rr, spec, mbnt):
    """
    Return server side metrics ONLY for non-saturated regimes. 
    Write non-saturated processed data into files that can be compared against the simulator.
    """
    e2e_server = []
    experiment_metrics = {}
    spec_small = spec.lower()
    total_active_steps = 0
    rr = f"{float(rr):.2f}"
    results_folder = f"../results_new/scenario4/{model_name}/{mode}/{spec_small}/mbnt_{mbnt}/rr_{rr}/"
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
    full_results["e2e latency list"] = e2e_server
    full_results["Mean E2E(ms)"] = mean_value*1e3
    full_results["Median E2E(ms)"] = median_value*1e3
    full_results["P99 E2E(ms)"] = p99_value*1e3
    full_results["Mean Active Steps"] = mean_active_steps

    return full_results

def save_unsaturated_results(model_name, mode, rr, spec, mbnt, full_results):
    # determine if saturated or not

    # Option A: Saturation in terms of overall RPS
    saturated = saturation_RPS(full_results["Request throughput (req/s)"], full_results["Request Rate(req/s)"])
    # Option B: Saturation in terms of e2e latency evolution
    # saturated = saturation_latency(full_results["e2e latency list"])

    if saturated:
        print(f"Saturated scenario, rr={rr}, spec={spec}, mbnt={mbnt}, skipping...")
        return
    # if not saturated, save processed results
    results_folder = f"results_server_side/{model_name}/{mode}"
    os.makedirs(results_folder, exist_ok=True)
    rr = f"{float(rr):.2f}"
    results_filename = f"vllm_{rr}r_{spec}_{mbnt}.json"
    full_results_filename = os.path.join(results_folder, results_filename)
    with open(full_results_filename, 'w+') as f:
        json.dump(full_results, f, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Postprocess vllm results to get aggregate metrics/saturation filtering")
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help="Specify the mode of operation (e.g., 'train', 'test').")

    args = parser.parse_args()
    if args.mode == "train":
        from experiment_configs_constants_train import *
    elif args.mode == "test":
        from experiment_configs_constants_test import *

    model_name = MODEL.split("/")[-1].replace(".", "_")

    # get server side final metrics for comparison against sim
    for spec in SPECS:
        for rr in REQUEST_RATES[spec]:
            for mbnt in MAX_NUM_BATCHED_TOKENS:
                    print(f"scenario, rr={rr}, spec={spec}, mbnt={mbnt}")
                    processed_results = process_server_side_metrics(model_name, args.mode, rr, spec, mbnt)
                    save_unsaturated_results(model_name, args.mode, rr, spec, mbnt, processed_results)
    # get Total KV Blocks from logs as input to simulator for test mode
    if args.mode == "test":
        print(f"Total KV Blocks: {extract_total_kv_blocks(model_name)}")

