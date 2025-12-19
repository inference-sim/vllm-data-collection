import argparse
import json
import os
import re
import sys
import yaml
import pandas as pd

from postprocessing_utils import BLIS_REQGEN_CONFIG_FOLDER, BLIS_TRAINING_FILEPATH, BLIS_TESTING_FILEPATH, SWEEP_INFO_FILENAME
from postprocessing_utils import read_traces_jsonl, get_server_side_metrics_from_traces

def extract_total_kv_blocks(log_file_path):
    """
    Finds the value of "Total KV blocks" for input to the simulator during testing. 
    Extracts the number following 'num_gpu_blocks is:' from vllm's server log file for BLIS to use.

    Parameters:
        log_file_path (str): Path to the log file.

    Returns:
        int: Total KV Blocks for input to simulator (testing phase)
    """
    total_kv_blocks = 0
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                match = re.search(r'num_gpu_blocks is:\s*(\d+)', line)
                if match:
                    total_kv_blocks += int(match.group(1))
    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
    return total_kv_blocks


def construct_BLIS_reqgenconfig(guidellm_profile, rps):
    """
    Given GuideLLM profile and request rates,
    construct BLIS-style request gen-config file.
    """
    blis_reqgen_config = {
        "format": "GuideLLM",
        "seed": guidellm_profile["random-seed"], 
        "rate": []
    }
    blis_reqgen_config["rate"] = {
        "arrival-type": "Constant",
        "rate": rps,
        "max-requests": guidellm_profile["max-requests"]
    }
    blis_reqgen_config["data"] = guidellm_profile["data"]
    return blis_reqgen_config

def get_metrics_per_benchmark(benchmark_df, rps, guidellm_profile, vllm_config):
    """
    Get BLIS-style benchmark metrics in milliseconds.
    """
    benchmark_metrics = {"rps": rps}
    # server-side values for cost fn
    benchmark_metrics["e2e_mean_ms"] = benchmark_df["e2e_latency"].mean() * 1e3
    benchmark_metrics["e2e_p90_ms"] = benchmark_df["e2e_latency"].quantile(0.9) * 1e3
    benchmark_metrics["ttft_mean_ms"] = benchmark_df["ttft"].mean() * 1e3
    benchmark_metrics["ttft_p90_ms"] = benchmark_df["ttft"].quantile(0.9) * 1e3
    benchmark_metrics["itl_mean_ms"] = (benchmark_df["prefill_time"] + benchmark_df["decode_time"]).sum()/benchmark_df["output_tokens"].sum() * 1e3
    # server-side values for bounds, and alpha regression
    benchmark_metrics["all_processing_times(s)"] = (benchmark_df["e2e_latency"] - (benchmark_df["queued_time"] + benchmark_df["prefill_time"] + benchmark_df["decode_time"])).tolist()
    benchmark_metrics["all_input_lens"] = benchmark_df["input_tokens"].tolist()
    benchmark_metrics["all_output_lens"] = benchmark_df["output_tokens"].tolist()
    # metrics needed for heuristic beta bounds
    benchmark_metrics["sum_prefill_time(s)"] = benchmark_df["prefill_time"].sum()
    benchmark_metrics["sum_decode_time(s)"] = benchmark_df["decode_time"].sum()
    benchmark_metrics["sum_inference_time(s)"] = (benchmark_df["prefill_time"] + benchmark_df["decode_time"]).sum()
    # assume all requests have the exact same number of prefix tokens
    benchmark_metrics["sum_prefill_tokens"] = int((benchmark_df["input_tokens"] - guidellm_profile["data"]["prefix_tokens"]).sum())
    benchmark_metrics["sum_output_tokens"] = int(benchmark_df["output_tokens"].sum())
    benchmark_metrics["mean_output_tokens"] = int(benchmark_df["output_tokens"].mean())
    chunk_size = vllm_config["max_num_batched_tokens"]
    benchmark_metrics["sum_steps"] = int(((benchmark_df["input_tokens"] - guidellm_profile["data"]["prefix_tokens"])/chunk_size + benchmark_df["output_tokens"]).sum())
    return benchmark_metrics

def get_heuristic_bounds(heuristic_aggs):
    # all bounds should be in ticks (microseconds)
    beta0_bound = heuristic_aggs["sum_inference_time(s)"] / heuristic_aggs["sum_steps"] * 1e6
    beta1_bound = heuristic_aggs["sum_prefill_time(s)"] / heuristic_aggs["sum_prefill_tokens"] * 1e6
    beta2_bound = heuristic_aggs["sum_decode_time(s)"] / heuristic_aggs["sum_output_tokens"] * 1e6
    return beta0_bound, beta1_bound, beta2_bound

def perform_postprocessing_blis(guidellm_profile_path, traces_path, vllm_config_path, results_path, vllm_logs, train = True):
    """
    Perform BLIS-style postprocessing to generate a BLIS_train.json and other necessary files.
    """
    sweep_info_filepath = os.path.join(results_path, "sweep_info.json")

    # read GuideLLM sweep info
    try:
        with open(sweep_info_filepath, 'r') as f:
            sweep_info = json.load(f)
    except:
        print("Could not read sweep info file.")
        sys.exit()

    # read GuideLLM profile file
    try:
        with open(guidellm_profile_path, 'r') as f:
            guidellm_profile = yaml.safe_load(f)
            if "prefix_tokens" not in guidellm_profile["data"]:
                guidellm_profile["data"]["prefix_tokens"] = 0 # set to default if unspecified
    except:
        print("Could not read GuideLLM profile file.")
        sys.exit()

    # read vllm YAML config file
    try:
        with open(vllm_config_path, 'r') as f:
            vllm_config = yaml.safe_load(f)
    except:
        print("Could not read vllm config file.")
        sys.exit()

    # populate total_kv_blocks if absent
    if "total_kv_blocks" not in vllm_config:
        vllm_config["total_kv_blocks"] = extract_total_kv_blocks(vllm_logs)

    # process traces to get server-side latencies
    traces_raw_data = read_traces_jsonl(traces_path)
    all_requests = get_server_side_metrics_from_traces(traces_raw_data)
    requests_df = pd.DataFrame(all_requests)

    # read GuideLLM sweep info
    sweep_info_filepath = os.path.join(results_path, SWEEP_INFO_FILENAME)
    try:
        with open(sweep_info_filepath, 'r') as f:
            sweep_info = json.load(f)
    except:
        print("Could not read sweep info file.")
        sys.exit()
    
    # check if blis_reqgen_config_folder exists, otherwise create
    blis_reqgen_config_folder = os.path.join(results_path, BLIS_REQGEN_CONFIG_FOLDER)
    os.makedirs(blis_reqgen_config_folder, exist_ok=True)

    blis_data = {}
    all_benchmarks = [] # record metrics for each benchmark
    heuristic_aggs = {"sum_prefill_time(s)": 0, "sum_decode_time(s)": 0, "sum_inference_time(s)": 0, "sum_output_tokens": 0,
                        "sum_prefill_tokens": 0, "sum_steps": 0, "max_output_delay(s)": 0} # heuristic aggregates across benchmarks
    for sweep in sweep_info:
        rps = sweep["rps"]

        # construct request gen-config for BLIS Golang sim,
        blis_reqgen_config = construct_BLIS_reqgenconfig(guidellm_profile, rps)
        blis_reqgen_config_filename = os.path.join(
            blis_reqgen_config_folder, 
            f"requestgenconfig_RPS={round(rps, 3)}.yaml"
        )

        # save to YAML file - one per RPS
        with open(blis_reqgen_config_filename, 'w+') as f:
            yaml.dump(blis_reqgen_config, f)
            # print(f"Request gen config saved to {blis_reqgen_config_filename}")

        # benchmark-wise metrics and heuristic totals
        benchmark_request_ids = sweep["requestIDs"]
        benchmark_df = requests_df[requests_df["request_id"].isin(benchmark_request_ids)].copy()
        benchmark_metrics = get_metrics_per_benchmark(benchmark_df, rps, guidellm_profile, vllm_config)
        all_benchmarks.append(benchmark_metrics)
        # sum heuristics are server-side - for betas
        for heuristic in heuristic_aggs:
            if "sum" in heuristic:
                heuristic_aggs[heuristic] += benchmark_metrics[heuristic]
    
    # calculate heuristic bounds for betas in blackbox optimizer
    beta0_bound, beta1_bound, beta2_bound = get_heuristic_bounds(heuristic_aggs)

    # combine all training data - metrics, bounds, vllm config etc. into file
    blis_data["bounds"] = {"beta0": beta0_bound, "beta1": beta1_bound, "beta2": beta2_bound}
    blis_data["benchmarks"] = all_benchmarks
    blis_data["vllm_config"] = vllm_config

    # save postprocessed JSON
    if train:
        blis_final_filename = os.path.join(results_path, BLIS_TRAINING_FILEPATH)
    else:
        blis_final_filename = os.path.join(results_path, BLIS_TESTING_FILEPATH)
    with open(blis_final_filename, 'w+') as f:
        json.dump(blis_data, f, indent=4)
    print(f"BLIS Postprocessing complete. Data saved to {blis_final_filename}")
    return blis_data

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--guidellm_profile", 
                        help="Path to the GuideLLM YAML profile file to be read.")
    parser.add_argument("--traces", 
                        help="Path to the vllm traces file to be read.")
    parser.add_argument("--vllm_config", 
                        help="Path to vllm server config file.")
    parser.add_argument("--vllm_logs", 
                        default="vllm.log",
                        help="Path to vllm logs file.")
    parser.add_argument("--results_path",
                        default=".",
                        help="Location to save intermediate files")
    
    args = parser.parse_args()
    perform_postprocessing_blis(args.guidellm_profile, args.traces, args.vllm_config, args.results_path, args.vllm_logs)
    
    