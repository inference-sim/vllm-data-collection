import argparse
import json
import os
import sys
import yaml
import pandas as pd

from postprocessing_utils import BLIS_REQGEN_CONFIG_FOLDER, BLIS_TRAINING_FILEPATH, SWEEP_INFO_FILENAME
from postprocessing_utils import read_traces_jsonl, get_server_side_metrics_from_traces

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
    benchmark_metrics["Mean E2E(ms)"] = benchmark_df['e2e_latency'].mean() * 1000
    benchmark_metrics["Median E2E(ms)"] = benchmark_df['e2e_latency'].median() * 1000
    benchmark_metrics["P99 E2E(ms)"] = benchmark_df['e2e_latency'].quantile(0.99) * 1000
    # metrics needed for heuristic beta bounds
    chunk_size = vllm_config["max-num-batched-tokens"]
    benchmark_metrics["sum_prefill_time"] = benchmark_df["prefill_time"].sum()
    benchmark_metrics["sum_decode_time"] = benchmark_df["decode_time"].sum()
    benchmark_metrics["sum_inference_time"] = (benchmark_df["prefill_time"] + benchmark_df["decode_time"]).sum()
    # assume all requests have the exact same number of prefix tokens
    benchmark_metrics["sum_prefill_tokens"] = (benchmark_df["input_tokens"] - guidellm_profile["data"]["prefix_tokens"]).sum()
    benchmark_metrics["sum_output_tokens"] = benchmark_df["output_tokens"].sum()
    benchmark_metrics["sum_steps"] = (benchmark_df["input_tokens"] - guidellm_profile["data"]["prefix_tokens"])/chunk_size + benchmark_df["output_tokens"]
    return benchmark_metrics

def get_heuristic_bounds(heuristic_totals):
    beta0_bound = heuristic_totals["sum_inference_time"] / heuristic_totals["sum_steps"]
    beta1_bound = heuristic_totals["sum_prefill_time"] / heuristic_totals["sum_prefill_tokens"]
    beta2_bound = heuristic_totals["sum_decode_time"] / heuristic_totals["sum_output_tokens"]
    return beta0_bound, beta1_bound, beta2_bound

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--guidellm_profile", 
                        help="Path to the GuideLLM YAML profile file to be read.")
    parser.add_argument("--traces", 
                        help="Path to the vllm traces file to be read.")
    parser.add_argument("--vllm_config", 
                        help="Path to vllm server config file.")
    parser.add_argument("--results_path",
                        default=".",
                        help="Location to save intermediate files")
    
    args = parser.parse_args()

    # read GuideLLM sweep info
    try:
        with open(args.sweep_info_path, 'r') as f:
            sweep_info = json.load(f)
    except:
        print("Could not read sweep info file.")
        sys.exit()

    # read GuideLLM profile file
    guidellm_profile_filepath = args.guidellm_profile
    try:
        with open(guidellm_profile_filepath, 'r') as f:
            guidellm_profile = yaml.safe_load(f)
            if "prefix_tokens" not in guidellm_profile["data"]:
                guidellm_profile["data"]["prefix_tokens"] = 0 # set to default if unspecified
    except:
        print("Could not read GuideLLM profile file.")
        sys.exit()

    # read vllm config file
    vllm_config_filepath = args.vllm_config
    try:
        with open(vllm_config_filepath, 'r') as f:
            vllm_config = json.load(f)
    except:
        print("Could not read vllm config file.")
        sys.exit()
    

    # process traces to get server-side latencies
    traces_raw_data = read_traces_jsonl(args.traces)
    all_requests = get_server_side_metrics_from_traces(traces_raw_data)
    requests_df = pd.Dataframe(all_requests)

    # read GuideLLM sweep info
    sweep_info_filepath = os.path.join(args.results_path, SWEEP_INFO_FILENAME)
    try:
        with open(sweep_info_filepath, 'r') as f:
            sweep_info = json.load(f)
    except:
        print("Could not read sweep info file.")
        sys.exit()
    
    # check if args.blis_reqgen_config_folder exists, otherwise create
    blis_reqgen_config_folder = os.path.join(args.results_path, BLIS_REQGEN_CONFIG_FOLDER)
    os.makedirs(blis_reqgen_config_folder, exist_ok=True)

    blis_training_data = {}
    all_benchmarks = [] # record metrics for each benchmark
    heuristic_totals = {"sum_prefill_time": 0, "sum_decode_time": 0, "sum_inference_time": 0, "sum_output_tokens": 0,
                        "sum_prefill_tokens": 0, "sum_steps": 0} # heuristic sums across benchmarks
    for sweep in sweep_info:
        rps = sweep["rps"]

        # construct request gen-config for BLIS Golang sim,
        blis_reqgen_config = construct_BLIS_reqgenconfig(guidellm_profile, rps)
        blis_reqgen_config_filename = os.path.join(
            args.blis_reqgen_config_folder, 
            f"requestgenconfig_RPS={round(rps, 3)}.yaml"
        )

        # save to YAML file - one per RPS
        with open(blis_reqgen_config_filename, 'w+') as f:
            yaml.dump(blis_reqgen_config, f)
            print(f"Request gen config saved to {blis_reqgen_config_filename}")

        # benchmark-wise metrics and heuristic totals
        benchmark_request_ids = sweep["requestIDs"]
        benchmark_df = requests_df[requests_df["request_id"].isin(benchmark_request_ids)].copy()
        benchmark_metrics = get_metrics_per_benchmark(benchmark_df, rps, guidellm_profile, vllm_config)
        all_benchmarks.append(benchmark_metrics)
        for heuristic in heuristic_totals:
            heuristic_totals[heuristic] += benchmark_metrics[heuristic]
    
    # calculate heuristic bounds for betas in blackbox optimizer
    beta0_bound, beta1_bound, beta2_bound = get_heuristic_bounds(heuristic_totals)

    # combine all training data - metrics, bounds, vllm config etc. into file
    blis_training_data["bounds"] = {"beta0": beta0_bound, "beta1": beta1_bound, "beta2": beta2_bound}
    blis_training_data["benchmarks"] = all_benchmarks
    blis_training_data["vllm_config"] = vllm_config

    # save postprocessed JSON
    blis_training_filename = os.path.join(args.results_path, BLIS_TRAINING_FILEPATH)
    with open(blis_training_filename, 'w+') as file:
        json.dump(blis_training_data, file, indent=4)
    print(f"BLIS Postprocessing complete. Training data saved to {blis_training_filename}")