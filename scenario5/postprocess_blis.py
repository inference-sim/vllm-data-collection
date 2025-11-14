import argparse
import json
import os
import sys
import yaml
import pandas as pd

from postprocessing_utils import BLIS_REQGEN_CONFIG_FOLDER, SWEEP_INFO_FILENAME
from postprocessing_utils import read_traces_jsonl, get_server_side_info

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

def get_average_metrics_per_benchmark(all_requests):
    pass


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--guidellm_profile", 
                        help="Path to the GuideLLM YAML profile file to be read.")
    parser.add_argument("--traces", 
                        help="Path to the vllm traces file to be read.")
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

    guidellm_profile_filepath = args.guidellm_profile
    # read GuideLLM profile file
    try:
        with open(guidellm_profile_filepath, 'r') as f:
            guidellm_profile = yaml.safe_load(f)
    except:
        print("Could not read GuideLLM profile file.")
        sys.exit()
    
    # Check if args.blis_reqgen_config_folder exists, otherwise create
    blis_reqgen_config_folder = os.path.join(args.results_path, BLIS_REQGEN_CONFIG_FOLDER)
    os.makedirs(blis_reqgen_config_folder, exist_ok=True)

    # Construct BLIS-style request gen-config
    for sweep in sweep_info:
        rps = sweep["rps"]
        blis_reqgen_config = construct_BLIS_reqgenconfig(guidellm_profile, rps)
        blis_reqgen_config_filename = os.path.join(
            args.blis_reqgen_config_folder, 
            f"requestgenconfig_RPS={round(rps, 3)}.yaml"
        )

        # Save to YAML file - one per RPS
        with open(blis_reqgen_config_filename, 'w+') as f:
            yaml.dump(blis_reqgen_config, f)

    # Process traces to get server-side latencies
    traces_raw_data = read_traces_jsonl(args.traces)
    all_requests = get_server_side_info(traces_raw_data)
    requests_df = pd.Dataframe(all_requests)

    # read GuideLLM sweep info
    sweep_info_filepath = os.path.join(args.results_path, SWEEP_INFO_FILENAME)
    try:
        with open(sweep_info_filepath, 'r') as f:
            sweep_info = json.load(f)
    except:
        print("Could not read sweep info file.")
        sys.exit()

    for sweep in sweep_info:
        # each request-rate forms a new benchmark
        rps = sweep["rps"]
        benchmark_request_ids = sweep["requestIDs"]
        benchmark_df = requests_df[requests_df["request_id"].isin(benchmark_request_ids)].copy()
        benchmark_averages = get_average_metrics_per_benchmark(benchmark_df, rps)

