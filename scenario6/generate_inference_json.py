import argparse
import json
import os
import re
import pandas as pd
import numpy as np

from postprocessing_utils import run_go_binary

GO_BINARY_NAME = "simulation_worker"
GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)

SPEC_DEFINITION = ["model_hf_repo", "hardware", "hardware_count", "docker_image","framework", "framework_version",
                   "prompt_tokens", "prompt_tokens_stdev", "prompt_tokens_min", "prompt_tokens_max",
                   "output_tokens", "output_tokens_stdev", "output_tokens_min", "output_tokens_max"]

SATURATION_RATIO = 0.98
MIN_RPS_FOR_BINARY_SEARCH = 1
MAX_RPS_FOR_BINARY_SEARCH = 100000
NUM_SWEEPS = 10

def run_blis(df_groups, spec, coeffs_filepath, rate):
    """
    Returns a tuple - (sim results: Dict, saturated: bool)
    """
    sim_results = {}
    saturated = False
    blis_cmd = df_groups.loc[spec, "blis_cmd"][0]
    pattern = r'--rate\s+\d+'
    blis_cmd = re.sub(pattern, f'--rate {rate}', blis_cmd)
    exp_dict = {SPEC_DEFINITION[i]: spec[i] for i in range(len(SPEC_DEFINITION))}
    exp_dict["blis_cmd"] = blis_cmd
    exp_dict["requests_per_second"] = rate
    blis_args = ["run"]
    blis_args.extend(blis_cmd.split(" "))
    extra_args_with_coeffs = {
        "block-size-in-tokens": 16,
        "long-prefill-token-threshold": 0,
        "horizon": "922337203685477580", # Golang int64 max value
        "coeffs-filepath": coeffs_filepath,
        "max-prompts": 100,
        "log": "fatal"
    }
    for key in extra_args_with_coeffs:
        blis_args.extend([f"--{key}", str(extra_args_with_coeffs[key])])
    try:
        sim_metrics = run_go_binary(blis_args, GO_BINARY_PATH)
    except:
        return None, saturated
    benchmark_metrics_list = ["model_hf_repo", "hardware", "hardware_count", "framework", 
                              "framework_version", "prompt_tokens", "prompt_tokens_stdev", "output_tokens", "output_tokens_stdev"]
    sim_metrics_list = ["ttft_mean", "ttft_p90", "ttft_p95", "ttft_p99", "itl_mean", "itl_p90", "itl_p95", "itl_p99",
                        "e2e_mean", "e2e_p90", "e2e_p95", "e2e_p99"]
    for key in benchmark_metrics_list:
        sim_results[key] = exp_dict[key]
    for key in sim_metrics_list:
        sim_results[key] = sim_metrics[f"{key}_ms"]
    sim_results["mean_input_tokens"] = sim_results["prompt_tokens"]
    sim_results["mean_output_tokens"] = sim_results["output_tokens"]
    sim_results["requests_per_second"] = float(rate)
    sim_results["responses_per_second"] = sim_metrics["responses_per_sec"]
    if sim_results["responses_per_second"] < SATURATION_RATIO * rate:
        saturated = True
    sim_results["tokens_per_second"] = sim_metrics["tokens_per_sec"]
    return sim_results, saturated
        
def determine_saturation_rps(df_groups, spec, coeffs_filepath):
    # binary search to find saturation rps
    rps_low = MIN_RPS_FOR_BINARY_SEARCH
    rps_high = MAX_RPS_FOR_BINARY_SEARCH
    saturation_rps = -1
    while rps_low <= rps_high:
        rps_mid = (rps_low + rps_high)//2
        _, is_saturated = run_blis(df_groups, spec, coeffs_filepath, rps_mid)
        if is_saturated:
            saturation_rps = rps_mid
            rps_high = rps_mid - 1
        else:
            rps_low = rps_mid + 1
    if saturation_rps == -1:
        return MAX_RPS_FOR_BINARY_SEARCH
    return saturation_rps

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--coeffs-filepath",
                        default="coefficients.yaml", 
                        help="Path to trained BLIS coeffs.")
    parser.add_argument("--testing-filepath",
                        default="blis_rh_final.xlsx",
                        help="Path to Excel file with GuideLLM RH data.")
    parser.add_argument("--synthetic-results-filepath",
                        default="benchmarks_BLIS.json",
                        help="Path to save formatted JSON sim results to.")
    parser.add_argument("--specs-filepath",
                        default="training_specs.csv",
                        help="Path to all combinations to train.")
    args = parser.parse_args()
    results = {
        "_metadata": {
            "description": "Synthetic benchmark data for development and testing",
            "version": "1.0-BLIS",
            "schema_changes": [
            "Generated from BLIS simulated trained on realistic data",
            "Performance values simulated and vary by model/GPU/tensor_parallel",
            "E2E calculated as: TTFT + (ITL \u00d7 output_tokens)",
            ],
            "generation_method": "BLIS simulated data",
            "source": "BLIS simulator",
            "variation": "",
            "random_seed": 42
        },
        "benchmarks": []
    }
    df = pd.read_excel(args.testing_filepath)
    specs = pd.read_csv(args.specs_filepath)

    # determine saturation req/s
    for spec in specs.itertuples():
        df_filtered = df[(df["train_test"]=="test") & (df["model_hf_repo"] == spec.LLM_name) & (df["hardware_count"] == int(spec.tp)) & (df["hardware"] == spec.GPU) & (df["docker_image"] == spec.vllm_version)]
        df_groups = df_filtered.groupby(by=SPEC_DEFINITION).agg(list)
        for idx in df_groups.index:
            saturation_rps = determine_saturation_rps(df_groups, idx, args.coeffs_filepath)
            print(f"model={spec.LLM_name}, tp={int(spec.tp)}, GPU={spec.GPU}, vllm_version={spec.vllm_version}, saturation_rps={saturation_rps}")
            rates_to_report = np.sort(np.unique(np.linspace(start=1, stop=saturation_rps, num=NUM_SWEEPS)))
            for rate in rates_to_report:
                sim_results, _ = run_blis(df_groups, idx, args.coeffs_filepath, rate)
                if sim_results:
                    results["benchmarks"].append(sim_results)
    with open(args.synthetic_results_filepath, "w+") as f:
        json.dump(results, f)