import argparse
import json
import os
import pandas as pd

from postprocessing_utils import run_go_binary

GO_BINARY_NAME = "simulation_worker"
GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)

def run_one_exp(exp_dict, coeffs_filepath):
    sim_results = {}
    blis_cmd = exp_dict["blis_cmd"]
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
        print(f"Found trained coefficients for model={exp_dict["model_hf_repo"]}, \
tp={exp_dict["hardware_count"]}, GPU={exp_dict["hardware"]}, vllm_version={exp_dict["framework_version"]}")
    except:
        return None
    benchmark_metrics_list = ["model_hf_repo", "hardware", "hardware_count", "framework", 
                              "framework_version", "mean_input_tokens", "mean_output_tokens",
                              "prompt_tokens", "prompt_tokens_stdev", "output_tokens", "output_tokens_stdev"]
    sim_metrics_list = ["ttft_mean", "ttft_p90", "ttft_p95", "ttft_p99", "itl_mean", "itl_p90", "itl_p95", "itl_p99",
                        "e2e_mean", "e2e_p90", "e2e_p95", "e2e_p99"]
    for key in benchmark_metrics_list:
        sim_results[key] = exp_dict[key]
    for key in sim_metrics_list:
        sim_results[key] = sim_metrics[f"{key}_ms"]
    sim_results["requests_per_second"] = float(exp_dict["requests_per_second"])
    sim_results["responses_per_second"] = sim_metrics["responses_per_sec"]
    sim_results["tokens_per_second"] = sim_metrics["tokens_per_sec"]
    return sim_results

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

    test_df = df[(df["train_test"] == "test") & (df["saturated"] == False)]
    for idx in range(len(test_df)):
        exp_dict = test_df.iloc[idx].to_dict()
        sim_results = run_one_exp(exp_dict, args.coeffs_filepath)
        if sim_results:
            results["benchmarks"].append(sim_results)
    with open(args.synthetic_results_filepath, "w+") as f:
        json.dump(results, f)