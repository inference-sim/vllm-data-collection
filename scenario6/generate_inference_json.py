import json
import os
import pandas as pd

from postprocessing_utils import run_go_binary

GO_BINARY_NAME = "simulation_worker"
GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)
TRAINING_DATA_FILEPATH = "blis_rh_final.xlsx"
SYNTHETIC_JSON_PATH = "benchmarks_BLIS.json"

def run_one_exp(exp_dict):
    sim_results = {}
    blis_cmd = exp_dict["blis_cmd"]
    blis_args = ["run"]
    blis_args.extend(blis_cmd.split(" "))
    extra_args_with_coeffs = {
        "block-size-in-tokens": 16,
        "long-prefill-token-threshold": 0,
        "horizon": "922337203685477580", # Golang int64 max value
        # do not provide alpha/beta coeffs - let the sim pick it
        "max-prompts": 100,
        "log": "fatal"
    }
    for key in extra_args_with_coeffs:
        blis_args.extend([f"--{key}", str(extra_args_with_coeffs[key])])
    try:
        sim_metrics = run_go_binary(blis_args, GO_BINARY_PATH)
    except:
        print(f"Could not find trained coefficients for model={exp_dict["model_hf_repo"]}, \
tp={exp_dict["hardware_count"]}, GPU={exp_dict["hardware"]}, vllm_version={exp_dict["framework_version"]}")
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
    sim_results["responses_per_second"] = sim_metrics["responses_per_sec"]
    sim_results["tokens_per_second"] = sim_metrics["tokens_per_sec"]
    return sim_results

if __name__=="__main__":
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
    df = pd.read_excel(TRAINING_DATA_FILEPATH)

    test_df = df[(df["train_test"] == "test") & (df["saturated"] == False)]
    for idx in range(len(test_df)):
        exp_dict = test_df.iloc[idx].to_dict()
        sim_results = run_one_exp(exp_dict)
        if sim_results:
            results["benchmarks"].append(sim_results)
    with open(SYNTHETIC_JSON_PATH, "w+") as f:
        json.dump(results, f)