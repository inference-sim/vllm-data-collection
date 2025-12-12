import argparse
import glob
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
import yaml

from postprocess_guidellm_common import perform_postprocessing_common
from postprocess_blis import perform_postprocessing_blis
from postprocessing_utils import run_go_binary
from blis_beta_model import GO_BINARY_PATH

METRICS_TO_COMPARE = ["e2e_mean_ms", "e2e_p90_ms", "ttft_mean_ms", "ttft_p90_ms", "itl_mean_ms", "itl_p90_ms"]

def plot_vllm_vs_sim(data_df, groupby = "tp"):
    grouped_df = data_df.groupby(groupby).mean(numeric_only=True)
    print(grouped_df)
    overall_errors = {}
    for group_idx in grouped_df.index:
        if groupby == "model_path":
            plot_title = f"{groupby}={group_idx.split("/")[1]}"
        else:
            plot_title = f"{groupby}={group_idx}"
        for metric in METRICS_TO_COMPARE:
            metric_error = f"{metric} MAPE"
            overall_errors[metric_error] = grouped_df.loc[group_idx, metric_error]
        plt.figure(figsize=(10, 6))
        colors = ['orange', 'red', 'green', 'blue', 'purple']
        plt.bar(list(overall_errors.keys()), list(overall_errors.values()), label=list(overall_errors.keys()), color=colors)
        
        plt.title(f'MAPE error - vllm vs sim - {plot_title}')
        plt.xlabel("Metrics")
        plt.xticks(rotation=90)
        plt.ylabel("Error %")
        plt.legend()
        plots_folder = f"test_plots"
        os.makedirs(plots_folder, exist_ok=True)
        plt.savefig(f'{plots_folder}/{plot_title}_error.png')

def get_alpha_beta_gamma_coeffs(model_tp_path):
    # read alpha coeffs
    alpha_metrics_file = os.path.join(model_tp_path, "BLIS_alpha_metrics.json")
    try:
        with open(alpha_metrics_file, 'r') as f:
            alpha_metrics = json.load(f)
            alpha_coeffs = list(map(str, alpha_metrics["coeffs"]))
    except:
        print("Could not load BLIS alpha coeffs.")
        sys.exit()
    
    # read beta coeffs
    beta_metrics_file = os.path.join(model_tp_path, "BLIS_beta_metrics.json")
    alpha2 = 0
    try:
        with open(beta_metrics_file, 'r') as f:
            beta_metrics = json.load(f)
            alpha2 = beta_metrics["best_params"]["alpha2"]
            beta_coeffs = list(map(str, list(beta_metrics["best_params"].values())))
            gamma = beta_metrics["best_params"]["gamma"]
    except:
        print("Could not load BLIS beta coeffs.")
        sys.exit()
    alpha_coeffs.append(str(alpha2))
    beta_coeffs = beta_coeffs[1:]
    return alpha_coeffs, beta_coeffs, gamma

def get_per_test_exp_result(model_path, test_full_path):
    guidellm_profile_path = os.path.join(test_full_path, "profile.yaml")
    guidellm_results_path = os.path.join(test_full_path, "guidellm-results.json")
    vllm_config_path = os.path.join(test_full_path, "exp-config.yaml")

    traces_path = os.path.join(test_full_path, "traces.json")
    perform_postprocessing_common(guidellm_results_path, test_full_path)
    perform_postprocessing_blis(guidellm_profile_path, traces_path, vllm_config_path, test_full_path, train=False)

    # get TP value from vllm_config
    try:
        with open(vllm_config_path, 'r') as f:
            vllm_config = yaml.safe_load(f)
            model = vllm_config["model"].split("/")[1].lower()
            tp = vllm_config["tensor_parallelism"]
    except:
        print("Could not load vllm config data.")
        sys.exit()

    # get pretrained BLIS coeffs
    model_tp_path = os.path.join(model_path, f"model_{model}_tp_{tp}")
    alpha_coeffs, beta_coeffs, gamma = get_alpha_beta_gamma_coeffs(model_tp_path)
    benchmark_file_path = os.path.join(test_full_path, "BLIS_test.json")
    try:
        with open(benchmark_file_path, 'r') as f:
            benchmark_data = json.load(f)
    except:
        print("Could not load BLIS benchmark data.")
        sys.exit()

    # compare sim metrics against benchmark metrics
    all_benchmark_mapes = []
    for benchmark_metrics in benchmark_data["benchmarks"]:
        row = {}
        rps = benchmark_metrics["rps"]
        # get sim metrics
        reqgen_config_folder = os.path.join(test_full_path, "blis_reqgenconfigs")
        reqgen_config_file = os.path.join(reqgen_config_folder, 
                                          f"requestgenconfig_RPS={round(rps, 3)}.yaml")
        args = {
            "max-num-running-reqs": benchmark_data["vllm_config"]["max_num_seqs"], 
            "total-kv-blocks": benchmark_data["vllm_config"]["total_kv_blocks"],
            "model": benchmark_data["vllm_config"]["model"],
            "max-num-scheduled-tokens": benchmark_data["vllm_config"]["max_num_batched_tokens"], 
            "block-size-in-tokens": 16,
            "horizon": "922337203685477580", # Golang int64 max value
            "beta-coeffs": ','.join(beta_coeffs),
            "gamma-coeffs": gamma,
            "flop-per-token": benchmark_data["vllm_config"]["f_tokens"],
            "long-prefill-token-threshold": 0,
            "alpha-coeffs": ','.join(alpha_coeffs),
            "log": "error"
        }
        args_list = ["run"]
        for key in args:
            args_list.extend([f"--{key}", str(args[key])])
        with open(reqgen_config_file, "r+") as f:
            workload_config = yaml.safe_load(f)
        for config in workload_config["data"]:
            if config != "prefix_tokens":
                config_field = f"--{config.replace("_", "-")}"
                args_list.extend([config_field, str(workload_config["data"][config])])
        args_list.extend(["--rate", str(workload_config["rate"]["rate"])])
        args_list.extend(["--max-prompts", str(workload_config["rate"]["max-requests"])])
        print(" ".join(list(map(str, args_list))))
        sim_metrics = run_go_binary(args_list, GO_BINARY_PATH, rps)
        if not sim_metrics:
            return None
        for idx, metric in enumerate(METRICS_TO_COMPARE):
            mape = abs(sim_metrics[metric] - benchmark_metrics[metric])/benchmark_metrics[metric] * 100
            row[f"{metric} MAPE"] = mape
        row["rps"] = rps
        row["tp"] = tp
        row["chunk_size"] = benchmark_data["vllm_config"]["max_num_batched_tokens"]
        row["app"] = vllm_config["app"]
        row["model_path"] = model_tp_path
        row["beta_coeffs"] = beta_coeffs
        row["alpha_coeffs"] = alpha_coeffs
        all_benchmark_mapes.append(row)
    return pd.DataFrame(all_benchmark_mapes)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--model_path", 
                        help="Path to trained beta models folder")
    parser.add_argument("--test_results_path", 
                        help="Path to the root test results folder")
    parser.add_argument("--groupby_field", default = "model_path",
                        help="Field to group and mean errors by - app/chunk_size/tp/rps/model_path")
    args = parser.parse_args()

    # Evaluate across all test experiments
    test_results_path = os.path.join(args.test_results_path, "*/")
    all_tests = glob.glob(test_results_path)
    all_error_dfs_list = []
    for test_path in all_tests:
        test_mape_df = get_per_test_exp_result(args.model_path, test_path)
        all_error_dfs_list.append(test_mape_df)
    combined_error_df = pd.concat(all_error_dfs_list, ignore_index=True)
    combined_error_df.to_csv("combined_test_errors.csv", index=False)
    plot_vllm_vs_sim(combined_error_df, args.groupby_field)
        




