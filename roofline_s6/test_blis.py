import argparse
import os
import pandas as pd

from postprocessing_utils import run_go_binary
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

GO_BINARY_NAME = "simulation_worker"
GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)
# modify this list to change which metrics to calculate error over
METRICS_TO_COMPARE = ["ttft_mean", "ttft_p90", "itl_mean", "itl_p90", "e2e_mean", "e2e_p90"]

def download_model_config_from_hf(model_id: str, save_directory: str = "."):
    """
    Downloads the config.json file of a given LLM from Hugging Face.
    
    Args:
        model_id (str): The repo ID on HF (e.g., "meta-llama/Llama-2-7b-hf").
        save_directory (str): Where to save the downloaded file.
        
    Returns:
        str: The local path to the downloaded config file.
    """
    try:
        # hf_hub_download returns the local path to the cached file
        file_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_dir=save_directory,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded config to: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example Usage:
# path = download_model_config_from_hf("google/gemma-2b")

def plot_vllm_vs_sim(error_df, mode = "train", groupby_fields = ["model"]):
    grouped_df = error_df.groupby(groupby_fields).mean(numeric_only=True)
    overall_errors = {}
    for group_idx in grouped_df.index:
        plot_title = f"{group_idx}"
        for metric in METRICS_TO_COMPARE:
            metric_error = f"{metric} MAPE"
            overall_errors[metric_error] = grouped_df.loc[group_idx, metric_error].mean()
        plt.figure(figsize=(10, 6))
        colors = ['orange', 'red', 'green', 'blue', 'purple', 'yellow']
        plt.bar(list(overall_errors.keys()), list(overall_errors.values()), label=list(overall_errors.keys()), color=colors)
        
        plt.title(f'MAPE error - vllm vs sim - {plot_title}')
        plt.xlabel("Metrics")
        plt.ylabel("Error %")
        plt.legend()
        plots_folder = f"{mode}_plots/blis"
        os.makedirs(plots_folder, exist_ok=True)
        plt.savefig(f'{plots_folder}/{plot_title}_error.png')

def get_per_exp_error(exp_dict):
    per_exp_error = {}
    per_exp_real = {}
    per_exp_sim = {}
    blis_cmd = exp_dict["blis_cmd"]
    blis_args = ["run"]
    blis_args.extend(blis_cmd.split(" "))
    model_config_folder = os.path.join("model_configs", exp_dict["model_hf_repo"].split("/")[1].lower())
    if not os.path.exists(model_config_folder):
        os.makedirs(model_config_folder)
        download_model_config_from_hf(exp_dict["model_hf_repo"], model_config_folder)
    extra_args_with_coeffs = {
        "block-size-in-tokens": 16,
        "long-prefill-token-threshold": 0,
        "horizon": "922337203685477580", # Golang int64 max value
        "max-prompts": 100,
        "log": "fatal",
        "model-config-folder": model_config_folder,
    }
    for key in extra_args_with_coeffs:
        blis_args.extend([f"--{key}", str(extra_args_with_coeffs[key])])
    # print(" ".join(map(str, blis_args)))
    try:
        sim_metrics = run_go_binary(blis_args, GO_BINARY_PATH)
        # print(f"Found trained coefficients for model={exp_dict["model_hf_repo"]}, \
# tp={exp_dict["hardware_count"]}, GPU={exp_dict["hardware"]}, vllm_version={exp_dict["framework_version"]}")
    except:
        return None, None, None
    for idx, metric in enumerate(METRICS_TO_COMPARE):
        mape = abs(sim_metrics[f"{metric}_ms"] - exp_dict[metric])/exp_dict[metric] * 100
        per_exp_error[f"{metric} MAPE"] = mape
        per_exp_real[f"{metric}_real"] = exp_dict[metric]
        per_exp_sim[f"{metric}_sim"] = sim_metrics[f"{metric}_ms"]
    per_exp_error["tp"] = exp_dict["hardware_count"]
    per_exp_error["GPU"] = exp_dict["hardware"]
    per_exp_error["model"] = exp_dict["model_hf_repo"].split("/")[1].lower()
    per_exp_error["request_rate"] = exp_dict["requests_per_second"]
    per_exp_error["mean_isl"] = exp_dict["mean_input_tokens"]
    per_exp_error["mean_osl"] = exp_dict["mean_output_tokens"]
    return per_exp_real, per_exp_sim, per_exp_error

def test_blis_model(testing_filepath, mode = "train", LLM_name = None, tp = None, gpu = None, vllm_version = None):
    # read testing CSV and filter to only train rows for LLM
    df = pd.read_excel(testing_filepath)
    filter_values = {
        "model_hf_repo": LLM_name,
        "hardware_count": int(tp) if tp is not None and str(tp).isdigit() else None,
        "hardware": gpu,
        "docker_image": vllm_version
    }

    mandatory_conditions = (df["train_test"] == mode) & (df["saturated"] == False)

    optional_conditions = [
        df[col] == val for col, val in filter_values.items() if val is not None
    ]

    final_mask = mandatory_conditions
    for cond in optional_conditions:
        final_mask = final_mask & cond

    test_df = df[final_mask]
    # test_df = test_df.groupby(["model_hf_repo", "hardware", "hardware_count", "prompt_tokens", "output_tokens"]).first().reset_index() # take only the first request rate for a group
    # test_df = test_df[test_df["model_hf_repo"] != "mistralai/mistral-small-3.1-24b-instruct-2503"]
    all_exp_real = []
    all_exp_sim = []
    all_exp_mapes = []
    for idx in range(len(test_df)):
        exp_dict = test_df.iloc[idx].to_dict()
        exp_real, exp_sim, exp_error = get_per_exp_error(exp_dict)
        if exp_real:
            all_exp_real.append(exp_real)
        if exp_sim:
            all_exp_sim.append(exp_sim)
        if exp_error:
            all_exp_mapes.append(exp_error)
    return pd.DataFrame(all_exp_real), pd.DataFrame(all_exp_sim), pd.DataFrame(all_exp_mapes)

    # Parallel Grid sampling
    # optimizer.search_space = {
    #     'beta0': list(np.arange(0, heuristics_bounds["beta0"][1], heuristics_bounds["beta0"][1]/20)),
    #     'beta1': list(np.arange(0, heuristics_bounds["beta1"][1], heuristics_bounds["beta1"][1]/20)),
    #     'beta2': list(np.arange(0, heuristics_bounds["beta2"][1], heuristics_bounds["beta2"][1]/20)),
    # }
    # num_GS_iters = len(optimizer.search_space["beta0"]) * len(optimizer.search_space["beta1"]) * len(optimizer.search_space["beta2"])
    
    # with Pool(processes=MAX_NUM_PROCESSES) as pool: 
    #     pool.map(with_inp, ((i, optimizer) for i in range(num_GS_iters)))

    # best_params = optimizer.get_best_trial()

    # save best optimizer parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--mode", 
                        default="train",
                        help="train/test")
    parser.add_argument("--testing-filepath",
                        default="blis_rh_final.xlsx",
                        help="Path to Excel file with GuideLLM RH data.")
    parser.add_argument("--specs-filepath",
                        default="specs.csv", 
                        help="Path to all combinations to train.")
    args = parser.parse_args()
    df = pd.read_csv(args.specs_filepath)
    all_error_dfs_list = []
    for idx in range(len(df)):
        row_dict = df.iloc[idx].to_dict()
        print("############################################################################")
        print(f"Running BLIS for LLM={row_dict["LLM_name"]}, tp={row_dict["tp"]}, GPU={row_dict["GPU"]}, vllm-version={row_dict["vllm_version"]}")
        print("############################################################################")
        real_df, sim_df, error_df = test_blis_model(args.testing_filepath, args.mode, row_dict["LLM_name"], row_dict["tp"], row_dict["GPU"], row_dict["vllm_version"])
        print("REAL")
        print(real_df)
        print("SIM")
        print(sim_df)
        print("ERROR")
        print(error_df)
        all_error_dfs_list.append(error_df)
    combined_error_df = pd.concat(all_error_dfs_list, ignore_index=True)
    print(combined_error_df[combined_error_df['tp']==8])
    plot_vllm_vs_sim(combined_error_df, args.mode, groupby_fields=["tp"])
