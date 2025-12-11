import argparse
import os
import pandas as pd

from postprocessing_utils import run_go_binary
import matplotlib.pyplot as plt

GO_BINARY_NAME = "simulation_worker"
GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)
# modify this list to change which metrics to calculate error over
METRICS_TO_COMPARE = ["ttft_mean", "itl_mean", "e2e_mean", "ttft_p90", "itl_p90", "e2e_p90"]

def plot_vllm_vs_sim(real_df, sim_df, error_df, plot_title=""):
    overall_errors = {}
    for metric in METRICS_TO_COMPARE:
        metric_error = f"{metric} MAPE"
        overall_errors[metric_error] = error_df.loc[:, metric_error].mean()
    plt.figure(figsize=(10, 6))
    colors = ['orange', 'red', 'green', 'blue', 'purple', 'yellow']
    plt.bar(list(overall_errors.keys()), list(overall_errors.values()), label=list(overall_errors.keys()), color=colors)
    
    plt.title(f'MAPE error - vllm vs sim - {plot_title}')
    plt.xlabel("Metrics")
    plt.ylabel("Error %")
    plt.legend()
    plots_folder = f"test_plots/blis"
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(f'{plots_folder}/{plot_title}_error.png')

    for metric in METRICS_TO_COMPARE[:3]:
        plt.figure(figsize=(10, 6))
        plt.plot(real_df.loc[:, f"{metric}_real"], color = "blue")
        plt.plot(sim_df.loc[:, f"{metric}_sim"], color = "red")
        plt.title(f'Real vs BLIS - {metric}')
        plt.xlabel("Exp idx")
        plt.ylabel("Error %")
        plt.legend(["Real", "Predicted - BLIS"])
        plt.savefig(f'{plots_folder}/blis_{metric}.png')

    plt.title(f'MAPE error - vllm vs sim - {plot_title}')
    plt.xlabel("Metrics")
    plt.ylabel("Error %")
    plt.legend()
    plots_folder = f"test_plots"
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(f'{plots_folder}/{plot_title}_error.png')

def get_per_exp_error(exp_dict, coeffs_filepath):
    per_exp_error = {}
    per_exp_real = {}
    per_exp_sim = {}
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
        return None, None, None
    for idx, metric in enumerate(METRICS_TO_COMPARE):
        mape = abs(sim_metrics[f"{metric}_ms"] - exp_dict[metric])/exp_dict[metric] * 100
        per_exp_error[f"{metric} MAPE"] = mape
        per_exp_real[f"{metric}_real"] = exp_dict[metric]
        per_exp_sim[f"{metric}_sim"] = sim_metrics[f"{metric}_ms"]
    per_exp_error["tp"] = exp_dict["hardware_count"]
    per_exp_error["GPU"] = exp_dict["hardware"]
    per_exp_error["model"] = exp_dict["model_hf_repo"]
    per_exp_error["vllm-version"] = exp_dict["docker_image"]
    return per_exp_real, per_exp_sim, per_exp_error

def test_blis_model(training_filepath, coeffs_filepath, LLM_name = None, tp = None, gpu = None, vllm_version = None):
    # read training CSV and filter to only train rows for LLM
    df = pd.read_excel(training_filepath)
    filter_values = {
        "model_hf_repo": LLM_name,
        "hardware_count": int(tp) if tp is not None and str(tp).isdigit() else None,
        "hardware": gpu,
        "docker_image": vllm_version
    }

    mandatory_conditions = (df["train_test"] == "test") & (df["saturated"] == False)

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
        exp_real, exp_sim, exp_error = get_per_exp_error(exp_dict, coeffs_filepath)
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
    parser.add_argument("--LLM-name", 
                        help="LLM to train BLIS coefficients for, pick one from the Excel file")
    parser.add_argument("--tp", 
                        help="TP value to train BLIS coefficients for, pick one from the Excel file")
    parser.add_argument("--GPU", 
                        help="GPU to train BLIS coefficients for, pick one from the Excel file")
    parser.add_argument("--vllm-version", 
                        help="vllm version to train BLIS coefficients for, pick one from the Excel file")
    parser.add_argument("--training-filepath",
                        default="blis_rh_final.xlsx",
                        help="Path to Excel file with GuideLLM RH data.")
    parser.add_argument("--coeffs-filepath",
                        default="coefficients.yaml", 
                        help="Path to trained BLIS coeffs.")
    args = parser.parse_args()
    real_df, sim_df, error_df = test_blis_model(args.training_filepath, args.coeffs_filepath, args.LLM_name, args.tp, args.GPU, args.vllm_version)
    print(error_df)
    model_name_for_plot_name = "*"
    vllm_version_for_plot_name = "*"
    tp = "*"
    gpu = "*"
    if args.LLM_name:
        model_name_for_plot_name = args.LLM_name.split("/")[1]
    if args.vllm_version:
        vllm_version_for_plot_name = args.vllm_version.split("/")[1]
    if args.tp:
        tp = args.tp
    if args.GPU:
        gpu = args.GPU
    plot_vllm_vs_sim(real_df, sim_df, error_df, plot_title=f"LLM={model_name_for_plot_name} TP={tp} GPU={gpu} vllmVersion={vllm_version_for_plot_name}")
