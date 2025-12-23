import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os

METRICS_TO_COMPARE = ["ttft", "itl", "e2e"]

def plot_vllm_vs_sim(data_df, plot_title=""):
    overall_errors = {}
    for metric in METRICS_TO_COMPARE:
        metric_error = f"{metric} MAPE"
        overall_errors[metric_error] = data_df.loc[:, metric_error].mean()
    plt.figure(figsize=(10, 6))
    colors = ['orange', 'red', 'green', 'blue', 'purple', 'yellow']
    plt.bar(list(overall_errors.keys()), list(overall_errors.values()), label=list(overall_errors.keys()), color=colors)
    
    plt.title(f'MAPE error - vllm vs sim - {plot_title}')
    plt.xlabel("Metrics")
    plt.ylabel("Error %")
    plt.legend()
    plots_folder = f"test_plots"
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(f'{plots_folder}/{plot_title}_error.png')

def generate_config():
    df = pd.read_excel("blis_rh_final.xlsx")
    with open("all_coefficients.yaml", "r+") as f:
        data = yaml.safe_load(f)
    df = df[(df["train_test"] == "test") & (df["saturated"] == False)]
    df = df.groupby(["model_hf_repo", "hardware", "hardware_count", "prompt_tokens", "output_tokens"]).first().reset_index() # take only the first request rate for a group

    aiconfigurator_config = {"exps": []}
    exp_count = 1
    gpu_mapping = {"H100": "h100_sxm", "A100-80": "a100_sxm"}
    vllm_versions = ["v0.8.4", "v0.10.1.1"]
    llm_mapping = {"MIXTRAL-8X7B-INSTRUCT-V0.1": "MOE_Mixtral8x7B"}

    for idx, row in enumerate(df.itertuples()):
        llm = row.model_hf_repo
        gpu = row.hardware
        tp = row.hardware_count
        isl = row.prompt_tokens
        osl = row.output_tokens
        model_exists = False
        for model in data["models"]:
            if model["GPU"] == gpu and model["id"] == llm and model["tensor_parallelism"] == tp and row.framework_version in vllm_versions:
                model_exists = True
                break
        if model_exists:
            print(llm, gpu, tp, isl, osl)
            aiconfigurator_config["exps"].append(f"exp_{exp_count}")
            llm_name = llm.split("/")[1].upper().rstrip("-INSTRUCT").replace("-", "_")
            if llm_name in llm_mapping:
                llm_name = llm_mapping[llm_name]
            aiconfigurator_config[f"exp_{exp_count}"] = {
                "mode": "patch",
                "serving_mode": "agg",
                "model_name": llm_name,
                "total_gpus": tp,
                "system_name": gpu_mapping[gpu],
                "backend_name": "vllm",
                "backend_version": "0.11.0",
                "isl": isl,
                "osl": osl,
                "prefix": 0,
                "config": {
                    "worker_config": {
                        "tp_list": [tp]
                    }
                }
            }
            df.loc[idx, "aiconfig_exp"] = f"exp_{exp_count}"
            exp_count += 1
        else:
            df.loc[idx, "aiconfig_exp"] = "N/A"

    with open("exp-aiconfig.yaml", "w+") as f:
        yaml.safe_dump(aiconfigurator_config, f)
    df.to_excel("blis_rh_final_test_unsaturated.xlsx", index=False)

def test_aiconfig():
    df_real = pd.read_excel("blis_rh_final_test_unsaturated.xlsx")
    df_aiconfig = pd.read_excel("aiconfig_results.xlsx")
    merged_df = pd.merge(df_real, df_aiconfig, on='aiconfig_exp', how='left')

    mask = merged_df['aiconfig_exp'] == 'N/A'
    merged_df.loc[mask, :] = np.nan
    merged_df["ttft MAPE"] = abs(merged_df[f"ttft_mean"] - merged_df["ttft"])/merged_df["ttft_mean"] * 100
    merged_df["itl MAPE"] = abs(merged_df[f"itl_mean"] - merged_df["itl"])/merged_df["itl_mean"] * 100
    merged_df["e2e MAPE"] = abs(merged_df[f"e2e_mean"] - merged_df["e2e"])/merged_df["e2e_mean"] * 100
    plot_vllm_vs_sim(merged_df, "aiconfig_LLM=* TP=* GPU=* vllmVersion=*")

# generate_config()
# now go and run aiconfigurator on the generated config exp-aiconfig.yaml
test_aiconfig()


