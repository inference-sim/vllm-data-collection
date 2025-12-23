import pandas as pd
import subprocess
import yaml
import re
import matplotlib.pyplot as plt
import os

METRICS_TO_COMPARE = ["ttft", "itl", "e2e"]

def plot_vllm_vs_sim(errors, plot_title=""):
    overall_errors = {}
    for metric in METRICS_TO_COMPARE:
        metric_error = f"{metric}_MAPE"
        overall_errors[metric_error] = errors[metric_error].mean()
    plt.figure(figsize=(10, 6))
    colors = ['orange', 'red', 'green', 'blue', 'purple', 'yellow']
    plt.bar(list(overall_errors.keys()), list(overall_errors.values()), label=list(overall_errors.keys()), color=colors)
    
    plt.title(f'MAPE error - vllm vs sim - {plot_title}')
    plt.xlabel("Metrics")
    plt.ylabel("Error %")
    plt.legend()
    plots_folder = f"test_plots/bentoml"
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(f'{plots_folder}/{plot_title}_error.png')

    for metric in METRICS_TO_COMPARE:
        plt.figure(figsize=(10, 6))
        plt.plot(errors[f"{metric}_real"], color = "blue")
        plt.plot(errors[f"{metric}_pred"], color = "red")
        plt.title(f'Real vs BentoML - {metric.upper()}')
        plt.xlabel("Exp idx")
        plt.ylabel("Error %")
        plt.legend(["Real", "Predicted - BentoML"])
        plt.savefig(f'{plots_folder}/bentoml_{metric.upper()}.png')
    
    plt.title(f'MAPE error - vllm vs sim - {plot_title}')
    plt.xlabel("Metrics")
    plt.ylabel("Error %")
    plt.legend()
    plots_folder = f"test_plots"
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(f'{plots_folder}/{plot_title}_error.png')

def test_bento():
    df = pd.read_excel("blis_rh_final.xlsx")
    with open("all_coefficients.yaml", "r+") as f:
        data = yaml.safe_load(f)
    df = df[(df["train_test"] == "test") & (df["saturated"] == False)]
    df = df.groupby(["model_hf_repo", "hardware", "hardware_count", "prompt_tokens", "output_tokens"]).first().reset_index() # take only the first request rate for a group
    errors = {"ttft_real": [], "itl_real": [], "e2e_real": [],
              "ttft_pred": [], "itl_pred": [], "e2e_pred": [],
              "ttft_MAPE": [], "itl_MAPE": [], "e2e_MAPE": []}

    for idx, row in enumerate(df.itertuples()):
        llm = row.model_hf_repo
        gpu = row.hardware
        tp = row.hardware_count
        isl = row.prompt_tokens
        osl = row.output_tokens
        model_exists = False
        for model in data["models"]:
            if model["GPU"] == gpu and model["id"] == llm and model["tensor_parallelism"] == tp:
                model_exists = True
                break
        if model_exists:
            print(llm, gpu, tp, isl, osl)
            cmd = [
                "llm-optimizer", 
                "estimate", 
            ]
            cmd.extend(["--model", llm])
            cmd.extend(["--gpu", gpu.split("-80")[0]]) # A100-80 -> A100
            cmd.extend(["--input-len", str(isl)])
            cmd.extend(["--output-len", str(osl)])
            cmd.extend(["--num-gpus", str(tp)])
            cmd_str = " ".join(cmd)
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
            
                ttft = get_metric("TTFT", r"TTFT:\s*([\d\.]+)", result.stdout)
                itl = get_metric("ITL", r"ITL:\s*([\d\.]+)", result.stdout)
                e2e = get_metric("E2E", r"E2E:\s*([\d\.]+)", result.stdout) * 1e3

                print(f"TTFT: {ttft}")
                print(f"ITL: {itl}")
                print(f"E2E: {e2e}")
                errors["ttft_real"].append(row.ttft_mean)
                errors["itl_real"].append(row.itl_mean)
                errors["e2e_real"].append(row.e2e_mean)
                errors["ttft_pred"].append(ttft)
                errors["itl_pred"].append(itl)
                errors["e2e_pred"].append(e2e)
                errors["ttft_MAPE"].append(abs(ttft - row.ttft_mean)/row.ttft_mean * 100)
                errors["itl_MAPE"].append(abs(itl - row.itl_mean)/row.itl_mean * 100)
                errors["e2e_MAPE"].append(abs(e2e - row.e2e_mean)/row.e2e_mean * 100)
            except Exception as e:
                print("Failed to run BentoML", cmd_str, e)
                pass
    errors = pd.DataFrame(errors)
    print(errors)
    plot_vllm_vs_sim(errors, "bentoml_LLM=* TP=* GPU=* vllmVersion=*")

def get_metric(name, pattern, text):
    """Helper to extract float values via regex or raise error."""
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Error: The metric '{name}' was not found in the output.")
    return float(match.group(1))

test_bento()


