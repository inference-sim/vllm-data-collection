import pandas as pd
import yaml

specs_filename = "training_specs.csv"

df = pd.read_excel("blis_rh_final.xlsx")
with open("coefficients.yaml", "r+") as f:
    data = yaml.safe_load(f)
models = data["models"]
models_to_train = []

df_grouped = df.groupby(["model_hf_repo", "hardware_count", "hardware", "docker_image"]).count()

for group in df_grouped.index:
    exists = False
    for model in models:
        if group[0] == model["id"] and group[1] == model["tensor_parallelism"] and group[2] == model["GPU"] and group[3] == model["vllm_version"]:
            exists = True
            # models_to_train.append({"LLM_name": group[0],"tp": group[1], "GPU": group[2], "vllm_version": group[3]})
            break
    if not exists:
        df_filtered = df[(df["model_hf_repo"] == group[0]) & (df["hardware_count"] == group[1]) & (df["hardware"] == group[2]) & (df["docker_image"] == group[3])]
        train_counts = df_filtered[(df_filtered["train_test"]=="train") & (df_filtered["saturated"] == False)].shape[0]
        test_counts = df_filtered[(df_filtered["train_test"]=="test") & (df_filtered["saturated"] == False)].shape[0]
        if train_counts >= 4 and test_counts >= 2:
            models_to_train.append({"LLM_name": group[0],"tp": group[1], "GPU": group[2], "vllm_version": group[3]})
        else:
            print("Insufficient data for: ", group)

specs_df = pd.DataFrame(models_to_train)
specs_df.to_csv(specs_filename, index=False)

