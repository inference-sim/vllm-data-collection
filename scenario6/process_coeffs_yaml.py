import pandas as pd
import yaml
from collections import defaultdict

df = pd.read_excel("blis_rh_final.xlsx")
specs = pd.read_csv("old_training_specs.csv")
with open("coefficients.yaml", "r+") as f:
    data = yaml.safe_load(f)
data["defaults"] = {}
models = data["models"]
max_count = defaultdict(lambda: 0)
for spec in specs.itertuples():
    df_filtered = df[(df["model_hf_repo"] == spec.LLM_name) & (df["hardware_count"] == int(spec.tp)) & (df["hardware"] == spec.GPU) & (df["docker_image"] == spec.vllm_version)]
    if df_filtered.shape[0] > max_count[spec.LLM_name]:
        max_count[spec.LLM_name] = df_filtered.shape[0]
        data["defaults"][spec.LLM_name] = {
            "GPU": spec.GPU,
            "tensor_parallelism": int(spec.tp),
            "vllm_version": spec.vllm_version,
            "max_count": max_count[spec.LLM_name],
        }
for default in data["defaults"]:
    data["defaults"][default].pop("max_count")

with open("coefficients.yaml", "w+") as f:
    yaml.safe_dump(data, f)
