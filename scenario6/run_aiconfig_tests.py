import re
import subprocess
import yaml
from time import sleep
import pandas as pd
from aiconfigurator.sdk.utils import HuggingFaceDownloadError

def get_metric(name, pattern, text):
    """Helper to extract float values via regex or raise error."""
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Error: The metric '{name}' was not found in the output.")
    return float(match.group(1))

with open("exp-aiconfig.yaml", "r+") as f:
    data = yaml.safe_load(f)

num_exps = len(data) - 1

results = {"aiconfig_exp": [], "ttft": [], "itl": [], "e2e": []}

for id in range(1, num_exps):
    exp_id = f"exp_{id}"
    data["exps"] = [exp_id]

    with open("exp-aiconfig.yaml", "w+") as f:
        yaml.safe_dump(data, f)

    sleep(0.5)

    command = [
        "aiconfigurator", 
        "cli", 
        "exp", 
        "--yaml_path", 
        "exp-aiconfig.yaml"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stderr)
    
        ttft = get_metric("TTFT", r"TTFT:\s*([\d\.]+)", result.stderr)
        tpot = get_metric("TPOT", r"TPOT:\s*([\d\.]+)", result.stderr)

        print(f"TTFT: {ttft}")
        print(f"TPOT: {tpot}")
        print(f"OSL: {data[exp_id]["osl"]}")
        print(f"E2E: {ttft + tpot * data[exp_id]["osl"]}")
        results["aiconfig_exp"].append(exp_id)
        results["ttft"].append(ttft)
        results["itl"].append(tpot) # because we don't know itl
        results["e2e"].append(ttft + tpot * data[exp_id]["osl"])
    except HuggingFaceDownloadError as e:
        pass
    except Exception as e:
        pass

df = pd.DataFrame(results)
df.to_excel("aiconfig_results.xlsx", index=False)