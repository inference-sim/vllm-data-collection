import requests
import time
import yaml
from kubernetes import config
from kubernetes.client import Configuration
from jinja2 import Environment, FileSystemLoader
from kr8s.objects import Job

from experiment_configs_constants import *

def start_vllm_server_client(benchmark_config, exp_folder, mode, model, chunk_size):
    """Start vLLM server with config parameters"""
    vllm_params = benchmark_config['vllm']

    model_alias = model.split("/")[-1].replace(".", "_")
    exp_results_path = f"/mnt/scenario3/results/{model_alias}/{exp_folder}/chunk_size_{chunk_size}"
    server_log_path = f"{exp_results_path}/scenario3_server_{mode}.log"
    client_log_path = f"{exp_results_path}/scenario3_client_{mode}.log"

    server_args = f"""              mkdir -p {exp_results_path}/
                touch {server_log_path}
                python3 -m vllm.entrypoints.openai.api_server --model {model} \\
                --gpu-memory-utilization {str(vllm_params['gpu_memory_utilization'])} \\
                --block-size {str(vllm_params['block_size'])} --max-model-len {str(vllm_params['max_model_len'])} \\
                --max-num-batched-tokens {str(vllm_params['max_num_batched_tokens'])} --max-num-seqs {str(vllm_params['max_num_seqs'])} \\
                --long-prefill-token-threshold {str(vllm_params['long_prefill_token_threshold'])} --seed {str(vllm_params['seed'])} \\
                --disable-log-requests > {server_log_path}
"""

    client_args = f"""              set -ex
                apt-get update && apt-get install -y git curl
                git clone -b scenario3 https://github.com/inference-sim/vllm-data-collection
                pip install -r vllm-data-collection/requirements.txt
                cd vllm-data-collection/scenario3
                touch {client_log_path}
                python generate_client_config.py --model {model}
                sleep 15
                python scenario3_client.py --model {model} --mode {mode} --chunk_size {chunk_size} --results_folder {exp_folder} > {client_log_path}
                sleep 30000000
"""

    print(f"Starting vLLM server: {server_args}")

    config.load_kube_config()

    model_name_for_pod = model.split("/")[-1].replace(".", "-").lower()

    job_name = f"scenario3-{mode}-{model_name_for_pod}-{chunk_size}"
    environment = Environment(loader=FileSystemLoader("../"))
    template = environment.get_template("benchmark-job.yaml")

    rendered = template.render(server_args=server_args,
                            client_args=client_args,
                            job_name=job_name)
    print(rendered)
    resource_dict = yaml.safe_load(rendered)
    job = Job(resource_dict)
    job.create()

    return job_name

def wait_for_server(model: str):
    """Wait for vLLM server to be ready"""
    url = "http://localhost:8000/v1/models"
    max_attempts = 50  # 60 seconds total

    print("Waiting for server to be ready...")
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=5)
            if model in str(response.json()):
                print("Server is ready")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(5)

    print("Server failed to start within timeout")
    return False

def run_experiment(model, mode, remote_exp_folder: str):
    server_config_file = f"scenario3_server_config_{mode}.yaml"

    with open(server_config_file, "r") as f:
       server_config = yaml.safe_load(f) # read necessary configs and seed files

    # Set up K8s client
    config.load_kube_config()

    try:
        c = Configuration().get_default_copy()
        c.verify_ssl = False
    except AttributeError:
        c = Configuration()
        c.assert_hostname = False
        c.verify_ssl = False
    Configuration.set_default(c)

    # Run the workload
    print(f"\n{'='*50}")
    print(f"STARTING SCENARIO 3 benchmark in mode = {mode}")
    print(f"{'='*50}")

    for experiment in server_config["experiments"]:
        chunk_size = experiment["chunk_size"]
        job_name = start_vllm_server_client(experiment, remote_exp_folder, mode, model, chunk_size)
        print(f"Created pod '{job_name}'")

def main():
    num_runs = 1
    models = ["google/gemma-7b"]
    # models = ["Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b", "meta-llama/Llama-3.1-8B","ibm-granite/granite-3.3-8b-instruct", "Qwen/Qwen3-14B", "mistralai/Mistral-Small-24B-Instruct-2501", "Qwen/Qwen3-32B"]

    for run in range(num_runs):
        for idx, model in enumerate(models):
            benchmark_name = "scenario3"
            remote_exp_folder = f"{time.strftime('%Y%m%d-%H%M%S')}_{benchmark_name}"
            for mode in MODES:
                run_experiment(model, mode, remote_exp_folder)
            if idx % 2 == 1:
                time.sleep(2400)
        

if __name__=="__main__":
    main()