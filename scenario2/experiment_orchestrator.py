import requests
import time
import yaml
from kubernetes import config
from kubernetes.client import Configuration
from jinja2 import Environment, FileSystemLoader
from kr8s.objects import Job

NAMESPACE = "blis"

def start_vllm_server_client(benchmark_config, exp_folder, mode, model, chunk_size):
    """Start vLLM server with config parameters"""
    vllm_params = benchmark_config['vllm']

    model_alias = model.split("/")[-1].replace(".", "_")
    exp_results_path = f"/mnt/scenario2/results/{model_alias}/{exp_folder}/chunk_size_{chunk_size}"
    server_log_path = f"{exp_results_path}/scenario2_server_{mode}.log"
    client_log_path = f"{exp_results_path}/scenario2_client_{mode}.log"

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
                git clone -b scenario2 https://github.com/inference-sim/vllm-data-collection
                pip install -r vllm-data-collection/requirements.txt
                cd vllm-data-collection/scenario2
                touch {client_log_path}
                python scenario2_client.py --model {model} --mode {mode} --chunk_size {chunk_size} --results_folder {exp_folder} > {client_log_path}
                sleep 30000000
"""

    print(f"Starting vLLM server: {server_args}")

    config.load_kube_config()

    model_name_for_pod = model.split("/")[-1].replace(".", "-").lower()

    job_name = f"scenario2-{mode}-{model_name_for_pod}-{chunk_size}"
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
    config_file = f"scenario2_config_{mode}.yaml"

    with open(config_file, "r") as f:
       full_config = yaml.safe_load(f) # read necessary configs and seed files

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
    print(f"STARTING SCENARIO 1 benchmark in mode = {mode}")
    print(f"{'='*50}")

    experiment_configs = full_config["experiments"]
    for experiment in experiment_configs:
        chunk_size = experiment["chunk_size"]
        job_name = start_vllm_server_client(experiment, remote_exp_folder, mode, model, chunk_size)
        print(f"Created pod '{job_name}'")

def main():
    modes = ["train", "test"]
    # models = ["mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b", "meta-llama/Llama-3.1-8B","ibm-granite/granite-3.3-8b-instruct", "mistralai/Mistral-Small-24B-Instruct-2501", "Qwen/Qwen3-32B"]
    # models = ["ibm-granite/granite-3.3-8b-instruct", "mistralai/Mistral-Small-24B-Instruct-2501"]
    # models = ["mistralai/Mistral-Small-24B-Instruct-2501"]
    num_runs = 2
    models = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b", "meta-llama/Llama-3.1-8B","ibm-granite/granite-3.3-8b-instruct", "Qwen/Qwen3-14B", "mistralai/Mistral-Small-24B-Instruct-2501", "Qwen/Qwen3-32B"]

    for run in range(num_runs):
        for idx, model in enumerate(models):
            benchmark_name = "scenario2"
            remote_exp_folder = f"{time.strftime('%Y%m%d-%H%M%S')}_{benchmark_name}"
            for mode in modes:
                run_experiment(model, mode, remote_exp_folder)
            if idx < 4:
                time.sleep(360)
            elif idx >=4 and idx <=7:
                time.sleep(600)
            else:
                time.sleep(1200)
        

if __name__=="__main__":
    main()