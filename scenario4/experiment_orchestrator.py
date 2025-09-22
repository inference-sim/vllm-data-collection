import requests
import time
import yaml
from kubernetes import config
from kubernetes.client import Configuration
from jinja2 import Environment, FileSystemLoader
from kr8s.objects import Job

from experiment_configs_constants import *

def start_vllm_server_client(exp_folder, mode, model, chunk_size, request_rate, specs):
    """Start vLLM server with config parameters"""
    model_alias = model.split("/")[-1].replace(".", "_")
    spec_type = specs["TYPE"].lower()
    exp_results_path = f"/mnt/scenario4/results/{model_alias}/{spec_type}/chunk_size_{chunk_size}/rr_{request_rate}/{exp_folder}/"
    server_log_path = f"{exp_results_path}/scenario4_server_{mode}.log"
    client_log_path = f"{exp_results_path}/scenario4_client_{mode}.log"
    input_len_min = specs["INPUT_LEN_MIN"]
    input_len_max = specs["INPUT_LEN_MAX"]
    output_len_min = specs["OUTPUT_LEN_MIN"]
    output_len_max = specs["OUTPUT_LEN_MAX"]

    server_args = f"""              mkdir -p {exp_results_path}/
                touch {server_log_path}
                git clone -b inst-benchmark https://github.com/toslali-ibm/vllm.git
                cp /usr/local/lib/python3.12/dist-packages/vllm/*.so /vllm-workspace/vllm/vllm/
                cp -r /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn /vllm-workspace/vllm/vllm/
                PYTHONPATH=/vllm-workspace/vllm METRICS_FILENAME=/mnt2/tmp/metrics_{mode} FLUSH_INTERVAL_STEPS=1000 python3 -m vllm.entrypoints.openai.api_server \\
                --model {model} --gpu-memory-utilization {str(GPU_MEM_UTIL)} \\
                --block-size {str(BLOCK_SIZE)} --max-model-len {str(CONTEXT_LENGTH)} \\
                --max-num-batched-tokens {str(MAX_NUM_BATCHED_TOKENS)} --max-num-seqs {str(MAX_NUM_SEQS)} \\
                --long-prefill-token-threshold {str(chunk_size)} --seed {str(SEED)} \\
                --disable-log-requests > {server_log_path}
"""

    client_args = f"""              set -ex
                git clone -b inst-benchmark https://github.com/toslali-ibm/vllm.git
                cp /usr/local/lib/python3.12/dist-packages/vllm/*.so /vllm-workspace/vllm/vllm/
                cp -r /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn /vllm-workspace/vllm/vllm/
                touch {client_log_path}
                PYTHONPATH=/vllm-workspace/vllm python3 vllm/benchmarks/benchmark_serving_scenario4.py --num-prompts {NUM_PROMPTS} \\
                --dataset-path {DATASET_PATH} --model {model} --save-result --save-detailed --seed {SEED} --experiment-mode {mode} \\
                --sharegpt-input-len-min {input_len_min} --sharegpt-input-len-max {input_len_max} --sharegpt-output-len-min {output_len_min} \\
                --sharegpt-output-len-max {output_len_max} --request-rate {request_rate} --result-filename detailed_results_{mode}.json --result-dir {exp_results_path} > {client_log_path}
                sleep 30000000
"""

    print(f"Starting vLLM server: {server_args}")

    config.load_kube_config()

    model_name_for_pod = model.split("/")[-1].replace(".", "-").lower()
    spec_type = specs["TYPE"].lower()

    job_name = f"scenario4-{mode}-{model_name_for_pod}-{spec_type}-{chunk_size}-{request_rate}"
    environment = Environment(loader=FileSystemLoader("../"))
    template = environment.get_template("benchmark-job.yaml")

    rendered = template.render(server_args=server_args,
                            client_args=client_args,
                            job_name=job_name,
                            exp_results_path=exp_results_path)
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
    print(f"STARTING SCENARIO 4 benchmark in mode = {mode}")
    print(f"{'='*50}")

    for chunk_size in CHUNK_SIZES:
        for request_rate in REQUEST_RATES:
            job_name = start_vllm_server_client(remote_exp_folder, mode, model, chunk_size, request_rate, LH_SPECS)
            print(f"Created pod '{job_name}'")

def main():
    for idx, model in enumerate(MODELS):
        benchmark_name = "scenario4"
        remote_exp_folder = f"{time.strftime('%Y%m%d-%H%M%S')}_{benchmark_name}"
        for mode in MODES:
            run_experiment(model, mode, remote_exp_folder)
        # if idx % 2 == 1:
        #     time.sleep(2400)
        

if __name__=="__main__":
    main()