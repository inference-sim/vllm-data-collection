import requests
import time
import yaml
from kubernetes import config
from kubernetes.client import Configuration
from jinja2 import Environment, FileSystemLoader
from kr8s.objects import Job

import argparse

def start_vllm_server_client(exp_folder, mode, model, request_rate, specs, max_num_batched_tokens):
    """Start vLLM server with config parameters"""
    model_alias = model.split("/")[-1].replace(".", "_")
    spec_type = specs["TYPE"].lower()
    exp_results_path = f"/mnt/scenario4/results/{model_alias}/{mode}/{spec_type}/mbnt_{max_num_batched_tokens}/rr_{request_rate}/{exp_folder}/"
    server_log_path = f"{exp_results_path}/scenario4_server_{mode}.log"
    client_log_path = f"{exp_results_path}/scenario4_client_{mode}.log"
    input_len_mean = specs["INPUT_LEN_MEAN"]
    output_len_mean = specs["OUTPUT_LEN_MEAN"]
    num_prefixes = specs["NUM_PREFIXES"]
    prefix_hit_ratio_mean = specs["PREFIX_HIT_RATIO_MEAN"]

    server_args = f"""              mkdir -p {exp_results_path}/
                touch {server_log_path}
                git clone -b inst-benchmark-prefix https://github.com/toslali-ibm/vllm.git
                cp /usr/local/lib/python3.12/dist-packages/vllm/*.so /vllm-workspace/vllm/vllm/
                cp -r /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn /vllm-workspace/vllm/vllm/
                PYTHONPATH=/vllm-workspace/vllm METRICS_FILENAME=/mnt2/tmp/metrics_{mode} FLUSH_INTERVAL_STEPS=1000 python3 -m vllm.entrypoints.openai.api_server \\
                --model {model} --gpu-memory-utilization {str(GPU_MEM_UTIL)} \\
                --max-num-batched-tokens {max_num_batched_tokens} \\
                --seed {SEED} --disable-log-requests > {server_log_path} 2>&1 &
                VLLM_PID=$!
                echo "Started vllm server with PID $VLLM_PID"
                echo $VLLM_PID > /mnt2/tmp/vllm.pid
                sleep infinity
"""

#     client_args = f"""              set -ex
#                 git clone -b inst-benchmark-prefix https://github.com/toslali-ibm/vllm.git
#                 cp /usr/local/lib/python3.12/dist-packages/vllm/*.so /vllm-workspace/vllm/vllm/
#                 cp -r /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn /vllm-workspace/vllm/vllm/
#                 touch {client_log_path}
#                 PYTHONPATH=/vllm-workspace/vllm python3 vllm/benchmarks/benchmark_serving_scenario4.py --num-prompts {NUM_PROMPTS} \\
#                 --dataset-path {DATASET_PATH} --model {model} --save-result --save-detailed --seed {SEED} --experiment-mode {mode} \\
#                 --sharegpt-input-len-min {input_len_min} --sharegpt-input-len-max {input_len_max} --sharegpt-output-len-min {output_len_min} \\
#                 --sharegpt-output-len-max {output_len_max} --request-rate {request_rate} --result-filename detailed_results_{mode}.json \\
#                 --sharegpt-prefix-hit-ratio {prefix_hit_ratio} --result-dir {exp_results_path} > {client_log_path}
#                 sleep 30000000
# """
    if DATASET_NAME == "random":
        client_args = f"""              set -ex
                    git clone -b inst-benchmark-prefix https://github.com/toslali-ibm/vllm.git
                    cp /usr/local/lib/python3.12/dist-packages/vllm/*.so /vllm-workspace/vllm/vllm/
                    cp -r /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn /vllm-workspace/vllm/vllm/
                    touch {client_log_path}
                    PYTHONPATH=/vllm-workspace/vllm python3 vllm/benchmarks/benchmark_serving_scenario4.py \\
                    --num-prompts {NUM_PROMPTS} --model {model} --save-result --save-detailed --seed {SEED} \\
                    --dataset-name prefix_repetition_with_random_lengths \\
                    --prefix-repetition-random-len-input-len-mean {input_len_mean} --prefix-repetition-random-len-input-len-std {input_len_mean//2} \\
                    --prefix-repetition-random-len-output-len-mean {output_len_mean} --prefix-repetition-random-len-output-len-std {output_len_mean//2} \\
                    --prefix-repetition-random-len-prefix-hit-ratio-mean {prefix_hit_ratio_mean} --prefix-repetition-random-len-prefix-hit-ratio-std {prefix_hit_ratio_mean//2} \\
                    --prefix-repetition-random-len-num-prefixes {num_prefixes} --request-rate {request_rate} --result-filename detailed_results_{mode}.json \\
                    --result-dir {exp_results_path} > {client_log_path}
                    sleep 30000000
"""
    print(f"Starting vLLM server: {server_args}")

    config.load_kube_config()

    model_name_for_pod = model.split("/")[-1].replace(".", "-").lower()
    mode_name_for_pod = mode.replace("_", "-").lower()
    spec_type = specs["TYPE"].lower()

    job_name = f"scenario4-{mode_name_for_pod}-{model_name_for_pod}-{spec_type}-{max_num_batched_tokens}-{request_rate}"
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

def run_experiment(model, mode, spec, remote_exp_folder: str):
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
    spec_constants_mapping = {"LL": LL_SPECS, "LH": LH_SPECS, "HL": HL_SPECS, "HH": HH_SPECS}

    for mbnt in MAX_NUM_BATCHED_TOKENS:
        for request_rate in REQUEST_RATES[spec]:
            job_name = start_vllm_server_client(remote_exp_folder, mode, model, request_rate, spec_constants_mapping[spec], mbnt)
            print(f"Created pod '{job_name}'")

def run(mode, spec):
    for idx, model in enumerate(MODELS):
        benchmark_name = "scenario4"
        remote_exp_folder = f"{time.strftime('%Y%m%d-%H%M%S')}_{mode}_{benchmark_name}"
        run_experiment(model, mode, spec, remote_exp_folder)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="A script that requires a mode to be specified.")
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help="Specify the mode of operation (e.g., 'train', 'val', 'test').")
    parser.add_argument('-s', '--spec', type=str, required=True,
                        help="Specify the spec (e.g., 'LL', 'LH', 'HL', 'HH').")

    args = parser.parse_args()
    if args.mode == "train":
        from experiment_configs_constants_train import *
    # elif args.mode == "val":
    #     from experiment_configs_constants_val import *
    elif args.mode == "test":
        from experiment_configs_constants_test import *
    run(args.mode, args.spec)