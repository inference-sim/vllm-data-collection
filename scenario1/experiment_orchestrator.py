from datetime import datetime
import os
import requests
import subprocess
import time
import yaml
from kubernetes import config
from kubernetes.client import Configuration
from kubernetes.client.api import core_v1_api
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream
from kr8s.objects import Pod

NAMESPACE = "blis"

def start_vllm_server_client(benchmark_config, exp_folder, k_client, mode, model):
    """Start vLLM server with config parameters"""
    vllm_params = benchmark_config['vllm']

    server_args = [model,
        '--gpu-memory-utilization', str(vllm_params['gpu_memory_utilization']),
        '--block-size', str(vllm_params['block_size']),
        '--max-model-len', str(vllm_params['max_model_len']),
        '--max-num-batched-tokens', str(vllm_params['max_num_batched_tokens']),
        '--max-num-seqs', str(vllm_params['max_num_seqs']),
        '--long-prefill-token-threshold', str(vllm_params['long_prefill_token_threshold']),
        '--seed', str(vllm_params['seed'])
    ]

    if vllm_params.get('enable_prefix_caching'):
        server_args.append('--enable-prefix-caching')

    server_args.append('--disable-log-requests')

    client_args = [
        f"""set -ex
apt-get update && apt-get install -y git curl
git clone -b scenario1_enhancements https://github.com/inference-sim/vllm-data-collection
cd vllm-data-collection/scenario1
pip install -r requirements.txt
python generate_prompts_fixedlen.py --model {model} --mode {mode}
python scenario1_client.py --model {model} --mode {mode} --results_folder {exp_folder}
sleep 30000000
        """
    ]

    print(f"Starting vLLM server: vllm serve {' '.join(server_args)}")

    config.load_kube_config()

    model_name_for_pod = model.split("/")[-1].replace(".", "-").lower()

    pod_name = f"vllm-benchmark-collection-scenario1-{model_name_for_pod}-{mode}"

    # Create a pod manifest for vllm
    pod_manifest = {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name': pod_name,
            },
            'spec': {
                'affinity': {
                    'nodeAffinity': {
                        'requiredDuringSchedulingIgnoredDuringExecution': {
                            'nodeSelectorTerms': [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "nvidia.com/gpu.product",
                                            "operator": "In",
                                            "values": [
                                                vllm_params['gpu_type']      # land pod on the node with this GPU
                                            ]
                                        },
                                        {
                                            "key": "nvidia.com/gpu.memory",
                                            "operator": "Gt",
                                            "values": [
                                                str(vllm_params['gpu_memory_min']) # land pod on the node at least this much GPU memory (in MB)
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                },
                # Temporary for now until cluster is fixed
                'securityContext': {
                    'runAsUser': 0
                },
                'initContainers': [
                    {
                        'name': 'vllm-server',
                        'image': "vllm/vllm-openai:v0.10.0",
                        'command': ['vllm', 'serve'],
                        'restartPolicy': 'Always',
                        'args': server_args,
                        'env': [
                            {
                                "name": "HF_TOKEN",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "key": "HF_TOKEN",
                                        "name": "hf-secret"
                                        }
                                    }
                            },
                            {
                                "name": "HF_HOME",
                                "value": "/mnt/.cache/huggingface/hub"
                            }
                        ],

                        # Got this info from node
                        'resources': {
                            'requests': {
                                'memory': '50Gi',
                                'cpu': '1',
                                'nvidia.com/gpu': '1',
                            },
                            'limits': {
                                'memory': '50Gi',
                                'cpu': '8',
                                'nvidia.com/gpu': '1',
                            }
                        },
                        'startupProbe': {
                            "httpGet": {
                                "path": "/v1/models",
                                "port": 8000,
                            },

                            # Max 5 minutes (50 * 10) to finish startup
                            "failureThreshold": 50,
                            "periodSeconds": 10,
                        }
                    }
                ],
                'containers': [
                    {
                        'name': 'vllm-client',
                        'image': "python:3.11-slim",
                        'command': ['bash', '-c'],
                        'args': client_args,
                        'env': [
                            {
                                "name": "HF_TOKEN",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "key": "HF_TOKEN",
                                        "name": "hf-secret"
                                        }
                                    }
                            }
                        ],
                    }
                ],
                'volumes': [
                    {
                        'name': 'model-storage',
                        'persistentVolumeClaim': {
                            'claimName': 'blis-pvc'
                        }
                    }
                ],
            }
        }
    
    # Apply the pod manifest
    resp = k_client.create_namespaced_pod(body=pod_manifest,
                                            namespace=NAMESPACE)      # edit ns if needed

    return pod_name

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

def collect_pod_logs_metrics(k_client, benchmark_name, pod_name, mode):
    """Stop vLLM server process and returns vllm pod log file and metrics json file"""

    # Get pod logs
    pod_log_filename = f"vllm_server_{benchmark_name}_pod_{mode}.log"
    with open(pod_log_filename, 'w') as log_file:
        pod = Pod.get(pod_name, namespace=NAMESPACE)
        log_file.write("\n".join(pod.logs()))

    # Get metrics
    # Metrics is plain text format
    metrics_filename = f"vllm_server_{benchmark_name}_metrics_{mode}.txt"
    with open(metrics_filename, 'w') as metrics_file:
        url = "http://localhost:8000/metrics"
        try:
            response = requests.get(url, timeout=10)
            metrics_file.write(str(response.text))

        except Exception as e:
            print(f"Cannot get metrics response, {str(e)}")

    return pod_log_filename, metrics_filename

def copy_file_from_pod_using_kubectl_cp(pod_name, namespace, container_name, remote_file_path, local_file_path):
    try:
        command = [
            'oc', 'cp', 
            f'{namespace}/{pod_name}:{remote_file_path}', 
            '-c', container_name,
            local_file_path,
        ]
        print(command)
        result = subprocess.run(command, capture_output=True, text=True)
        print(result)

        if result.returncode == 0:
            print(f"File '{remote_file_path}' copied to '{local_file_path}' successfully.")
            return True
        else:
            print(f"Error copying file: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error executing oc cp command: {e}")
        return False
    
def stop_vllm_server(k_client, pod_name):
    """Stop vLLM server process and returns vllm pod log file and metrics json file"""

    print("Stopping vLLM server...")

    # Delete pod
    try:
        api_response = k_client.delete_namespaced_pod(pod_name, NAMESPACE, grace_period_seconds=0)
        print(f"vllm pod {pod_name} has been deleted: api response {api_response}")

    except ApiException as e:
        print("Exception when deleting vllm pod: %s\n" % e)

    print("Server stopped")

def run_experiment(model, mode, dir_name: str):
    config_file = f"scenario1_config_{mode}.yaml"

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
    core_v1 = core_v1_api.CoreV1Api()

    # Run the workload
    print(f"\n{'='*50}")
    print(f"STARTING SCENARIO 1 benchmark in mode = {mode}")
    print(f"{'='*50}")

    benchmark_name = "scenario1"
    exp_folder = f"{time.strftime("%Y%m%d-%H%M%S")}_{benchmark_name}"

    # Start server
    pod_name = start_vllm_server_client(full_config, exp_folder, core_v1, mode, model)
    print(f"Created pod '{pod_name}'")

    while True:
        remote_file_path = f"/mnt/{exp_folder}/results/scenario1_output_{mode}.json"
        local_file_path = f"./{dir_name}/results_{mode}.json"

        command = ["sh", "-c", f"test -f {remote_file_path} && echo 'EXISTS' || echo 'NOT_EXISTS'"]

        resp = None
        # Execute the command
        try:
            resp = stream(core_v1.connect_get_namespaced_pod_exec,
                name=pod_name,
                namespace=NAMESPACE,
                container="vllm-client", # Specify if needed
                command=command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
            )
        except KeyboardInterrupt:
            break
        except ApiException:
            continue
        
        if resp and "NOT" not in resp:
            print(f"File '{remote_file_path}' exists in container vllm-client of pod '{pod_name}'.")
            time.sleep(20)
            copy_file_from_pod_using_kubectl_cp(pod_name, NAMESPACE, "vllm-client", remote_file_path, local_file_path)
            break
        time.sleep(10)

    stop_vllm_server(core_v1, pod_name)

def main():
    modes = ["train", "test"]
    # models = ["facebook/opt-125m", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2-1.5B", "Qwen/Qwen2-7B", "Qwen/Qwen3-14B", "mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b", "meta-llama/Llama-3.1-8B","ibm-granite/granite-3.3-8b-instruct", "mistralai/Mistral-Small-24B-Instruct-2501"]
    # models = ["ibm-granite/granite-3.3-8b-instruct", "mistralai/Mistral-Small-24B-Instruct-2501"]
    # models = ["Qwen/Qwen2.5-3B"]
    models = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2-7B", "Qwen/Qwen3-14B", "mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b", "meta-llama/Llama-3.1-8B","ibm-granite/granite-3.3-8b-instruct", "mistralai/Mistral-Small-24B-Instruct-2501", "Qwen/Qwen3-32B"]

    # models = ["facebook/opt-125m"]
    for model in models:
        for mode in modes:
            dir_name = "results/" + model.split("/")[-1].replace(".", "_")
            os.makedirs(dir_name, exist_ok=True)
            run_experiment(model, mode, dir_name)

if __name__=="__main__":
    main()