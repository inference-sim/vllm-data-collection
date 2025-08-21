import argparse
from datetime import datetime
import os
import requests
import subprocess
import time
import yaml
from pathlib import Path
from kubernetes import config
from kubernetes.client import Configuration
from kubernetes.client.api import core_v1_api
import kubernetes.client as client
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream
from kr8s.objects import Pod

def start_vllm_server_client(benchmark_config, benchmark_name, k_client, mode, model):
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

    if vllm_params.get('disable_log_requests'):
        server_args.append('--disable-log-requests')

    client_args = [
        f"""git clone https://github.com/inference-sim/vllm-data-collection
cd vllm-data-collection/scenario1
pip install -r requirements.txt
python scenario1_client.py --model {model}
        """
    ]

    print(f"Starting vLLM server: vllm serve {' '.join(server_args)}")

    config.load_kube_config()

    pod_name = f"vllm-benchmark-collection-{benchmark_name}-{mode}"

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
                    }
                ]
            }
        }
    
    # Apply the pod manifest
    resp = k_client.create_namespaced_pod(body=pod_manifest,
                                            namespace='llmdbench')      # edit ns if needed

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
        pod = Pod.get(pod_name, namespace='llmdbench')
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

def main():
    parser = argparse.ArgumentParser(description='Simple vLLM Benchmark Runner')
    parser.add_argument('--mode', help='train/test',  default="train")
    args = parser.parse_args()
    config_file = f"scenario1_config_{args.mode}.yaml"
    model = "facebook/opt-125m"

    with open(config_file, "r") as f:
       full_config = yaml.safe_load(f) # read necessary configs and seed files

    prompts_generation_command = [
        "python",
        "generate_prompts_fixedlen.py",
        "--model", model,
        "--mode", args.mode
    ]

    subprocess.run(prompts_generation_command, check=True)

    # Set up K8s client
    config.load_kube_config()
    try:
        c = Configuration().get_default_copy()
    except AttributeError:
        c = Configuration()
        c.assert_hostname = False
    Configuration.set_default(c)
    core_v1 = core_v1_api.CoreV1Api()

    # Run the workload
    print(f"\n{'='*50}")
    print(f"STARTING SCENARIO 1 benchmark in mode = {args.mode}")
    print(f"{'='*50}")

    benchmark_name = "scenario1"

    # Start server
    pod_name = start_vllm_server_client(full_config, benchmark_name, core_v1, args.mode, model)
    try:
        time.sleep(15)

        # Wait for server
        if not wait_for_server(full_config['model']):
            print("Skipping this run due to server startup failure")

    except KeyboardInterrupt:
        # stop server
        pod_log_file, metrics_log_file = collect_pod_logs_metrics(core_v1, benchmark_name, pod_name, args.mode)
        print(f"vLLM logs saved to: {pod_log_file}")
        print(f"vLLM metrics saved to: {metrics_log_file}")
        time.sleep(2) # give 2 seconds for server pod to spin down

if __name__=="__main__":
    main()