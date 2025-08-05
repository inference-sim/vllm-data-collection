from datetime import datetime
import argparse
import json
import os
import requests
import subprocess
import time
from pathlib import Path
from kubernetes import config
from kubernetes.client import Configuration
from kubernetes.client.api import core_v1_api
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream
from kr8s.objects import Pod


def download_dataset(url, filename):
    if not os.path.exists(filename):
        try:
            print(f"Downloading {filename}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"Dataset Download complete! File saved as {filename}")

        except Exception as e:
            print(f"An error occurred: {e}")

def start_vllm_server(config, run, k_client):
    """Start vLLM server with config parameters"""
    model = config['model']
    vllm_params = config['vllm']

    args = [model,
        '--gpu-memory-utilization', str(vllm_params['gpu_memory_utilization']),
        '--block-size', str(vllm_params['block_size']),
        '--max-model-len', str(vllm_params['max_model_len']),
        '--max-num-batched-tokens', str(vllm_params['max_num_batched_tokens']),
        '--max-num-seqs', str(vllm_params['max_num_seqs']),
        '--long-prefill-token-threshold', str(vllm_params['long_prefill_token_threshold']),
        '--seed', str(vllm_params['seed'])
    ]

    if vllm_params.get('enable_prefix_caching'):
        args.append('--enable-prefix-caching')

    if vllm_params.get('disable_log_requests'):
        args.append('--disable-log-requests')

    print(f"Starting vLLM server: vllm serve {' '.join(args)}")

    with open(f'vllm_server_{run}.log', 'w') as log_file:

        pod_name = "vllm-benchmark-collection"

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
                                                "NVIDIA-H100-80GB-HBM3"      # edit this to land pod on the node with GPUs
                                            ]
                                        },
                                        # Short-term solution
                                        # TODO: ask how to set gpu memory for affinity
                                        # https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
                                        {
                                            "key": "kubernetes.io/hostname",
                                            "operator": "In",
                                            "values": [
                                                "pokprod-b93r38s2"      # edit this to land pod on the node with GPUs
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                },
                'containers': [
                    {
                        'name': 'benchmark',
                        'image': "vllm/vllm-openai:v0.10.0",
                        'command': ['vllm', 'serve'],
                        'args': args
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

def run_benchmark(config, output_folder, run_number):
    """Run benchmark and return output"""
    benchmark_params = config['benchmark']
    model = config['model']

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    cmd = [
        'python', 'benchmark_serving_simulator.py',
        '--backend', benchmark_params['backend'],
        '--model', model,
        '--dataset-name', benchmark_params['dataset_name'],
        '--num-prompts', str(benchmark_params['num_prompts']),
        '--request-rate', str(benchmark_params['request_rate']),
        '--temperature', str(benchmark_params['temperature']),
        '--seed', str(benchmark_params['seed']),
        '--save-result',
    ]

    # save request rate, temperature, dataset, distribution to env

    with open('benchmark_params.json', 'w') as f:
        json.dump({
            'request_rate': str(benchmark_params['request_rate']),
            'temperature': str(benchmark_params['temperature']),
            'distribution': "poisson",
            'dataset_name': "sharegpt"
        }, f, indent=4)

    if benchmark_params.get('dataset_path'):
        cmd.extend(['--dataset-path', benchmark_params['dataset_path']])

    if benchmark_params['dataset_name'] == "sharegpt" and int(benchmark_params['sharegpt_output_len']) > 0:
        cmd.extend(['--sharegpt-output-len', str(benchmark_params['sharegpt_output_len'])])

    print(f"Running benchmark (run {run_number}): {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Benchmark run {run_number} completed successfully")
        return result.stdout
    else:
        print(f"Benchmark run {run_number} failed: {result.stderr}")
        return None

def port_forward(k_client, pod_name):
    """
    Port forwards the vllm pod
    """

    ns = 'llmdbench'

    print("Waiting for vLLM pod to be Ready...")
    pod = Pod.get(pod_name, namespace=ns)

    pod.wait("condition=Ready")
    time.sleep(120)

    print("vLLM pod is Ready, port-forwarding now...")

    port = 8000
    pf = pod.portforward(remote_port=port, local_port=port)
    pf.start()
    return pf

def stop_vllm_server(k_client, pod_name, pf):
    """Stop vLLM server process"""

    print("Stopping vLLM server...")

    # Close port-forward
    pf.stop()

    # Delete pod
    try:
        api_response = k_client.delete_namespaced_pod(pod_name, 'llmdbench')
        print(f"vllm pod {pod_name} has been deleted: api response {api_response}")

    except ApiException as e:
        print("Exception when deleting vllm pod: %s\n" % e)

    print("Server stopped")

def save_results(outputs, params, output_folder):
    """Save all benchmark outputs to file"""
    benchmark_params = params['benchmark']

    config_name = f"{params['name']}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_folder) / f"{config_name}_{timestamp}.txt"

    with open(output_file, 'w') as f:
        f.write(f"Benchmark Results: {params['name']}\n")
        f.write(f"Config: {config_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Request Rate: {benchmark_params['request_rate']}\n")
        f.write(f"Num Prompts: {benchmark_params['num_prompts']}\n")
        f.write(f"Temperature: {benchmark_params['temperature']}\n")
        f.write(f"Dataset: {benchmark_params['dataset_name']}\n")
        f.write(f"Total Runs: {len(outputs)}\n")
        f.write("="*60 + "\n\n")

        for i, output in enumerate(outputs, 1):
            f.write(f"RUN {i}\n")
            f.write("-" * 20 + "\n")
            f.write(output)
            f.write("\n" + "="*60 + "\n\n")

    print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Run Container with different experiments")

    parser.add_argument(
        "--benchmark",
        type=str,
        default="baseline0",
        help="Baseline name"
    )

    parser.add_argument('--params', type=json.loads)

    args = parser.parse_args()

    # Download ShareGPT dataset
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    filename = "ShareGPT_V3_unfiltered_cleaned_split.json"
    download_dataset(url, filename)

    # Create output folder
    output_path = './outputs_vllm'
    os.makedirs(output_path, exist_ok=True)

    # Set up K8s client
    config.load_kube_config()
    try:
        c = Configuration().get_default_copy()
    except AttributeError:
        c = Configuration()
        c.assert_hostname = False
    Configuration.set_default(c)
    core_v1 = core_v1_api.CoreV1Api()

    # Run the baseline
    outputs = []
    runs = args.params['runs']
    for run in range(1, runs + 1):
        print(f"\n{'='*50}")
        print(f"STARTING RUN {run}/{args.params['runs']} for benchmark: {args.benchmark}")
        print(f"{'='*50}")

        # Start server
        pod_name = start_vllm_server(args.params, run, core_v1)
        try:

            time.sleep(15)

            # Port forward the vllm pod
            pf = port_forward(core_v1, pod_name)

            # Wait for server
            if not wait_for_server(args.params['model']):
                print("Skipping this run due to server startup failure")
                continue

            # Run benchmark
            output = run_benchmark(args.params, output_path, run)
            if output:
                outputs.append(output)

        finally:
            # Always stop server
            stop_vllm_server(core_v1, pod_name, pf)
            time.sleep(2)  # Brief pause between runs

    # Save all results
    if outputs:
        save_results(outputs, args.params, output_path)
    else:
        print("No successful runs to save")

if __name__ == '__main__':
    main()