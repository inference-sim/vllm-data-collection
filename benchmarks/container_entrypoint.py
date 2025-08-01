from datetime import datetime
import argparse
import json
import os
import requests
import subprocess
import time
from pathlib import Path

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

def start_vllm_server(config, run):
    """Start vLLM server with config parameters"""
    model = config['model']
    vllm_params = config['vllm']
    
    cmd = [
        'vllm', 'serve', model,
        '--gpu-memory-utilization', str(vllm_params['gpu_memory_utilization']),
        '--block-size', str(vllm_params['block_size']),
        '--max-model-len', str(vllm_params['max_model_len']),
        '--max-num-batched-tokens', str(vllm_params['max_num_batched_tokens']),
        '--max-num-seqs', str(vllm_params['max_num_seqs']),
        '--long-prefill-token-threshold', str(vllm_params['long_prefill_token_threshold']),
        '--seed', str(vllm_params['seed'])
        ]
    
    if vllm_params.get('enable_prefix_caching'):
        cmd.append('--enable-prefix-caching')
    
    if vllm_params.get('disable_log_requests'):
        cmd.append('--disable-log-requests')
        
    print(f"Starting vLLM server: {' '.join(cmd)}")

    with open(f'vllm_server_{run}.log', 'w') as log_file:
        # Redirect stdout and stderr to log file
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    # process = subprocess.Popen(cmd)
    return process

def wait_for_server(model: str):
    """Wait for vLLM server to be ready"""
    url = "http://127.0.0.1:8000/v1/models"
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

def stop_vllm_server(process):
    """Stop vLLM server process"""
    print("Stopping vLLM server...")
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
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

    # Download ShareGPT dataset into container
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    filename = "ShareGPT_V3_unfiltered_cleaned_split.json"
    download_dataset(url, filename)

    # Create output folder
    output_path = './outputs_vllm'
    os.makedirs(output_path, exist_ok=True)

    outputs = []
    runs = int(args.params['runs'])
    for run in range(1, runs + 1):
        print(f"\n{'='*50}")
        print(f"STARTING RUN {run}/{args.params['runs']} for benchmark: {args.benchmark}")
        print(f"{'='*50}")
        
        # Start server
        server_process = start_vllm_server(args.params, run)
        run += 1
        try:
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
            stop_vllm_server(server_process)
            time.sleep(2)  # Brief pause between runs

    # Save all results
    if outputs:
        save_results(outputs, args.params, output_path)
    else:
        print("No successful runs to save")

if __name__ == '__main__':
    main()