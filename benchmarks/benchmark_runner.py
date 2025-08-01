#!/usr/bin/env python3
"""
Simple vLLM Benchmark Runner
"""

import argparse
import os
import subprocess
import time
import yaml
import requests
from pathlib import Path
from datetime import datetime
import json


def load_config(config_file):
    """Load YAML configuration file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

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
        'python', 'benchmark_serving.py',
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

def save_results(outputs, config, config_file, output_folder):
    """Save all benchmark outputs to file"""
    benchmark_params = config['benchmark']

    config_name = f"{config['name']}_{benchmark_params['request_rate']}_{benchmark_params['num_prompts']}_{benchmark_params['temperature']}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_folder) / f"{config_name}_{timestamp}.txt"
    
    
    with open(output_file, 'w') as f:
        f.write(f"Benchmark Results: {config['name']}\n")
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
    parser = argparse.ArgumentParser(description='Simple vLLM Benchmark Runner')
    parser.add_argument('config', help='Path to YAML config file')
    # parser.add_argument('--runs', type=int, default=1, help='Number of benchmark runs')
    parser.add_argument('--output', default='./outputs', help='Output folder for results')
    
    args = parser.parse_args()
    
    # Create output folder
    os.makedirs(args.output, exist_ok=True)
    
    # Load config
    config = load_config(args.config)
    config = config['test']
    
    outputs = []
    curr_run = 1
    
    for benchmark, params in config.items():
        runs = params['runs']
        outputs = []
        for run in range(1, runs + 1):

            print(f"\n{'='*50}")
            print(f"STARTING RUN {run}/{params['runs']} for benchmark: {benchmark}")
            print(f"{'='*50}")
            
            # Start server
            server_process = start_vllm_server(params, curr_run)
            curr_run += 1
            try:
                # Wait for server
                if not wait_for_server(params['model']):
                    print("Skipping this run due to server startup failure")
                    continue
                
                # Run benchmark
                output = run_benchmark(params, args.output, run)
                if output:
                    outputs.append(output)
                
            finally:
                # Always stop server
                stop_vllm_server(server_process)
                time.sleep(2)  # Brief pause between runs
    
        # Save all results
        if outputs:
            save_results(outputs, params, args.config, args.output)
        else:
            print("No successful runs to save")

if __name__ == '__main__':
    main()