#!/usr/bin/env python3
"""
Simple vLLM Benchmark Runner
"""

import argparse
import json
import time
import yaml

from container_entrypoint import benchmark_wrapper

def load_config(config_file):
    """Load YAML configuration file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Simple vLLM Benchmark Runner')
    parser.add_argument('--config', help='Path to YAML config file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config = config['test']
    n = 1
    
    for benchmark, params in config.items():
        # spin up each benchmark with arguments to the entrypoint script
        # if n >= 156:
        params_json = json.dumps(params)
        benchmark_wrapper(params_json, benchmark)
        time.sleep(10)
        n+=1

if __name__ == '__main__':
    main()