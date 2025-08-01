#!/usr/bin/env python3
"""
Simple vLLM Benchmark Runner
"""

import argparse
import json
import subprocess
import yaml


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
    
    for benchmark, params in config.items():
        # spin up each container with arguments to the entrypoint script
        params_json = json.dumps(params)
        cmd = [
            'python', 'container_entrypoint.py',
            '--params', str(params_json),
            '--benchmark', benchmark, # benchmark name, e.g: baseline, baseline0, etc.
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:", result.stdout)

if __name__ == '__main__':
    main()