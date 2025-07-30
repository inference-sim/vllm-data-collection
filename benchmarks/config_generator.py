"""
Script to generate vLLM benchmark YAML configurations by sweeping over parameters.
"""

import yaml
from itertools import product

def generate_config(num_prompts_list, request_rate_list, temperature_list, datasets_list, 
                    max_num_batched_tokens_list, long_prefill_token_threshold_list, models):
    """
    Generate YAML configuration by sweeping over parameter combinations.
    
    Args:
        num_prompts_list: List of num_prompts values
        request_rate_list: List of request_rate values  
        temperature_list: List of temperature values
        datasets_list: List of dataset dictionaries with 'name' and 'path' keys
    """
    
    # Base configuration template
    base_config = {
        'test': {}
    }
    
    # Counter for experiment naming
    exp_counter = 1
    
    # Generate all parameter combinations
    for num_prompts, request_rate, temperature, dataset, max_num_batched_tokens, long_prefill_token_threshold, model in product(
        num_prompts_list, request_rate_list, temperature_list, datasets_list, 
        max_num_batched_tokens_list, long_prefill_token_threshold_list, models
    ):
        if long_prefill_token_threshold > max_num_batched_tokens:
            continue
        # Generate experiment name based on parameters
        exp_name = f"exp_{num_prompts}p_{request_rate}r_{temperature}t_{max_num_batched_tokens}mbt_{long_prefill_token_threshold}lpt_{dataset['name']}_{model.replace('/', '_')}"
        
        # Create experiment configuration
        exp_config = {
            'name': exp_name,
            'description': "Basic vLLM performance test",
            'model': model,
            'runs': 1,
            'vllm': {
                'gpu_memory_utilization': 0.9,
                'enable_prefix_caching': True,
                'disable_log_requests': False,
                'block_size': 16,
                'max_model_len': 2048,
                'max_num_batched_tokens': max_num_batched_tokens,
                'max_num_seqs': 256,
                'long_prefill_token_threshold': long_prefill_token_threshold,
                'seed': 42
            },
            'benchmark': {
                'backend': "vllm",
                'dataset_name': dataset['name'],
                'dataset_path': dataset['path'],
                'num_prompts': num_prompts,
                'request_rate': request_rate,
                'sharegpt_output_len': 0,
                'temperature': temperature,
                'seed': 42
            }
        }
        
        # Add to base config with baseline key
        baseline_key = f"baseline{exp_counter if exp_counter > 1 else ''}"
        base_config['test'][baseline_key] = exp_config
        exp_counter += 1
    
    return base_config

def main():
    # Define parameter sweep ranges
    num_prompts_list = [100, 400, 800, 1600]
    request_rate_list = [4, 8, 16, 32, 64, 128]
    temperature_list = [0.0]
    max_num_batched_tokens = [256, 1024, 8192]
    long_prefill_token_threshold = [16, 256, 1024]
    datasets_list = [
        {'name': 'sharegpt', 'path': 'ShareGPT_V3_unfiltered_cleaned_split.json'},
    ]
    # models = ['Qwen/Qwen2.5-0.5B', 'google/gemma-2b']
    models = ['Qwen/Qwen2-7B']
    
    # Generate configuration
    config = generate_config(num_prompts_list, request_rate_list, temperature_list, datasets_list, 
                             max_num_batched_tokens, long_prefill_token_threshold, models)
    
    # Write to YAML file
    with open('vllm_benchmark_config.yaml', 'w') as f:
        # Add header comment
        f.write("# vLLM Benchmark Configuration\n\n")
        f.write("# Model to benchmark\n\n\n")
        f.write("# Generated test configurations\n")
        
        # Write YAML content
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"Generated configuration with {len(config['test'])} experiments")
    print("Saved to: vllm_benchmark_config.yaml")

if __name__ == "__main__":
    main()