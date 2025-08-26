from jinja2 import Template, Environment, FileSystemLoader


job_name = "random-name"
server_args = """               mkdir -p
                touch
                python3 -m vllm.entrypoints.openai.api_server --model \\
                --gpu-memory-utilization {str(vllm_params['gpu_memory_utilization'])} \\
                --block-size {str(vllm_params['block_size'])} --max-model-len {str(vllm_params['max_model_len'])} \\
                --max-num-batched-tokens {str(vllm_params['max_num_batched_tokens'])} --max-num-seqs {str(vllm_params['max_num_seqs'])} \\
                --long-prefill-token-threshold {str(vllm_params['long_prefill_token_threshold'])} --seed {str(vllm_params['seed'])} \\
                --disable-log-requests
"""


client_args = """               set -ex
                apt-get update && apt-get install -y git curl
                git clone https://github.com/inference-sim/vllm-data-collection
                cd vllm-data-collection/scenario1
                pip install -r requirements.txt
                python generate_prompts_fixedlen.py --model {model} --mode {mode}
                touch {client_log_path}
                python scenario1_client.py --model {model} --mode {mode} --results_folder {exp_folder} > {client_log_path}
                sleep 30000000
"""

environment = Environment(loader=FileSystemLoader("./"))
template = environment.get_template("benchmark-job.yaml")

rendered = template.render(server_args=server_args,
                           client_args=client_args,
                           job_name=job_name)

print(rendered)