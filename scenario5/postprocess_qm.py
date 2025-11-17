import argparse
import json
import yaml
import os
import sys
import pandas as pd
from postprocessing_utils import QM_TRAINING_FILEPATH, SWEEP_INFO_FILENAME
from postprocessing_utils import read_traces_jsonl, get_server_side_metrics_from_traces

def get_metrics_per_benchmark(benchmark_df, request_rate, vllm_config):
    """
    Get QM-style benchmark metrics in seconds.
    """
    benchmark_df['ITL'] = benchmark_df['decode_time'] / benchmark_df['output_tokens']
    all_means = benchmark_df[['input_tokens', 'output_tokens', 'queued_time', 'prefill_time', 'ITL']].mean()
    benchmark_averages = {
        "requestRate":    request_rate,
        "inputTokens":    all_means['input_tokens'],
        "outputTokens":   all_means['output_tokens'],
        "avgWaitTime":    all_means['queued_time'],
        "avgPrefillTime": all_means['prefill_time'],
        "avgITLTime":     all_means['ITL'],
        "maxBatchSize":   vllm_config['max_num_seqs']
    }
    return benchmark_averages

def perform_postprocessing_qm(traces_path, vllm_config_path, results_path):
    traces_raw_data = read_traces_jsonl(traces_path)
    all_requests = get_server_side_metrics_from_traces(traces_raw_data)
    requests_df = pd.DataFrame(all_requests)

    # read GuideLLM sweep info
    sweep_info_filepath = os.path.join(results_path, SWEEP_INFO_FILENAME)
    try:
        with open(sweep_info_filepath, 'r') as f:
            sweep_info = json.load(f)
    except:
        print("Could not read sweep info file.")
        sys.exit()

    # read vllm YAML config file
    try:
        with open(vllm_config_path, 'r') as f:
            vllm_config = yaml.safe_load(f)
    except:
        print("Could not read vllm config file.")
        sys.exit()

    qm_training_data = []
    for sweep in sweep_info:
        # each request-rate forms a new benchmark
        rps = sweep["rps"]
        benchmark_request_ids = sweep["requestIDs"]
        benchmark_df = requests_df[requests_df["request_id"].isin(benchmark_request_ids)].copy()
        benchmark_averages = get_metrics_per_benchmark(benchmark_df, rps, vllm_config)
        qm_training_data.append(benchmark_averages)
    
    # save postprocessed JSON
    qm_training_filename = os.path.join(results_path, QM_TRAINING_FILEPATH)
    with open(qm_training_filename, 'w+') as file:
        json.dump(qm_training_data, file, indent=4)
    print(f"QM Postprocessing complete. Saved to {qm_training_filename}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--traces", 
                        help="Path to the vllm traces file to be read.")
    parser.add_argument("--vllm_config", 
                        help="Path to vllm server config file.")
    parser.add_argument("--results_path",
                        default=".", 
                        help="Location to load intermediate files from")
    args = parser.parse_args()
    perform_postprocessing_qm(args.traces, args.vllm_config, args.results_path)